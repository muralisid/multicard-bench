"""E1c: does an LLM-designed, workload-conditioned taxonomy fix the SciFact loss?

The owner's conjecture, stated after reviewing the programme: the card taxonomy
was never meant to be static. In the production system the views are chosen with
an LLM, conditioned on the corpus and on the insights to be extracted, and the
static rhetorical taxonomy this programme hand-wrote for SciFact is precisely the
kind of misalignment the original validity analysis warned about. If that is
right, an LLM given the corpus and the query workload should design a taxonomy
that closes some or all of the gap to chunking.

Design. One generative call, before any evaluation: the model sees a sample of
corpus documents and a sample of TRAINING-split queries, and proposes 4 to 7
views with anchor phrases. Test queries are never shown to the designer, so the
evaluation stays clean. The proposed taxonomy then drives the same partitioning
card builder as every other arm, and everything downstream is identical.

Arms, all at document level, all with the same encoder:
  dense-pooled          one vector per document
  dense-chunk           blind windows, width matched to the LLM cards' units/doc
  cards-static          the hand-written rhetorical taxonomy (the arm that lost)
  cards-llm             the workload-conditioned taxonomy under test
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, AnchorCardBuilder, build_raw_chunks
from ..index.encoder import Encoder
from ..index.lexical import dense_rank, fold_to_docs
from ..llm.costmeter import CostMeter
from ..llm.vertex import GenerativeClient
from ..metrics.ranking import judged_at_k, ndcg_at_k, recall_at_k
from ..metrics.stats import compare, holm
from ..utils.seeds import set_seed
from .e1_beir import SCIENTIFIC_ASPECTS, _load

DESIGN_PROMPT = """You are designing an embedding index for a retrieval system.

Each document in the corpus will receive several separate embeddings, one per
"view". A view is a purpose-specific facet of a document. At query time the
system scores queries against every view embedding and takes the best match per
document. Views help when each query tends to target one facet of a document;
they hurt when the facets cut across what queries ask about.

Here is a sample of the corpus:

{docs}

Here is a sample of the query workload this index must serve:

{queries}

Design between 4 and 7 views for this corpus and this workload. For each view
give a short key, a label, and exactly 4 anchor phrases. An anchor phrase is a
short representative sentence of the kind of content that belongs to the view,
written in the corpus's own register. Choose views so that each query in the
workload has one view it naturally targets.

Reply with only a JSON array, no other text:
[{{"key": "...", "label": "...", "anchors": ["...", "...", "...", "..."]}}]"""


def design_taxonomy(gen, corpus_docs: list[str], train_queries: list[str],
                    seed: int) -> list[Aspect]:
    rng = np.random.default_rng(seed)
    doc_sample = [corpus_docs[i][:600] for i in
                  rng.choice(len(corpus_docs), size=min(30, len(corpus_docs)),
                             replace=False)]
    q_sample = [train_queries[i] for i in
                rng.choice(len(train_queries), size=min(25, len(train_queries)),
                           replace=False)]
    prompt = DESIGN_PROMPT.format(
        docs="\n".join(f"- {d}" for d in doc_sample),
        queries="\n".join(f"- {q}" for q in q_sample))
    r = gen.generate(prompt, max_output_tokens=2000)
    m = re.search(r"\[.*\]", r.text, re.S)
    if not m:
        raise RuntimeError(f"taxonomy designer returned no JSON: {r.text[:200]}")
    spec = json.loads(m.group(0))
    aspects = [Aspect(a["key"], a.get("label", a["key"]), list(a["anchors"])[:4])
               for a in spec if a.get("anchors")]
    if not 3 <= len(aspects) <= 8:
        raise RuntimeError(f"designer proposed {len(aspects)} views")
    return aspects


def run(n_docs: int = 0, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        design_model: str = "gemini-2.5-flash", collection: str = "scifact",
        max_usd: float = 2.0, out_dir: str | None = None) -> dict:
    set_seed(seed)
    out = Path(out_dir or f"results/e1_llm_taxonomy_{collection}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    corpus, queries, qrels = _load(collection)
    doc_ids = [str(r["_id"]) for r in corpus]
    doc_text = {
        str(r["_id"]): (f"{r['title']} {r['text']}" if r.get("title") else r["text"]).strip()
        for r in corpus
    }
    q_text = {str(r["_id"]): r["text"] for r in queries}
    test_qids = [q for q in q_text if q in qrels]

    # Train queries for the designer: the split the qrels loader does not serve.
    from datasets import load_dataset
    train_rows = load_dataset(f"BeIR/{collection}-qrels")["train"]
    # Sorted: iterating a set of strings follows the per-process hash seed, so
    # the seeded draw below sampled different queries on every run, the prompt
    # never hit the cache, and each run designed a different taxonomy.
    train_qids = sorted({str(r["query-id"]) for r in train_rows})
    train_queries = [q_text[q] for q in train_qids
                     if q in q_text and q not in qrels]
    print(f"designer sees {len(train_queries)} train queries; "
          f"evaluation uses {len(test_qids)} held-out test queries")

    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model=design_model, meter=meter, tier="vertex-flash")
    try:
        llm_aspects = design_taxonomy(gen, [doc_text[d] for d in doc_ids],
                                      train_queries, seed)
    finally:
        gen.close()
    print("LLM-designed views:", ", ".join(a.key for a in llm_aspects))
    (out / "llm_taxonomy.json").write_text(json.dumps(
        [{"key": a.key, "label": a.label, "anchors": a.anchors}
         for a in llm_aspects], indent=2))

    enc = Encoder(model_name=model)
    pairs = [(d, doc_text[d]) for d in doc_ids]

    def card_arm(aspects):
        b = AnchorCardBuilder(aspects, enc, partition=True)
        cards = b.build_many(pairs)
        ids = [c.card_id for c in cards]
        own = {c.card_id: c.doc_id for c in cards}
        vecs = enc.encode([c.text for c in cards])
        return ids, vecs, own, len(cards) / len(doc_ids)

    llm_ids, llm_vecs, llm_own, llm_upd = card_arm(llm_aspects)
    st_ids, st_vecs, st_own, st_upd = card_arm(SCIENTIFIC_ASPECTS)

    words = int(np.mean([len(doc_text[d].split()) for d in doc_ids]))
    chunk_size = max(20, int(words / max(1.0, llm_upd)))
    chunk_ids, chunk_texts, chunk_own = [], [], {}
    for d in doc_ids:
        for c in build_raw_chunks(d, doc_text[d], chunk_size, chunk_size // 4):
            chunk_ids.append(c.card_id)
            chunk_texts.append(c.text)
            chunk_own[c.card_id] = d
    chunk_vecs = enc.encode(chunk_texts)

    pooled = enc.encode([doc_text[d] for d in doc_ids])
    qv = enc.encode([q_text[q] for q in test_qids]).astype(np.float64)

    sims = {
        "dense-pooled": (qv @ pooled.astype(np.float64).T, None, None),
        "dense-chunk": (qv @ chunk_vecs.astype(np.float64).T, chunk_ids, chunk_own),
        "cards-static": (qv @ st_vecs.astype(np.float64).T, st_ids, st_own),
        "cards-llm": (qv @ llm_vecs.astype(np.float64).T, llm_ids, llm_own),
    }
    SYSTEMS = list(sims)
    depth = 100

    per_query, scores = [], {s: {"ndcg10": [], "recall10": [], "recall100": [],
                                 "judged10": []} for s in SYSTEMS}
    for qi, q in enumerate(test_qids):
        rel = qrels[q]
        row = {"query_id": q, "n_relevant": len(rel)}
        for s in SYSTEMS:
            sim, ids, own = sims[s]
            if ids is None:
                r = dense_rank(sim[qi], doc_ids, k=depth)
            else:
                pairs_q = sorted(zip(ids, sim[qi]), key=lambda kv: -kv[1])[:depth * 6]
                r = fold_to_docs(pairs_q, own)
            vals = {"ndcg10": ndcg_at_k(r, rel, 10),
                    "recall10": recall_at_k(r, rel, 10),
                    "recall100": recall_at_k(r, rel, 100),
                    "judged10": judged_at_k(r, rel, 10)}
            for m, v in vals.items():
                scores[s][m].append(v)
                row[f"{s}_{m}"] = round(v, 6)
        per_query.append(row)

    summary = {s: {m: float(np.mean(v)) for m, v in ms.items()}
               for s, ms in scores.items()}
    tests, pvals = {}, {}
    for name, a, b in (("llm_vs_chunk", "cards-llm", "dense-chunk"),
                       ("llm_vs_static", "cards-llm", "cards-static"),
                       ("llm_vs_pooled", "cards-llm", "dense-pooled"),
                       ("static_vs_chunk", "cards-static", "dense-chunk")):
        t = compare(scores[a]["ndcg10"], scores[b]["ndcg10"])
        tests[name] = t.as_dict()
        pvals[name] = t.p_value
    sig = holm(pvals)
    for k in tests:
        tests[k]["significant_holm_0.05"] = sig[k]

    result = {
        "experiment": f"e1_llm_taxonomy_{collection}",
        "collection": collection, "model": model, "design_model": design_model,
        "seed": seed, "n_docs": len(doc_ids), "n_test_queries": len(per_query),
        "n_train_queries_shown_to_designer": len(train_queries),
        "llm_taxonomy": [a.key for a in llm_aspects],
        "units_per_doc": {"cards_llm": llm_upd, "cards_static": st_upd,
                          "chunks": len(chunk_ids) / len(doc_ids)},
        "cost": meter.as_dict(),
        "wall_seconds": round(time.time() - t0, 1),
        "summary": summary, "significance": tests,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)

    print(f"\nunits/doc: llm {llm_upd:.1f}, static {st_upd:.1f}, "
          f"chunk {len(chunk_ids)/len(doc_ids):.1f}")
    print(f"{'system':<16}{'nDCG@10':>9}{'R@10':>8}{'R@100':>8}")
    for s in SYSTEMS:
        d = summary[s]
        print(f"{s:<16}{d['ndcg10']:>9.3f}{d['recall10']:>8.3f}{d['recall100']:>8.3f}")
    print()
    for k, t in tests.items():
        print(f"  {k:<18} {t['mean_delta']:+.3f} [{t['ci_low']:+.3f},{t['ci_high']:+.3f}]"
              f" p={t['p_value']:.4f} {t['wins']}W/{t['ties']}T/{t['losses']}L")
    return result

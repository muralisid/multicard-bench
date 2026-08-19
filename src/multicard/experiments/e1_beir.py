"""E1b: multi-card retrieval on real corpora with human relevance labels.

The synthetic corpus fixes the number of aspects by construction and LIMIT hands
over a perfect decomposition. Neither says what happens on prose that people
wrote for their own purposes, where the aspect structure is real but implicit and
the taxonomy has to be designed and may be wrong. This experiment supplies that
missing case using BEIR collections, which carry human relevance judgements, so
no model labels anything and the run costs nothing.

Default collection is SciFact: 5,183 scientific abstracts judged against 300 real
claims. A scientific abstract is genuinely multi-aspect, moving through
background, method, result and implication, and a claim usually matches one of
those parts rather than the whole, which is precisely the situation the pattern
addresses. Unlike LIMIT the boundaries are not marked, so the card builder has to
find them.

Systems, all scored at document level and all using the same encoder:
  bm25, dense-pooled, dense-chunk (the control), dense-multicard variants
  (extractive anchors and template framing), and hybrid fusion.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, AnchorCardBuilder, build_raw_chunks
from ..index.encoder import Encoder
from ..index.lexical import BM25, dense_rank, fold_to_docs
from ..metrics.ranking import (judged_at_k, ndcg_at_k, recall_at_k, rrf,
                               success_at_k)
from ..metrics.stats import compare, holm
from ..utils.seeds import set_seed

# A rhetorical taxonomy for scientific abstracts. Written from the standard
# structure of an abstract, not tuned on the evaluation, and reported in the
# paper as a design input that a reader can criticise.
SCIENTIFIC_ASPECTS = [
    Aspect("background", "Background and motivation", [
        "previous studies have suggested that",
        "the underlying mechanism remains poorly understood",
        "this condition affects a large population",
        "prior work established the context for this question",
    ]),
    Aspect("objective", "Objective and hypothesis", [
        "we hypothesised that the treatment would change the outcome",
        "the aim of this study was to determine whether",
        "we set out to test whether the association holds",
        "this study investigates the relationship between the two",
    ]),
    Aspect("method", "Methods and design", [
        "participants were randomly assigned to two groups",
        "we conducted a retrospective cohort analysis",
        "samples were analysed using the assay described",
        "the model was trained and evaluated on held out data",
    ]),
    Aspect("result", "Results and measurements", [
        "the treatment group showed a significant increase",
        "we observed a reduction compared with controls",
        "the difference was statistically significant",
        "levels were higher in the exposed group",
    ]),
    Aspect("conclusion", "Conclusion and implication", [
        "these findings suggest that the intervention is effective",
        "we conclude that the association is causal",
        "the results have implications for clinical practice",
        "further research is required to confirm this",
    ]),
]

SYSTEMS = ["bm25", "dense-pooled", "dense-chunk", "dense-multicard-extractive",
           "dense-multicard-template", "hybrid-rrf"]


def _load(collection: str):
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{collection}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{collection}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{collection}-qrels")
    split = "test" if "test" in qrels_ds else list(qrels_ds)[0]
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for r in qrels_ds[split]:
        if float(r["score"]) > 0:
            qrels[str(r["query-id"])][str(r["corpus-id"])] = float(r["score"])
    return corpus, queries, qrels


def run(n_docs: int = 0, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection: str = "scifact", chunk_words: int = 60, chunk_overlap: int = 20,
        spans_per_card: int = 2, out_dir: str | None = None) -> dict:
    set_seed(seed)
    out = Path(out_dir or f"results/e1_{collection}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    corpus, queries, qrels = _load(collection)
    doc_ids = [str(r["_id"]) for r in corpus]
    doc_text = {
        str(r["_id"]): (f"{r['title']} {r['text']}" if r.get("title") else r["text"]).strip()
        for r in corpus
    }
    q_text = {str(r["_id"]): r["text"] for r in queries}
    scored_qids = [q for q in q_text if q in qrels]

    enc = Encoder(model_name=model)

    # ---- representations -----------------------------------------------------
    pooled = enc.encode([doc_text[d] for d in doc_ids])

    chunk_ids, chunk_texts, chunk_own = [], [], {}
    for d in doc_ids:
        for c in build_raw_chunks(d, doc_text[d], chunk_words, chunk_overlap):
            chunk_ids.append(c.card_id)
            chunk_texts.append(c.text)
            chunk_own[c.card_id] = d
    chunks = enc.encode(chunk_texts)

    builder = AnchorCardBuilder(SCIENTIFIC_ASPECTS, enc, spans_per_card=spans_per_card)
    pairs = [(d, doc_text[d]) for d in doc_ids]
    card_sets = {}
    for method in ("extractive_anchor", "template_frame"):
        cards = builder.build_many(pairs, method=method)
        ids = [c.card_id for c in cards]
        own = {c.card_id: c.doc_id for c in cards}
        card_sets[method] = (ids, enc.encode([c.text for c in cards]), own)

    bm25 = BM25(doc_ids, [doc_text[d] for d in doc_ids])
    qv = enc.encode([q_text[q] for q in scored_qids]).astype(np.float64)

    sim_pooled = qv @ pooled.astype(np.float64).T
    sim_chunks = qv @ chunks.astype(np.float64).T
    sim_cards = {m: qv @ v.astype(np.float64).T for m, (_, v, _) in card_sets.items()}

    # ---- evaluate ------------------------------------------------------------
    depth = 100
    per_query, scores = [], {s: defaultdict(list) for s in SYSTEMS}

    def unit_rank(ids, sim_row, owner, take):
        pairs = sorted(zip(ids, sim_row), key=lambda kv: -kv[1])[:take]
        return fold_to_docs(pairs, owner)

    for qi, q in enumerate(scored_qids):
        rel = qrels[q]
        ranks = {
            "bm25": bm25.rank(q_text[q], k=depth),
            "dense-pooled": dense_rank(sim_pooled[qi], doc_ids, k=depth),
            "dense-chunk": unit_rank(chunk_ids, sim_chunks[qi], chunk_own, depth * 4),
            "dense-multicard-extractive": unit_rank(
                card_sets["extractive_anchor"][0], sim_cards["extractive_anchor"][qi],
                card_sets["extractive_anchor"][2], depth * 5),
            "dense-multicard-template": unit_rank(
                card_sets["template_frame"][0], sim_cards["template_frame"][qi],
                card_sets["template_frame"][2], depth * 5),
        }
        ranks["hybrid-rrf"] = rrf([ranks["dense-multicard-extractive"], ranks["bm25"]], k=60)

        row = {"query_id": q, "n_relevant": len(rel)}
        for s in SYSTEMS:
            r = ranks[s]
            vals = {
                "ndcg10": ndcg_at_k(r, rel, 10),
                "recall10": recall_at_k(r, rel, 10),
                "recall100": recall_at_k(r, rel, 100),
                "success10": success_at_k(r, rel, 10),
                "judged10": judged_at_k(r, rel, 10),
            }
            for m, v in vals.items():
                scores[s][m].append(v)
                row[f"{s}_{m}"] = round(v, 6)
        per_query.append(row)

    summary = {s: {m: float(np.mean(v)) for m, v in ms.items()} for s, ms in scores.items()}

    tests, pvals = {}, {}
    for s in SYSTEMS:
        if s == "dense-pooled":
            continue
        r = compare(scores[s]["ndcg10"], scores["dense-pooled"]["ndcg10"])
        tests[s] = r.as_dict()
        pvals[s] = r.p_value
    # The claim that separates alignment from capacity gets its own comparison.
    r = compare(scores["dense-multicard-extractive"]["ndcg10"], scores["dense-chunk"]["ndcg10"])
    tests["multicard_vs_chunk"] = r.as_dict()
    pvals["multicard_vs_chunk"] = r.p_value
    sig = holm(pvals)
    for s in tests:
        tests[s]["significant_holm_0.05"] = sig[s]

    result = {
        "experiment": f"e1_{collection}",
        "collection": collection,
        "model": model,
        "seed": seed,
        "n_docs": len(doc_ids),
        "n_queries_scored": len(per_query),
        "chunks_total": len(chunk_ids),
        "cards_total": len(card_sets["extractive_anchor"][0]),
        "cards_per_doc": len(card_sets["extractive_anchor"][0]) / max(1, len(doc_ids)),
        "chunks_per_doc": len(chunk_ids) / max(1, len(doc_ids)),
        "wall_seconds": round(time.time() - t0, 1),
        "summary": summary,
        "significance": tests,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)

    print(f"\n{collection}: {len(doc_ids)} docs, {len(per_query)} queries, "
          f"{result['cards_per_doc']:.1f} cards/doc, {result['chunks_per_doc']:.1f} chunks/doc\n")
    print(f"{'system':<28}{'nDCG@10':>9}{'R@10':>8}{'R@100':>8}{'J@10':>7}"
          f"{'vs pooled':>11}{'p':>9}")
    for s in SYSTEMS:
        d = summary[s]
        t = tests.get(s)
        extra = (f"{t['mean_delta']:>+11.3f}{t['p_value']:>9.4f}" if t
                 else f"{'baseline':>11}{'':>9}")
        print(f"{s:<28}{d['ndcg10']:>9.3f}{d['recall10']:>8.3f}{d['recall100']:>8.3f}"
              f"{d['judged10']:>7.3f}{extra}")
    mc = tests["multicard_vs_chunk"]
    print(f"\nalignment vs capacity: multicard minus chunk = {mc['mean_delta']:+.3f} "
          f"[{mc['ci_low']:+.3f},{mc['ci_high']:+.3f}], p={mc['p_value']:.4f}, "
          f"{mc['wins']}W/{mc['ties']}T/{mc['losses']}L")
    return result

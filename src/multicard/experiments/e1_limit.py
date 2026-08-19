"""E1a: the LIMIT capacity stress test.

LIMIT (Weller et al., arXiv:2508.21038) is built to expose the representational
ceiling of single-vector retrieval. Every document is a person followed by a long
list of things they like, and every query asks who likes one specific thing. With
only 46 documents in the small variant, the task looks trivial, yet single-vector
dense retrievers do badly on it, because one vector of fixed dimension cannot keep
every attribute of a many-attribute document separately addressable.

That makes it the cleanest possible test of the multi-card claim. The document
structure hands us the aspect decomposition: one card per attribute. If the
paper's thesis is right, splitting the same text into per-attribute cards and
scoring by the maximum should recover a large part of what the pooled vector
loses, at k times the storage and without any late-interaction machinery.

Systems compared, all scored at document level:
  bm25            lexical baseline
  dense-pooled    one embedding of the whole document
  dense-chunk     the document split into fixed-size word windows (the control
                  that separates "more embeddings" from "purpose-shaped cards")
  dense-multicard one embedding per attribute, maximum over cards
  hybrid-rrf      dense-multicard fused with bm25 by reciprocal rank fusion
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..index.encoder import Encoder
from ..index.lexical import BM25, dense_rank, fold_to_docs
from ..metrics.ranking import ndcg_at_k, recall_at_k, rrf, success_at_k
from ..metrics.stats import compare, holm
from ..utils.seeds import set_seed

SYSTEMS = ["bm25", "dense-pooled", "dense-chunk", "dense-multicard", "hybrid-rrf"]


def split_attributes(text: str) -> tuple[str, list[str]]:
    """Return (subject, attributes) for a LIMIT document.

    Documents read "<name> likes A, B, C". Splitting on the first "likes" keeps
    the subject attached to every card, which matters because a query names the
    attribute and expects the document about the person who has it.
    """
    m = re.split(r"\blikes\b", text, maxsplit=1)
    if len(m) != 2:
        return text.strip(), [text.strip()]
    subject = m[0].strip()
    attrs = [a.strip(" .,") for a in m[1].split(",")]
    return subject, [a for a in attrs if a]


def word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


def run(n_docs: int = 0, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dataset: str = "orionweller/LIMIT-small",
        chunk_words: int = 30, chunk_overlap: int = 10,
        out_dir: str = "results/e1_limit") -> dict:
    set_seed(seed)
    from datasets import load_dataset

    corpus = load_dataset(dataset, "corpus")["corpus"]
    queries = load_dataset(dataset, "queries")["queries"]
    qrels_raw = load_dataset(dataset)["test"]

    doc_ids = [r["_id"] for r in corpus]
    doc_text = {r["_id"]: (r["text"] or "").strip() for r in corpus}
    q_ids = [r["_id"] for r in queries]
    q_text = {r["_id"]: r["text"] for r in queries}

    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for r in qrels_raw:
        if float(r["score"]) > 0:
            qrels[r["query-id"]][r["corpus-id"]] = float(r["score"])

    enc = Encoder(model_name=model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- build the representations -----------------------------------------
    pooled = enc.encode([doc_text[d] for d in doc_ids])

    card_texts, card_owner_ids, card_ids = [], [], []
    n_attrs = []
    for d in doc_ids:
        subject, attrs = split_attributes(doc_text[d])
        n_attrs.append(len(attrs))
        for i, a in enumerate(attrs):
            card_ids.append(f"{d}##c{i}")
            card_texts.append(f"{subject} likes {a}")
            card_owner_ids.append(d)
    cards = enc.encode(card_texts)
    card_owner = dict(zip(card_ids, card_owner_ids))

    chunk_texts, chunk_ids, chunk_owner_ids = [], [], []
    for d in doc_ids:
        for i, c in enumerate(word_chunks(doc_text[d], chunk_words, chunk_overlap)):
            chunk_ids.append(f"{d}##k{i}")
            chunk_texts.append(c)
            chunk_owner_ids.append(d)
    chunks = enc.encode(chunk_texts)
    chunk_owner = dict(zip(chunk_ids, chunk_owner_ids))

    bm25 = BM25(doc_ids, [doc_text[d] for d in doc_ids])

    qv = enc.encode([q_text[q] for q in q_ids]).astype(np.float64)
    sim_pooled = qv @ pooled.astype(np.float64).T
    sim_cards = qv @ cards.astype(np.float64).T
    sim_chunks = qv @ chunks.astype(np.float64).T

    # ---- evaluate ------------------------------------------------------------
    depth = min(100, len(doc_ids))
    per_query: list[dict] = []
    scores: dict[str, dict[str, list[float]]] = {
        s: {"ndcg10": [], "recall10": [], "recall2": [], "success10": []} for s in SYSTEMS
    }

    for qi, q in enumerate(q_ids):
        rel = qrels.get(q, {})
        if not rel:
            continue
        ranks: dict[str, list[str]] = {}
        ranks["bm25"] = bm25.rank(q_text[q], k=depth)
        ranks["dense-pooled"] = dense_rank(sim_pooled[qi], doc_ids, k=depth)
        ranks["dense-chunk"] = fold_to_docs(
            sorted(zip(chunk_ids, sim_chunks[qi]), key=lambda kv: -kv[1])[:depth * 4],
            chunk_owner)
        ranks["dense-multicard"] = fold_to_docs(
            sorted(zip(card_ids, sim_cards[qi]), key=lambda kv: -kv[1])[:depth * 8],
            card_owner)
        ranks["hybrid-rrf"] = rrf([ranks["dense-multicard"], ranks["bm25"]], k=60)

        row = {"query_id": q, "n_relevant": len(rel)}
        for s in SYSTEMS:
            r = ranks[s]
            vals = {
                "ndcg10": ndcg_at_k(r, rel, 10),
                "recall10": recall_at_k(r, rel, 10),
                "recall2": recall_at_k(r, rel, 2),
                "success10": success_at_k(r, rel, 10),
            }
            for m, v in vals.items():
                scores[s][m].append(v)
                row[f"{s}_{m}"] = round(v, 6)
        per_query.append(row)

    summary = {
        s: {m: float(np.mean(v)) for m, v in ms.items()} for s, ms in scores.items()
    }

    # Significance against the pooled single-vector baseline.
    tests, pvals = {}, {}
    for s in SYSTEMS:
        if s == "dense-pooled":
            continue
        r = compare(scores[s]["ndcg10"], scores["dense-pooled"]["ndcg10"])
        tests[s] = r.as_dict()
        pvals[s] = r.p_value
    sig = holm(pvals)
    for s in tests:
        tests[s]["significant_holm_0.05"] = sig[s]

    result = {
        "experiment": "e1_limit",
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n_docs": len(doc_ids),
        "n_queries_scored": len(per_query),
        "attrs_per_doc_mean": float(np.mean(n_attrs)),
        "cards_total": len(card_ids),
        "chunks_total": len(chunk_ids),
        "chunk_words": chunk_words,
        "chunk_overlap": chunk_overlap,
        "wall_seconds": round(time.time() - t0, 1),
        "summary": summary,
        "significance_vs_dense_pooled": tests,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)

    print(f"\n{len(doc_ids)} documents, {np.mean(n_attrs):.1f} attributes each, "
          f"{len(card_ids)} cards, {len(chunk_ids)} chunks, "
          f"{len(per_query)} queries scored\n")
    print(f"{'system':<18}{'nDCG@10':>9}{'R@2':>8}{'R@10':>8}{'S@10':>8}"
          f"{'delta vs pooled':>18}{'p':>10}")
    for s in SYSTEMS:
        d = summary[s]
        if s in tests:
            t = tests[s]
            extra = f"{t['mean_delta']:>+18.3f}{t['p_value']:>10.4f}"
        else:
            extra = f"{'baseline':>18}{'':>10}"
        print(f"{s:<18}{d['ndcg10']:>9.3f}{d['recall2']:>8.3f}{d['recall10']:>8.3f}"
              f"{d['success10']:>8.3f}{extra}")
    return result

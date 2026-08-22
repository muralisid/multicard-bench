"""FollowIR loader (Weller et al., 2024, arXiv:2403.15246).

Three TREC collections re-annotated so that each query carries two instructions,
original and changed, with two sets of human relevance judgements. The changed
instruction only ever NARROWS relevance (changed relevant set is a subset of the
original), so the native question is instruction following: when the objective
narrows, does a retriever demote the documents that are no longer relevant.

The jhu-clsp release bundles a subsampled, judged corpus (tens of thousands of
documents, not the full TREC disks), which is redistributable and small enough to
run on a laptop. Nothing here needs an LDC or NIST licence.
"""

from __future__ import annotations

from collections import defaultdict

COLLECTIONS = ("core17", "news21", "robust04")


def load_followir(collection: str):
    if collection not in COLLECTIONS:
        raise ValueError(f"unknown FollowIR collection {collection!r}")
    from datasets import load_dataset

    name = f"jhu-clsp/{collection}-instructions"
    corpus_rows = load_dataset(name, "corpus")["corpus"]
    text = {}
    for r in corpus_rows:
        t = (f"{r['title']} {r['text']}" if r.get("title") else r["text"]).strip()
        text[str(r["_id"])] = t

    queries = []
    for r in load_dataset(name, "queries")["queries"]:
        queries.append({
            "qid": str(r["_id"]),
            "query": r["text"],
            "instruction_og": r["instruction_og"],
            "instruction_changed": r["instruction_changed"],
        })

    def _qrels(cfg):
        d = defaultdict(dict)
        for r in load_dataset(name, cfg)["test"]:
            s = float(r["score"])
            if s > 0:
                d[str(r["query-id"])][str(r["corpus-id"])] = s
        return d

    qrels_og = _qrels("qrels_og")
    qrels_changed = _qrels("qrels_changed")

    pools = defaultdict(list)
    for r in load_dataset(name, "top_ranked")["top_ranked"]:
        pools[str(r["qid"])].append(str(r["pid"]))
    # de-duplicate while keeping order, and keep only docs present in the corpus
    pools = {q: [d for d in dict.fromkeys(ids) if d in text] for q, ids in pools.items()}

    return {"text": text, "queries": queries, "qrels_og": qrels_og,
            "qrels_changed": qrels_changed, "pools": pools}

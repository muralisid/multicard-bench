"""Compute the MultiHop-RAG paper's own retrieval metrics on our arms.

The paper (arXiv 2401.15391, Table 5) reports MRR@10, MAP@10, Hits@10 and
Hits@4 over retrieved chunks against the ground-truth evidence, on the same
609 articles and the same non-null queries. Their chunk is 256 tokens; ours is
about 500 tokens with no overlap, so the chunk-level rows are close but not
identical in unit. The document-level rows have no such caveat: a document is
relevant when the query's evidence list names it.

These metrics read the ranking before any budget cut, which is what the paper
measures. Our own JointRecall is measured after the rendering budget, so the
two families answer different questions and both are reported.
"""
import json, sys
sys.path.insert(0, "src")
from multicard.data.multihoprag import load_queries

R = json.load(open("results/part1/mhrag/retrieve/rankings.json"))
queries = {q.qid: q for q in load_queries() if q.question_type != "null_query"}

def doc_of(unit_id):
    return unit_id.split("#", 1)[0]

def mrr(ranked, rel, k):
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0

def ap(ranked, rel, k):
    hits, s = 0, 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(rel), k) if rel else 0.0

def hits(ranked, rel, k):
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0

rows = []
for arm, per_q in R["4000"].items():
    m = {"mrr10": [], "map10": [], "h10": [], "h4": [], "n": 0,
         "u_mrr10": [], "u_map10": [], "u_h10": [], "u_h4": []}
    for qid, ranking in per_q.items():
        q = queries.get(qid)
        if q is None:
            continue
        rel_docs = set(q.evidence_doc_ids)
        # document level: fold the unit ranking to first appearance of each doc
        seen, doc_rank = set(), []
        for u in ranking:
            d = doc_of(u)
            if d not in seen:
                seen.add(d)
                doc_rank.append(d)
        m["mrr10"].append(mrr(doc_rank, rel_docs, 10))
        m["map10"].append(ap(doc_rank, rel_docs, 10))
        m["h10"].append(hits(doc_rank, rel_docs, 10))
        m["h4"].append(hits(doc_rank, rel_docs, 4))
        # unit level: a unit is relevant when its document is an evidence document
        rel_units = {u for u in ranking if doc_of(u) in rel_docs}
        m["u_mrr10"].append(mrr(ranking, rel_units, 10))
        m["u_map10"].append(ap(ranking, rel_units, 10))
        m["u_h10"].append(hits(ranking, rel_units, 10))
        m["u_h4"].append(hits(ranking, rel_units, 4))
        m["n"] += 1
    mean = lambda v: sum(v) / len(v) if v else 0.0
    rows.append((arm, m["n"], mean(m["mrr10"]), mean(m["map10"]), mean(m["h10"]), mean(m["h4"]),
                 mean(m["u_mrr10"]), mean(m["u_map10"]), mean(m["u_h10"]), mean(m["u_h4"])))

rows.sort(key=lambda r: -r[2])
print("Document level: a document is relevant when the query's evidence list names it.")
print(f"{'arm':22s} {'n':>5s} {'MRR@10':>7s} {'MAP@10':>7s} {'Hits@10':>8s} {'Hits@4':>7s}")
for a, n, mr, mp, h10, h4, *_ in rows:
    print(f"{a:22s} {n:5d} {mr:7.4f} {mp:7.4f} {h10:8.4f} {h4:7.4f}")
print()
print("Unit level: a retrieved chunk is relevant when it belongs to an evidence document.")
print(f"{'arm':22s} {'n':>5s} {'MRR@10':>7s} {'MAP@10':>7s} {'Hits@10':>8s} {'Hits@4':>7s}")
for a, n, _, _, _, _, mr, mp, h10, h4 in rows:
    print(f"{a:22s} {n:5d} {mr:7.4f} {mp:7.4f} {h10:8.4f} {h4:7.4f}")
print()
print("Published, MultiHop-RAG paper Table 5, best row (voyage-02 with bge-reranker-large):")
print(f"{'voyage-02 + reranker':22s} {2255:5d} {0.5860:7.4f} {0.4795:7.4f} {0.7467:8.4f} {0.6625:7.4f}")

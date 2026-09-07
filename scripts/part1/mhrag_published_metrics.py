"""The MultiHop-RAG paper's own retrieval metrics on our arms.

arXiv 2401.15391 Table 5 reports MRR@10, MAP@10, Hits@10 and Hits@4 over the
same 609 articles and the same non-null queries. Its definitions, taken from
the paper and applied here exactly:

  Hit@K   the mean FRACTION of a query's evidence that appears in the top K.
          This is a recall, not a hit indicator. An earlier version of this
          script computed "at least one evidence item in the top K", which is
          a different and much easier quantity; the final verification of
          2026-09-07 caught it and the numbers below are the corrected ones.
  MAP@K   mean average precision, precision summed at each relevant position
          and divided by the number of relevant items that exist (capped at K).
  MRR@K   reciprocal rank of the first relevant item.

Their chunk is 256 tokens; ours is about 500 with no overlap, so the unit-level
rows are close in kind but not identical. The document-level rows are a
different unit again and are labelled as such.

Everything is computed on the ranking before any budget cut, which is what the
paper measures.
"""
import json, sys
sys.path.insert(0, "src")
from multicard.data.multihoprag import load_queries

R = json.load(open("results/part1/mhrag/retrieve/rankings.json"))["4000"]
queries = {q.qid: q for q in load_queries() if q.question_type != "null_query"}
doc_of = lambda u: u.split("#", 1)[0]

def mrr(ranked, rel, k):
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0

def ap(ranked, rel, k):
    """Precision summed at relevant positions, over the relevant items that
    exist, capped at k. The standard denominator."""
    if not rel:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(rel), k)

def hit_fraction(ranked, rel, k):
    """The paper's Hit@K: the fraction of the evidence that appears in top k."""
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0

def any_hit(ranked, rel, k):
    """Kept only so the two quantities can be told apart in the output."""
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0

rows = []
for arm, per_q in R.items():
    acc = {k: [] for k in ("mrr10", "map10", "h10", "h4", "any10", "any4",
                           "u_mrr10", "u_map10", "u_h10", "u_h4")}
    n = 0
    for qid, ranking in per_q.items():
        q = queries.get(qid)
        if q is None:
            continue
        rel_docs = set(q.evidence_doc_ids)
        seen, doc_rank = set(), []
        for u in ranking:
            d = doc_of(u)
            if d not in seen:
                seen.add(d); doc_rank.append(d)
        acc["mrr10"].append(mrr(doc_rank, rel_docs, 10))
        acc["map10"].append(ap(doc_rank, rel_docs, 10))
        acc["h10"].append(hit_fraction(doc_rank, rel_docs, 10))
        acc["h4"].append(hit_fraction(doc_rank, rel_docs, 4))
        acc["any10"].append(any_hit(doc_rank, rel_docs, 10))
        acc["any4"].append(any_hit(doc_rank, rel_docs, 4))
        rel_units = {u for u in ranking if doc_of(u) in rel_docs}
        acc["u_mrr10"].append(mrr(ranking, rel_units, 10))
        acc["u_map10"].append(ap(ranking, rel_units, 10))
        acc["u_h10"].append(hit_fraction(ranking, rel_units, 10))
        acc["u_h4"].append(hit_fraction(ranking, rel_units, 4))
        n += 1
    mean = lambda v: sum(v) / len(v) if v else 0.0
    rows.append((arm, n, {k: mean(v) for k, v in acc.items()}))

rows.sort(key=lambda r: -r[2]["mrr10"])
print("Document level, the paper's definitions. n = 2,255.")
print(f"{'arm':22s} {'MRR@10':>7s} {'MAP@10':>7s} {'Hits@10':>8s} {'Hits@4':>7s}   "
      f"{'anyEv@10':>8s} {'anyEv@4':>8s}")
for a, n, m in rows:
    print(f"{a:22s} {m['mrr10']:7.4f} {m['map10']:7.4f} {m['h10']:8.4f} {m['h4']:7.4f}   "
          f"{m['any10']:8.4f} {m['any4']:8.4f}")
print("\nUnit level, closest in kind to the paper's chunk rows.")
print(f"{'arm':22s} {'MRR@10':>7s} {'MAP@10':>7s} {'Hits@10':>8s} {'Hits@4':>7s}")
for a, n, m in rows:
    print(f"{a:22s} {m['u_mrr10']:7.4f} {m['u_map10']:7.4f} {m['u_h10']:8.4f} {m['u_h4']:7.4f}")
print("\nPublished, Table 5, best row (voyage-02 with bge-reranker-large):")
print(f"{'voyage-02 + reranker':22s} {0.5860:7.4f} {0.4795:7.4f} {0.7467:8.4f} {0.6625:7.4f}")

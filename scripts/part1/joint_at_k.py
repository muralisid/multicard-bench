"""Separate ranking quality from the rendering budget on MultiHop-RAG.

Joint hit at k: EVERY evidence document of the query appears in the top k of
the ranking, before any budget cut. This is the ranking-level twin of our
JointRecall at a token budget, so the gap between them is exactly what the
budget costs.
"""
import json, sys, collections
sys.path.insert(0, "src")
from multicard.data.multihoprag import load_queries

R = json.load(open("results/part1/mhrag/retrieve/rankings.json"))["4000"]
queries = {q.qid: q for q in load_queries() if q.question_type != "null_query"}
doc_of = lambda u: u.split("#", 1)[0]

KS = [4, 10, 20, 50, 100]
print(f"{'arm':22s} " + " ".join(f"joint@{k:<3d}" for k in KS) + "  JR@4k  JR@8k")
budget = {}
for b in ("4000", "8000"):
    m = json.load(open("results/part1/mhrag/retrieve/metrics.json"))["summary"][b]
    budget[b] = {a: v.get("fact_joint_recall") for a, v in m.items() if isinstance(v, dict)}

by_type = collections.defaultdict(lambda: collections.defaultdict(list))
rows = []
for arm, per_q in R.items():
    scores = {k: [] for k in KS}
    for qid, ranking in per_q.items():
        q = queries.get(qid)
        if q is None:
            continue
        rel = set(q.evidence_doc_ids)
        seen, doc_rank = set(), []
        for u in ranking:
            d = doc_of(u)
            if d not in seen:
                seen.add(d)
                doc_rank.append(d)
        for k in KS:
            hit = 1.0 if rel <= set(doc_rank[:k]) else 0.0
            scores[k].append(hit)
            if k == 10:
                by_type[arm][q.question_type].append(hit)
    mean = lambda v: sum(v) / len(v) if v else 0.0
    rows.append((arm, [mean(scores[k]) for k in KS],
                 budget["4000"].get(arm), budget["8000"].get(arm)))
rows.sort(key=lambda r: -r[1][1])
for arm, s, j4, j8 in rows:
    print(f"{arm:22s} " + " ".join(f"{x:8.3f}" for x in s) +
          f"  {j4 if j4 is not None else float('nan'):.3f}  {j8 if j8 is not None else float('nan'):.3f}")

print("\njoint@10 by question type (n: comparison 856, inference 816, temporal 583)")
types = sorted({t for a in by_type for t in by_type[a]})
print(f"{'arm':22s} " + " ".join(f"{t.split('_')[0]:>12s}" for t in types))
for arm, *_ in rows:
    print(f"{arm:22s} " + " ".join(f"{sum(by_type[arm][t])/len(by_type[arm][t]):12.3f}" for t in types))

print("\nEvidence documents needed per query, and joint@10 for ours_cheap by that count:")
need = collections.defaultdict(list)
for qid, ranking in R["ours_cheap"].items():
    q = queries.get(qid)
    if q is None:
        continue
    rel = set(q.evidence_doc_ids)
    seen, doc_rank = set(), []
    for u in ranking:
        d = doc_of(u)
        if d not in seen:
            seen.add(d); doc_rank.append(d)
    need[len(rel)].append(1.0 if rel <= set(doc_rank[:10]) else 0.0)
for n in sorted(need):
    v = need[n]
    print(f"  {n} documents: {len(v):5d} queries, joint@10 {sum(v)/len(v):.3f}")

"""E3a: the retrieval side of consumer-dependent diversity.

The study's novel claim has two halves. The second half, that a generative
consumer benefits from coverage while a human reader is served better by
relevance, needs downstream judgement and therefore costs money. The first half
does not: whether a policy actually delivers coverage, and at what cost in
relevance, is measurable directly, and that is what this experiment settles.

Ground truth comes from construction rather than from a model. Each query is
associated with several distinct facts, and each document realises one of them,
so the subtopics a selection covers can be counted exactly. That makes
alpha-nDCG and subtopic recall meaningful rather than approximate, and it removes
any circularity that would follow from deriving subtopics with the same
clustering the diversity policy uses.

What the experiment cannot show on its own is which policy is right, because that
depends on the consumer. It measures the trade a decision-maker is choosing
between; the consumer study assigns it a value.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from ..data.synthetic import POOLS, POOL_NAMES, _passage
from ..index.encoder import Encoder
from ..metrics.diversity import alpha_ndcg, intra_list_distance, subtopic_recall
from ..metrics.ranking import ndcg_at_k, recall_at_k
from ..metrics.stats import compare, holm
from ..select.policies import (cluster_round_robin, coverage_stratified,
                               dpp_greedy, mmr, outlier_harvest, top_k)
from ..utils.seeds import rng, set_seed

POLICY_ORDER = ["top_k", "mmr_0.7", "mmr_0.3", "dpp", "cluster_rr",
                "outlier_0.2", "outlier_0.4"]


def build_diverse_task(n_queries: int = 120, docs_per_query: int = 60,
                       facts_per_query: int = 6, distractors_per_query: int = 40,
                       seed: int = 13):
    """A broad query whose relevant material spans several distinct facts.

    Each fact is realised by several documents, so a selection that takes only
    the most similar documents will cover few facts, while one that spends part
    of its budget away from the centre can cover many. Distractor documents are
    on topic but realise no queried fact, which is what a gate would leave in the
    pool and what a diversity policy risks promoting.
    """
    r = rng(seed)
    tasks = []
    for qi in range(n_queries):
        pool = POOL_NAMES[int(r.integers(len(POOL_NAMES)))]
        spec = POOLS[pool]
        facts = []
        for _ in range(facts_per_query):
            facts.append((pool, spec["a"][r.integers(len(spec["a"]))],
                          spec["b"][r.integers(len(spec["b"]))]))
        facts = list(dict.fromkeys(facts))

        docs, subtopics = [], {}
        per_fact = max(1, docs_per_query // max(1, len(facts)))
        for fi, fact in enumerate(facts):
            for j in range(per_fact):
                did = f"q{qi}_f{fi}_{j}"
                docs.append((did, _passage(pool, fact, r, sentences=3)))
                subtopics[did] = {fi}
        # A distractor pool large enough that a policy can actually lose by
        # selecting from it. An earlier version divided a corpus-wide count by the
        # query count and produced two per task, which saturated every relevance
        # metric and made "diversity is nearly free" an artefact of there being
        # almost nothing non-relevant to pick.
        for j in range(distractors_per_query):
            did = f"q{qi}_x{j}"
            other = (pool, spec["a"][r.integers(len(spec["a"]))],
                     spec["b"][r.integers(len(spec["b"]))])
            if other in facts:
                continue
            docs.append((did, _passage(pool, other, r, sentences=3)))
            subtopics[did] = set()

        # The query must be about the subtopics it is scored against. Drawing its
        # attribute from the pool at large left 27 percent of tasks asking about
        # something no subtopic-bearing document discussed.
        query_attr = facts[int(r.integers(len(facts)))][1]
        query = f"Summarise everything about {query_attr} in this material."
        tasks.append({
            "qid": f"q{qi:04d}",
            "query": query,
            "docs": docs,
            "subtopics": subtopics,
            "all_subtopics": set(range(len(facts))),
        })
    return tasks


def run(n_docs: int = 0, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        budget: int = 10, out_dir: str = "results/e3_diversity") -> dict:
    set_seed(seed)
    enc = Encoder(model_name=model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tasks = build_diverse_task(seed=seed)
    per_query, scores = [], {p: {"alpha_ndcg": [], "s_recall": [], "ndcg10": [],
                                 "recall10": [], "ild": [], "distractor_rate": []}
                             for p in POLICY_ORDER}

    for t in tasks:
        ids = [d for d, _ in t["docs"]]
        vecs = enc.encode([x for _, x in t["docs"]]).astype(np.float64)
        qv = enc.encode([t["query"]]).astype(np.float64)[0]
        rel_scores = vecs @ qv

        # A gated pool: keep the upper half by relevance, as pass one would.
        keep = np.argsort(-rel_scores)[: max(budget * 3, len(ids) // 2)]
        pool_ids = [ids[i] for i in keep]
        pool_vecs = vecs[keep]
        pool_rel = rel_scores[keep]

        # Cluster structure for the round-robin policy, from the pool itself.
        from sklearn.cluster import KMeans
        n_clusters = min(8, max(2, len(pool_ids) // 8))
        labels = KMeans(n_clusters=n_clusters, n_init=10,
                        random_state=seed).fit_predict(pool_vecs)

        selections = {
            "top_k": top_k(pool_rel, budget),
            "mmr_0.7": mmr(pool_rel, pool_vecs, budget, lambda_=0.7),
            "mmr_0.3": mmr(pool_rel, pool_vecs, budget, lambda_=0.3),
            "dpp": dpp_greedy(pool_rel, pool_vecs, budget),
            "cluster_rr": cluster_round_robin(pool_rel, labels, budget),
            "outlier_0.2": outlier_harvest(pool_rel, budget, fraction=0.2),
            "outlier_0.4": outlier_harvest(pool_rel, budget, fraction=0.4),
        }

        rel_qrels = {d: 1.0 for d, s in t["subtopics"].items() if s}
        vec_by_id = {pool_ids[i]: pool_vecs[i] for i in range(len(pool_ids))}
        row = {"query_id": t["qid"]}
        for name in POLICY_ORDER:
            sel = [pool_ids[i] for i in selections[name]]
            vals = {
                "alpha_ndcg": alpha_ndcg(sel, t["subtopics"], k=budget,
                                         candidates=pool_ids),
                "s_recall": subtopic_recall(sel, t["subtopics"],
                                            t["all_subtopics"], k=budget),
                "ndcg10": ndcg_at_k(sel, rel_qrels, budget),
                "recall10": recall_at_k(sel, rel_qrels, budget),
                "ild": intra_list_distance(sel, vec_by_id, k=budget),
                # Fraction of the selection that carries no queried fact: the
                # answerless-adjacent risk a diversity policy takes on.
                "distractor_rate": float(
                    np.mean([1.0 if not t["subtopics"].get(d) else 0.0 for d in sel])),
            }
            for m, v in vals.items():
                scores[name][m].append(v)
                row[f"{name}_{m}"] = round(float(v), 6)
        per_query.append(row)

    summary = {p: {m: float(np.mean(v)) for m, v in ms.items()}
               for p, ms in scores.items()}

    tests, pvals = {}, {}
    for p in POLICY_ORDER:
        if p == "top_k":
            continue
        for metric in ("s_recall", "ndcg10"):
            r = compare(scores[p][metric], scores["top_k"][metric])
            tests[f"{p}::{metric}"] = r.as_dict()
            pvals[f"{p}::{metric}"] = r.p_value
    sig = holm(pvals)
    for k in tests:
        tests[k]["significant_holm_0.05"] = sig[k]

    result = {
        "experiment": "e3_diversity",
        "model": model,
        "seed": seed,
        "budget": budget,
        "n_queries": len(tasks),
        "wall_seconds": round(time.time() - t0, 1),
        "summary": summary,
        "significance_vs_top_k": tests,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)

    print(f"\n{len(tasks)} queries, budget {budget} items per selection\n")
    print(f"{'policy':<14}{'S-recall':>10}{'a-nDCG':>9}{'nDCG':>8}{'recall':>8}"
          f"{'ILD':>7}{'distract':>10}")
    for p in POLICY_ORDER:
        d = summary[p]
        print(f"{p:<14}{d['s_recall']:>10.3f}{d['alpha_ndcg']:>9.3f}{d['ndcg10']:>8.3f}"
              f"{d['recall10']:>8.3f}{d['ild']:>7.3f}{d['distractor_rate']:>10.3f}")
    print("\ncoverage gained and relevance given up, against pure relevance:")
    for p in POLICY_ORDER:
        if p == "top_k":
            continue
        cov = tests[f"{p}::s_recall"]
        rel = tests[f"{p}::ndcg10"]
        print(f"  {p:<14} S-recall {cov['mean_delta']:+.3f} "
              f"[{cov['ci_low']:+.3f},{cov['ci_high']:+.3f}] p={cov['p_value']:.4f}   "
              f"nDCG {rel['mean_delta']:+.3f} p={rel['p_value']:.4f}")
    return result

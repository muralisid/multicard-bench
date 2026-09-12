"""Paired comparisons only on identical saved question IDs."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from multicard.version_a.core import read_jsonl
from .common import atomic_json


def bootstrap(pairs):
    groups = defaultdict(list)
    for group, difference in pairs:
        groups[group].append(difference)
    if len(groups) < 2:
        return None, None
    values = list(groups.values())
    sums = np.asarray([sum(v) for v in values])
    sizes = np.asarray([len(v) for v in values])
    samples = np.random.default_rng(13).integers(0, len(values), (2000, len(values)))
    means = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def report(output):
    output = Path(output)
    rows, details = [], []
    for directory in sorted(p.parent for p in output.glob("*/sample.json")):
        sample = json.loads((directory / "sample.json").read_text())
        qs = {q["id"]: q for q in sample["questions"]}
        group_sizes = defaultdict(int)
        for q in qs.values():
            group_sizes[q["group"]] += 1
        preparations = [json.loads(p.read_text()) for p in (directory / "groups").glob("*.json")]
        arms = {}
        for arm in "ABCD":
            path = directory / f"answers_{arm}.jsonl"
            arms[arm] = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
        cohorts = {"headline": [q for q in qs.values() if q.get("headline_eligible", True)]}
        if any(not q.get("headline_eligible", True) for q in qs.values()):
            cohorts["adversarial"] = [q for q in qs.values() if not q.get("headline_eligible", True)]
        for cohort, questions in cohorts.items():
            cohort_groups = defaultdict(int)
            for q in questions:
                cohort_groups[q["group"]] += 1
            setup_cost = sum(sum(call["usd"] for call in group["extraction"])
                             * cohort_groups[group["group"]] / group_sizes[group["group"]]
                             for group in preparations)
            for arm, reference in (("B", "A"), ("C", "B"), ("D", "C")):
                ids = [q["id"] for q in questions if q["id"] in arms[arm] and q["id"] in arms[reference]]
                differences = [(qs[i]["group"], arms[arm][i]["score"] - arms[reference][i]["score"]) for i in ids]
                lo, hi = bootstrap(differences)
                score = float(np.mean([arms[arm][i]["score"] for i in ids])) if ids else None
                base = float(np.mean([arms[reference][i]["score"] for i in ids])) if ids else None
                a_score = float(np.mean([arms["A"][i]["score"] for i in ids])) if ids and all(i in arms["A"] for i in ids) else None
                correct = sum(arms[arm][i].get("correct") or 0 for i in ids)
                cost = sum(arms[arm][i]["answer_usd"] for i in ids)
                row = {"dataset": directory.name, "cohort": cohort, "comparison": f"{arm}-{reference}",
                       "paired": len(ids), "requested": len(questions), "full_dataset": sample["full_count"],
                       "scope": "full" if sample["full_count"] == len(qs) else "screen",
                       "status": "complete" if len(ids) == len(questions) else "incomplete",
                       "metric": questions[0]["metric"], "reference_score": base, "score": score,
                       "delta": score - base if ids else None, "cluster_ci_low": lo, "cluster_ci_high": hi,
                       "A_score": a_score, "delta_vs_A": score - a_score if a_score is not None else None,
                       "answer_usd": cost, "judge_usd": sum(arms[arm][i]["judge_usd"] for i in ids),
                       "standalone_shared_fact_setup_usd": setup_cost,
                       "answer_plus_setup_usd": cost + setup_cost,
                       "total_serving_cost_per_correct": (cost + setup_cost) / correct if correct and len(ids) == len(questions)
                                                          and questions[0]["metric"] != "beam_rubric_mean" else None,
                       "cost_per_correct": cost / correct if correct and questions[0]["metric"] != "beam_rubric_mean" else None,
                       "correct_basis": "exact_match" if questions[0]["metric"] == "f1" else None
                                        if questions[0]["metric"] == "beam_rubric_mean" else "binary_correct",
                       "changed_contexts": sum(arms[arm][i].get("context_changed", False) for i in ids),
                       "empty_answers": sum(arms[arm][i].get("supplementary", {}).get("empty_answer", False) for i in ids)}
                rows.append(row)
                for kind in sorted({qs[i]["type"] for i in ids}):
                    chosen = [i for i in ids if qs[i]["type"] == kind]
                    details.append({"dataset": directory.name, "comparison": f"{arm}-{reference}", "type": kind,
                                    "paired": len(chosen), "delta": float(np.mean([arms[arm][i]["score"] - arms[reference][i]["score"] for i in chosen]))})
    budget = []
    db = output / "_meter/calls.sqlite"
    if db.exists():
        with sqlite3.connect(db) as c:
            budget = [dict(zip(("run", "role", "status", "calls", "usd"), r)) for r in c.execute(
                "SELECT run,role,status,count(*),sum(usd) FROM calls GROUP BY run,role,status")]
    atomic_json(output / "comparison.json", {"comparisons": rows, "by_type": details, "budget": budget})
    for name, values in (("COMPARISON.csv", rows), ("BY_TYPE.csv", details), ("COSTS.csv", budget)):
        if values:
            temporary = output / (name + ".tmp")
            with temporary.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
            temporary.replace(output / name)
    def fmt(value):
        return "pending" if value is None else f"{value:.4f}"
    lines = ["# B/C/D paired hypothesis screen", "", "BEAM 10M excluded. Same sampled questions, reader, judge and 4,000-token evidence budget. Larger-dataset samples are not full benchmark results.", "",
             "| Dataset | Cohort | Comparison | Scope | Paired/requested | Metric | Reference | New | Delta | Delta vs A | Fact setup USD | Answer USD | Judge USD |",
             "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['cohort']} | {r['comparison']} | {r['scope']}/{r['status']} | {r['paired']}/{r['requested']} | {r['metric']} | {fmt(r['reference_score'])} | {fmt(r['score'])} | {fmt(r['delta'])} | {fmt(r['delta_vs_A'])} | {fmt(r['standalone_shared_fact_setup_usd'])} | {fmt(r['answer_usd'])} | {fmt(r['judge_usd'])} |")
    by_role = defaultdict(float)
    for row in budget:
        by_role[row["role"] if row["status"] == "done" else "reserved"] += row["usd"]
    lines += ["", "Measured new API spend: " + ", ".join(f"{key}=${value:.4f}" for key, value in sorted(by_role.items())) + ".",
              "", "Fact setup is the shared extraction cost allocated to this cohort by history/question count. It is shown for each arm as a standalone deployment cost; the experiment pays it once across B/C/D, so do not sum it across arms. CSV includes answering plus setup cost per correct answer. Judge cost is evaluation overhead. Logical serving cost includes cache reuse; actual incremental spend is in COSTS.csv. Local compute is recorded in preparation artifacts, not priced as free.",
              "", "CSV includes exploratory 95% bootstrap intervals clustered by history, intervention counts and cost per correct. A single shared corpus has no cluster interval. F1 is separate from exact match; BEAM has no binary cost-per-correct. D is a graph of discovered evidence, not the entire corpus. No cross-dataset average or superiority claim is made."]
    temporary = output / "COMPARISON.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output / "COMPARISON.md")
    return rows

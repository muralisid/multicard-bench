"""Read-only recomputation of A and matched B/C/D results; no model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/version_a"
BCD = ROOT / "results/version_abcd"
OUT = ROOT / "outputs/benchmark-report-20260908"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def indexed(values):
    result = {v["id"]: v for v in values}
    assert len(result) == len(values), "Duplicate question IDs"
    return result


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def quantile(values, q):
    return float(np.quantile(values, q)) if values else None


def close(x, y):
    assert x is not None and y is not None and abs(x - y) < 1e-9, (x, y)


def ci(pairs):
    groups = defaultdict(list)
    for group, delta in pairs:
        groups[group].append(delta)
    if len(groups) < 2:
        return None, None
    values = list(groups.values())
    sums, sizes = np.array([sum(v) for v in values]), np.array([len(v) for v in values])
    choices = np.random.default_rng(13).integers(len(values), size=(2000, len(values)))
    return [float(v) for v in np.quantile(sums[choices].sum(1) / sizes[choices].sum(1), [.025, .975])]


def label(name):
    known = {"locomo": "LoCoMo", "musique": "MuSiQue", "2wikimultihopqa": "2WikiMultiHopQA",
             "longmemeval_s": "LongMemEval S", "longmemeval_oracle": "LongMemEval oracle",
             "multihoprag": "MultiHop-RAG", "personamem_v2_32k": "PersonaMem v2 32K",
             "personamem_v2_128k": "PersonaMem v2 128K", "beam_100k": "BEAM 100K",
             "beam_500k": "BEAM 500K", "beam_1m": "BEAM 1M"}
    if name.startswith("factconsolidation_"):
        _, hop, size = name.split("_")
        return f"FactConsolidation {hop.upper()} {size.upper()}"
    return known[name]


def metric_label(metric):
    return {"f1": "F1", "judge": "Judge accuracy", "mcq": "MCQ accuracy",
            "substring_exact_match": "Substring EM", "beam_rubric_mean": "Rubric mean"}[metric]


def display(value, fmt="text"):
    if value is None:
        return "N/A"
    if fmt == "score":
        return str((Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if fmt == "delta":
        rounded = (Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return f"{rounded:+.2f}"
    if fmt == "usd":
        return f"${value:.4f}"
    if fmt == "int":
        return f"{value:,.0f}"
    if fmt == "decimal":
        return f"{value:,.2f}"
    return str(value).replace("|", "/").replace("\n", " ")


def table(rows, columns):
    header = "| " + " | ".join(c[1] for c in columns) + " |"
    separator = "| " + " | ".join("---" if c[2] == "text" else "---:" for c in columns) + " |"
    return "\n".join([header, separator] + ["| " + " | ".join(display(r.get(c[0]), c[2]) for c in columns) + " |" for r in rows])


SETTINGS = [
    {"variable": "Arm A", "value": "BM25 + local MiniLM dense retrieval; raw source evidence", "purpose": "Zero ingestion LLM calls; baseline"},
    {"variable": "Arm B", "value": "A + cached facts from detected source text", "purpose": "Test whether fact compression improves evidence per token"},
    {"variable": "Arm C", "value": "B + validity intervals, supersession and source replacement", "purpose": "Test whether handling changed facts reduces stale answers"},
    {"variable": "Arm D", "value": "C + topic index, community index and relation traversal", "purpose": "Test whether graph links improve evidence ranking"},
    {"variable": "Reader / provider", "value": "Gemini 3.6 Flash / Google Vertex AI", "purpose": "Same reader across every arm"},
    {"variable": "Judge / extractor", "value": "Gemini 2.5 Flash Lite / Google Vertex AI", "purpose": "Judging only where required; B/C/D fact extraction"},
    {"variable": "Evidence budget", "value": "4,000 MiniLM WordPiece tokens including source labels", "purpose": "Same reader evidence allowance; nominal corpus size is separate"},
    {"variable": "Answer allowance", "value": "768 provider output tokens; includes hidden reasoning", "purpose": "Empty answers score zero; no quality retries"},
    {"variable": "Judge allowance", "value": "64 output tokens; BEAM rubric 512", "purpose": "Preserve dataset-specific metric"},
    {"variable": "A retrieval", "value": "BM25 + all-MiniLM-L6-v2; equal RRF weights; k=60; top 100/channel", "purpose": "Local lexical and embedding search"},
    {"variable": "Dense windows", "value": "200 tokens, overlap 30; max window score folded to parent source", "purpose": "Long source units remain retrievable"},
    {"variable": "Sample", "value": "100 questions/dataset; proportional type strata; seed 13", "purpose": "Same IDs in A/B/C/D; all 100-question MAB variants are full"},
    {"variable": "B detection", "value": "Up to 8 units from first 30 A ranks; at most 12 fixed 500-token pieces/query", "purpose": "Bound extraction cost; not exhaustive fact coverage"},
    {"variable": "B extraction", "value": "At most 16 facts/piece; 2,048 output tokens; units over 1,000 tokens stay raw", "purpose": "Quote/object grounding required; invalid output falls back to raw"},
    {"variable": "Fact cache", "value": "Key includes namespace, source piece, metadata, prompt and model parameters", "purpose": "Reuse extraction across arms and questions; new protocol means new key"},
    {"variable": "Memory state", "value": "Discover all sampled queries' evidence per history before answering", "purpose": "Warm shared memory; setup charged separately; not online cold-start replay"},
    {"variable": "C supersession", "value": "Matching subject/predicate/scope; exclusive state; strict date or benchmark serial", "purpose": "Preserve history/events/multi-value facts; no guessed temporal order"},
    {"variable": "D topics", "value": "TF-IDF + MiniBatchKMeans, seed 13, up to 32 topics", "purpose": "Local topic membership and separate inverted index"},
    {"variable": "D communities", "value": "Leiden on source projection with shared entities; filter entity hubs", "purpose": "Local community membership and separate inverted index"},
    {"variable": "D relationships", "value": "Typed source/entity/fact nodes and edges; two-hop traversal", "purpose": "Graph covers discovered evidence, not all corpus facts"},
    {"variable": "D fusion weights", "value": "Base 1.0; topic 0.2; community 0.2; relation 0.4; RRF k=60", "purpose": "Fixed before screening; no evaluation-answer tuning"},
    {"variable": "API price inputs", "value": "Reader input/output $0.75/$3.75 per million; Lite $0.10/$0.40", "purpose": "Recorded-token cost estimates; not a reconciled cloud invoice"},
    {"variable": "Concurrency / caps", "value": "16 QA workers; A cap $150; B/C/D shared cap $60; program cap $500", "purpose": "Persistent reservation ledger before each request"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scope = read(A / "SCOPE.json")
    names = scope["active_datasets"]
    assert "beam_10m" not in names
    identity = read(BCD / "run.json")
    declared = read(BCD / "SAMPLE-IDS.json")
    for name, digest in identity["code_sha256"].items():
        assert hashlib.sha256((ROOT / "src/multicard/version_abcd" / name).read_bytes()).hexdigest() == digest
    for name, digest in identity["baseline_code_sha256"].items():
        assert hashlib.sha256((ROOT / "src/multicard/version_a" / name).read_bytes()).hexdigest() == digest
    if args.require_complete:
        for name, files in identity["data_sha256"].items():
            for filename, expected in files.items():
                with (ROOT / "data/version_a" / name / filename).open("rb") as handle:
                    assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, (name, filename)
    summary, quality, paired, bytype, mechanisms, runtime, sources, retrieval = [], [], [], [], [], [], [], []
    audit = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "sources": {}}
    completed, total_a, total_bcd = [], 0, Counter()
    for name in names:
        metrics = read(A / name / "metrics.json")
        manifest = read(ROOT / "data/version_a" / name / "manifest.json")
        aq = indexed(lines(ROOT / "data/version_a" / name / "questions.jsonl"))
        answers = indexed(lines(A / name / "answers.jsonl"))
        assert set(aq) == set(answers), name
        assert metrics["status"] == "complete"
        total_a += len(answers)
        for file in [A / name / "metrics.json", A / name / "answers.jsonl", ROOT / "data/version_a" / name / "manifest.json"]:
            audit["sources"][str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        sample_file = BCD / name / "sample.json"
        sample = read(sample_file) if sample_file.exists() else {"questions": [], "full_count": len(aq)}
        qs = indexed(sample["questions"])
        if sample_file.exists():
            assert set(qs) == set(declared[name]["ids"]), (name, "Sample differs from preregistration")
        arms = {arm: indexed(lines(BCD / name / f"answers_{arm}.jsonl")) for arm in "ABCD"}
        for arm, rows in arms.items():
            assert set(rows) <= set(qs)
            total_bcd[arm] += len(rows)
        complete = bool(qs) and all(set(rows) == set(qs) for rows in arms.values())
        if complete:
            completed.append(name)
        for qid, row in arms["A"].items():
            assert row == answers[qid], (name, qid, "Baseline was changed")
        groups = [read(p) for p in sorted((BCD / name / "groups").glob("*.json"))]
        for path in (BCD / name / "groups").glob("*.json"):
            audit["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        contexts = indexed(lines(BCD / name / "contexts.jsonl"))
        group_n = Counter(q["group"] for q in qs.values())
        logical_setup = sum(call["usd"] for g in groups for call in g["extraction"])
        cohorts = metrics.get("cohorts") or [metrics]
        for cohort in cohorts:
            cohort_name = "adversarial" if cohort["cohort"] == "adversarial" else "headline"
            pred = lambda q: bool(q.get("headline_eligible", True)) == (cohort_name == "headline")
            ids_a = [i for i, q in aq.items() if pred(q)]
            assert len(ids_a) == cohort["questions"]
            close(mean(answers[i]["score"] for i in ids_a), cohort["mean_score"])
            close(sum(answers[i]["answer_usd"] for i in ids_a), cohort["answer_cost_usd"])
            close(sum(answers[i]["judge_usd"] for i in ids_a), cohort["evaluation_cost_usd"])
            chosen = [i for i, q in qs.items() if pred(q)]
            kind = aq[ids_a[0]]["metric"]
            display_name = label(name) + (" adversarial" if cohort_name == "adversarial" else "")
            s = {"dataset": name, "test": display_name, "cohort": cohort_name, "metric": metric_label(kind),
                 "full_n": len(ids_a), "paired_n": len(chosen), "A_full": cohort["mean_score"],
                 "status": "complete" if complete else "running", "scope": "full" if len(qs) == len(aq) else "screen"}
            for arm in "ABCD":
                s[arm] = mean(arms[arm][i]["score"] for i in chosen) if chosen and all(i in arms[arm] for i in chosen) else None
            s["D_minus_A"] = s["D"] - s["A"] if s["D"] is not None else None
            best = max(s[a] for a in "ABCD") if complete else None
            s["best"] = "/".join(a for a in "ABCD" if abs(s[a] - best) < 1e-10) if complete else "pending"
            summary.append(s)
            cg = Counter(qs[i]["group"] for i in chosen)
            setup = sum(sum(c["usd"] for c in g["extraction"]) * cg[g["group"]] / group_n[g["group"]] for g in groups)
            for arm, pop in [("A full", ids_a)] + [(a, chosen) for a in "ABCD"]:
                rows = answers if arm == "A full" else arms[arm]
                available = [i for i in pop if i in rows]
                if not available:
                    continue
                selected = [rows[i] for i in available]
                binary = kind != "beam_rubric_mean"
                for r in selected:
                    assert 0 <= r["score"] <= 1
                    assert 0 <= r.get("context_tokens", 0)
                    if binary:
                        assert r["correct"] in (0, 1)
                        if kind != "f1":
                            close(r["score"], r["correct"])
                    else:
                        assert r["correct"] is None
                correct = sum(r.get("correct") or 0 for r in selected) if binary else None
                answer_usd, judge_usd = sum(r["answer_usd"] for r in selected), sum(r["judge_usd"] for r in selected)
                setup_usd = setup if arm in "BCD" else 0
                fresh = [r["answer_seconds"] for r in selected if not r.get("answer_cached") and r["answer_seconds"] > 0]
                quality.append({"dataset": name, "test": display_name, "cohort": cohort_name, "arm": arm,
                                "scope": "full" if arm == "A full" else s["scope"], "n": len(selected), "requested": len(pop),
                                "metric": metric_label(kind), "score": mean(r["score"] for r in selected),
                                "correct": correct, "correct_basis": "Exact match" if kind == "f1" else "Binary" if binary else "N/A",
                                "exact_match": correct / len(selected) if kind == "f1" else None,
                                "setup_usd": setup_usd, "answer_usd": answer_usd, "judge_usd": judge_usd,
                                "serving_usd": setup_usd + answer_usd, "per_question_usd": (setup_usd + answer_usd) / len(selected),
                                "per_correct_usd": (setup_usd + answer_usd) / correct if correct else None,
                                "input_tokens": sum(r["answer_tokens_in"] for r in selected),
                                "output_tokens": sum(r["answer_tokens_out"] for r in selected),
                                "mean_context_tokens": mean(r.get("context_tokens", 0) for r in selected),
                                "max_context_tokens": max(r.get("context_tokens", 0) for r in selected),
                                "context_over_budget": sum(r.get("context_tokens", 0) > 4000 for r in selected),
                                "cached": sum(bool(r.get("answer_cached")) for r in selected),
                                "empty": sum(r.get("supplementary", {}).get("empty_answer", False) for r in selected),
                                "reader_p50_s": quantile(fresh, .5), "reader_p95_s": quantile(fresh, .95)})
            if complete:
                for new, ref in [("B", "A"), ("C", "B"), ("D", "C"), ("C", "A"), ("D", "A")]:
                    differences = [(qs[i]["group"], arms[new][i]["score"] - arms[ref][i]["score"]) for i in chosen]
                    low, high = ci(differences)
                    paired.append({"test": display_name, "dataset": name, "cohort": cohort_name, "comparison": f"{new}-{ref}",
                                   "n": len(chosen), "histories": len(cg), "metric": metric_label(kind),
                                   "reference": s[ref], "new": s[new], "delta": mean(d for _, d in differences),
                                   "ci_low": low, "ci_high": high, "wins": sum(d > 1e-10 for _, d in differences),
                                   "losses": sum(d < -1e-10 for _, d in differences), "ties": sum(abs(d) <= 1e-10 for _, d in differences)})
            retrieval.append({"test": display_name, "gold_n": cohort.get("retrieval_gold_questions", 0),
                              "excluded": cohort.get("retrieval_incomplete_evidence_excluded", 0),
                              **cohort.get("retrieval_metrics", {})})
        for typ in sorted({q["type"] for q in aq.values()}):
            full_ids = [i for i in aq if aq[i]["type"] == typ]
            sample_ids = [i for i in qs if qs[i]["type"] == typ]
            row = {"test": label(name), "type": typ, "metric": metric_label(aq[full_ids[0]]["metric"]),
                   "full_n": len(full_ids), "sample_n": len(sample_ids), "A_full": mean(answers[i]["score"] for i in full_ids)}
            for arm in "ABCD":
                row[arm] = mean(arms[arm][i]["score"] for i in sample_ids) if sample_ids and all(i in arms[arm] for i in sample_ids) else None
            bytype.append(row)
        node_types = Counter(n["type"] for g in groups for n in g["graph"]["nodes"])
        status_counts = Counter(c["status"] for g in groups for c in g["extraction"])
        mechanisms.append({"test": label(name), "prepared_queries": len(contexts), "groups": len(groups),
                           "extraction_pieces": sum(status_counts.values()), "ok_pieces": status_counts["ok"],
                           "fallback_pieces": sum(v for k, v in status_counts.items() if k != "ok"),
                           "rejected_facts": sum(g["rejected_facts"] for g in groups),
                           "facts": sum(len(g["facts"]) for g in groups), "compressed_units": sum(len(g["compressed_units"]) for g in groups),
                           "superseded_facts": sum(bool(f.get("superseded_by")) for g in groups for f in g["facts"]),
                           "reversed_intervals": sum(bool(f.get("valid_from") and f.get("valid_to") and f["valid_to"] < f["valid_from"]) for g in groups for f in g["facts"]),
                           "queries_hiding_facts": sum(c["hidden_facts"] > 0 for c in contexts.values()),
                           "B_changed": sum(r.get("context_changed", False) for r in arms["B"].values()),
                           "C_changed": sum(r.get("context_changed", False) for r in arms["C"].values()),
                           "D_changed": sum(r.get("context_changed", False) for r in arms["D"].values()),
                           "topics": node_types["topic"], "communities": node_types["community"],
                           "entities": node_types["entity"], "edges": sum(len(g["graph"]["edges"]) for g in groups),
                           "setup_usd": logical_setup})
        runtime.append({"test": label(name), "source_groups": metrics["source_groups"],
                        "source_documents": metrics["source_documents_indexed"], "source_tokens": metrics["source_tokens"],
                        "mean_history_tokens": metrics["source_tokens"] / metrics["source_groups"],
                        "index_minutes": metrics["index_seconds"] / 60, "index_mb": metrics["index_storage_bytes"] / 1e6,
                        "retrieval_p50_ms": metrics["retrieval_p50_seconds"] * 1000,
                        "retrieval_p95_ms": metrics["retrieval_p95_seconds"] * 1000,
                        "prep_minutes": sum(g["preparation_seconds"] for g in groups) / 60,
                        "graph_seconds": sum(g["graph"]["build_seconds"] for g in groups),
                        "index_llm_usd": metrics["index_llm_usd"]})
        source = manifest.get("source_url") or manifest.get("source") or manifest.get("sources") or "See manifest.json"
        protocol = list(dict.fromkeys(metrics.get("protocol_differences", []) + manifest.get("protocol_differences", []) + manifest.get("protocol_notes", [])))
        protocol += [f"{k}: {v}" for k, v in manifest.get("protocol", {}).items()]
        sources.append({"test": label(name), "variant": metrics["variant"], "source": json.dumps(source) if not isinstance(source, str) else source,
                        "revision": manifest.get("source_revision") or manifest.get("revision") or "See source manifest",
                        "protocol": " / ".join(protocol),
                        "manifest_sha256": audit["sources"][f"data/version_a/{name}/manifest.json"]})
        for path in [sample_file, BCD / name / "contexts.jsonl"] + [BCD / name / f"answers_{a}.jsonl" for a in "ABCD"]:
            if path.exists():
                audit["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert total_a == scope["active_question_count"] == 20142
    if args.require_complete:
        assert completed == names, [n for n in names if n not in completed]
        assert all(total_bcd[a] == 1900 for a in "ABCD")
    budget, actual, reservations = [], {}, {}
    for name, path in [("A", A), ("B/C/D", BCD)]:
        with sqlite3.connect((path / "_meter/calls.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute("SELECT role,status,model,count(*),sum(usd),sum(tin),sum(tout) FROM calls GROUP BY role,status,model").fetchall()
        actual[name] = sum(r[4] for r in rows if r[1] == "done")
        reservations[name] = sum(r[4] for r in rows if r[1] != "done")
        for role, status, model, count, usd, tin, tout in rows:
            assert model in {"gemini-3.6-flash", "gemini-2.5-flash-lite"}
            if status == "done":
                pin, pout = (.75, 3.75) if model == "gemini-3.6-flash" else (.1, .4)
                close(usd, (tin * pin + tout * pout) / 1e6)
            budget.append({"campaign": name, "role": role or "preflight/smoke", "status": status, "model": model, "calls": count,
                           "usd": usd, "input_tokens": tin, "output_tokens": tout})
        if args.require_complete:
            assert all(r[1] == "done" for r in rows), rows
    assert actual["A"] <= 150 and actual["B/C/D"] + reservations["B/C/D"] <= 60
    assert sum(actual.values()) + sum(reservations.values()) + 127 < 500
    previous = read(BCD / "comparison.json")["comparisons"]
    our = {(r["dataset"], r["cohort"], r["comparison"]): r for r in paired}
    for row in previous:
        key = row["dataset"], row["cohort"], row["comparison"]
        if key in our and row["status"] == "complete":
            close(our[key]["delta"], row["delta"])
    audit["checks"] = ["A: unique IDs, complete declared questions, raw scores/costs match generated metrics",
                       "Matched A answers identical to full A answers", "BCD IDs are subsets of preregistered sample",
                       "Paired deltas independently match existing comparison output", "Frozen BCD source fingerprints unchanged",
                       "Ledger costs independently match token counts and configured model prices",
                       "Budget caps respected; BEAM 10M excluded"]
    audit.update({"A_questions": total_a, "BCD_answers_by_arm": dict(total_bcd), "BCD_complete_datasets": completed,
                  "BCD_pending_datasets": [n for n in names if n not in completed], "actual_api_usd": actual, "reservations_usd": reservations})
    repairs = lines(BCD / "judge_reason_repairs.jsonl")
    audit["judge_reason_recoveries"] = repairs
    for path in [BCD / "JUDGE-JSON-RECOVERY.json", BCD / "judge_reason_repairs.jsonl"]:
        if path.exists():
            audit["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    notes = [
        {"topic": "Report scope", "note": "This report covers the current Version A and B/C/D campaigns. Older Part 1 variants, scores and spending are not mixed into these comparisons."},
        {"topic": "Evidence budget audit", "note": "; ".join(f"{r['test']} / {r['arm']}: {r['context_over_budget']} answers over 4,000 tokens, maximum {r['max_context_tokens']}" for r in quality if r["context_over_budget"]) or "All recorded contexts fit the 4,000-token evidence target."},
        {"topic": "Completion", "note": f"A: {total_a:,} questions across 19 active variants. B/C/D: {len(completed)}/19 variants complete; 100 requested per variant per arm. BEAM 10M deferred."},
        {"topic": "Metric scale", "note": "Scores are on 0-100 display scale. F1 and BEAM rubric mean are not percentages of questions answered correctly. Correct counts use exact match for F1 datasets."},
        {"topic": "Scope", "note": "A full means the selected released benchmark variant. MuSiQue/2Wiki use HippoRAG 2 released 1,000-question samples. Only FactConsolidation B/C/D are full 100-question variants; others are screens."},
        {"topic": "Oracle", "note": "LongMemEval oracle supplies relevant sessions and is a diagnostic; it does not establish full-history retrieval performance."},
        {"topic": "Accounting", "note": "USD estimates use recorded provider token usage and fixed configured prices. Actual API spend counts each completed ledger request once, including preflights and recovered attempts. This is not an invoice reconciliation."},
        {"topic": "Logical cost", "note": "Per-arm serving cost = reader cost + shared fact setup allocation. Cache hits retain standalone logical cost. B/C/D share setup in the experiment; do not sum setup three times. Judges are separate evaluation overhead."},
        {"topic": "Quality-adjusted cost", "note": "Cost per correct = logical serving dollars / correct count. Uses exact match for MuSiQue/2Wiki, binary correctness elsewhere; undefined for BEAM. It includes setup amortized over this sample, not a proven production workload."},
        {"topic": "Runtime", "note": "Index minutes sum saved group build durations, including cached group loads; exclude lost interrupted work. Preparation minutes sum group extraction/graph durations. Neither is end-to-end campaign elapsed time or CPU core-hours. Cache reconciliation timestamps are not initial runtimes."},
        {"topic": "Latency", "note": "Reader latency quantiles exclude cached answers. They omit fact setup, retrieval, judging and production serving overhead. A retrieval latency is shown separately."},
        {"topic": "Local compute", "note": "Local embedding, graph compute, storage, energy, engineering and SaaS operations are not monetized. Zero ingestion LLM dollars does not mean zero total indexing cost."},
        {"topic": "Uncertainty", "note": "Paired 95% bootstrap intervals resample histories (2,000 draws, seed 13). Single-corpus datasets have no history-cluster interval. Results are exploratory, with no multiple-comparison correction or repeated model seeds."},
        {"topic": "Causal limits", "note": "B changes representation; C adds validity text, suppression and ranking replacement together; D adds all graph channels together. No B-plus-graph without C, topics-only, communities-only or relationships-only ablation has run."},
        {"topic": "Lazy scope", "note": "Facts are cached once per exact namespace/source/protocol identity. This is not one extraction forever for changing text or extraction versions. Warm discovery uses only source text selected by sample queries; no gold answers/options/rubrics enter extraction."},
        {"topic": "Validity limits", "note": "Temporal updates operate on discovered, validated facts with strict matching and ordering. No full entity resolution, exhaustive contradiction detection or production transaction-time API is established."},
        {"topic": "Fact validation limits", "note": "The parser checks normalized quote and object text against the source. It does not independently verify the subject/predicate interpretation, exclusivity classification or date meaning. Extracted dates are checked for ISO-shaped text, not source entailment or calendar validity. Those fields still depend on extractor accuracy."},
        {"topic": "Observed temporal defect", "note": f"{sum(r['reversed_intervals'] for r in mechanisms)} stored facts have valid_to earlier than valid_from. Missing starts default to the source message date, and the code has no interval-order guard. C and D results retain this defect as run; no post-hoc correction or paid rerun is included. Its contribution to score changes has not been isolated."},
        {"topic": "Failures", "note": "Empty reader answers score zero. Invalid extraction JSON or grounding uses raw source fallback. BEAM invalid JSON escapes were repaired deterministically on cached text. No quality retries were used to improve answers."},
        {"topic": "Judge JSON recovery", "note": f"{len(repairs)} malformed BEAM reason strings were recovered using a separately recorded parser overlay. Only a sole leading numeric score in {{0,0.5,1}} is accepted; ambiguous/duplicate scores are rejected. Cached raw text and token usage are preserved. Recovery did not ask the model to judge again."},
        {"topic": "Retrieval metric", "note": "Evidence recall and all-evidence coverage are diagnostics. MultiHop-RAG uses source/article coverage, not the paper's fact Hit@K. Missing gold evidence is disclosed. Do not compare unlike Hit@K definitions."},
        {"topic": "External comparisons", "note": "This campaign compares our A/B/C/D. It does not rerun Chandan, Zep, Mem0 or other providers under the same models and costs; it does not establish industry-best quality or a total-cost advantage over those systems."},
        {"topic": "Remaining work", "note": "No BEAM 10M, full B/C/D expansion on larger variants, PersonaMem open-ended scoring, cold-start/online replay, production load test, independent judge audit or complete external-provider controlled comparison."},
        {"topic": "Validation", "note": "Saved validation: 429 passed / 1 skipped full suite; subsequent BCD targeted check 19 passed. These overlap and must not be added. Report values are independently recomputed from saved answers."},
        {"topic": "Lineage", "note": "A: metrics.json and answers.jsonl; B/C/D: sample.json, answers_A/B/C/D.jsonl, contexts.jsonl, groups/*.json, run.json; API accounting: calls.sqlite. Report audit JSON records exact source hashes."},
        {"topic": "Reproducibility scope", "note": "New A/B/C/D implementation is identified by saved working-tree code hashes. The report recomputes from local saved artifacts. A fresh clone rerun, all publication gates and public release have not been completed."},
    ]
    def result(name, arm):
        return next(r[arm] for r in summary if r["dataset"] == name and r["cohort"] == "headline")
    def change_text(name, before, after):
        return f"{label(name)}: {before} {result(name, before) * 100:.2f} to {after} {result(name, after) * 100:.2f}"
    outside_mab = [r for r in mechanisms if not r["test"].startswith("FactConsolidation")]
    findings = [
        {"hypothesis": "A: inexpensive indexing", "measurement": f"All 19 active variants indexed with zero ingestion LLM calls. Recorded local index/build-load duration totals {sum(r['index_minutes'] for r in runtime):.1f} minutes across groups.", "interpretation": "The zero-ingestion-LLM design is implemented and measured. Indexing still needs local compute and storage. This run does not establish a vendor total-cost comparison."},
        {"hypothesis": "B: lazy fact compression", "measurement": change_text("personamem_v2_32k", "A", "B") + "; " + change_text("locomo", "A", "B") + "; " + change_text("factconsolidation_sh_6k", "A", "B"), "interpretation": "Mixed quality. B is useful on some tests, but bounded fact extraction can lose useful source information. The screen does not isolate which losses came from extraction versus evidence packing."},
        {"hypothesis": "C: temporal validity", "measurement": change_text("factconsolidation_sh_262k", "B", "C") + "; " + change_text("factconsolidation_sh_64k", "B", "C") + "; " + change_text("longmemeval_oracle", "B", "C"), "interpretation": "The current temporal implementation does not consistently improve answers. It hurts all four single-hop FactConsolidation variants compared with B. Improvement in oracle cannot establish full-history retrieval gains."},
        {"hypothesis": "C: actual invalidation coverage", "measurement": f"Only {sum(r['superseded_facts'] for r in outside_mab):,} stored facts are superseded outside FactConsolidation in prepared datasets. Both LongMemEval variants and both PersonaMem variants have zero.", "interpretation": "C also changes validity labels and filters by supplied dates. Score movement without supersession is not evidence that old-fact invalidation helped."},
        {"hypothesis": "C: interval correctness", "measurement": f"{sum(r['reversed_intervals'] for r in mechanisms)} stored facts have an end date before their start date. " + "; ".join(f"{r['test']}: {r['reversed_intervals']}" for r in mechanisms if r["reversed_intervals"]), "interpretation": "This is an implementation defect. Correct start-date fallback and interval validation before relying on C or D for temporal memory. The current measurements preserve the defect; its accuracy effect is not isolated."},
        {"hypothesis": "D: combined graph ranking", "measurement": change_text("factconsolidation_mh_32k", "C", "D") + "; " + change_text("multihoprag", "A", "D") + "; " + change_text("2wikimultihopqa", "A", "D"), "interpretation": "The combined graph helps some tests and hurts others. Recovery from C does not always beat A. Separate topic, community and relationship contributions have not been measured."},
        {"hypothesis": "Extraction robustness", "measurement": "; ".join(f"{r['test']}: {r['fallback_pieces']}/{r['extraction_pieces']} pieces fell back to raw" for r in mechanisms if r["test"] in {"LongMemEval S", "MultiHop-RAG", "MuSiQue"}), "interpretation": "Grounding/format checks frequently reject extraction on natural text. These are retained raw-source fallbacks, not missing answers. Improve extraction coverage before assuming the fact layer is fully exercised."},
        {"hypothesis": "Cost plus quality", "measurement": "Per-arm tables include questions correct and setup-plus-answer dollars per correct answer. BEAM uses graded rubric scores and has no binary cost-per-correct figure.", "interpretation": "Lower cost is useful only alongside sufficient quality. Shared-cache experiment savings are not the same as standalone deployment economics."},
    ]
    data = {"generated_at": audit["generated_at_utc"], "summary": summary, "quality": quality, "paired": paired,
            "by_type": bytype, "mechanisms": mechanisms, "runtime": runtime, "retrieval": retrieval,
            "budget": budget, "settings": SETTINGS, "sources": sources, "notes": notes, "findings": findings, "audit": audit}
    (OUT / "report-data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    for key, values in data.items():
        if isinstance(values, list) and values:
            keys = list(dict.fromkeys(k for r in values for k in r))
            with (OUT / f"{key}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(values)
    summary_cols = [("test", "Test", "text"), ("metric", "Metric", "text"), ("full_n", "A full N", "int"),
                    ("A_full", "A full", "score"), ("paired_n", "Matched N", "int"), ("A", "A matched", "score"),
                    ("B", "B", "score"), ("C", "C", "score"), ("D", "D", "score"), ("best", "Best matched", "text"), ("scope", "B/C/D scope", "text")]
    quality_cols = [("test", "Test", "text"), ("arm", "Arm", "text"), ("n", "N", "int"), ("score", "Score /100", "score"),
                    ("correct", "Correct", "int"), ("exact_match", "EM /100", "score"), ("setup_usd", "Fact setup", "usd"),
                    ("answer_usd", "Answers", "usd"), ("judge_usd", "Judging", "usd"), ("per_question_usd", "Serving/Q", "usd"), ("per_correct_usd", "Serving/correct", "usd")]
    md = ["# TG-VGRAG experiment results", "", f"Generated from saved run outputs at {data['generated_at']}.", "",
          f"Version A is complete: **{total_a:,} questions across 19 active variants**. B/C/D are complete on **{len(completed)}/19 variants**. The B/C/D design is a 100-question paired screen per variant, except the eight 100-question FactConsolidation variants, which are full tests. BEAM 10M remains deferred.", "",
          "The results are mixed. The baseline remains competitive. Fact compression, temporal validity and graph ranking each help some cases and hurt others. These measurements do not establish an overall winner or industry superiority.", "",
          f"The audit found **{sum(r['reversed_intervals'] for r in mechanisms)} stored fact intervals with an end before the start**. C and D include this known defect. Their scores below are the measured results before any correction, and the defect's effect on accuracy has not been isolated.", "",
          "## One summary table", "", "All scores use a 0-100 display scale. F1 and rubric mean are graded scores, not percent correct. Compare B/C/D with **A matched**, not A full. LoCoMo adversarial is a separate cohort, not another dataset.", "", table(summary, summary_cols), "",
          "## Test variables", "", table(SETTINGS, [("variable", "Variable", "text"), ("value", "Setting", "text"), ("purpose", "Purpose", "text")]), "",
          "## Hypothesis findings", "", table(findings, [("hypothesis", "Hypothesis", "text"), ("measurement", "Observed result", "text"), ("interpretation", "What it means", "text")]), "",
          "## Full Version A quality and cost", "", "USD below are logical model costs. Answer generation is serving cost; judging is evaluation overhead. A has zero ingestion LLM calls, with local indexing time and storage recorded separately.", "", table([r for r in quality if r["arm"] == "A full"], quality_cols), "",
          "## Matched A/B/C/D quality and cost", "", "Fact setup is shown for each arm as its standalone allocation. The experiment pays shared extraction once across B/C/D. Serving/Q and serving/correct include that setup, exclude judging and unpriced local compute. Correct counts on F1 datasets use exact match; BEAM has no binary correct count.", "", table([r for r in quality if r["arm"] != "A full"], quality_cols), "",
          "## Paired changes", "", "Score differences are points on a 0-100 scale. Wins/losses/ties count per-question score changes. Intervals are exploratory history-cluster bootstrap intervals; N/A means only one shared corpus, not zero uncertainty.", "", table(paired, [("test", "Test", "text"), ("comparison", "Change", "text"), ("n", "N", "int"), ("histories", "Histories", "int"), ("delta", "Delta", "delta"), ("ci_low", "95% low", "delta"), ("ci_high", "95% high", "delta"), ("wins", "Wins", "int"), ("losses", "Losses", "int"), ("ties", "Ties", "int")]), "",
          "## Results by question type", "", table(bytype, [("test", "Test", "text"), ("type", "Question type", "text"), ("metric", "Metric", "text"), ("full_n", "Full N", "int"), ("A_full", "A full", "score"), ("sample_n", "Matched N", "int")] + [(a, a, "score") for a in "ABCD"]), "",
          "## What the mechanisms actually did", "", "Counts below are sums across prepared histories. Superseded facts are counted once per stored history, not once per question. Fallback counts are extraction pieces, not whole source units. Changed context compares B-A, C-B and D-C respectively.", "", table(mechanisms, [("test", "Test", "text"), ("extraction_pieces", "Pieces", "int"), ("fallback_pieces", "Fallback pieces", "int"), ("facts", "Facts", "int"), ("compressed_units", "Compressed units", "int"), ("superseded_facts", "Superseded", "int"), ("queries_hiding_facts", "Q hiding facts", "int"), ("B_changed", "B changed", "int"), ("C_changed", "C changed", "int"), ("D_changed", "D changed", "int")]), "",
          table(mechanisms, [("test", "Test", "text"), ("topics", "Topics", "int"), ("communities", "Communities", "int"), ("entities", "Entities", "int"), ("edges", "Typed edges", "int"), ("setup_usd", "Shared fact setup", "usd")]), "",
          "Temporal interval audit: end-before-start facts are invalid intervals in the stored timeline. This count is separate from supersession and includes intervals on events. C and D were measured without repairing them.", "", table([r for r in mechanisms if r["reversed_intervals"]], [("test", "Test", "text"), ("reversed_intervals", "End before start", "int")]), "",
          "## Context, indexing and runtime", "", "Corpus tokens below use the MiniLM WordPiece tokenizer. Published size labels such as 32K or 100K may use other tokenizers. Source totals sum all indexed histories, including related variants that share source material. Index minutes are saved build/load durations, not total campaign elapsed time. B/C/D reuse A indexes.", "", table(runtime, [("test", "Test", "text"), ("source_groups", "Groups", "int"), ("source_documents", "Source units", "int"), ("source_tokens", "Source tokens", "int"), ("mean_history_tokens", "Mean/group", "int"), ("index_minutes", "A index min", "decimal"), ("index_mb", "Index MB", "decimal"), ("prep_minutes", "BCD prep min", "decimal"), ("graph_seconds", "Graph sec", "decimal")]), "",
          "Reader latency excludes cached answers, extraction, retrieval and judging. All token counts below are provider reader tokens except mean evidence tokens, which use MiniLM WordPiece. Hidden reasoning consumes the output allowance.", "", table(quality, [("test", "Test", "text"), ("arm", "Arm", "text"), ("n", "N", "int"), ("input_tokens", "Input tokens", "int"), ("output_tokens", "Output tokens", "int"), ("mean_context_tokens", "Mean evidence", "decimal"), ("cached", "Cached Q", "int"), ("empty", "Empty Q", "int"), ("reader_p50_s", "Reader p50 sec", "decimal"), ("reader_p95_s", "Reader p95 sec", "decimal")]), "",
          "## Version A retrieval diagnostics", "", "These are evidence coverage diagnostics, not interchangeable vendor Hit@K scores. MultiHop-RAG coverage uses source/article evidence. N/A means no applicable gold evidence or metric.", "", table(retrieval, [("test", "Test", "text"), ("gold_n", "Gold Q", "int"), ("excluded", "Incomplete excluded", "int"), ("recall_at_5", "Recall@5 /100", "score"), ("recall_at_10", "Recall@10 /100", "score"), ("recall_at_100", "Recall@100 /100", "score"), ("evidence_recall_at_budget", "Recall in 4K /100", "score"), ("all_evidence_at_budget", "All evidence /100", "score")]), "",
          "## Actual API accounting", "", f"Completed-request ledger: A **${actual['A']:.4f}**, B/C/D **${actual['B/C/D']:.4f}**, combined **${sum(actual.values()):.4f}**. Outstanding reservations: **${sum(reservations.values()):.4f}**. Costs use recorded token usage and configured prices, not a reconciled cloud invoice. These campaign totals exclude earlier Part 1 work. The B/C/D total includes $0.38770325 of imported preflight requests. Cache reuse and recovered attempts mean actual spend differs from summed logical per-arm serving costs.", "", table(budget, [("campaign", "Campaign", "text"), ("role", "Role", "text"), ("status", "Status", "text"), ("model", "Model", "text"), ("calls", "Calls", "int"), ("usd", "USD", "usd"), ("input_tokens", "Input", "int"), ("output_tokens", "Output", "int")]), "",
          "## What is and is not established", "", table(notes, [("topic", "Topic", "text"), ("note", "Finding or limit", "text")]), "",
          "## Dataset provenance and protocol differences", "", table(sources, [("test", "Test", "text"), ("variant", "Variant", "text"), ("source", "Source", "text"), ("revision", "Revision", "text"), ("protocol", "Protocol differences", "text")]), "",
          "## Recompute", "", "This report is generated by `scripts/build_full_results_report.py` from saved answers, metrics, preparation artifacts and read-only API ledgers. Companion CSV tables and `AUDIT.json` contain unrounded numbers and source hashes. No model calls are needed to regenerate it.", ""]
    (OUT / "FULL-REPORT.md").write_text("\n".join(md))
    print(json.dumps({"output": str(OUT), "A_questions": total_a, "BCD_complete": len(completed), "BCD_by_arm": dict(total_bcd), "actual_usd": actual, "table_rows": {k: len(v) for k, v in data.items() if isinstance(v, list)}}))


if __name__ == "__main__":
    main()

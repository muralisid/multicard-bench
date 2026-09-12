"""Generate measured Version A tables without mixing metrics or denominators."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean


def _records(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def _unique(records, key="id"):
    return {record[key]: record for record in records}


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values):
    values = [float(value) for value in values if _number(value)]
    return mean(values) if values else None


def _percentile(values, fraction):
    values = sorted(float(value) for value in values if _number(value))
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _sum(records, key):
    return sum(float(record.get(key, 0) or 0) for record in records)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path, rows, columns=None):
    path = Path(path)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def read_budget(db_path, dataset):
    """Read the persistent meter, including calls with no successful QA record."""
    path = Path(db_path)
    if not path.exists():
        return {"available": False, "dataset": dataset, "actual_api_usd": None,
                "unresolved_reserved_usd": None, "committed_usd": None}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM calls")]
    def aggregate(selected):
        done = [row for row in selected if row["status"] == "done"]
        unresolved = [row for row in selected if row["status"] != "done"]
        result = {"calls": len(selected), "done_calls": len(done),
                  "pending_calls": sum(row["status"] == "pending" for row in selected),
                  "error_calls": sum(row["status"] == "error" for row in selected),
                  "actual_api_usd": _sum(done, "usd"),
                  "unresolved_reserved_usd": _sum(unresolved, "usd"),
                  "committed_usd": _sum(selected, "usd")}
        if all(row.get("role") in {"answer", "judge"} for row in done) and selected:
            result["actual_answer_usd"] = _sum([row for row in done if row.get("role") == "answer"], "usd")
            result["actual_judge_usd"] = _sum([row for row in done if row.get("role") == "judge"], "usd")
            result["role_attribution_available"] = True
        else:
            result.update(actual_answer_usd=None, actual_judge_usd=None, role_attribution_available=False)
        return result
    return {"available": True, "dataset": dataset,
            **aggregate([row for row in rows if row["run"] == dataset]),
            "global": aggregate(rows),
            "note": "Done costs are metered usage; unresolved reservations are budget holds, not confirmed spend. Cached cross-dataset calls belong to the first paying run."}


def _cohort(name, full_questions, requested_ids, answers_by_id, retrieval_by_id):
    requested = [question for question in full_questions if question["id"] in requested_ids]
    answers = [answers_by_id[question["id"]] for question in requested if question["id"] in answers_by_id]
    retrieval = [retrieval_by_id[question["id"]] for question in requested if question["id"] in retrieval_by_id]
    by_id = {question["id"]: question for question in full_questions}
    scored = [answer for answer in answers if _number(answer.get("score"))]
    metrics = sorted({question["metric"] for question in full_questions})
    binary_allowed = bool(metrics) and all(metric in {"judge", "mcq", "f1", "substring_exact_match"} for metric in metrics)
    binary = [answer for answer in scored if answer.get("correct") in (0, 1, 0.0, 1.0)] if binary_allowed else []
    correct = _sum(binary, "correct") if binary_allowed and binary else None
    logical_answer = _sum(answers, "answer_usd")
    actual_answer = sum(float(answer.get("incremental_answer_usd", 0 if answer.get("answer_cached") else answer.get("answer_usd", 0)) or 0) for answer in answers)
    # cash_usd includes judges. Keep it separate from the serving-only amount.
    recorded_cash = _sum(answers, "cash_usd")
    clean_retrieval = [record for record in retrieval
                       if by_id[record["id"]].get("evidence_complete", True)
                       and by_id[record["id"]].get("evidence_ids")]
    retrieval_keys = sorted({key for record in clean_retrieval for key in record.get("metrics", {})})
    result = {
        "cohort": name, "questions": len(full_questions), "requested_questions": len(requested),
        "retrieved": len(retrieval), "answered": len(answers), "scored": len(scored),
        "unanswered": len(full_questions) - len(answers), "unscored": len(full_questions) - len(scored),
        "status": "complete" if len(scored) == len(full_questions) and full_questions else "incomplete",
        "score_metric": metrics,
        "mean_score": _mean(answer["score"] for answer in scored) if len(metrics) == 1 else None,
        "mean_score_denominator": len(scored), "binary_scored": len(binary), "correct": correct,
        "accuracy_answered": correct / len(binary) if binary else None,
        "accuracy": correct / len(binary) if binary else None,
        "verified_correct_fraction_of_full": correct / len(full_questions) if binary and full_questions else None,
        "f1_mean": _mean(answer["score"] for answer in scored) if metrics == ["f1"] else None,
        "exact_match_accuracy": correct / len(binary) if binary and metrics == ["f1"] else None,
        "answer_cost_usd": logical_answer, "logical_answer_cost_usd": logical_answer,
        "evaluation_cost_usd": _sum(answers, "judge_usd"),
        "incremental_answer_usd_recorded": actual_answer,
        "incremental_api_usd_recorded": recorded_cash,
        "answer_cost_per_correct_usd": logical_answer / correct if correct else None,
        "cost_per_correct_basis": "exact_match" if metrics == ["f1"] else ("binary_correct" if binary_allowed else None),
        "retrieval_gold_questions": len(clean_retrieval),
        "retrieval_incomplete_evidence_excluded": sum(not by_id[r["id"]].get("evidence_complete", True) for r in retrieval),
        "retrieval_metrics": {key: _mean(r.get("metrics", {}).get(key) for r in clean_retrieval) for key in retrieval_keys},
        "mean_context_tokens": _mean(r.get("context_tokens") for r in retrieval),
        "retrieval_p50_seconds": _percentile([r.get("retrieval_seconds") for r in retrieval], .5),
        "retrieval_p95_seconds": _percentile([r.get("retrieval_seconds") for r in retrieval], .95),
        "answer_p50_seconds": _percentile([a.get("answer_seconds") for a in answers], .5),
        "answer_p95_seconds": _percentile([a.get("answer_seconds") for a in answers], .95),
        "serving_p50_seconds": _percentile([a.get("answer_seconds", 0) + retrieval_by_id.get(a["id"], {}).get("retrieval_seconds", 0) for a in answers], .5),
        "serving_p95_seconds": _percentile([a.get("answer_seconds", 0) + retrieval_by_id.get(a["id"], {}).get("retrieval_seconds", 0) for a in answers], .95),
        "answer_cached_questions": sum(bool(a.get("answer_cached")) for a in answers),
        "empty_answer_questions": sum(bool(a.get("supplementary", {}).get("empty_answer")) for a in answers),
        "answer_tokens_in": _sum(answers, "answer_tokens_in"),
        "answer_tokens_out": _sum(answers, "answer_tokens_out"),
    }
    result["score_label"] = ("F1 / exact match" if metrics == ["f1"] else
                             "graded rubric mean" if any("rubric" in metric for metric in metrics) else
                             "MCQ accuracy" if metrics == ["mcq"] else
                             "substring exact match" if metrics == ["substring_exact_match"] else
                             "judge accuracy" if metrics == ["judge"] else ",".join(metrics))
    return result


def _summarise_unlocked(data, out, args):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    questions = _records(data / "questions.jsonl")
    question_by_id = _unique(questions)
    if len(question_by_id) != len(questions):
        raise ValueError("Duplicate question IDs in dataset")
    limit = getattr(args, "limit", None)
    requested = questions[:limit] if limit is not None else questions
    requested_ids = {q["id"] for q in requested}
    answers = {key: value for key, value in _unique(_records(out / "answers.jsonl")).items() if key in requested_ids}
    retrieval = {key: value for key, value in _unique(_records(out / "retrieval.jsonl")).items() if key in requested_ids}
    index_log = _records(out / "index.jsonl")
    index = list(_unique(index_log, "group").values())
    run = _load(out / "run.json")
    identity = run.get("identity", {})
    manifest = _load(data / "manifest.json")
    frozen_manifest = identity.get("manifest", manifest)
    config = identity.get("config", {})
    excluded = [q for q in questions if not q.get("headline_eligible", True)]
    headline = [q for q in questions if q.get("headline_eligible", True)]
    primary = _cohort("headline" if excluded else "all", headline, requested_ids, answers, retrieval)
    cohorts = [primary]
    if excluded:
        cohorts.append(_cohort("adversarial" if data.name == "locomo" else "supplementary", excluded, requested_ids, answers, retrieval))
    by_type = {kind: _cohort(kind, [q for q in questions if q["type"] == kind], requested_ids, answers, retrieval)
               for kind in sorted({q["type"] for q in questions})}
    budget = read_budget(Path(getattr(args, "output", out.parent)) / "_meter/calls.sqlite", data.name)
    source_groups = {q["group"] for q in questions}
    processed_groups = {record["group"] for record in index}
    source = {"source_groups": len(source_groups), "indexed_groups": len(processed_groups),
              "source_tokens": _sum(index, "source_tokens"),
              "source_documents_indexed": int(_sum(index, "documents")),
              "source_tokens_complete": source_groups <= processed_groups,
              "source_reported_history_tokens": frozen_manifest.get("source_reported_history_tokens"),
              "source_corpus_bytes": (data / "corpus.jsonl").stat().st_size if (data / "corpus.jsonl").exists() else None,
              "index_seconds": _sum(index_log, "build_seconds"),
              "index_uncached_build_seconds": _sum([r for r in index_log if not r.get("cache_hit")], "build_seconds"),
              "index_cache_groups": sum(bool(r.get("cache_hit")) for r in index),
              "index_storage_bytes": int(_sum(index, "storage_bytes")),
              "index_llm_usd": _sum(index_log, "llm_usd"),
              "index_llm_calls": int(_sum(index_log, "llm_calls"))}
    metrics = {"dataset": data.name, "variant": frozen_manifest.get("variant", data.name),
               "version": config.get("version", "unknown"), "reader": identity.get("reader", getattr(args, "reader", None)),
               "judge": identity.get("judge", getattr(args, "judge", None)),
               "scope": "smoke" if limit is not None else "full", "full_dataset_questions": len(questions),
               "full_dataset_requested_questions": len(requested), "full_dataset_answered": len(answers),
               "config_sha256": _digest(identity) if identity else None,
               "git_sha": run.get("git_sha"), "run_started": run.get("started"),
               "budget_limit_usd": run.get("budget_usd"), "dataset_budget_limit_usd": run.get("run_budget_usd"),
               "context_budget_tokens": config.get("context_tokens"), "context_tokenizer": config.get("context_tokenizer"),
               "source_url": frozen_manifest.get("source_url"), "source_revision": frozen_manifest.get("source_revision"),
               "protocol_differences": frozen_manifest.get("protocol_differences", []),
               **primary, **source, "cohorts": cohorts, "by_type": by_type, "budget": budget,
               "all_cohorts_logical_answer_usd": _sum(list(answers.values()), "answer_usd"),
               "all_cohorts_logical_judge_usd": _sum(list(answers.values()), "judge_usd"),
               "error_events": len(_records(out / "errors.jsonl")),
               "notes": [
                   "Partial scores describe scored questions only; requested and full denominators remain visible.",
                   "Verified correct/full is a coverage lower bound, not an accuracy estimate with missing answers scored wrong.",
                   "F1 and exact-match accuracy are separate. F1 cost/correct uses exact matches.",
                   "Empty model answers receive zero score and retain their billed cost; they are never retried for quality.",
                   "BEAM rubric scores have no manufactured binary threshold or cost-per-correct value.",
                   "Logical answer cost prices completed answers including cache hits; incremental recorded costs exclude cached calls.",
                   "Persistent meter includes failed/orphan API attempts; reservations are not confirmed spend.",
                   "Judge cost is evaluation overhead and excluded from serving cost. Local compute/storage are not priced as free.",
                   "Serving latency combines retrieval and answer generation, excluding judging; cache hits remain visible.",
                   "Vendor comparisons require matching dataset variant, model, judge and metric; no cross-dataset average is produced.",
               ]}
    _write_json(out / "metrics.json", metrics)
    _write_csv(out / "per_type.csv", [{"dataset": data.name, "type": kind, **row} for kind, row in by_type.items()])
    _write_csv(out / "per_query.csv", list(answers.values()))
    return metrics


def summarise(data, out, args):
    """Serialize dataset snapshots if retrieval and QA finish together."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".summary.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _summarise_unlocked(data, out, args)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# Accept either spelling for integration with existing scripts.
summarize = summarise


def _fmt(value, digits=4):
    return "pending" if value is None else f"{value:.{digits}f}"


def _report_unlocked(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    datasets = [_load(path) for path in sorted(root.glob("*/metrics.json"))]
    campaign_scope = _load(root / "SCOPE.json")
    if campaign_scope.get("active_datasets"):
        active = set(campaign_scope["active_datasets"])
        datasets = [dataset for dataset in datasets if dataset["dataset"] in active]
    rows, type_rows, budget_rows = [], [], []
    metadata_keys = ["dataset", "variant", "scope", "reader", "judge", "config_sha256", "source_tokens",
                     "source_tokens_complete", "source_corpus_bytes", "source_groups", "indexed_groups",
                     "index_llm_usd", "index_seconds", "index_storage_bytes", "context_budget_tokens"]
    for dataset in datasets:
        metadata = {key: dataset.get(key) for key in metadata_keys}
        for cohort in dataset.get("cohorts", [dataset]):
            rows.append({**metadata, **cohort})
        for kind, cohort in dataset.get("by_type", {}).items():
            type_rows.append({**metadata, "type": kind, **cohort})
        budget = dataset.get("budget", {})
        budget_rows.append({"dataset": dataset["dataset"], **{key: value for key, value in budget.items() if key != "global"},
                            "budget_limit_usd": dataset.get("dataset_budget_limit_usd"),
                            "error_events": dataset.get("error_events", 0)})
    _write_csv(root / "SUMMARY.csv", rows)
    _write_csv(root / "BY_TYPE.csv", type_rows)
    _write_csv(root / "BUDGET.csv", budget_rows)
    lines = ["# Version A measured results", "", "Generated from saved records. Partial scores apply only to scored questions. No cross-dataset average.", "",
             "| Dataset | Cohort | Scope/status | Answered / scored / requested / full | Metric | Score | Correct | Answer USD | Judge USD | USD/correct |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    if campaign_scope.get("deferred"):
        lines[4:4] = ["Deferred at the user's request: " + ", ".join(campaign_scope["deferred"])
                      + ". Cached work is preserved and excluded from the active results below. See SCOPE.json.", ""]
    for row in rows:
        score = _fmt(row.get("mean_score"))
        if row.get("f1_mean") is not None:
            score += " / " + _fmt(row.get("exact_match_accuracy"))
        correct = "N/A" if row.get("correct") is None else str(int(row["correct"]))
        lines.append(f"| {row['dataset']} | {row.get('cohort', 'all')} | {row.get('scope')}/{row.get('status')} | "
                     f"{row.get('answered', 0)} / {row.get('scored', 0)} / {row.get('requested_questions', 0)} / {row.get('questions', 0)} | "
                     f"{row.get('score_label', '')} | {score} | {correct} | {_fmt(row.get('answer_cost_usd'))} | "
                     f"{_fmt(row.get('evaluation_cost_usd'))} | {_fmt(row.get('answer_cost_per_correct_usd'))} |")
    lines += ["", "Answer USD is logical serving cost, including reuse from cache. Judge USD is evaluation overhead. For F1, USD/correct uses exact matches. BEAM graded scores have no binary cost/correct.", "",
              "| Dataset | Indexed groups | Source tokens indexed | Mean context tokens | Retrieval p50/p95 seconds | Serving p50/p95 seconds | Config SHA256 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for dataset in datasets:
        lines.append(f"| {dataset['dataset']} | {dataset.get('indexed_groups', 0)}/{dataset.get('source_groups', 0)} | "
                     f"{int(dataset.get('source_tokens', 0)):,} | {_fmt(dataset.get('mean_context_tokens'), 1)} | "
                     f"{_fmt(dataset.get('retrieval_p50_seconds'))}/{_fmt(dataset.get('retrieval_p95_seconds'))} | "
                     f"{_fmt(dataset.get('serving_p50_seconds'))}/{_fmt(dataset.get('serving_p95_seconds'))} | "
                     f"{(dataset.get('config_sha256') or 'unavailable')[:12]} |")
    lines += ["", "| Dataset | Metered API USD | Unresolved reservation USD | Committed USD | Done / pending / error calls | Recorded QA errors |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in budget_rows:
        lines.append(f"| {row['dataset']} | {_fmt(row.get('actual_api_usd'))} | {_fmt(row.get('unresolved_reserved_usd'))} | "
                     f"{_fmt(row.get('committed_usd'))} | {row.get('done_calls', 0)} / {row.get('pending_calls', 0)} / "
                     f"{row.get('error_calls', 0)} | {row.get('error_events', 0)} |")
    lines += ["", "The persistent meter includes both answering and judging, plus orphan calls. Reservations are holds, not measured spending. "
              "Full hashes, source variants, protocol differences, per-type scores, token usage and latency are retained in JSON/CSV files."]
    temporary = root / "SUMMARY.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "SUMMARY.md")
    return rows


def report(root):
    """One root-level snapshot writer across concurrent dataset workers."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _report_unlocked(root)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

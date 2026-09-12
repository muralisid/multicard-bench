"""Actual-like fixtures protect score, cohort and persistent-cost accounting."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from multicard.version_a.reporting import read_budget, report, summarise


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def setup_dataset(tmp_path, name, questions, answers, limit=None):
    data, out = tmp_path / "data" / name, tmp_path / "results" / name
    write_jsonl(data / "questions.jsonl", questions)
    write_jsonl(data / "corpus.jsonl", [{"id": "g:d1", "group": "g", "text": "history"}])
    manifest = {"variant": name + "_official_split", "source_revision": "pinned-revision", "protocol_differences": ["Reader differs from vendor."]}
    (data / "manifest.json").write_text(json.dumps(manifest))
    write_jsonl(out / "answers.jsonl", answers)
    write_jsonl(out / "retrieval.jsonl", [{"id": q["id"], "context_tokens": 50, "retrieval_seconds": .1,
                                            "metrics": {"recall_at_5": .5}} for q in questions])
    write_jsonl(out / "index.jsonl", [{"group": "g", "documents": 1, "source_tokens": 100,
                                       "build_seconds": 2, "cache_hit": False, "storage_bytes": 64,
                                       "llm_usd": 0, "llm_calls": 0}])
    identity = {"manifest": manifest, "config": {"version": "a1", "context_tokens": 4000},
                "reader": "frozen-reader", "judge": "frozen-judge", "limit": limit}
    (out / "run.json").write_text(json.dumps({"identity": identity, "git_sha": "abc", "budget_usd": 10, "run_budget_usd": 4}))
    args = SimpleNamespace(limit=limit, output=tmp_path / "results", reader="mutable-reader", judge="mutable-judge")
    return data, out, args


def question(uid, kind="single_hop", metric="judge", eligible=True):
    return {"id": uid, "group": "g", "type": kind, "metric": metric,
            "headline_eligible": eligible, "evidence_ids": ["g:d1"]}


def answer(uid, score=1, correct=1, cost=1, cached=False, metric="judge"):
    return {"id": uid, "type": "single_hop", "metric": metric, "answer": "response",
            "score": score, "correct": correct, "answer_usd": cost, "judge_usd": cost / 10,
            "cash_usd": 0 if cached else cost * 1.1, "answer_cached": cached,
            "answer_seconds": 0 if cached else 2, "answer_tokens_in": 50, "answer_tokens_out": 10}


def create_meter(root, rows, with_role=False):
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "calls.sqlite") as connection:
        connection.execute("CREATE TABLE calls (key TEXT PRIMARY KEY, run TEXT, model TEXT, status TEXT, text TEXT, tin INT, tout INT, usd REAL, seconds REAL" + (",role TEXT" if with_role else "") + ")")
        for index, (run, status, cost) in enumerate(rows):
            values = (str(index), run, "reader", status, "", 1, 1, cost, 1)
            if with_role:
                values += ("answer",)
            connection.execute("INSERT INTO calls VALUES (" + ",".join("?" for _ in values) + ")", values)


def test_deferred_dataset_is_disclosed_and_excluded_without_deleting_results(tmp_path):
    for name in ("beam_1m", "beam_10m"):
        data, out, args = setup_dataset(tmp_path, name, [question("q")], [answer("q")])
        summarise(data, out, args)
    root = tmp_path / "results"
    (root / "SCOPE.json").write_text(json.dumps({"active_datasets": ["beam_1m"], "deferred": ["beam_10m"]}))
    rows = report(root)
    assert {row["dataset"] for row in rows} == {"beam_1m"}
    assert "Deferred at the user's request: beam_10m" in (root / "SUMMARY.md").read_text()
    assert (root / "beam_10m/metrics.json").exists()


def test_locomo_headline_and_adversarial_separate_with_full_denominators(tmp_path):
    questions = [question("q1"), question("q2"), question("q3", "temporal"), question("q4", "adversarial", eligible=False)]
    questions[0]["evidence_complete"] = False
    data, out, args = setup_dataset(tmp_path, "locomo", questions, [answer("q1"), answer("q4", cost=.5, cached=True)])
    create_meter(args.output / "_meter", [("locomo", "done", 1), ("locomo", "done", .1),
                                         ("locomo", "done", .2), ("locomo", "pending", .4),
                                         ("locomo", "error", .3), ("other", "done", .7)])
    metrics = summarise(data, out, args)
    assert metrics["questions"] == 3
    assert metrics["full_dataset_questions"] == 4
    assert metrics["answered"] == 1
    assert metrics["accuracy_answered"] == 1
    assert metrics["verified_correct_fraction_of_full"] == pytest.approx(1 / 3)
    assert metrics["status"] == "incomplete"
    assert metrics["answer_cost_usd"] == 1
    assert metrics["answer_cost_per_correct_usd"] == 1
    assert metrics["retrieval_incomplete_evidence_excluded"] == 1
    assert metrics["retrieval_gold_questions"] == 2
    assert metrics["reader"] == "frozen-reader"
    assert len(metrics["config_sha256"]) == 64
    adversarial = metrics["cohorts"][1]
    assert adversarial["cohort"] == "adversarial"
    assert adversarial["status"] == "complete"
    assert adversarial["answer_cost_usd"] == .5
    assert adversarial["incremental_answer_usd_recorded"] == 0
    assert metrics["budget"]["actual_api_usd"] == pytest.approx(1.3)
    assert metrics["budget"]["unresolved_reserved_usd"] == pytest.approx(.7)
    assert metrics["budget"]["committed_usd"] == pytest.approx(2)
    assert metrics["budget"]["global"]["committed_usd"] == pytest.approx(2.7)
    assert metrics["by_type"]["temporal"]["answered"] == 0
    rows = report(args.output)
    assert len(rows) == 2
    text = (args.output / "SUMMARY.md").read_text()
    assert "headline" in text and "adversarial" in text
    assert "1 / 3 / 3" in text
    assert (args.output / "BUDGET.csv").exists()


def test_f1_is_not_accuracy_and_cost_correct_uses_exact_match(tmp_path):
    questions = [question("q1", metric="f1"), question("q2", metric="f1"), question("q3", metric="f1")]
    answers = [answer("q1", .8, 0, metric="f1"), answer("q2", 1, 1, metric="f1")]
    data, out, args = setup_dataset(tmp_path, "musique", questions, answers, limit=2)
    metrics = summarise(data, out, args)
    assert metrics["scope"] == "smoke"
    assert metrics["status"] == "incomplete"
    assert metrics["requested_questions"] == 2
    assert metrics["questions"] == 3
    assert metrics["f1_mean"] == pytest.approx(.9)
    assert metrics["exact_match_accuracy"] == .5
    assert metrics["correct"] == 1
    assert metrics["answer_cost_per_correct_usd"] == 2
    assert metrics["cost_per_correct_basis"] == "exact_match"


def test_beam_graded_scores_never_manufacture_binary_accuracy(tmp_path):
    questions = [question("q1", metric="beam_rubric_mean")]
    # Even a stale producer's binary field must not be used for this metric.
    data, out, args = setup_dataset(tmp_path, "beam_100k", questions, [answer("q1", .5, 1, metric="beam_rubric_mean")])
    metrics = summarise(data, out, args)
    assert metrics["mean_score"] == .5
    assert metrics["status"] == "complete"
    assert metrics["correct"] is None
    assert metrics["accuracy_answered"] is None
    assert metrics["answer_cost_per_correct_usd"] is None
    assert metrics["score_label"] == "graded rubric mean"


def test_persistent_meter_role_attribution_and_missing_meter(tmp_path):
    assert not read_budget(tmp_path / "missing.sqlite", "x")["available"]
    create_meter(tmp_path / "_meter", [("x", "done", .3), ("x", "error", .2)], with_role=True)
    budget = read_budget(tmp_path / "_meter/calls.sqlite", "x")
    assert budget["actual_answer_usd"] == .3
    assert budget["actual_judge_usd"] == 0
    assert budget["error_calls"] == 1
    assert budget["committed_usd"] == .5


def test_rerun_records_deduplicated_but_index_wall_time_is_cumulative(tmp_path):
    data, out, args = setup_dataset(tmp_path, "test", [question("q1")], [answer("q1", cost=.2), answer("q1", cost=.3)])
    with (out / "index.jsonl").open("a") as handle:
        handle.write(json.dumps({"group": "g", "documents": 1, "source_tokens": 100, "build_seconds": .1,
                                 "cache_hit": True, "storage_bytes": 64, "llm_usd": 0}) + "\n")
    metrics = summarise(data, out, args)
    assert metrics["answered"] == 1
    assert metrics["source_tokens"] == 100
    assert metrics["index_storage_bytes"] == 64
    assert metrics["index_seconds"] == 2.1
    assert metrics["answer_cost_usd"] == .3
    assert metrics["serving_p95_seconds"] == 2.1
    first_hash = metrics["config_sha256"]
    assert summarise(data, out, args)["config_sha256"] == first_hash


def test_no_partial_rubric_score_marked_complete(tmp_path):
    data, out, args = setup_dataset(tmp_path, "beam_1m", [question("q1", metric="beam_rubric_mean")], [answer("q1", None, None)])
    metrics = summarise(data, out, args)
    assert metrics["answered"] == 1
    assert metrics["scored"] == 0
    assert metrics["status"] == "incomplete"
    assert metrics["mean_score"] is None


def test_concurrent_root_reports_are_serialized(tmp_path):
    data, out, args = setup_dataset(tmp_path, "test", [question("q1")], [answer("q1")])
    summarise(data, out, args)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(report, [args.output] * 8))
    assert all(len(rows) == 1 for rows in results)
    assert (args.output / ".report.lock").exists()
    assert (args.output / "SUMMARY.md").read_text().endswith("\n")

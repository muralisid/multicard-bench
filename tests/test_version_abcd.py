import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from multicard.version_abcd.facts import FactStore, parse_facts, PROMPT
from multicard.version_abcd.temporal import build_timeline, visible_facts, date_value
from multicard.version_abcd.graph import EvidenceGraph
from multicard.version_abcd.runner import sample_questions, render
from multicard.version_abcd.report import bootstrap
from multicard.version_abcd.report import report
from multicard.version_a.core import append_jsonl
from test_version_a_core import WordTokenizer


def fact(uid, obj, date="2023-01-01", **extra):
    return {"id": uid, "subject": "Alice", "predicate": "lives_in", "object": obj,
            "quote": "Alice lives in " + obj, "kind": "state", "cardinality": "one", "scope": "home",
            "namespace": "alice-history", "unit_id": uid, "source_date": date,
            "valid_from": None, "valid_to": None, "negated": False, "serial": None, **extra}


def test_fact_validation_rejects_fabricated_quotes_and_objects():
    facts = [fact("a", "Paris"), fact("b", "Rome"), dict(fact("c", "Paris"), object="Berlin")]
    result, rejected = parse_facts(json.dumps({"facts": facts}), "Alice lives in Paris")
    assert len(result) == 1 and result[0]["object"] == "Paris" and rejected == 2


def test_serial_is_grounded_in_source_and_not_invented_by_extractor():
    result, _ = parse_facts(json.dumps({"facts": [fact("a", "Paris", serial=999)]}), "42. Alice lives in Paris")
    assert result[0]["serial"] == 42


def test_predicate_aliases_and_null_scope_normalize_before_temporal_comparison():
    values = [fact("a", "Paris", predicate="resides in", scope=None), fact("b", "Rome", predicate="lives_in", scope="")]
    got, _ = parse_facts(json.dumps({"facts": values}), "Alice lives in Paris. Alice lives in Rome.")
    assert {f["predicate"] for f in got} == {"lives_in"}
    assert {f["scope"] for f in got} == {""}


def test_once_only_cache_under_concurrent_queries_and_namespace_isolation(tmp_path):
    calls = []
    def generate(prompt, allowance, role):
        calls.append(prompt)
        return {"text": json.dumps({"facts": [fact("a", "Paris")]}), "key": "api-key", "usd": .01,
                "seconds": 1, "tokens_in": 10, "tokens_out": 20}
    client = SimpleNamespace(generate=generate)
    store = FactStore(tmp_path)
    doc = {"id": "source", "text": "Alice lives in Paris", "speaker": "Alice"}
    piece = {"text": doc["text"], "start": 0, "end": len(doc["text"])}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.extract("alice", doc, piece, client), range(8)))
    assert len(calls) == 1 and sum(not cached for _, cached in results) == 1
    store.extract("different-history", doc, piece, client)
    assert len(calls) == 2
    assert "Question:" not in calls[0] and "reference_answers" not in calls[0]


def test_invalid_extraction_is_cached_and_falls_back_without_quality_retry(tmp_path):
    calls = []
    def generate(*args, **kwargs):
        calls.append(1)
        return {"text": "broken", "key": "k", "usd": .1, "seconds": 1, "tokens_in": 1, "tokens_out": 1}
    store = FactStore(tmp_path)
    for _ in range(2):
        result, _ = store.extract("scope", {"id": "x"}, {"text": "text"}, SimpleNamespace(generate=generate))
        assert result["status"] == "invalid_output_raw_fallback"
    assert len(calls) == 1


def test_temporal_current_and_history_and_as_of_keep_correct_versions():
    timeline = build_timeline([fact("old", "Paris"), fact("new", "Rome", "2024-01-01")])
    assert timeline[0]["valid_to"] == "2024-01-01"
    assert [f["id"] for f in visible_facts(timeline, "Where does Alice live?")] == ["new"]
    assert [f["id"] for f in visible_facts(timeline, "Where does Alice live?", "2023-08-01")] == ["old"]
    assert len(visible_facts(timeline, "Where did Alice live before?")) == 2


@pytest.mark.parametrize("change", [{"namespace": "other"}, {"scope": "holiday"}, {"cardinality": "many"}, {"kind": "event"}])
def test_no_cross_scope_or_multivalued_event_invalidation(change):
    old = fact("old", "Paris", **change)
    new = fact("new", "Rome", "2024-01-01", **change)
    if "namespace" in change or "scope" in change:
        new = fact("new", "Rome", "2024-01-01")
    timeline = build_timeline([old, new])
    assert all(f["superseded_by"] is None for f in timeline)


def test_undated_or_equal_time_conflicts_are_not_silently_overwritten():
    for date in (None, "2023-01-01"):
        timeline = build_timeline([fact("a", "Paris", date), fact("b", "Rome", date)])
        assert all(f["superseded_by"] is None for f in timeline)


def test_numbered_counterfactual_updates_override_real_world_and_keep_history():
    timeline = build_timeline([fact("a", "Paris", None, serial=10), fact("b", "Mars", None, serial=20)], numbered=True)
    assert [f["object"] for f in visible_facts(timeline, "Where?", numbered=True)] == ["Mars"]
    assert len(timeline) == 2


def test_source_date_formats_are_stable():
    assert date_value("2023/05/30 (Tue) 23:40") == "2023-05-30"
    assert date_value("March-15-2024") == "2024-03-15"
    assert date_value(None) is None


def test_topic_community_and_relation_indexes_reach_supported_bridge():
    docs = [{"id": "a", "text": "Alice works at Acme research laboratory"},
            {"id": "b", "text": "Acme research laboratory headquarters are in Paris"},
            {"id": "c", "text": "Banana bread recipe and kitchen"}]
    fs = [fact("a", "Acme", predicate="works_for"), fact("b", "Paris", subject="Acme", predicate="headquarters")]
    graph = EvidenceGraph(docs, fs)
    ranking, channels = graph.rank("Where are Alice headquarters located?", ["a", "c"])
    assert "b" in ranking and "b" in channels["relations"]
    assert graph.topics and graph.communities
    assert {edge[1] for edge in graph.edges} >= {"subject", "object", "supported_by", "belongs_to_topic", "belongs_to_community"}


def test_personal_pronoun_can_enter_the_users_relation_graph():
    docs = [{"id": "a", "text": "I work at Acme"}, {"id": "b", "text": "Acme headquarters are Paris"}]
    graph = EvidenceGraph(docs, [fact("a", "Acme", subject="user"), fact("b", "Paris", subject="Acme")])
    _, channels = graph.rank("Where are my company's headquarters?", ["a"])
    assert "b" in channels["relations"]


def test_sampling_is_stratified_deterministic_and_score_blind():
    qs = [{"id": str(i), "type": "rare" if i < 2 else "common", "answers": ["old"]} for i in range(200)]
    first = sample_questions(qs, 100)
    second = sample_questions([dict(q, answers=["new"]) for q in qs], 100)
    assert len(first) == 100 and {q["id"] for q in first} == {q["id"] for q in second}
    assert any(q["type"] == "rare" for q in first)


def test_render_never_restores_superseded_raw_text_and_respects_token_budget():
    docs = {"old": {"text": "Alice lives in Paris"}, "new": {"text": "Rome " * 5000}}
    result = render(["old", "new"], docs, {}, {"old"}, WordTokenizer(), temporal=True)
    assert "Paris" not in result["context"] and result["context_tokens"] <= 4000


def test_cluster_interval_does_not_fake_independent_histories():
    assert bootstrap([("one", 1), ("one", -1)]) == (None, None)
    lo, hi = bootstrap([("one", .1), ("two", .1)])
    assert lo == pytest.approx(.1) and hi == pytest.approx(.1)


def test_paired_report_counts_shared_setup_without_mixing_questions(tmp_path):
    out = tmp_path / "dataset"
    out.mkdir()
    (out / "sample.json").write_text(json.dumps({"full_count": 3, "questions": [
        {"id": "q1", "group": "g1", "type": "one", "metric": "f1"},
        {"id": "q2", "group": "g2", "type": "one", "metric": "f1"}]}))
    for arm in "ABCD":
        for uid in ("q1", "q2"):
            if arm == "D" and uid == "q2":
                continue
            append_jsonl(out / f"answers_{arm}.jsonl", {"id": uid, "score": 1 if arm != "A" else 0,
                         "correct": 1, "answer_usd": .1, "judge_usd": .01, "supplementary": {}})
    (out / "groups").mkdir()
    (out / "groups/g1.json").write_text(json.dumps({"group": "g1", "extraction": [{"usd": .5}]}))
    rows = report(tmp_path)
    assert rows[0]["paired"] == 2 and rows[0]["delta_vs_A"] == 1
    assert rows[0]["answer_plus_setup_usd"] == pytest.approx(.7)
    assert rows[0]["total_serving_cost_per_correct"] == pytest.approx(.35)
    assert rows[-1]["paired"] == 1 and rows[-1]["status"] == "incomplete"
    assert rows[-1]["total_serving_cost_per_correct"] is None

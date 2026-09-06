"""Tests for the Graphiti runner part1-tools/graphiti/run_groups.py (design
sections 2, 3 and 5, graphiti arm), on fixtures and without Neo4j, the proxy
or graphiti_core.

The runner lives outside the bench package because it needs graphiti-core's
own venv; it is loaded here by file path and imports cleanly without
graphiti_core. What matters: the three ingestion variants render the episodes
the design describes with the session date at 00:00 UTC; the pilot takes the
first N sessions in date order; the section 3 normalisation; evidence presence
per marked turn; USD at the models.json prices and the length-scaled
projection; the choice rule with its tie-break; Zep's template; the raised
limit ladder; the parquet export schema; the meter's attribution, tags and cap;
the Pilot section written into graphiti.md; and the character lint.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from multicard.data.longmemeval import Instance

RUNNER = Path("/Users/muralisid/github_other/part1-tools/graphiti/run_groups.py")
pytestmark = pytest.mark.skipif(not RUNNER.exists(), reason="run_groups.py not present")


@pytest.fixture(scope="module")
def rg():
    spec = importlib.util.spec_from_file_location("run_groups", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_groups"] = mod  # dataclasses looks the module up by name
    spec.loader.exec_module(mod)
    return mod


MODELS = {"chat_model": "gemini-2.5-flash-lite", "embed_model": "gemini-embedding-001", "embed_dim": 3072,
          "prices": {"gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
                     "gemini-embedding-001": {"in": 0.15, "out": 0.0}}}

SESSIONS = {
    "s1": {"date": "2023-05-22", "turns": [{"role": "user", "content": "I spent $75 at SaveMart."},
                                           {"role": "assistant", "content": "Noted."}]},
    "s2": {"date": "2023-05-20", "turns": [{"role": "user", "content": "Hello there"},
                                           {"role": "assistant", "content": "   "},
                                           {"role": "user", "content": "Plan my week"}]},
    "s3": {"date": "2023-05-20", "turns": [{"role": "user", "content": "Second on the same date"}]},
    "s4": {"date": "2023-02-01", "turns": []},
}


def _inst(**kw):
    base = dict(qid="q1", qtype="multi-session", question="How much cashback?", answer="$0.75",
                question_date="2023-05-30", abstention=False, session_ids=["s1", "s2", "s3", "s4"],
                evidence_sessions={"s1", "s2"}, evidence_turns={"s1#0", "s2#2"})
    base.update(kw)
    return Instance(**base)


def test_episode_plan_three_variants(rg):
    text = rg.episode_plan("text_session", "s1", SESSIONS["s1"])
    msg = rg.episode_plan("message_session", "s1", SESSIONS["s1"])
    turns = rg.episode_plan("per_turn", "s1", SESSIONS["s1"])
    assert [e.source for e in text] == ["text"] and [e.source for e in msg] == ["message"]
    assert text[0].body == "user: I spent $75 at SaveMart.\nassistant: Noted."
    assert text[0].name == "s1" and text[0].turn_index is None and text[0].session_id == "s1"
    assert text[0].reference_time.isoformat() == "2023-05-22T00:00:00+00:00"
    assert [e.name for e in turns] == ["s1#0", "s1#1"]
    assert [e.body for e in turns] == ["user: I spent $75 at SaveMart.", "assistant: Noted."]
    assert [e.turn_index for e in turns] == [0, 1] and all(e.source == "message" for e in turns)
    # empty session gives no episode; a blank turn is skipped under per_turn
    assert rg.episode_plan("per_turn", "s4", SESSIONS["s4"]) == []
    assert [e.turn_index for e in rg.episode_plan("per_turn", "s2", SESSIONS["s2"])] == [0, 2]
    with pytest.raises(ValueError):
        rg.episode_plan("other", "s1", SESSIONS["s1"])


def test_session_order_and_pilot_selection(rg):
    ids = ["s1", "s2", "s3", "s4"]
    assert rg.session_order(ids, SESSIONS) == ["s4", "s2", "s3", "s1"]  # date, then haystack position
    assert rg.session_order(["s3", "s2"], SESSIONS) == ["s3", "s2"]
    assert rg.pilot_sessions(ids, SESSIONS, 2) == ["s4", "s2"]
    plan, records = rg.plan_question(["s4", "s2"], SESSIONS, "per_turn")
    assert [r["empty"] for r in records] == [True, False]
    assert records[1]["episodes"] == 2 and len(plan) == 2


def test_normalise_section_3(rg):
    assert rg.normalise("  $0.75 ") == "0.75"
    assert rg.normalise('"Hello,  World!"') == "hello, world"
    assert rg.normalise("\uff21\uff22\uff23 x") == "abc x"   # NFKC folds fullwidth letters
    assert rg.normalise("\u201cquoted\u201d") == "quoted"    # curly quotes stripped
    assert rg.normalise("a \n\t b") == "a b"
    assert rg.normalise("") == "" and rg.normalise(None) == ""


def test_evidence_presence_rules(rg):
    inst = _inst()
    episodes = [{"uuid": "e10", "session_id": "s1", "turn_index": 0},
                {"uuid": "e11", "session_id": "s1", "turn_index": 1}]
    # cited through the session's second turn episode, answer not in any fact
    ev = rg.evidence_presence(inst, [{"fact": "user shops at SaveMart", "episodes": ["e11"]}], episodes)
    rows = {r["turn"]: r for r in ev["turns"]}
    assert rows["s1#0"]["cited"] and rows["s1#0"]["cited_turn_episode"] is False and rows["s1#0"]["present"]
    assert rows["s2#2"]["session_in_pilot"] is False and not rows["s2#2"]["present"]
    assert ev["n_turns"] == 2 and ev["n_present"] == 1 and ev["presence"] == 0.5
    assert ev["n_ingested"] == 1 and ev["presence_over_ingested"] == 1.0
    assert ev["gold_normalised"] == "0.75" and ev["answer_in_any_fact"] is False
    # the gold answer inside a fact counts for every marked turn, cited or not
    ev2 = rg.evidence_presence(inst, [{"fact": "User earned $0.75 cashback.", "episodes": []}], episodes)
    assert all(r["present"] and r["answer_in_fact"] and not r["cited"] for r in ev2["turns"])
    assert ev2["presence"] == 1.0
    # nothing extracted
    ev3 = rg.evidence_presence(inst, [], [])
    assert ev3["presence"] == 0.0 and ev3["presence_over_ingested"] is None


def test_cost_projection_and_choice(rg, tmp_path):
    usage = dict(rg.fresh_usage(), llm_tokens_in=1_000_000, llm_tokens_out=100_000, embed_tokens_in=200_000)
    assert rg.cost_usd(MODELS, usage) == pytest.approx(0.10 + 0.04 + 0.03)
    # USD per rendered character in the pilot times the rendered characters of the subset's slots
    assert rg.project(0.01, 5034, 74_150_000) == pytest.approx(0.01 / 5034 * 74_150_000)
    assert rg.project(1.0, 0, 100) is None and rg.project(1.0, 10, 0) is None
    # the basis is read from subsets.json and the loader at pilot time: haystack slots, not unique sessions
    f = tmp_path / "subsets.json"
    f.write_text(json.dumps({"ORDER": ["q1", "q2"], "GRAPHITI_150": ["q1", "q2"]}))
    data = {"instances": [_inst(qid="q1", session_ids=["s1", "s2", "s4"]), _inst(qid="q2", session_ids=["s2", "s3"])],
            "sessions": SESSIONS}
    basis = rg.graphiti_basis(f, data)
    chars = {s: len(rg.render_session(SESSIONS[s]["turns"])) if SESSIONS[s]["turns"] else 0 for s in SESSIONS}
    assert basis["n_questions"] == 2 and basis["slots"] == 5 and basis["unique_sessions"] == 4
    assert basis["chars"] == chars["s1"] + 2 * chars["s2"] + chars["s3"] + chars["s4"]
    assert basis["mean_chars_per_slot"] == pytest.approx(basis["chars"] / 5)

    def row(v, pres, usd, partial=False):
        return {"variant": v, "evidence_presence": pres, "projected_usd": usd, "partial": partial}

    # highest presence among those that fit; the best one does not fit
    choice, why = rg.choose_variant([row("text_session", 0.5, 10), row("message_session", 1.0, 80),
                                     row("per_turn", 0.75, 30)])
    assert choice == "per_turn" and "highest" in why
    # tie goes to per_turn
    choice, why = rg.choose_variant([row("text_session", 1.0, 10), row("per_turn", 1.0, 59.9)])
    assert choice == "per_turn" and "tie" in why
    # tie without per_turn goes to message_session
    assert rg.choose_variant([row("text_session", 0.5, 10), row("message_session", 0.5, 10),
                              row("per_turn", 0.0, 10)])[0] == "message_session"
    # nothing fits, and a partial row is never chosen
    assert rg.choose_variant([row("per_turn", 1.0, 61)])[0] is None
    assert rg.choose_variant([row("per_turn", 1.0, 10, partial=True)])[0] is None
    assert rg.choose_variant([row("per_turn", None, 10)])[0] == "per_turn"


def test_render_context_zep_template(rg):
    edges = [{"fact": "SaveMart has a sale", "valid_at": "2023-05-22T00:00:00+00:00", "invalid_at": None},
             {"fact": "Old price", "valid_at": None, "invalid_at": "2023-05-23T00:00:00+00:00"}]
    nodes = [{"name": "SaveMart", "summary": "A grocery store."}, {"name": "user", "summary": None}]
    ctx = rg.render_context(edges, nodes)
    lines = ctx.split("\n")
    assert lines[0] == "FACTS and ENTITIES represent relevant context to the current conversation."
    assert "# format: FACT (Date range: from - to)" in lines
    assert "# ENTITY_NAME: entity summary" in lines
    assert "  - SaveMart has a sale (2023-05-22 00:00:00+00:00 - present)" in lines
    assert "  - Old price (date unknown - 2023-05-23 00:00:00+00:00)" in lines
    assert "  - SaveMart: A grocery store." in lines and "  - user: " in lines
    assert ctx.index("<FACTS>") < ctx.index("</FACTS>") < ctx.index("<ENTITIES>") < ctx.index("</ENTITIES>")
    assert "<FACTS>\n\n</FACTS>" in rg.render_context([], [])  # empty scopes keep the frame


def test_pick_ladder(rg):
    tokens = {20: 1200, 40: 4300, 80: 7500}
    assert rg.pick_ladder(tokens, 4000) == (40, True)
    assert rg.pick_ladder(tokens, 8000) == (80, False)      # the top step did not reach B (design gap)
    assert rg.pick_ladder({20: 9000}, 4000) == (20, True)
    assert rg.pick_ladder({20: 9000}, 8000) == (20, True)
    assert rg.pick_ladder({}, 4000) == (None, False)
    assert rg.BUDGET_TOKENS == {"raised_4k": 4000, "raised_8k": 8000} and rg.CONCURRENCY_DEFAULT == 1


def test_rendered_tokens_match_the_bench_counter_and_lines(rg):
    from multicard.llm.costmeter import TokenCounter
    from multicard.part1 import competitors as C

    edges = [{"uuid": "e1", "fact": "SaveMart has a sale", "valid_at": "2023-05-22T00:00:00+00:00", "invalid_at": None},
             {"uuid": "e1", "fact": "SaveMart has a sale", "valid_at": "2023-05-22T00:00:00+00:00", "invalid_at": None},
             {"uuid": "e2", "fact": "Old price", "valid_at": None, "invalid_at": "2023-05-23T00:00:00+00:00"}]
    nodes = [{"uuid": "n1", "name": "SaveMart", "summary": "A grocery store."}, {"uuid": "n2", "name": "user", "summary": None}]
    assert rg.fact_line(edges[0]) == C.fact_line(edges[0]) and rg.entity_summary_line(nodes[1]) == C.entity_summary_line(nodes[1])
    bench = TokenCounter()
    ours = rg.BenchTokenCounter()
    assert ours.count(rg.fact_line(edges[0])) == bench.count(rg.fact_line(edges[0]))
    # summed per unit, the duplicate uuid counted once
    expect = sum(bench.count(rg.fact_line(e)) for e in edges[1:]) + sum(bench.count(rg.entity_summary_line(n)) for n in nodes)
    assert rg.rendered_tokens(edges, nodes, ours) == expect
    assert rg.rendered_tokens([], [], ours) == 0


def test_pilot_spend_sources(rg, tmp_path):
    f = tmp_path / "subsets.json"
    f.write_text(json.dumps({"GRAPHITI_150": ["q1"], "derived": {"GRAPHITI_PILOT": "q1"}}))
    p = tmp_path / "pilot.json"
    p.write_text(json.dumps({"usd_total": 0.75}))
    assert rg.pilot_spend(SimpleNamespace(spent_usd=1.5, pilot_json=None, subsets=str(f))) == 1.5
    assert rg.pilot_spend(SimpleNamespace(spent_usd=None, pilot_json=str(p), subsets=str(f))) == 0.75
    assert rg.pilot_spend(SimpleNamespace(spent_usd=None, pilot_json=str(tmp_path / "none.json"), subsets=str(f))) == 0.0
    args = rg.parse_args(["run", "--variant", "per_turn", "--spent-usd", "2", "--subsets", str(f)])
    assert args.k == 1 and args.max_usd == 60.0 and rg.pilot_spend(args) == 2.0


def test_parquet_export_schema(rg, tmp_path):
    edges = [{"uuid": "e1", "fact": "f", "name": "HAS", "source_node_uuid": "a", "target_node_uuid": "b",
              "episodes": ["p1", "p2"], "valid_at": "2023-05-22T00:00:00+00:00", "invalid_at": None,
              "created_at": "2026-09-06T00:00:00+00:00", "expired_at": None}]
    nodes = [{"uuid": "a", "name": "SaveMart", "summary": "", "labels": ["Entity"], "created_at": None}]
    eps = [{"uuid": "p1", "name": "s1#0", "source_description": "d", "reference_time": "2023-05-22T00:00:00+00:00",
            "session_id": "s1", "turn_index": 0},
           {"uuid": "p2", "name": "s1", "source_description": "d", "reference_time": "2023-05-22T00:00:00+00:00",
            "session_id": "s1", "turn_index": None}]
    assert rg.write_parquet(edges, rg.EDGE_SCHEMA, tmp_path / "edges.parquet") == 1
    rg.write_parquet(nodes, rg.NODE_SCHEMA, tmp_path / "nodes.parquet")
    rg.write_parquet(eps, rg.EPISODE_SCHEMA, tmp_path / "episodes.parquet")
    e = pq.read_table(tmp_path / "edges.parquet")
    assert e.column_names == ["uuid", "fact", "name", "source_node_uuid", "target_node_uuid", "episodes",
                              "valid_at", "invalid_at", "created_at", "expired_at"]
    assert e.column("episodes").to_pylist() == [["p1", "p2"]] and e.column("invalid_at").to_pylist() == [None]
    n = pq.read_table(tmp_path / "nodes.parquet")
    assert n.column_names == ["uuid", "name", "summary", "labels", "created_at"]
    p = pq.read_table(tmp_path / "episodes.parquet")
    assert p.column_names == ["uuid", "name", "source_description", "reference_time", "session_id", "turn_index"]
    assert p.column("turn_index").to_pylist() == [0, None] and str(p.schema.field("turn_index").type) == "int64"
    assert rg.write_parquet([], rg.EDGE_SCHEMA, tmp_path / "empty.parquet") == 0


def test_load_subsets_forms(rg, tmp_path):
    f = tmp_path / "subsets.json"
    f.write_text(json.dumps({"ORDER": ["c", "a", "b", "d"], "GRAPHITI_150": ["a", "b", "c", "z"]}))
    assert rg.load_subsets(f) == ["c", "a", "b", "z"]  # ORDER first, unknown ids last in place
    f.write_text(json.dumps({"ORDER": {"longmemeval": ["b", "a"], "multihoprag": ["m1"]},
                             "GRAPHITI_150": {"longmemeval": ["a", "b"]}}))
    assert rg.load_subsets(f) == ["b", "a"]
    f.write_text(json.dumps({"GRAPHITI_150": ["x", "y"]}))
    assert rg.load_subsets(f) == ["x", "y"]


def test_rpm_stats(rg):
    times = [(100.0 + i, ("g1",)) for i in range(30)] + [(100.0 + 200 + i, ("g1",)) for i in range(5)]
    times += [(100.0, ("g2",))]
    s = rg.rpm_stats(times, 100.0, 100.0 + 220, ("g1",))
    assert s["requests"] == 35 and s["peak_rpm_60s"] == 30
    assert s["mean_rpm"] == pytest.approx(35 / (220 / 60), abs=0.1)
    assert rg.rpm_stats(times, 100.0, 100.0 + 220)["requests"] == 36


class _Resp:
    def __init__(self, pt, ct):
        self.usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)
        self.choices = []


class _Client:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._emb)

    async def _chat(self, **kw):
        self.calls.append(kw)
        return _Resp(1000, 100)

    async def _emb(self, **kw):
        self.calls.append(kw)
        return _Resp(50, 0)


def test_meter_attribution_tags_and_cap(rg):
    client = _Client()
    m = rg.Meter(MODELS, cap_usd=0.0002, job_tag="graphiti-test")
    m.wrap(client)

    async def go():
        m.label.set(("g1", "s1", "s1#0"))
        await client.chat.completions.create(model="x", messages=[{"role": "system", "content": "sys"}])
        await client.embeddings.create(model="e", input=["a", "b"])
        m.label.set(("g2", "s9", "s9"))
        await client.chat.completions.create(model="x", messages=[{"role": "system", "content": "sys"}])

    asyncio.run(go())
    g1 = m.total(("g1",))
    assert g1["llm_calls"] == 1 and g1["llm_tokens_in"] == 1000 and g1["llm_tokens_out"] == 100
    assert g1["embed_calls"] == 1 and g1["embed_tokens_in"] == 50 and g1["embed_inputs"] == 2
    assert m.total(("g1", "s1"))["llm_calls"] == 1 and m.total(("g2",))["embed_calls"] == 0
    assert m.total()["llm_calls"] == 2
    assert m.usd(("g1",)) == pytest.approx(1000 * 0.10 / 1e6 + 100 * 0.40 / 1e6 + 50 * 0.15 / 1e6)
    assert m.over_cap()  # 0.0002875 USD over the three calls, above the 0.0002 cap
    assert client.calls[0]["user"] == "graphiti-test"
    assert client.calls[0]["extra_body"] == {"metadata": {"tags": ["graphiti-test", "g1"]}}
    assert client.calls[2]["extra_body"]["metadata"]["tags"] == ["graphiti-test", "g2"]
    assert len(m.times) == 3 and m.times[2][1] == ("g2", "s9", "s9")


def test_meter_cap_boundary(rg):
    client = _Client()
    m = rg.Meter(MODELS, cap_usd=1.0, job_tag="t")
    m.wrap(client)
    asyncio.run(client.chat.completions.create(model="x", messages=[]))
    assert not m.over_cap()
    m.cap_usd = m.usd()
    assert m.over_cap()


def test_pilot_table_and_md_section(rg, tmp_path):
    rows = [{"variant": v, "sessions": 3, "episodes": 3, "chars": 100, "llm_calls": 6, "tokens_in": 1000,
             "tokens_out": 50, "embed_calls": 4, "embed_tokens": 20, "wall_s": 12.3, "rpm_mean": 40.0,
             "rpm_peak": 12, "usd": 0.001, "edges": 2, "nodes": 5, "evidence_presence": 0.5,
             "evidence_n_present": 1, "evidence_n": 2, "projected_usd": 12.5, "fits_cap": True,
             "projected_wall_h": 3.0, "partial": v == "per_turn"} for v in rg.VARIANTS]
    table = rg.pilot_table_md(rows)
    assert table.count("\n") == 4 and "| per_turn (partial) |" in table and "1/2 (0.50)" in table
    md = tmp_path / "graphiti.md"
    md.write_text("# Title\n\n## 1. Versions\n\nold text\n\n## 9. Files\n\n- a\n")
    rg.write_md_section(md, "Pilot", "first body\n")
    text = md.read_text()
    assert text.index("## 9. Files") < text.index("## Pilot") and text.endswith("first body\n")
    rg.write_md_section(md, "Pilot", "second body\n")
    text = md.read_text()
    assert "first body" not in text and text.count("## Pilot") == 1 and "## 9. Files\n\n- a\n" in text
    # replacing a section in the middle keeps what follows
    md.write_text("## A\n\na\n\n## Pilot\n\nold\n\n## B\n\nb\n")
    rg.write_md_section(md, "Pilot", "new")
    assert md.read_text() == "## A\n\na\n\n## Pilot\n\nnew\n\n## B\n\nb\n"


def test_pilot_md_mentions_choice_and_time_of_day(rg):
    ctx = {"date": "2026-09-06", "qid": "q1", "qtype": "multi-session", "question": "Q?", "answer": "$0.75",
           "n": 3, "n_ingested": 3, "n_empty": 0, "evidence_sessions": 2, "evidence_in_pilot": 2,
           "session_override": True, "chat_model": "m", "embed_model": "e", "embed_dim": 3072,
           "temperature": 1, "max_tokens": 16384, "graphiti_core": "0.30.1", "previous_episodes": "library default",
           "job_tag": "tag", "partial": False}
    rows = [{"variant": "per_turn", "sessions": 3, "episodes": 3, "chars": 100, "llm_calls": 6, "tokens_in": 1,
             "tokens_out": 1, "embed_calls": 1, "embed_tokens": 1, "wall_s": 1.0, "rpm_mean": 1.0, "rpm_peak": 1,
             "usd": 0.001, "edges": 1, "nodes": 1, "evidence_presence": 1.0, "evidence_n_present": 2,
             "evidence_n": 2, "projected_usd": 5.0, "fits_cap": True, "projected_wall_h": 1.0}]
    ev = {"per_turn": {"turns": [{"turn": "s1#0", "session_in_pilot": True, "cited": True,
                                  "answer_in_fact": False, "present": True}]}}
    body = rg.pilot_md(ctx, rows, "per_turn", "highest evidence presence", ev)
    assert "Choice: per_turn" in body and "time of day" in body and "| per_turn | s1#0 | yes | yes | no | yes |" in body
    body2 = rg.pilot_md(ctx, rows, None, "no variant fits", ev)
    assert "Choice: none" in body2


def test_recipe_record_and_no_banned_characters(rg):
    assert rg.RECIPE["bfs"] is False and rg.RECIPE["cross_encoder"] is False
    assert rg.RECIPE["edge"]["methods"] == ["bm25", "cosine_similarity"]
    banned = "\u2014\u2013\u2192\u2026\u2018\u2019\u201c\u201d"
    for path in (RUNNER, Path(__file__)):
        text = path.read_text()
        hits = [c for c in banned if c in text]
        assert not hits, f"{path.name} contains {hits}"

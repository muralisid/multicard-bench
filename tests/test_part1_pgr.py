"""Tests for the post-graph-rag runner (part1-tools/pgr/run_spaces.py), design section 4 item 6 and 5.

Only the pure helpers are tested here: document building and turn spans, chunk
mapping, the relation line, the raised top_k rule, shards, the projection, the
usage meter, the id list reader, the parquet writer and the turn-span check.
No database, no proxy, no post_graph_rag import.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

PGR = pathlib.Path("/Users/muralisid/github_other/part1-tools/pgr")
sys.path.insert(0, str(PGR))

import run_spaces as rs  # noqa: E402

PRICES = {"gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
          "gemini-3.6-flash": {"in": 0.75, "out": 3.75},
          "gemini-embedding-001": {"in": 0.15, "out": 0.0}}

TURNS = [{"role": "user", "content": "I moved to Leeds\nlast May."},
         {"role": "assistant", "content": "Nice.\n\nHow is it?"},
         {"role": "user", "content": "Cold."}]
HEADER = "[Conversation on 2023-05-20]\n"


def test_session_document_one_line_per_turn_with_spans():
    text, spans = rs.build_session_document(TURNS, HEADER)
    lines = text.split("\n")
    assert lines[0] == "[Conversation on 2023-05-20]"
    assert lines[1] == "user: I moved to Leeds last May."
    assert lines[2] == "assistant: Nice.  How is it?"
    assert lines[3] == "user: Cold."
    assert len(lines) == 4
    assert len(spans) == 3
    for i, (a, b) in enumerate(spans):
        assert text[a:b] == f"{TURNS[i]['role']}: {rs.oneline(TURNS[i]['content'])}"
    assert spans[0][0] == len(HEADER)
    assert spans[2][1] == len(text)


def test_map_chunk_whole_document_and_partial_and_unmapped():
    text, spans = rs.build_session_document(TURNS, HEADER)
    whole = rs.map_chunk(text, spans, text)
    assert whole == {"char_start": 0, "char_end": len(text), "turn_start": 0, "turn_end": 2}
    a, b = spans[1][0] + 3, spans[2][0] + 4
    part = rs.map_chunk(text, spans, text[a:b])
    assert part["turn_start"] == 1 and part["turn_end"] == 2
    assert part["char_start"] == a and part["char_end"] == b
    assert rs.map_chunk(text, spans, "not in the document") is None
    assert rs.map_chunk(text, spans, "") is None
    header_only = rs.map_chunk(text, spans, HEADER[:10])
    assert header_only["turn_start"] is None and header_only["turn_end"] is None


def test_relation_line_matches_his_rendering():
    r = {"src_id": "User", "tgt_id": "Leeds", "relation_type": "lives_in", "weight": 2,
         "negated": False, "description": "moved last May", "valid_from": "2023-05", "valid_to": None}
    assert rs.relation_line(r) == "- (User) --[lives_in (weight=2)]--> (Leeds) [from 2023-05]: moved last May"
    r.update(negated=True, valid_from="2022", valid_to="2023")
    assert rs.relation_line(r).startswith("- (User) --[NOT lives_in (weight=2)]--> (Leeds) [valid 2022 to 2023]:")
    r.update(valid_from=None, valid_to="2023")
    assert " [until 2023]: " in rs.relation_line(r)
    r.update(valid_to=None)
    assert rs.relation_line(r, render_validity=False) == "- (User) --[NOT lives_in (weight=2)]--> (Leeds): moved last May"


def _data(n_chunks: int, chunk_chars: int, n_rels: int = 0, n_ents: int = 0) -> dict:
    return {"data": {"chunks": [{"chunk_id": str(i), "content": "x" * chunk_chars, "metadata": {"document": "s1"}}
                                for i in range(n_chunks)],
                     "entities": [{"entity_name": f"e{i}", "entity_type": "T", "description": "d"} for i in range(n_ents)],
                     "relationships": [{"edge_id": str(i), "src_id": "a", "tgt_id": "b", "relation_type": "r", "weight": 1,
                                        "negated": False, "description": "d"} for i in range(n_rels)]}}


class Chars:
    """One token per character, so the budget is a character budget."""

    def count(self, text):
        return len(text)


def test_context_chars_counts_chunks_and_relation_lines():
    c = rs.context_chars(_data(2, 100, 1))
    assert c["chunks"] == 200 and c["n_chunks"] == 2 and c["n_relations"] == 1
    assert c["relation_lines"] == len(rs.relation_line(_data(0, 0, 1)["data"]["relationships"][0]))
    assert c["total"] == c["chunks"] + c["relation_lines"]


def test_rendered_units_follow_the_bench_rendering():
    from multicard.part1 import competitors as C

    d = _data(2, 10, 1, 1)
    units = rs.rendered_units(d)
    assert [u for u, _ in units] == ["chunk:0", "chunk:1", "entity:e0", "rel:0"]
    data = d["data"]
    assert units[0][1] == C.chunk_line(1, "s1", "x" * 10) == "Chunk [1] (s1): " + "x" * 10
    assert units[2][1] == C.entity_line(data["entities"][0])
    assert units[3][1] == C.relation_line(data["relationships"][0]) == rs.relation_line(data["relationships"][0])
    # a duplicated chunk id is rendered once, as the bench's render deduplicates by unit id
    d["data"]["chunks"].append(dict(d["data"]["chunks"][0]))
    assert len(rs.rendered_units(d)) == 4
    assert rs.rendered_tokens(d, Chars()) == sum(len(line) for _, line in rs.rendered_units(d))


def test_bench_token_counter_equals_the_bench_tokenizer():
    from multicard.llm.costmeter import TokenCounter

    ours = rs.BenchTokenCounter()
    bench = TokenCounter()
    for text in ("Hello world", "[2023-05-20, user] I bought a red bike. It cost $200!",
                 "Chunk [1] (s1): user: Cafe naive resume emoji\nassistant: ok", "x" * 3000, "", "word " * 700,
                 rs.relation_line({"src_id": "User", "tgt_id": "Leeds", "relation_type": "lives_in", "weight": 2,
                                   "negated": False, "description": "moved last May", "valid_from": "2023-05",
                                   "valid_to": None})):
        assert ours.count(text) == bench.count(text), text[:40]
    assert rs.BUDGET_TOKENS == {"raised_4k": 4000, "raised_8k": 8000}


def test_raise_until_steps_and_reuse():
    calls = []
    counter = Chars()
    prefix = len("Chunk [1] (s1): ")

    async def run_k(k):
        calls.append(k)
        return _data(k, 250)          # about k * (250 + prefix) tokens: 16 -> 4256, 32 -> 8512

    shipped = _data(8, 250)           # about 2128 tokens, below both budgets
    assert rs.rendered_tokens(shipped, counter) == 8 * (250 + prefix)
    out = asyncio.run(rs.raise_until(run_k, shipped, counter))
    assert calls == [16, 32]
    assert out["raised_4k"]["top_k"] == 16 and out["raised_4k"]["reached"] and out["raised_4k"]["reused_from"] is None
    assert out["raised_4k"]["rendered_tokens"] == rs.rendered_tokens(_data(16, 250), counter) >= 4000
    assert out["raised_4k"]["target_tokens"] == 4000
    assert out["raised_8k"]["top_k"] == 32 and out["raised_8k"]["reached"] and out["raised_8k"]["reused_from"] is None

    # Shipped already over both budgets: no call, both variants reuse it.
    calls.clear()
    big = _data(8, 10_000)
    out = asyncio.run(rs.raise_until(run_k, big, counter))
    assert calls == []
    assert out["raised_4k"] == {"top_k": 8, "data": big, "reached": True, "reused_from": "shipped",
                                "rendered_tokens": rs.rendered_tokens(big, counter), "target_tokens": 4000}
    assert out["raised_8k"]["reused_from"] == "shipped" and out["raised_8k"]["top_k"] == 8

    # One call serves both budgets: raised_8k reuses raised_4k.
    calls.clear()

    async def run_big(k):
        calls.append(k)
        return _data(k, 3000)         # 16 -> about 48000

    out = asyncio.run(rs.raise_until(run_big, shipped, counter))
    assert calls == [16]
    assert out["raised_4k"]["reused_from"] is None and out["raised_8k"]["reused_from"] == "raised_4k"

    # Never reached: stops at the top step (design gap) and says so.
    calls.clear()

    async def run_small(k):
        calls.append(k)
        return _data(1, 10)

    out = asyncio.run(rs.raise_until(run_small, shipped, counter))
    assert calls == [16, 32, 64, 128]
    assert out["raised_4k"]["top_k"] == 128 and out["raised_4k"]["reached"] is False
    assert out["raised_8k"]["top_k"] == 128 and out["raised_8k"]["reused_from"] == "raised_4k"


def test_cap_defaults_question_forms_and_shipped_top_k():
    a = rs.parse_args(["--corpus", "lme", "--smoke"])
    assert a.cap_usd == 130.0 and "section 11" in a.cap_usd_source
    assert a.shipped_top_k == 8 and a.question_form == "reader_sweep"
    a = rs.parse_args(["--corpus", "lme", "--smoke", "--calibration"])
    assert a.cap_usd == 70.0 and a.index_model == "gemini-3.7-flash" and a.answer_model == "gemini-3.6-flash"
    a = rs.parse_args(["--corpus", "mhrag", "--smoke", "--cap-usd", "2.5", "--shipped-top-k", "5",
                       "--question-form", "run_py"])
    assert a.cap_usd == 2.5 and a.cap_usd_source == "--cap-usd override" and a.shipped_top_k == 5
    assert rs.question_sent("What is my cat's name?", "2023-05-30") == "Today is 2023-05-30. What is my cat's name?"
    assert rs.question_sent("Q?", None) == "Q?"
    run_py = rs.question_sent("Q?", "2023-05-30", "run_py")
    assert run_py.startswith("Answer from the facts and conversation excerpts provided.\n") and run_py.endswith("Today is 2023-05-30. Q?")
    with pytest.raises(ValueError):
        rs.question_sent("Q?", None, "other")
    assert rs.CAP_USD == {"lme": 130.0, "calibration": 70.0, "mhrag": 10.0}


def test_plan_shards_cuts_on_question_boundaries():
    counts = [("a", 50), ("b", 55), ("c", 49), ("d", 52), ("e", 40)]
    assert rs.plan_shards(counts, 100) == [["a", "b"], ["c", "d"], ["e"]]
    assert rs.plan_shards(counts, 1000) == [["a", "b", "c", "d", "e"]]
    assert rs.plan_shards(counts, 1) == [["a"], ["b"], ["c"], ["d"], ["e"]]
    assert rs.plan_shards([], 100) == []


def test_projection_arithmetic_and_empty_shard():
    p = rs.projection(usd=0.5, sessions=100, seconds=3600, run_sessions=1000, project_sessions=25112)
    assert p["usd_per_session"] == pytest.approx(0.005)
    assert p["projected_usd_run"] == pytest.approx(5.0)
    assert p["projected_usd_full"] == pytest.approx(125.56)
    assert p["projected_hours_full"] == pytest.approx(25112 * 36 / 3600)
    empty = rs.projection(0.0, 0, 0.0, 1000, 25112)
    assert empty["usd_per_session"] is None and empty["projected_usd_run"] is None


def test_meter_prices_windows_and_cap():
    m = rs.UsageMeter(PRICES, cap_usd=0.01)
    m.record("chat", "gemini-2.5-flash-lite", 1_000_000, 1_000_000, reasoning=10)
    assert m.total_usd == pytest.approx(0.5)
    assert m.exceeded is True
    with pytest.raises(rs.BudgetExceeded):
        m.check()
    m2 = rs.UsageMeter(PRICES)
    m2.record("chat", "gemini-2.5-flash-lite", 2000, 500)
    mark = m2.mark()
    m2.record("embed", "gemini-embedding-001", 1000, 0)
    s = m2.summary(mark)
    assert s == {"calls": 1, "chat_calls": 0, "embed_calls": 1, "tokens_in": 1000, "tokens_out": 0,
                 "reasoning_tokens": 0, "usd": pytest.approx(0.00015),
                 "by_model": {"gemini-embedding-001": {"calls": 1, "tokens_in": 1000, "tokens_out": 0,
                                                       "usd": pytest.approx(0.00015)}}}
    whole = m2.summary()
    assert whole["calls"] == 2 and whole["chat_calls"] == 1
    assert whole["usd"] == pytest.approx(2000 * 0.10 / 1e6 + 500 * 0.40 / 1e6 + 0.00015)
    assert rs.price_for(PRICES, "gemini-3.7-flash") == PRICES["gemini-3.6-flash"]
    with pytest.raises(KeyError):
        rs.price_for(PRICES, "no-such-model")


def test_metered_completions_tag_and_usage():
    seen = {}

    class Inner:
        async def create(self, **kw):
            seen.update(kw)
            usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3,
                                    completion_tokens_details=SimpleNamespace(reasoning_tokens=2))
            return SimpleNamespace(usage=usage)

    meter = rs.UsageMeter(PRICES)
    captured = []
    c = rs._MeteredCompletions(Inner(), meter, "tag-1", captured)
    asyncio.run(c.create(model="gemini-2.5-flash-lite", messages=[{"role": "user", "content": "hi"}]))
    assert seen["user"] == "tag-1"
    assert seen["extra_body"]["metadata"]["tags"] == ["tag-1"]
    assert meter.records[0]["tokens_in"] == 7 and meter.records[0]["tokens_out"] == 3
    assert meter.records[0]["reasoning_tokens"] == 2
    assert captured[0]["messages"][0]["content"] == "hi"


def test_read_id_list_and_restrict(tmp_path):
    p = tmp_path / "subsets.json"
    p.write_text(json.dumps({"ORDER": {"lme": ["b", "a", "c"], "mhrag": ["q2", "q1"]},
                             "GRAPHITI_150": ["c", "b"]}))
    assert rs.read_id_list(p, "lme") == ["b", "a", "c"]
    assert rs.read_id_list(p, "mhrag") == ["q2", "q1"]
    assert rs.read_id_list(p, "lme", "GRAPHITI_150") == ["c", "b"]
    assert rs.restrict(["b", "a", "c"], ["c", "b"]) == ["b", "c"]
    assert rs.restrict(["b", "a", "c"], None, limit=2) == ["b", "a"]
    flat = tmp_path / "ids.json"
    flat.write_text(json.dumps(["x", "y"]))
    assert rs.read_id_list(flat, "lme") == ["x", "y"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"ORDER": {"other": []}}))
    with pytest.raises(KeyError):
        rs.read_id_list(bad, "lme")


def test_write_tables_and_turn_span_check(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    text, spans = rs.build_session_document(TURNS, HEADER)
    loc = rs.map_chunk(text, spans, text)
    chunks = [{"chunk_id": "1", "content": text, "doc_id": "s1", "doc_date": "2023-05-20", **loc},
              {"chunk_id": "2", "content": "stray", "doc_id": "s1", "doc_date": "2023-05-20",
               "char_start": -1, "char_end": -1, "turn_start": None, "turn_end": None}]
    entities = [{"entity_key": "10", "name": "User", "aliases": ["me"], "description": "d", "entity_type": None}]
    relations = [{"edge_id": "5", "src_key": "10", "tgt_key": "11", "relation_type": "lives_in",
                  "description": "d", "weight": 1.0, "negated": False, "confidence": None,
                  "valid_from": "2023-05", "valid_to": None, "t_created": "2026", "t_expired": None,
                  "superseded_by": None, "asserted_at": None, "sources": ["1"], "embedding": [0.5, 0.25]}]
    mentions = [{"entity_key": "10", "chunk_id": "1"}]
    counts = rs.write_tables(tmp_path, entities, relations, chunks, mentions)
    assert counts == {"entities": 1, "relations": 1, "chunks": 2, "doc_mentions": 1}
    rel = pq.read_table(tmp_path / "relations.parquet").to_pylist()[0]
    assert rel["sources"] == ["1"] and rel["embedding"] == pytest.approx([0.5, 0.25])
    assert rel["confidence"] is None and rel["negated"] is False
    ch = pq.read_table(tmp_path / "chunks.parquet").to_pylist()
    assert ch[0]["turn_start"] == 0 and ch[0]["turn_end"] == 2 and ch[1]["turn_start"] is None
    check = rs.check_turn_spans(tmp_path, {"s1": {"date": "2023-05-20", "turns": TURNS}})
    assert check["chunks"] == 2 and check["mapped_to_bench_turns"] == 1
    assert check["share"] == pytest.approx(0.5) and check["failed_chunk_ids"] == ["2"]


def test_oneline_replaces_every_newline_kind():
    assert rs.oneline("a\r\nb\rc\nd") == "a b c d"
    assert rs.oneline(None) == ""

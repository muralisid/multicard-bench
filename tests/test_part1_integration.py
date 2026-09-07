"""Tests for the Part 1 integration: multicard.experiments.part1 and
multicard.part1.competitors, on synthetic inputs. No model call, no
encoder, no tokenizer (a word counter stands in for the bench TokenCounter).

Covered: his session document rebuilt with turn spans; his query output as
native units in his block order with the section 5 date keys and the
provenance spans; the Graphiti FACT and ENTITY lines with the sessions a
fact cites; the contexts.jsonl round trip (a rendered context described
without its unit texts and rebuilt from the text); the variant choice when
contexts are read back; the question selection in ORDER and the sample;
the absent-arm rule in run_tests and the absent rows of the report; the
score row round trip; the run.py registry and options; the character lint.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multicard.experiments import part1 as PT
from multicard.part1 import competitors as C
from multicard.part1 import evaluate as E
from multicard.part1 import render as RD
from multicard.part1 import report as RP
from multicard.part1 import retrieve as R
from multicard.part1 import score as S
from multicard.part1 import units as U

BANNED = re.compile("[\u2014\u2013\u2192\u2018\u2019\u201c\u201d\u2026]")


class Words:
    def count(self, text):
        return len(text.split())


SESSIONS = {
    "s1": {"date": "2023-05-20", "turns": [
        {"role": "user", "content": "I bought a red bike yesterday.\nIt cost two hundred dollars."},
        {"role": "assistant", "content": "Great choice for the city."}]},
    "s2": {"date": "2023-05-18", "turns": [
        {"role": "user", "content": "My cat is called Tom and he is three years old."},
        {"role": "assistant", "content": "Tom is a fine name for a cat."}]},
}


def tables():
    return U.build_tables(SESSIONS, ["s1", "s2"])


def pgr_for(tabs):
    """An export shaped like the runner's: one chunk per session, one relation."""
    chunks = {}
    for k, sid in enumerate(("s1", "s2"), start=1):
        doc, spans = C.session_document(tabs[sid])
        chunks[str(k)] = {"chunk_id": str(k), "content": doc, "doc_id": sid, "doc_date": tabs[sid].date,
                          "turn_start": 0, "turn_end": 1, "char_start": 0, "char_end": len(doc)}
    rels = [{"edge_id": "7", "src_key": "1", "tgt_key": "2", "relation_type": "bought", "description": "bike",
             "weight": 1.0, "negated": False, "confidence": 1.0, "valid_from": None, "valid_to": None,
             "sources": ["1"], "embedding": None}]
    ents = {"1": {"entity_key": "1", "name": "The user", "aliases": ["me"], "description": "", "entity_type": None},
            "2": {"entity_key": "2", "name": "bike", "aliases": [], "description": "", "entity_type": None}}
    return R.PgrTables(ents, rels, chunks, {"1": ["1"], "2": ["1"]})


def query_payload():
    return {"variant": "shipped", "top_k": 8, "reused_from": None, "meter": {"calls": 3, "tokens_in": 100, "tokens_out": 10, "usd": 0.001},
            "query_data": {"data": {
                "chunks": [{"chunk_id": "1", "content": C.session_document(tables()["s1"])[0],
                            "metadata": {"document": "s1", "session_date": "2023-05-20"}}],
                "entities": [{"entity_name": "bike", "entity_type": "Object", "description": "a red bike"}],
                "relationships": [{"src_id": "The user", "tgt_id": "bike", "edge_id": "7", "relation_type": "bought",
                                   "description": "bike", "weight": 1, "negated": False, "valid_from": None,
                                   "valid_to": "2023-06-01"}],
                "references": [], "communities": []}}}


# ----------------------------------------------------------------------------
# His session document and units
# ----------------------------------------------------------------------------
def test_session_document_and_turn_spans():
    t = tables()["s1"]
    doc, spans = C.session_document(t)
    assert doc.startswith("[Conversation on 2023-05-20]\nuser: I bought a red bike yesterday. It cost")
    assert doc[spans[0][0]:spans[0][1]] == "I bought a red bike yesterday. It cost two hundred dollars."
    assert doc[spans[1][0]:spans[1][1]] == "Great choice for the city."
    row = {"turn_start": 0, "turn_end": 1, "char_start": 0, "char_end": len(doc)}
    ts = C.chunk_turn_spans(row, doc, t)
    assert set(ts) == {"s1#0", "s1#1"}
    lo, hi = ts["s1#1"]
    assert doc[lo:hi] == "Great choice for the city."
    # a chunk that is not an exact substring falls back to a search per turn
    ts2 = C.chunk_turn_spans({**row, "char_end": len(doc) - 3}, "xx " + doc, t)
    assert doc[ts["s1#0"][0]:ts["s1#0"][1]] in ("xx " + doc)[ts2["s1#0"][0]:ts2["s1#0"][1]]
    assert C.chunk_turn_spans({"turn_start": None, "turn_end": None}, doc, t) == {}


def test_chandan_units_block_order_date_keys_and_provenance():
    tabs = tables()
    pgr = pgr_for(tabs)
    units = C.chandan_units(query_payload(), pgr, tabs, "lme")
    assert [u.kind for u in units] == ["chunk", "entity", "relation"]
    chunk, ent, rel = units
    assert chunk.unit_id == "chunk:1" and chunk.container == "s1" and chunk.date == "2023-05-20"
    assert chunk.text.startswith("Chunk [1] (s1): [Conversation on 2023-05-20]")
    lo, hi = chunk.turn_spans["s1#0"]
    assert chunk.text[lo:hi] == "I bought a red bike yesterday. It cost two hundred dollars."
    assert chunk.sessions == ("s1",)
    assert ent.date is None and ent.text == "- Entity bike (Object): a red bike" and ent.sessions == ()
    # the relation has no valid_from: it takes its provenance chunk's session date and spans
    assert rel.date == "2023-05-20" and rel.container == "s1" and rel.sessions == ("s1",)
    assert set(rel.provenance_spans) == {"s1#0", "s1#1"}
    assert rel.text == "- (The user) --[bought (weight=1)]--> (bike) [until 2023-06-01]: bike"
    # index units: every exported chunk and relation with full spans
    idx = C.chandan_index_units(pgr, tabs, "lme")
    assert [u.kind for u in idx] == ["chunk", "chunk", "relation"]
    assert idx[0].turn_chars["s1#0"] == len("I bought a red bike yesterday. It cost two hundred dollars.")
    assert idx[2].provenance_chars["s1#1"] == len("Great choice for the city.") and idx[2].sessions == ("s1",)


def test_chandan_units_render_and_score_under_the_coverage_rule():
    tabs = tables()
    pgr = pgr_for(tabs)
    units = C.chandan_units(query_payload(), pgr, tabs, "lme")
    ctx = RD.render_native(units, 1000, Words())
    sc = S.score_longmemeval("q", {"s1#0"}, {"s1"}, ctx, tabs)
    assert sc.joint_recall == 1.0 and sc.session_joint_recall == 1.0 and sc.candidate_joint_recall == 1.0
    # display: the undated entity line first, then the dated units oldest first
    assert ctx.units[0].kind == "entity"
    ctx2 = RD.render_native(units, 1000, Words(), keep_native_order=True)
    assert [u.kind for u in ctx2.units] == ["chunk", "entity", "relation"]


def test_relation_line_validity_forms():
    base = {"src_id": "a", "tgt_id": "b", "relation_type": "likes", "weight": 1, "description": "d"}
    assert C.relation_line({**base, "valid_from": "2023-01-01", "valid_to": "2023-02-01"}).endswith("[valid 2023-01-01 to 2023-02-01]: d")
    assert C.relation_line({**base, "valid_from": "2023-01-01"}).endswith("[from 2023-01-01]: d")
    assert C.relation_line({**base, "negated": True}) == "- (a) --[NOT likes (weight=1)]--> (b): d"


def test_load_query_fills_query_data_from_the_reused_variant(tmp_path):
    d = tmp_path / "lme" / "q1" / "queries"
    d.mkdir(parents=True)
    (d / "shipped.json").write_text(json.dumps(query_payload()))
    (d / "raised_4k.json").write_text(json.dumps({"variant": "raised_4k", "reused_from": "shipped", "top_k": 8}))
    q = C.load_query("lme", "q1", "q1", "raised_4k", tmp_path)
    assert q["query_data"]["data"]["chunks"][0]["chunk_id"] == "1" and q["reused_from"] == "shipped"
    assert C.load_query("lme", "q1", "q1", "raised_8k", tmp_path) is None
    assert C.pgr_query_dir("mhrag", "corpus", "mhr_q0001", tmp_path) == tmp_path / "mhrag" / "corpus" / "queries" / "mhr_q0001"


# ----------------------------------------------------------------------------
# Graphiti units
# ----------------------------------------------------------------------------
def graphiti_group(tmp_path):
    d = tmp_path / "g1"
    d.mkdir()
    episodes = [{"uuid": "e1", "name": "s1", "source_description": "x", "reference_time": "2023-05-20T00:00:00+00:00",
                 "session_id": "s1", "turn_index": None},
                {"uuid": "e2", "name": "s2", "source_description": "x", "reference_time": "2023-05-18T00:00:00+00:00",
                 "session_id": "s2", "turn_index": None}]
    pq.write_table(pa.Table.from_pylist(episodes), d / "episodes.parquet")
    edges = [{"uuid": "f1", "fact": "The user bought a bike.", "name": "BOUGHT", "source_node_uuid": "n1",
              "target_node_uuid": "n2", "episodes": ["e1"], "valid_at": "2023-05-20T00:00:00+00:00",
              "invalid_at": None, "created_at": "x", "expired_at": None},
             {"uuid": "f2", "fact": "Tom is a cat.", "name": "IS", "source_node_uuid": "n3",
              "target_node_uuid": "n4", "episodes": ["e2", "e1"], "valid_at": None, "invalid_at": None,
              "created_at": "x", "expired_at": None}]
    pq.write_table(pa.Table.from_pylist(edges), d / "edges.parquet")
    (d / "meta.json").write_text(json.dumps({"partial": False, "refused_episodes": []}))
    (d / "search").mkdir()
    search = {"limit": 20, "edges": [{**e, "score": 1.0} for e in edges],
              "nodes": [{"uuid": "n1", "name": "The user", "summary": "Bought a bike.", "labels": ["Entity"], "score": 1.0}],
              "episodes": episodes, "context": "", "usage": {"llm_calls": 0, "embed_calls": 1, "embed_tokens_in": 5}}
    (d / "search" / "shipped.json").write_text(json.dumps(search))
    return d


def test_graphiti_units_and_index_units(tmp_path):
    graphiti_group(tmp_path)
    assert C.graphiti_available("g1", tmp_path) and not C.graphiti_available("g2", tmp_path)
    eps = C.graphiti_episodes("g1", tmp_path)
    units = C.graphiti_units(C.load_search("g1", "shipped", tmp_path), eps)
    assert [u.kind for u in units] == ["fact", "fact", "entity"]
    assert units[0].text == "  - The user bought a bike. (2023-05-20 00:00:00+00:00 - present)"
    assert units[0].date == "2023-05-20" and units[0].sessions == ("s1",)
    # no valid_at: the reference_time of the first cited episode; sessions of every cited episode
    assert units[1].text == "  - Tom is a cat. (date unknown - present)"
    assert units[1].date == "2023-05-18" and units[1].sessions == ("s2", "s1")
    assert units[2].text == "  - The user: Bought a bike." and units[2].date is None and units[2].sessions == ()
    ctx = RD.render_native(units, 100, Words())
    sc = S.score_longmemeval("q", {"s2#0"}, {"s2"}, ctx, tables(), session_level_only=True)
    assert sc.session_joint_recall == 1.0 and sc.turn_r10 is None and sc.sess_r5 == 1.0
    idx = C.graphiti_index_units("g1", tmp_path)
    assert [u.sessions for u in idx] == [("s1",), ("s2", "s1")]
    assert C.graphiti_status(["g1"], tmp_path) == "run"
    assert C.graphiti_status(["g1", "g2"], tmp_path) == "partial"
    assert C.graphiti_status(["g2"], tmp_path) == "dropped"


def test_refused_rules():
    assert C.chandan_refused({"refused": [{"doc_id": "s1"}]}, {"s1", "s3"})
    assert not C.chandan_refused({"refused": [{"doc_id": "s9"}]}, {"s1"})
    assert not C.chandan_refused(None, {"s1"})
    assert C.graphiti_refused({"refused_episodes": [{"session_id": "s2"}]}, {"s2"})
    assert not C.graphiti_refused({"refused_episodes": []}, {"s2"})


# ----------------------------------------------------------------------------
# contexts.jsonl round trip and the variant choice
# ----------------------------------------------------------------------------
def test_context_row_round_trip_rebuilds_the_bodies():
    tabs = tables()
    ctx = RD.render(["s1#0", "s2#1", "s1#1"], 30, tabs, counter=Words())
    row = PT.context_row("S4_static", "q", 30, ctx, [])
    back = PT.rendered_from_row(row)
    assert back.text == ctx.text and back.unit_ids == ctx.unit_ids
    assert [u.body for u in back.units] == [u.body for u in ctx.units]
    assert [u.line for u in back.units] == [u.line for u in ctx.units]
    assert [u.turn_chars for u in back.units] == [u.turn_chars for u in ctx.units]
    assert back.tokens == ctx.tokens and back.n_truncated == ctx.n_truncated
    # native units too, with a truncated unit
    units = C.chandan_units(query_payload(), pgr_for(tabs), tabs, "lme")
    nctx = RD.render_native(units, 12, Words())
    assert nctx.n_truncated >= 1
    nback = PT.rendered_from_row(PT.context_row("chandan_live", "q", 12, nctx, ["shipped"]))
    assert [u.body for u in nback.units] == [u.body for u in nctx.units]
    assert [u.truncated for u in nback.units] == [u.truncated for u in nctx.units]
    # the 2,000-character cut flag survives the round trip; an older row without it reads False
    long = {"L": {"date": "2023-01-01", "turns": [{"role": "user", "content": "x " * 1500}]}}
    lctx = RD.render(["L#0"], 10 ** 6, U.build_tables(long, ["L"]), counter=Words())
    lrow = PT.context_row("S5_primary", "q", 10 ** 6, lctx, [])
    assert lrow["units"][0]["cut"] is True and PT.rendered_from_row(lrow).units[0].cut is True
    del lrow["units"][0]["cut"]
    assert PT.rendered_from_row(lrow).units[0].cut is False


def test_pgr_build_subset_reads_the_runner_logs(tmp_path):
    subsets = {"ORDER": {"longmemeval": ["a", "b"]}, "GRAPHITI_150": ["a"]}
    root = tmp_path / "pgr"
    (root / "lme").mkdir(parents=True)
    assert C.pgr_build_subset("lme", subsets, root) is None
    (root / "lme" / "run_pgr-lme-full-s0.json").write_text(json.dumps({"started": "2026-09-06T10:00:00", "args": {"subset": None}}))
    assert C.pgr_build_subset("lme", subsets, root) is None
    (root / "lme" / "run_pgr-lme-g150.json").write_text(json.dumps({"started": "2026-09-06T12:00:00", "args": {"subset": "GRAPHITI_150"}}))
    assert C.pgr_build_subset("lme", subsets, root) == "GRAPHITI_150"
    (root / "lme" / "run_bad.json").write_text("{not json")
    assert C.pgr_build_subset("lme", subsets, root) == "GRAPHITI_150"
    # a subset name that is not a key of subsets.json is ignored
    (root / "lme" / "run_later.json").write_text(json.dumps({"started": "2026-09-06T13:00:00", "args": {"subset": "other.json"}}))
    assert C.pgr_build_subset("lme", subsets, root) == "GRAPHITI_150"
    assert [log["started"] for log in C.pgr_run_logs("lme", root)][0] == "2026-09-06T10:00:00"
    # the answering population per arm follows the same rule
    ids = ["a", "b"]
    sub = {"GRAPHITI_150": ["a"], "CHANDAN_CAL_18": ["b"]}
    assert PT.arm_population("graphiti", "lme", ids, sub, ids, None, None) == ["a"]
    assert PT.arm_population("chandan_live_cal", "lme", ids, sub, ids, None, None) == ["b"]
    assert PT.arm_population("chandan_live", "lme", ids, sub, ids, "GRAPHITI_150", None) == ["a"]
    assert PT.arm_population("chandan_live", "lme", ids, sub, ids, None, None) == ["a", "b"]
    assert PT.arm_population("S5_primary", "lme", ids, sub, ids, "GRAPHITI_150", None) == ["a", "b"]


def test_load_contexts_picks_the_chosen_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "OUT", tmp_path / "results")
    out = PT.stage_dir("t", "lme", "retrieve")
    out.mkdir(parents=True)
    rows = [{"arm": "chandan_live", "qid": "q1", "budget": 4000, "variants": ["shipped"], "text": "a"},
            {"arm": "chandan_live", "qid": "q1", "budget": 4000, "variants": ["raised_4k"], "text": "b"},
            {"arm": "graphiti", "qid": "q1", "budget": 4000, "variants": ["shipped", "raised_4k"], "text": "c"},
            {"arm": "S5_primary", "qid": "q1", "budget": 4000, "variants": [], "text": "d"}]
    (out / "contexts_4000.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = PT.load_contexts("lme", "t", 4000, {"chandan_live": "raised_4k", "graphiti": "shipped"})
    assert got[("chandan_live", "q1")]["text"] == "b" and got[("graphiti", "q1")]["text"] == "c"
    assert got[("S5_primary", "q1")]["text"] == "d"
    got2 = PT.load_contexts("lme", "t", 4000, {})
    assert got2[("chandan_live", "q1")]["text"] == "a"


def test_score_row_round_trip():
    sc = S.QuestionScore(qid="q", joint_recall=1.0, candidate_joint_recall=1.0, session_joint_recall=1.0,
                         turn_r10=0.5, turn_ndcg10=0.4, sess_r5=1.0, rendered_tokens=10, duplicate_share=0.0,
                         n_candidates=3, n_rendered=2, n_truncated=0, truncated_evidence=0, uncovered=["s1#1"])
    from dataclasses import asdict
    row = {"arm": "x", "variants": [], **asdict(sc)}
    assert PT._score_from_row("lme", row) == sc
    ms = S.MhrQuestionScore(qid="q", fact_joint_recall=0.0, doc_joint_recall=1.0, candidate_fact_joint_recall=1.0,
                            candidate_doc_joint_recall=1.0, all_located=True, n_facts=2, n_contained=1, n_fallback=0,
                            doc_hits={"d": True}, rendered_tokens=5, duplicate_share=0.0, n_candidates=1,
                            n_rendered=1, n_truncated=0)
    assert PT._score_from_row("mhrag", {"arm": "x", **asdict(ms)}) == ms


# ----------------------------------------------------------------------------
# Question selection and containers
# ----------------------------------------------------------------------------
def small_subsets():
    lme = ["a", "b_abs", "c", "d", "e"]
    mh = ["mhr_q0000", "mhr_q0001", "mhr_q0002", "mhr_q0003"]
    return {"ORDER": {"longmemeval": lme, "multihoprag": mh}, "abstention": {"longmemeval": ["b_abs"]},
            "types": {"longmemeval": {q: "multi-session" for q in lme},
                      "multihoprag": {"mhr_q0000": "null_query", "mhr_q0001": "temporal_query",
                                      "mhr_q0002": "inference_query", "mhr_q0003": "comparison_query"}}}


def test_select_questions_in_order_and_sample():
    s = small_subsets()
    assert PT.select_questions(s, "lme") == ["a", "b_abs", "c", "d", "e"]
    assert PT.select_questions(s, "lme", limit=2) == ["a", "b_abs"]
    assert PT.select_questions(s, "lme", limit=2, sample=True) == ["a", "c"]
    assert PT.select_questions(s, "mhrag", sample=True) == ["mhr_q0001", "mhr_q0002", "mhr_q0003"]
    assert PT.corpus_kind("longmemeval") == ("lme", "longmemeval") and PT.corpus_kind("mhrag") == ("mhrag", "multihoprag")


def test_container_ids_for_the_sample():
    from multicard.data.longmemeval import Instance
    from multicard.data.multihoprag import Document, Evidence, Query

    inst = {"a": Instance("a", "t", "q", "x", "2023-01-01", False, ["s2", "s1"]),
            "c": Instance("c", "t", "q", "x", "2023-01-01", False, ["s3"])}
    assert PT.lme_container_ids(inst, ["a"], True) == ["s1", "s2"]
    assert PT.lme_container_ids(inst, ["a"], False) == ["s1", "s2", "s3"]
    docs = [Document(f"d{i}", f"t{i}", "src", "cat", "2023-09-2{i}", "body") for i in range(4)]
    queries = {"q": Query("q", "?", "a", "temporal_query", [Evidence("t3", "f", doc_id="d3")])}
    old = PT.SAMPLE_DOCS
    PT.SAMPLE_DOCS = 2
    try:
        assert PT.mhr_container_ids(docs, queries, ["q"], True) == ["d0", "d1", "d3"]
    finally:
        PT.SAMPLE_DOCS = old
    assert PT.mhr_container_ids(docs, queries, ["q"], False) == ["d0", "d1", "d2", "d3"]


# ----------------------------------------------------------------------------
# Absent arms in the tests and the report
# ----------------------------------------------------------------------------
def eval_subsets():
    lme = [f"q{i}" for i in range(8)]
    types = {q: ("single-session-user" if i % 2 else "multi-session") for i, q in enumerate(lme)}
    mh = [f"mhr_q{i:04d}" for i in range(4)]
    return {"meta": {"seed": 13, "corpora": {}}, "types": {"longmemeval": types, "multihoprag": {q: "temporal_query" for q in mh}},
            "abstention": {"longmemeval": []}, "ORDER": {"longmemeval": lme, "multihoprag": mh},
            "GRAPHITI_150": lme[:4], "CHANDAN_CAL_18": lme[:2], "MHRAG_ANSWER": mh, "READER_B_MHRAG": mh[:2],
            "JUDGE_AUDIT": {"share": 0.1, "arms": {}, "cells": []},
            "derived": {"GRAPHITI_PILOT": lme[0], "LOCAL_120": [q for q in lme if types[q] == "single-session-user"]}}


def qscore(q, v):
    return S.QuestionScore(qid=q, joint_recall=v, candidate_joint_recall=1.0, session_joint_recall=v, turn_r10=v,
                           turn_ndcg10=v, sess_r5=1.0, rendered_tokens=100, duplicate_share=0.0, n_candidates=5,
                           n_rendered=2, n_truncated=0, truncated_evidence=0)


def test_run_tests_marks_absent_arms_not_run_and_report_prints_absent_rows(tmp_path):
    subsets = eval_subsets()
    ids = subsets["ORDER"]["longmemeval"]
    lme = {"S5_primary": {q: qscore(q, 1.0 if i % 3 else 0.0) for i, q in enumerate(ids)},
           "ours_cheap": {q: qscore(q, 1.0 if i % 2 else 0.0) for i, q in enumerate(ids)},
           "S4_static": {q: qscore(q, 1.0) for q in ids}}
    absent = {"longmemeval": ["chandan_live", "graphiti", "S5_overlay_R0", "S5_overlay_P0"],
              "multihoprag": ["chandan_live", "S5_primary", "ours_cheap"]}
    tests = E.run_tests(lme, {}, [], subsets, graphiti_status="dropped", absent=absent)
    t1, t2, t8a, t8b = tests["family_A"]
    assert not t1["ran"] and "chandan_live" in t1["reason"]
    assert not t2["ran"] and not t8a["ran"] and not t8b["ran"]
    assert tests["holm"]["A"] == {}
    t3, t4, t5a, t5b, t6 = tests["family_B"]
    assert t3["ran"] and t4["ran"] and t6["ran"] and not t5a["ran"] and not t5b["ran"]
    assert tests["T5"]["branch"] == "not run"
    assert tests["T7"]["ran"] and tests["pass_rule"]["passed"] is False
    assert tests["pass_rule"]["T1_label"] == "not run"
    payload = E.build_metrics(lme={a: {4000: d} for a, d in lme.items()}, mhr={}, records=[],
                              audit={"n": 0, "n_agree": 0, "agreement": 0.0, "threshold": 0.9, "cheap": "c",
                                     "strong": "s", "primary": "s", "second": "c", "cells": []},
                              buckets=[], subsets=subsets, subsets_sha256="abc", commits={"head": "x"},
                              graphiti_status="dropped", absent=absent)
    assert payload["absent_arms"]["longmemeval"] == sorted(absent["longmemeval"])
    text = RP.render_report(payload)
    block = text.split("## Retrieval")[1].split("## By question type")[0]
    assert "| chandan_live | absent |" in block and "| graphiti | absent |" in block
    assert "Part 1 passed: **no**" in text
    assert "not run: no output from chandan_live on longmemeval" in text
    assert not BANNED.search(text)


# ----------------------------------------------------------------------------
# Registry and options
# ----------------------------------------------------------------------------
def test_registry_entries_and_options_match_the_signatures():
    import os

    os.environ.setdefault("PYTHONHASHSEED", "13")
    from multicard import run as RUN

    for name in ("part1_index", "part1_retrieve", "part1_qa", "part1_report"):
        mod, fn = RUN.REGISTRY[name]
        assert mod == "multicard.experiments.part1"
        f = getattr(PT, fn)
        params = inspect.signature(f).parameters
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        for opt in ("limit", "corpus", "tag"):
            assert opt in params, (name, opt)
    for opt in RUN.PART1_OPTIONS:
        assert any(opt in inspect.signature(getattr(PT, fn)).parameters
                   for fn in ("part1_index", "part1_retrieve", "part1_qa", "part1_report")), opt
    assert PT.out_root("smoke") == Path("results/part1_smoke") and PT.out_root("") == Path("results/part1")


def test_reader_b_order_follows_the_withdrawal_rule():
    # section 5: withdrawn in the order closed_book, oracle_full, chandan_full, ours_cheap, graphiti;
    # the head-to-head pair keeps Reader B last, so it runs first
    assert PT.READER_B_ARMS[:2] == ("S5_primary", "chandan_live")
    assert list(reversed(PT.READER_B_ARMS[2:])) == ["closed_book", "oracle_full", "chandan_full", "ours_cheap", "graphiti"]


def test_new_files_have_no_banned_characters():
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/multicard/experiments/part1.py", "src/multicard/part1/competitors.py",
                "src/multicard/run.py", "docs/part1/RUNBOOK.md", "tests/test_part1_integration.py"):
        p = root / rel
        if p.exists():
            assert not BANNED.search(p.read_text()), rel


# ----------------------------------------------------------------------------
# Reporting fixes: the cost ledger, the audit history, the build state, the
# arm states, the planner attribution, the bucket records and the report
# ----------------------------------------------------------------------------
from types import SimpleNamespace

import pytest

from multicard.llm.costmeter import CostMeter


def mscore(q, v):
    return S.MhrQuestionScore(qid=q, fact_joint_recall=v, doc_joint_recall=v, candidate_fact_joint_recall=1.0,
                              candidate_doc_joint_recall=1.0, all_located=True, n_facts=2, n_contained=int(2 * v),
                              n_fallback=0, doc_hits={"d1": True}, rendered_tokens=3800, duplicate_share=0.0,
                              n_candidates=100, n_rendered=7, n_truncated=1)


def test_iso_datetime_keeps_the_time_of_day():
    from multicard.data.longmemeval import iso_date, iso_datetime

    assert iso_datetime("2023/05/20 (Sat) 02:21") == "2023-05-20 02:21"
    assert iso_date("2023/05/20 (Sat) 02:21") == "2023-05-20"
    assert iso_datetime("2023/05/20") == "2023-05-20" and iso_datetime("") == ""
    assert iso_datetime("2023/05/20 (Sat) 02:21") < iso_datetime("2023/05/20 (Sat) 14:05")


def test_cost_ledger_append_read_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "OUT", tmp_path / "results" / "part1")
    monkeypatch.setenv("MCB_JOB_TAG", "pass_x")
    meter = CostMeter(max_usd=5.0)
    meter.record("vertex-flash", 100000, 10000)          # 0.01 in, 0.004 out
    row = PT.append_cost_ledger("", "lme", "retrieve", meter, arms={"S5_noPGR", "ours_cheap"}, budgets=[4000, 8000],
                                max_usd=35.0, n_questions=500, extra={"query_cost": {"planner": {"uncached": 500}}})
    assert row["job_tag"] == "pass_x" and row["arms"] == ["S5_noPGR", "ours_cheap"] and row["usd"] == pytest.approx(0.014)
    assert row["cost"]["by_tier"]["vertex-flash"]["calls"] == 1 and row["git"] and row["stage"] == "retrieve"
    assert row["invocation_id"].startswith("retrieve-") and row["corpora"] == ["lme"]
    p = PT.ledger_path("", "lme", "retrieve")
    assert p == tmp_path / "results" / "part1" / "lme" / "retrieve" / "cost_ledger.jsonl" and p.exists()
    # a capped rerun appends, never rewrites; two invocations in the same second stay two invocations
    row2 = PT.append_cost_ledger("", "lme", "retrieve", meter, arms={"S5_noPGR"}, budgets=[4000], max_usd=35.0,
                                 stopped="spend 35.01 USD exceeds the cap of 35.00")
    assert row2["invocation_id"] != row["invocation_id"]
    # one qa invocation over both corpora writes the same id to each corpus ledger
    qm = CostMeter(max_usd=100.0)
    qm.record("azure-gpt54", 1_000_000, 100_000)         # 1.25 in, 1.00 out
    for kind in ("lme", "mhrag"):
        PT.append_cost_ledger("", kind, "qa", qm, arms=["S5_noPGR"], readers=["reader_a", "reader_b"], budgets=[4000],
                              max_usd=40.0, corpora=["lme", "mhrag"], invocation_id="qa-1")
    ledgers = PT.read_cost_ledgers("", ["lme", "mhrag"])
    assert len(ledgers["retrieve"]["longmemeval"]) == 2
    assert ledgers["retrieve"]["longmemeval"][1]["stopped"].startswith("spend")
    assert "index" not in ledgers and set(ledgers["qa"]) == {"longmemeval", "multihoprag"}
    s = PT.ledger_summary(ledgers)
    r = s["stages"]["retrieve"]
    assert r["cap_usd"] == 35.0 and r["total_usd"] == pytest.approx(0.028) and r["n_invocations"] == 2
    assert not r["over_cap"] and r["by_corpus"]["longmemeval"]["n_invocations"] == 2
    q = s["stages"]["qa"]
    assert q["n_invocations"] == 1 and q["total_usd"] == pytest.approx(2.25) and q["cap_usd"] == 100.0
    assert q["by_corpus"]["longmemeval"]["usd"] == pytest.approx(2.25)
    assert q["by_corpus"]["multihoprag"]["usd"] == pytest.approx(2.25)
    inv = q["invocations"][0]
    assert inv["readers"] == ["reader_a", "reader_b"] and inv["by_tier"]["azure-gpt54"]["usd"] == pytest.approx(2.25)
    assert s["stages"]["index"]["n_invocations"] == 0 and s["stages"]["index"]["cap_usd"] == 15.0
    assert not BANNED.search(json.dumps(s))
    # over the cap is said so
    big = CostMeter(max_usd=1000.0)
    big.record("azure-gpt54", 100_000_000, 0)            # 125 USD
    PT.append_cost_ledger("", "lme", "qa", big, corpora=["lme"], invocation_id="qa-2")
    s = PT.ledger_summary(PT.read_cost_ledgers("", ["lme", "mhrag"]))
    assert s["stages"]["qa"]["over_cap"] and s["stages"]["qa"]["n_invocations"] == 2


def test_audit_history_seeds_from_the_old_file_and_appends(tmp_path):
    p = tmp_path / "qa_audit.json"
    e1 = {"timestamp": "t1", "n": 250, "agreement": 0.848, "primary": "gpt-5.4"}
    assert PT.audit_history(p, e1) == [e1]
    p.write_text(json.dumps({"judge_audit": {"n": 450, "agreement": 0.8, "primary": "gpt-5.4"}, "primary_judge": "gpt-5.4"}))
    h = PT.audit_history(p, {"timestamp": "t2", "n": 460, "agreement": 0.81, "primary": "gpt-5.4"})
    assert len(h) == 2 and h[0]["n"] == 450 and "seeded" in h[0]["note"] and h[0]["timestamp"] and h[1]["n"] == 460
    p.write_text(json.dumps({"judge_audit": {"n": 460}, "history": h}))
    h2 = PT.audit_history(p, {"timestamp": "t3", "n": 470, "agreement": 0.8, "primary": "gpt-5.4"})
    assert [x["n"] for x in h2] == [450, 460, 470]
    p.write_text("not json")
    assert PT.audit_history(p, e1) == [e1]


def _run_log(tag, started, ended, usd, stopped=None, subset=None):
    return json.dumps({"job_tag": tag, "started": started, "ended": ended, "stopped": stopped,
                       "args": {"subset": subset}, "meter": {"usd": usd}})


def test_pgr_build_state_reads_run_logs_and_counts_spaces(tmp_path):
    subsets = eval_subsets()
    ids = subsets["ORDER"]["longmemeval"]
    root = tmp_path / "pgr"
    (root / "lme").mkdir(parents=True)
    for q in ids[:3]:
        (root / "lme" / q).mkdir()
        (root / "lme" / q / "meta.json").write_text("{}")
    (root / "lme" / "run_pgr-lme-full-s0.json").write_text(_run_log("pgr-lme-full-s0", "2026-09-06T16:04:05+05:30", None, 14.04))
    (root / "lme" / "run_pgr-lme-full-s1.json").write_text(_run_log("pgr-lme-full-s1", "2026-09-06T16:04:15+05:30",
                                                                    "2026-09-06T20:00:00+05:30", 12.6))
    b = PT._pgr_build_state("lme", subsets, root)
    assert b["in_progress"] and not b["complete"] and b["running_job_tags"] == ["pgr-lme-full-s0"]
    assert b["n_spaces"] == 3 and b["n_wanted"] == 8
    assert b["label"] == "partial build snapshot, 3 of 8 spaces, build in progress"
    assert b["run_log_usd"] == pytest.approx(26.64) and b["subset"] is None and b["stopped"] == []
    # every runner ended and every space exported: complete
    (root / "lme" / "run_pgr-lme-full-s0.json").write_text(_run_log("pgr-lme-full-s0", "2026-09-06T16:04:05+05:30",
                                                                    "2026-09-07T01:00:00+05:30", 14.04))
    for q in ids[3:]:
        (root / "lme" / q).mkdir()
        (root / "lme" / q / "meta.json").write_text("{}")
    b = PT._pgr_build_state("lme", subsets, root)
    assert b["complete"] and not b["in_progress"] and b["label"] == "complete build, 8 of 8 spaces"
    # a build restricted to a subset counts against the subset
    (root / "lme" / "run_pgr-lme-g150.json").write_text(_run_log("pgr-lme-g150", "2026-09-07T02:00:00+05:30",
                                                                 "2026-09-07T03:00:00+05:30", 1.0, subset="GRAPHITI_150"))
    b = PT._pgr_build_state("lme", subsets, root)
    assert b["subset"] == "GRAPHITI_150" and b["n_wanted"] == 4 and b["complete"]
    # MultiHop-RAG: one corpus space; the runner stopped at its cap before exporting it
    (root / "mhrag").mkdir()
    (root / "mhrag" / "run_pgr-mhrag-full.json").write_text(_run_log("pgr-mhrag-full", "2026-09-06T16:05:38+05:30",
                                                                     "2026-09-07T00:24:38+05:30", 10.007077,
                                                                     stopped="cap: spend 10.0071 USD exceeds the cap of 10.00"))
    b = PT._pgr_build_state("mhrag", subsets, root)
    assert not b["complete"] and not b["in_progress"] and b["n_spaces"] == 0 and b["n_wanted"] == 1
    assert b["label"] == "partial build snapshot, 0 of 1 spaces, build stopped at its cap"
    assert b["stopped"][0]["job_tag"] == "pgr-mhrag-full" and b["run_log_usd"] == pytest.approx(10.007077)
    # no export and no log at all: a snapshot of nothing, not a complete build
    b = PT._pgr_build_state("mhrag", subsets, tmp_path / "nothing")
    assert not b["complete"] and b["label"] == "partial build snapshot, 0 of 1 spaces" and b["run_log_usd"] == 0.0


def test_arm_status_three_states(tmp_path):
    root, groot = tmp_path / "pgr", tmp_path / "graphiti"
    ids = ["q1", "q2"]
    st = PT._arm_status("lme", ids, {"ours_cheap", "S5_noPGR"}, root, groot)
    assert st["ours_cheap"] == PT.STATUS_RUN and st["S5_noPGR"] == PT.STATUS_RUN
    assert st["S5_primary"] == PT.STATUS_NOT_RUN and st["chandan_live"] == PT.STATUS_NO_EXPORT
    assert st["chandan_full"] == PT.STATUS_NO_EXPORT and st["graphiti"] == PT.STATUS_NO_EXPORT
    (root / "lme" / "q2").mkdir(parents=True)
    (root / "lme" / "q2" / "relations.parquet").write_bytes(b"")
    st = PT._arm_status("lme", ids, {"ours_cheap"}, root, groot)
    assert st["chandan_live"] == st["chandan_full"] == PT.STATUS_EXPORT_NOT_RUN
    assert st["graphiti"] == PT.STATUS_NO_EXPORT
    st = PT._arm_status("lme", ids, {"chandan_live"}, root, groot)
    assert st["chandan_live"] == PT.STATUS_RUN and st["chandan_full"] == PT.STATUS_EXPORT_NOT_RUN
    # the answer-only arms and his own reader's arm
    assert st["closed_book"] == st["oracle_full"] == PT.STATUS_NOT_RUN
    assert st["chandan_full_uncut"] == PT.STATUS_EXPORT_NOT_ANSWERED
    st0 = PT._arm_status("lme", ids, {"closed_book", "chandan_full_uncut"}, tmp_path / "none", groot)
    assert st0["closed_book"] == PT.STATUS_RUN and st0["chandan_full_uncut"] == PT.STATUS_RUN
    assert PT._arm_status("lme", ids, set(), tmp_path / "none", groot)["chandan_full_uncut"] == PT.STATUS_NO_EXPORT
    # the graphiti export
    g = groot / "q1"
    (g / "search").mkdir(parents=True)
    for f in ("edges.parquet", "episodes.parquet", "meta.json"):
        (g / f).write_bytes(b"")
    (g / "search" / "shipped.json").write_text("{}")
    assert PT._arm_status("lme", ids, set(), root, groot)["graphiti"] == PT.STATUS_EXPORT_NOT_RUN
    # MultiHop-RAG: the corpus space; graphiti has no row there
    st = PT._arm_status("mhrag", ids, set(), root, groot)
    assert "graphiti" not in st and st["chandan_live"] == PT.STATUS_NO_EXPORT and st["S5_primary"] == PT.STATUS_NOT_RUN
    assert "ours_cheap_norule" not in st
    (root / "mhrag" / "corpus").mkdir(parents=True)
    (root / "mhrag" / "corpus" / "relations.parquet").write_bytes(b"")
    assert PT._arm_status("mhrag", ids, set(), root, groot)["chandan_full"] == PT.STATUS_EXPORT_NOT_RUN
    for v in st.values():
        assert not BANNED.search(v)


def test_pgr_tables_at_retrieve_and_planner_attribution():
    assert PT._pgr_tables_at_retrieve({"pgr_tables": {"present": 0, "wanted": 1}}) == \
        {"present": 0, "wanted": 1, "source": "retrieve metrics pgr_tables"}
    old = PT._pgr_tables_at_retrieve({"index": {"steps": {"relation_vectors": "116 spaces encoded, 4 present, 500 wanted"}}})
    assert old["present"] == 120 and old["wanted"] == 500 and "index steps" in old["source"]
    assert PT._pgr_tables_at_retrieve({})["present"] is None
    rmeta = {"query_cost": {"planner": {"calls": 500, "uncached": 0, "tokens_in": 99021, "tokens_out": 873, "usd": 0.0}}}
    rows = [{"timestamp": "t0", "job_tag": "pass1", "usd": 0.0101, "arms": ["S5_noPGR"],
             "query_cost": {"planner": {"uncached": 500}}},
            {"timestamp": "t1", "job_tag": "pass1c", "usd": 0.0, "arms": ["S5_noPGR"],
             "query_cost": {"planner": {"uncached": 0}}},
            {"timestamp": "t2", "job_tag": "s5", "usd": 0.02, "arms": ["S5_primary"],
             "query_cost": {"S5_primary": {"uncached": 7}, "S5_noPGR": {"uncached": 0}, "S2_lazy": {"uncached": 900}}}]
    p = PT._planner_attribution(rmeta, rows, 0.1, 0.4)
    assert p["calls"] == 500 and p["uncached_last_pass"] == 0 and p["cached_last_pass"] == 500
    assert p["usd_at_study_price"] == pytest.approx(99021 * 0.1 / 1e6 + 873 * 0.4 / 1e6, abs=1e-6)
    assert [x["job_tag"] for x in p["paid_in"]] == ["pass1", "s5"] and p["paid_in"][1]["uncached_calls"] == 7
    assert p["ledger_covers_paying_pass"]
    assert set(p["arms"]) == {n for n, s in R.ARM_SPECS.items() if s.planner == "llm"}
    assert PT._planner_attribution({"query_cost": {}}, rows, 0.1, 0.4) is None
    assert PT._planner_attribution(rmeta, [], 0.1, 0.4)["ledger_covers_paying_pass"] is False


def test_bucket_cases_record_reader_budget_and_the_timestamp_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "OUT", tmp_path / "results")
    sessions = {"a": {"date": "2023-05-20", "turns": [{"role": "user", "content": "My bike is red now."},
                                                     {"role": "assistant", "content": "Noted."}]},
                "b": {"date": "2023-05-20", "turns": [{"role": "user", "content": "My bike is blue since today."},
                                                     {"role": "assistant", "content": "Okay."}]}}
    tabs = U.build_tables(sessions, ["a", "b"])
    ctx = RD.render(["a#0", "b#0"], 10 ** 6, tabs, counter=Words())
    out = PT.stage_dir("", "lme", "retrieve")
    out.mkdir(parents=True)
    (out / "contexts_4000.jsonl").write_text(json.dumps(PT.context_row("ours_cheap", "q", 4000, ctx, [])) + "\n")
    x = SimpleNamespace(qid="q", evidence_turns={"a#0", "b#0"}, evidence_sessions={"a", "b"}, session_ids=["a", "b"],
                        qtype="knowledge-update", abstention=False)
    rec = E.AnswerRecord(qid="q", corpus="longmemeval", arm="ours_cheap", reader="reader_b", budget=4000,
                         qtype="knowledge-update", abstention=False, question="What colour is my bike?", gold="red",
                         answer="blue", verdicts={"j": False}, rendered_tokens=10, context_sha256="x", context_chars=1)
    E.set_primary([rec], "j")
    # the gold turn (a#0, red) is displayed first on the shared date: the clause fires; by the time of
    # day it is the earlier turn, so the case is counted as fired backwards
    ts = {"a": "2023-05-20 02:21", "b": "2023-05-20 09:40"}
    res = PT._bucket_cases("lme", "", [rec], {}, tabs, {"q": x}, None, ts)
    assert len(res) == 1
    r = res[0]
    assert r.bucket == 3 and r.ku_clause_fired and r.ku_gold_turn_earlier is True
    assert r.reader == "reader_b" and r.budget == 4000 and r.arm == "ours_cheap"
    # with the times the other way round the gold turn is the later one
    r = PT._bucket_cases("lme", "", [rec], {}, tabs, {"q": x}, None, {"a": "2023-05-20 09:40", "b": "2023-05-20 02:21"})[0]
    assert r.bucket == 3 and r.ku_gold_turn_earlier is False
    # no timestamps: the session order of the haystack stands in
    r = PT._bucket_cases("lme", "", [rec], {}, tabs, {"q": x}, None)[0]
    assert r.bucket == 3 and r.ku_gold_turn_earlier is True
    d = r.as_dict()
    assert d["reader"] == "reader_b" and d["budget"] == 4000 and d["ku_clause_fired"] is True


def test_report_prints_ledger_planner_states_notes_and_reader_tables():
    subsets = eval_subsets()
    ids = subsets["ORDER"]["longmemeval"]
    mh = subsets["ORDER"]["multihoprag"]
    lme = {"S5_noPGR": {4000: {q: qscore(q, 1.0) for q in ids}}, "ours_cheap": {4000: {q: qscore(q, 0.5) for q in ids}}}
    mhr = {"S5_primary": {4000: {q: mscore(q, 1.0) for q in mh}}, "ours_cheap": {4000: {q: mscore(q, 0.0) for q in mh}}}
    absent = {"longmemeval": ["S5_primary", "chandan_live", "chandan_full", "graphiti"],
              "multihoprag": ["chandan_live", "chandan_full", "S4_static"]}
    status = {"longmemeval": {"S5_noPGR": PT.STATUS_RUN, "ours_cheap": PT.STATUS_RUN, "S5_primary": PT.STATUS_NOT_RUN,
                              "chandan_live": PT.STATUS_EXPORT_NOT_RUN, "chandan_full": PT.STATUS_EXPORT_NOT_RUN,
                              "graphiti": PT.STATUS_NO_EXPORT},
              "multihoprag": {"S5_primary": PT.STATUS_RUN, "ours_cheap": PT.STATUS_RUN, "chandan_live": PT.STATUS_NO_EXPORT,
                              "chandan_full": PT.STATUS_NO_EXPORT, "S4_static": PT.STATUS_NOT_RUN}}
    notes = {"multihoprag": {"S5_primary": PT.NOTE_WITHOUT_PGR}}
    rows = [{"stage": "retrieve", "corpus": "lme", "corpora": ["lme"], "timestamp": "2026-09-06T14:10:00+00:00",
             "invocation_id": "retrieve-1", "job_tag": "retrieve_lme_pass1", "tag": "", "arms": ["S5_noPGR"], "readers": [],
             "budgets": [4000, 8000], "max_usd": 35.0, "stopped": None, "usd": 0.0101, "calls": 500,
             "cost": {"by_tier": {"vertex-flash": {"usd": 0.0101}}}, "git": "09925ad",
             "query_cost": {"planner": {"uncached": 500}}},
            {"stage": "retrieve", "corpus": "lme", "corpora": ["lme"], "timestamp": "2026-09-06T15:30:00+00:00",
             "invocation_id": "retrieve-2", "job_tag": "chain_lme_pass1c", "tag": "", "arms": ["S5_noPGR"], "readers": [],
             "budgets": [4000, 8000], "max_usd": 35.0, "stopped": None, "usd": 0.0, "calls": 0, "cost": {"by_tier": {}},
             "git": "1b5614a", "query_cost": {"planner": {"uncached": 0}}}]
    qrow = {"stage": "qa", "corpus": "lme", "corpora": ["lme", "mhrag"], "timestamp": "2026-09-06T20:00:00+00:00",
            "invocation_id": "qa-1", "job_tag": "chain_qa_all", "tag": "", "arms": ["S5_noPGR", "ours_cheap"],
            "readers": ["reader_a", "reader_b"], "budgets": [4000, 8000], "max_usd": 40.0, "stopped": None, "usd": 9.5,
            "calls": 15200, "cost": {"by_tier": {"azure-gpt54": {"usd": 9.0}, "vertex-flash": {"usd": 0.5}}}, "git": "d8239df"}
    ledger = PT.ledger_summary({"retrieve": {"longmemeval": rows}, "qa": {"longmemeval": [qrow], "multihoprag": [qrow]}})
    rmeta = {"query_cost": {"planner": {"calls": 8, "uncached": 0, "tokens_in": 1600, "tokens_out": 16, "usd": 0.0}}}
    planner = PT._planner_attribution(rmeta, rows, 0.1, 0.4)
    label = "partial build snapshot, 220 of 500 spaces, build in progress"
    costs = {"index": {"longmemeval": {"pgr_build": {"usd": 28.5, "calls": 1, "tokens_in": 1, "tokens_out": 1, "seconds": 1}}},
             "query": {"chandan_live": {"longmemeval": {"calls": 660, "tokens_in": 100, "tokens_out": 10, "usd": 0.01,
                                                        "seconds_per_question": 3.0, "n_questions": 220}},
                       "planner": {"longmemeval": {"calls": 8, "tokens_in": 1600, "tokens_out": 16,
                                                   "usd": planner["usd_at_study_price"], "seconds_per_question": 0.0,
                                                   "n_questions": 8}}},
             "ledger": ledger, "planner": {"longmemeval": planner},
             "builds": {"longmemeval": {"complete": False, "in_progress": True, "label": label, "n_spaces": 220,
                                        "n_wanted": 500, "running_job_tags": ["pgr-lme-full-s0"], "stopped": [],
                                        "run_log_usd": 52.2, "subset": None},
                        "multihoprag": {"complete": False, "in_progress": False,
                                        "label": "partial build snapshot, 0 of 1 spaces, build stopped at its cap",
                                        "n_spaces": 0, "n_wanted": 1, "running_job_tags": [],
                                        "stopped": [{"job_tag": "pgr-mhrag-full", "stopped": "cap"}],
                                        "run_log_usd": 10.007, "subset": None}},
             "meters": {"retrieve_lme": {"total_usd": 0}}, "caps": {"retrieve_lme": 35.0}}
    buckets = [E.BucketResult("q1", "S5_noPGR", "longmemeval", "knowledge-update", 3, "context assembly failure", True,
                              "superseding turn a#0 rendered before superseded turn b#0 on the same date",
                              reader="reader_a", budget=4000, ku_clause_fired=True, ku_gold_turn_earlier=True),
               E.BucketResult("q2", "S5_noPGR", "longmemeval", "multi-session", 4, "reader failure", True,
                              "every evidence turn inside the rendered context", reader="reader_b", budget=4000)]
    record = {"n": 450, "n_agree": 360, "agreement": 0.8, "primary": "gpt-5.4", "timestamp": "2026-09-06T17:00:00+00:00",
              "history": [{"timestamp": "2026-09-06T15:00:00+00:00", "n": 250, "agreement": 0.848, "primary": "gpt-5.4",
                           "note": "seeded from the qa_audit.json written before the history was kept"},
                          {"timestamp": "2026-09-06T17:00:00+00:00", "n": 450, "agreement": 0.8, "primary": "gpt-5.4"}]}
    audit = {"n": 450, "n_agree": 360, "agreement": 0.8, "threshold": 0.9, "cheap": "gemini-2.5-flash-lite",
             "strong": "gpt-5.4", "primary": "gpt-5.4", "second": "gemini-2.5-flash-lite", "cells": []}
    payload = E.build_metrics(lme=lme, mhr=mhr, records=[], audit=audit, buckets=buckets, subsets=subsets,
                              subsets_sha256="abc", commits={"head": "x"}, costs=costs, graphiti_status="dropped",
                              absent=absent, arm_status=status, arm_notes=notes, qa_audit_record=record)
    assert payload["arm_status"] == status and payload["arm_notes"] == notes
    assert payload["tests"]["pass_rule"]["T8b_note"] == "S5_primary run without post-graph-rag tables"
    assert payload["cost"]["arms"]["chandan_live"]["longmemeval"]["index_usd"] == pytest.approx(28.5)
    assert payload["cost"]["arms"]["planner"]["longmemeval"]["query_usd"] == pytest.approx(0.0001664, abs=1e-6)
    text = RP.render_report(payload)
    # the retrieval rows keep the absent cell and print the state under the table
    block = text.split("## Retrieval")[1].split("## By question type")[0]
    assert "| chandan_live | absent |" in block and "| graphiti | absent |" in block
    assert "Absent rows: S5_primary (not run in the retrieve pass); chandan_full (export present, not run in the " \
           "retrieve pass); chandan_live (export present, not run in the retrieve pass); graphiti (no export)." in block
    assert "chandan_live (no export)" in block and "S4_static (not run in the retrieve pass)" in block
    # the MultiHop-RAG S5 row and the tests that read it carry the label
    assert "| note |" in block and "| run without post-graph-rag tables |" in block
    tests_block = text.split("## Pre-declared tests")[1].split("## Predictions")[0]
    assert "S5_primary run without post-graph-rag tables" in tests_block
    pass_block = text.split("## The pass rule")[1].split("## Setup")[0]
    assert "| T8b note | S5_primary run without post-graph-rag tables |" in pass_block
    assert "second-build rule: not evaluated, build incomplete" in pass_block
    # one cost row per arm, the build charged to chandan_live, the state beside it, no second absent row
    cost = text.split("## Cost and time")[1].split("## Missing outputs")[0]
    assert cost.count("| chandan_live | longmemeval |") == 1
    assert "| chandan_live | longmemeval | 28.50 | pgr_build (" + label + ") |" in cost
    assert "| export present, not run in the retrieve pass |" in cost and "| no export |" in cost
    assert cost.count("| chandan_full | longmemeval |") == 1 and "| planner | longmemeval |" in cost
    assert "| pseudo-arm |" in cost
    # the planner sentence and the pass that paid
    assert "The planner row on longmemeval is a pseudo-arm" in cost
    assert "S5_primary, S5_primary_norule, S5_noPGR" in cost
    assert "8 of 8 were served from the cache and 0 were paid" in cost
    assert "The pass that paid for them: 2026-09-06T14:10:00+00:00 (job tag retrieve_lme_pass1, 500 uncached calls, " \
           "USD 0.01 for the whole pass)." in cost
    # the build state and the ledger against the section 11 caps, the last-pass meter beside it
    assert "| multihoprag | partial build snapshot, 0 of 1 spaces, build stopped at its cap | no | none | pgr-mhrag-full: cap | 0 | 1 | ORDER | 10.01 |" in cost
    assert "Spend ledger (section 11)" in cost
    # the ledger column is what the ledger itself recorded; the stage total beside
    # it is what the stage's own metrics file holds (none in this payload)
    assert "| retrieve | 35.00 | planner 10 and S2 relevance tests 25 | 0.01 | 2 | longmemeval 0.01 (2 invocations) " \
           "| n/a | n/a | none | no |" in cost
    assert "| qa | 100.00 |" in cost and "| 9.50 | 1 | longmemeval 9.50 (1 invocations), multihoprag 9.50 " \
           "(1 invocations) | n/a | n/a | none | no |" in cost
    assert "| index | 15.00 | overlay generation 15 | 0.00 | 0 | none | n/a | n/a | none | no |" in cost
    assert "The ledger was added part way through the study" in cost
    assert "| retrieve | lme | 2026-09-06T14:10:00+00:00 | retrieve_lme_pass1 | none | S5_noPGR | none | 4,000, 8,000 | 35.00 | 0.01 | 500 | vertex-flash 0.01 | no | 09925ad |" in cost
    assert "| qa | lme, mhrag | 2026-09-06T20:00:00+00:00 | chain_qa_all |" in cost
    assert "Last-pass meter totals" in cost
    # one bucket table per arm and reader, with the knowledge-update reading
    bk = text.split("## Failure buckets")[1].split("## Judges")[0]
    assert "### S5_noPGR, longmemeval, reader_a, budget 4,000 tokens" in bk
    assert "### S5_noPGR, longmemeval, reader_b, budget 4,000 tokens" in bk
    assert "Knowledge-update bucket 3 cases where the clause fired: 1, of which the gold turn is the earlier of the two by timestamp: 1." in bk
    assert "all arms and readers: 1 of 1 cases where the clause fired" in bk
    assert "Counts pool both readers" not in bk
    # the judge decision as the qa stage recorded it, with the first pass
    jd = text.split("## Judges")[1].split("## Answering accuracy")[0]
    assert "Decision as recorded by the qa stage: made on 450 pooled verdicts, 360 agreeing, agreement 0.800, " \
           "primary gpt-5.4, at 2026-09-06T17:00:00+00:00." in jd
    assert "First pass: 250 pooled verdicts, agreement 0.848, primary gpt-5.4 (2026-09-06T15:00:00+00:00)." in jd
    assert "Absent on longmemeval: S5_primary (not run in the retrieve pass)" in text
    assert not BANNED.search(text)
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    assert "None" not in prose and "{" not in prose
    # an older payload without the states, notes, ledger or history renders as before
    old = E.build_metrics(lme=lme, mhr=mhr, records=[], audit=audit, buckets=[], subsets=subsets, subsets_sha256="abc",
                          commits={"head": "x"}, graphiti_status="dropped", absent=absent)
    old_text = RP.render_report(old)
    assert "No ledger rows yet" in old_text and "Absent rows: S5_primary (absent)" in old_text
    assert "No bucket data" in old_text and "Decision as recorded" not in old_text
    assert "| chandan_live | longmemeval | absent |" in old_text
    assert not BANNED.search(old_text)


# ----------------------------------------------------------------------------
# The report defects found by the independent recomputation of REPORT.md
# ----------------------------------------------------------------------------
def test_the_build_state_carries_every_attempt_and_names_the_owner_stop(tmp_path):
    """Defect 1: the MultiHop-RAG build was tried twice. The first attempt
    stopped at its USD 10 cap, the second was stopped by the owner and left no
    run log at all, so it is recorded in attempts.json beside the log. Both
    attempts appear, with their stop reasons and how far each got, and a test
    that could not run says the build was stopped rather than never tried."""
    subsets = eval_subsets()
    root = tmp_path / "pgr"
    (root / "mhrag").mkdir(parents=True)
    (root / "mhrag" / "run_pgr-mhrag-full.json").write_text(
        _run_log("pgr-mhrag-full", "2026-09-06T16:05:38+05:30", "2026-09-07T00:24:38+05:30", 10.007077,
                 stopped="cap: spend 10.0071 USD exceeds the cap of 10.00"))
    (root / "mhrag" / "attempts.json").write_text(json.dumps({"attempts": [
        {"job_tag": "pgr-mhrag-full", "n_indexed": 592, "n_wanted": 609, "unit": "articles", "source": "design"},
        {"job_tag": "pgr-mhrag-full3", "started": "2026-09-07T07:30:00+05:30",
         "ended": "2026-09-07T08:17:21+05:30", "n_indexed": 72, "n_wanted": 609, "unit": "articles",
         "stopped": "stopped by the owner on 2026-09-07 at 72 of 609 articles",
         "no_run_log": True, "proxy_log_requests": 467, "proxy_log_usd": 1.699976, "source": "proxy log"}]}))
    b = PT._pgr_build_state("mhrag", subsets, root)
    assert b["n_attempts"] == 2 and not b["complete"]
    first, second = b["attempts"]
    assert first["job_tag"] == "pgr-mhrag-full" and first["run_log"] and first["progress"] == "592 of 609 articles"
    assert first["usd"] == pytest.approx(10.007077) and first["usd_source"] == "runner meter"
    assert second["job_tag"] == "pgr-mhrag-full3" and second["run_log"] is False
    assert second["progress"] == "72 of 609 articles" and second["usd"] == pytest.approx(1.699976)
    assert second["usd_source"].startswith("proxy request log")
    assert [s["job_tag"] for s in b["stopped"]] == ["pgr-mhrag-full", "pgr-mhrag-full3"]
    assert b["label"] == ("partial build snapshot, 0 of 1 spaces, 2 attempts, the last stopped by the owner on "
                          "2026-09-07 at 72 of 609 articles")
    # a build with one cap stop and no attempts file keeps the old wording
    plain = tmp_path / "plain"
    (plain / "mhrag").mkdir(parents=True)
    (plain / "mhrag" / "run_pgr-mhrag-full.json").write_text(
        _run_log("pgr-mhrag-full", "2026-09-06T16:05:38+05:30", "2026-09-07T00:24:38+05:30", 10.0, stopped="cap: x"))
    assert PT._pgr_build_state("mhrag", subsets, plain)["label"].endswith("build stopped at its cap")
    # the not-run reason of a test on the absent arm names the stop
    why = ("the post-graph-rag build on that corpus never finished: pgr-mhrag-full cap: spend 10.0071 USD exceeds "
           "the cap of 10.00 (592 of 609 articles); pgr-mhrag-full3 stopped by the owner on 2026-09-07 at 72 of "
           "609 articles. No query was ever run over it")
    tests = E.run_tests({}, {}, [], subsets, absent={E.MULTIHOPRAG: ["chandan_live"]},
                        absent_reasons={E.MULTIHOPRAG: {"chandan_live": why}})
    t8a = next(t for t in tests["family_A"] if t["name"] == "T8a")
    assert t8a["ran"] is False and "stopped by the owner" in t8a["reason"] and "592 of 609" in t8a["reason"]
    text = RP.render_report({"tests": tests, "cost": {"builds": {"multihoprag": b}}})
    assert "stopped by the owner on 2026-09-07 at 72 of 609 articles" in text
    assert "| multihoprag | pgr-mhrag-full3 |" in text and "| multihoprag | pgr-mhrag-full |" in text
    assert not BANNED.search(text)


def test_the_ledger_is_read_against_the_stage_totals_of_the_metrics_files(tmp_path, monkeypatch):
    """Defect 2: the ledger was added part way through, so its sums cover a
    fraction of the invocations. The stage total comes from each stage's own
    metrics file, and the passes that neither the ledger nor a metrics file
    kept are reported from the run logs as a sum only."""
    monkeypatch.setattr(PT, "OUT", tmp_path / "results" / "part1")
    meter = CostMeter(max_usd=5.0)
    meter.record("vertex-flash", 100000, 10000)          # 0.014 USD
    PT.append_cost_ledger("", "lme", "qa", meter, corpora=["lme"], invocation_id="qa-1")
    stage_cost = {"index": {"longmemeval": {"total_usd": 0.281447, "total_calls": 2566},
                            "multihoprag": {"total_usd": 0.031848, "total_calls": 291}},
                  "qa": {"longmemeval": {"total_usd": 0.019596, "total_calls": 1002},
                         "multihoprag": {"total_usd": 10.9883, "total_calls": 17533}}}
    totals = PT.stage_totals_from_metrics(stage_cost)
    assert totals["index"]["usd"] == pytest.approx(0.313295) and totals["index"]["calls"] == 2857
    assert totals["qa"]["usd"] == pytest.approx(11.007896) and totals["qa"]["calls"] == 18535
    logs = tmp_path / "results" / "part1" / "logs"
    logs.mkdir(parents=True)
    (logs / "index_lme.log").write_text("[part1] index written to results/part1/lme/index in 111.6 min, USD 0.2814\n")
    (logs / "chain_qa.log").write_text("[part1] qa done in 118.6 min, USD 10.9883\n"
                                       "[part1] qa done in 35.6 min, USD 4.1136\n"
                                       "[part1] qa done in 6.1 min, USD 0.0196\n"
                                       "[part1] retrieve written to results/part1/mhrag/retrieve in 5 min, USD 0.4892\n")
    passes = PT.stage_log_passes(tmp_path / "results" / "part1")
    assert [p["usd"] for p in passes["qa"]] == [10.9883, 4.1136, 0.0196]
    assert passes["index"] == [{"log": "index_lme.log", "corpus": "lme", "usd": 0.2814}]
    assert passes["retrieve"][0]["corpus"] == "mhrag"
    s = PT.ledger_summary(PT.read_cost_ledgers("", ["lme", "mhrag"]), totals, passes)["stages"]
    assert s["qa"]["total_usd"] == pytest.approx(0.014) and s["qa"]["n_invocations"] == 1
    assert s["qa"]["metrics_total_usd"] == pytest.approx(11.007896) and s["qa"]["metrics_total_calls"] == 18535
    # the passes the metrics files lost: 15.1215 logged minus 11.0079 kept
    assert s["qa"]["not_in_any_metrics_file_usd"] == pytest.approx(4.1136, abs=1e-4)
    assert s["index"]["n_invocations"] == 0 and s["index"]["not_in_any_metrics_file_usd"] == pytest.approx(0.0)
    assert not s["qa"]["over_cap"]
    text = RP.render_report({"cost": {"ledger": {"stages": s, "note": "n"}}})
    cost = text.split("## Cost and time")[1]
    assert "| stage | section 11 cap USD | cap covers | recorded invocations USD | recorded invocations | " \
           "recorded per corpus | stage total from the metrics files USD | calls in that total | " \
           "stage total per corpus | over cap |" in cost
    assert "| qa | 100.00 |" in cost and "| 0.01 | 1 | longmemeval 0.01 (1 invocations) | 11.01 | 18,535 |" in cost
    assert "The ledger was added part way through the study, so most invocations were never itemised: they are " \
           "counted in the stage total" in cost
    assert "which puts the stage at USD 15.12 in total" in cost
    assert not BANNED.search(text)
    # over the cap is read against the fuller number, not the ledger sum alone
    big = PT.ledger_summary({}, {"qa": {"usd": 120.0, "calls": 1, "by_corpus": {}}}, {})["stages"]["qa"]
    assert big["over_cap"] and big["total_usd"] == 0.0


def test_the_ranking_arms_are_read_without_parsing_the_whole_file(tmp_path):
    """Defect 8: rankings.json reaches 155 MB, so the report reads the arm
    names off the file instead of loading it; a pass that recorded them uses
    the record."""
    p = tmp_path / "rankings.json"
    p.write_text(json.dumps({"4000": {"ours_cheap": {"q1": ["u1"]}, "chandan_live": {"q1": ["u2"]}},
                             "8000": {"ours_cheap": {"q1": ["u1"]}}}, indent=2) + "\n")
    assert PT.ranking_arms({}, p) == {"4000": ["chandan_live", "ours_cheap"], "8000": ["ours_cheap"]}
    assert PT.ranking_arms({"ranking_arms": {"4000": ["S5_primary"]}}, p) == {"4000": ["S5_primary"]}
    assert PT.ranking_arms({}, tmp_path / "missing.json") == {}

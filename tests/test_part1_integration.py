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

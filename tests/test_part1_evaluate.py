"""Tests for part1.evaluate and part1.report on synthetic per-question data.

No model is called: judges are fake callables, rendered contexts are built
from RenderedUnits directly, and scores are QuestionScore objects made by
hand. Covered: the reader and judge prompts, the record format and the
missing-output rule, the judge audit and the second-judge flow, the failure
buckets in the stated order, the exact McNemar test against a hand
computation, Holm within a family, the D1, D2, T5 and T7 readings, the pass
rule, the refused-id rule, the metrics payload, and the report rendering with
no missing key and no banned character.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from multicard.metrics.stats import holm, holm_adjusted, mcnemar_exact
from multicard.part1 import evaluate as E
from multicard.part1 import report as RP
from multicard.part1.render import RenderedContext, RenderedUnit
from multicard.part1.score import MhrQuestionScore, QuestionScore, _empty_score
from multicard.part1.units import FactLocation

BANNED = re.compile("[\u2014\u2013\u2192\u2018\u2019\u201c\u201d\u2026]")


# ----------------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------------
def unit(uid, body, turn_chars=None, truncated=False, sessions=None, provenance=None, kind="turn",
         container=None, index=0, order=0):
    container = container or uid.rsplit("#", 1)[0]
    return RenderedUnit(unit_id=uid, kind=kind, container=container, index=index, date="2023-05-20",
                        label="user", body=body, line=body, tokens=len(body.split()), truncated=truncated,
                        turn_chars=turn_chars if turn_chars is not None else {uid: len(body)},
                        provenance_chars=provenance or {}, sessions=tuple(sessions or (container,)),
                        take_order=order)


def context(units, candidates=None, candidate_units=None):
    return RenderedContext(text="\n\n".join(u.line for u in units), unit_ids=[u.unit_id for u in units],
                           units=list(units), truncated={u.unit_id: u.truncated for u in units},
                           tokens=sum(u.tokens for u in units), budget=4000, duplicate_share=0.0,
                           candidates=candidates or [u.unit_id for u in units],
                           n_candidates=len(candidates or units), n_taken=len(units),
                           n_truncated=sum(1 for u in units if u.truncated), candidate_units=candidate_units)


def qs(qid, jr, sjr=None, missing=False, session_level=False):
    if missing:
        return _empty_score(qid, session_level)
    return QuestionScore(qid=qid, joint_recall=jr, candidate_joint_recall=1.0,
                         session_joint_recall=jr if sjr is None else sjr,
                         turn_r10=None if session_level else jr, turn_ndcg10=None if session_level else jr,
                         sess_r5=1.0, rendered_tokens=3500, duplicate_share=0.05, n_candidates=40,
                         n_rendered=8, n_truncated=0, truncated_evidence=0)


def ms(qid, fjr, all_located=True):
    return MhrQuestionScore(qid=qid, fact_joint_recall=fjr, doc_joint_recall=fjr,
                            candidate_fact_joint_recall=1.0, candidate_doc_joint_recall=1.0,
                            all_located=all_located, n_facts=2, n_contained=int(2 * fjr), n_fallback=0,
                            doc_hits={"d1": True}, rendered_tokens=3800, duplicate_share=0.0,
                            n_candidates=100, n_rendered=7, n_truncated=1)


LME_TYPES = ("single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning",
             "knowledge-update", "single-session-preference")
MHR_TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")


def synthetic_subsets(n_lme=36, n_mhr=24):
    lme_ids = [f"q{i:03d}" for i in range(n_lme)]
    lme_types = {q: LME_TYPES[i % len(LME_TYPES)] for i, q in enumerate(lme_ids)}
    abstention = lme_ids[-3:]
    answerable = [q for q in lme_ids if q not in abstention]
    mhr_ids = [f"mhr_q{i:04d}" for i in range(n_mhr)]
    mhr_types = {q: MHR_TYPES[i % len(MHR_TYPES)] for i, q in enumerate(mhr_ids)}
    order = {E.LONGMEMEVAL: list(reversed(lme_ids)), E.MULTIHOPRAG: list(mhr_ids)}
    g150 = [q for q in order[E.LONGMEMEVAL] if q in answerable][:18]
    local = [q for q in order[E.LONGMEMEVAL] if q in answerable and lme_types[q] in E.LOCAL_TYPES]
    mh_answer = [q for q in mhr_ids][:16]
    reader_b = mh_answer[:8]
    cells = []
    for reader, arms in (("reader_a", ["S5_primary", "chandan_live", "graphiti", "ours_cheap"]),
                         ("reader_b", ["S5_primary", "chandan_live"])):
        for arm in arms:
            cells.append({"arm": arm, "corpus": E.LONGMEMEVAL, "reader": reader, "population": "x",
                          "n_population": n_lme, "n": 4, "ids": order[E.LONGMEMEVAL][:4]})
            if arm != "graphiti":
                cells.append({"arm": arm, "corpus": E.MULTIHOPRAG, "reader": reader, "population": "x",
                              "n_population": 16, "n": 3, "ids": mh_answer[:3]})
    return {
        "meta": {"seed": 13, "corpora": {E.LONGMEMEVAL: {"n": n_lme, "n_abstention": len(abstention)},
                                         E.MULTIHOPRAG: {"n": n_mhr}}},
        "types": {E.LONGMEMEVAL: lme_types, E.MULTIHOPRAG: mhr_types},
        "abstention": {E.LONGMEMEVAL: abstention},
        "ORDER": order,
        "GRAPHITI_150": g150,
        "CHANDAN_CAL_18": g150[:6],
        "MHRAG_ANSWER": mh_answer,
        "READER_B_MHRAG": reader_b,
        "JUDGE_AUDIT": {"share": 0.1, "arms": {}, "cells": cells},
        "derived": {"GRAPHITI_PILOT": g150[0], "LOCAL_120": local},
    }


def record(qid, arm, reader="reader_a", corpus=E.LONGMEMEVAL, qtype="multi-session", abstention=False,
           verdicts=None, missing=False, budget=4000):
    r = E.AnswerRecord(qid=qid, corpus=corpus, arm=arm, reader=reader, budget=budget, qtype=qtype,
                       abstention=abstention, question="What?", gold="42", answer="" if missing else "42",
                       verdicts=dict(verdicts or {}), rendered_tokens=3000, context_sha256=E.context_sha256("c"),
                       context_chars=1, missing=missing)
    return r


class FakeJudge:
    """Answers "yes" or "no" by a rule over the prompt, counting calls."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return SimpleNamespace(text="Yes." if self.rule(prompt) else "No.", tokens_in=1, tokens_out=1, cached=False)


# ----------------------------------------------------------------------------
# Prompts and records
# ----------------------------------------------------------------------------
def test_reader_prompt_follows_e5_layout_and_mhrag_substitutions():
    p = E.reader_prompt("Where?", "2023-05-30", "[2023-05-20, user] hi", E.LONGMEMEVAL)
    assert p.startswith("Today is 2023-05-30.\n\n")
    assert "\nExcerpts (oldest first):\n\n[2023-05-20, user] hi\n\nQuestion: Where?\nAnswer:" in p
    assert "conversation excerpts" in p and "who spoke" in p
    m = E.reader_prompt("Where?", "2023-12-25", "", E.MULTIHOPRAG)
    assert "news excerpts" in m and "which outlet" in m
    assert "conversation excerpts" not in m and "who spoke" not in m
    # the rest of the rules is untouched
    assert E.READER_RULES_MHRAG.count("\n") == E.READER_RULES.count("\n")
    assert E.mhrag_question_date([SimpleNamespace(published_at="2023-10-01T00:00:00"),
                                  SimpleNamespace(published_at="2023-12-25T09:00:00")]) == "2023-12-25"


def test_judge_prompts_by_corpus_and_type():
    lme = E.judge_prompt_for(E.LONGMEMEVAL, "temporal-reasoning", "Q", "A", "R", False)
    assert "off-by-one" in lme
    std = E.judge_prompt_for(E.MULTIHOPRAG, "comparison_query", "Q", "Gold", "R", False)
    assert "Correct Answer: Gold" in std and "Rubric" not in std
    null = E.judge_prompt_for(E.MULTIHOPRAG, "null_query", "Q", "Insufficient information.", "R", True)
    # the query's own gold answer is the explanation, as e5 judge_prompt passes it
    assert "unanswerable question" in null and "Explanation: Insufficient information." in null
    assert null == E.judge_prompt("single-session-user", "Q", "Insufficient information.", "R", True)
    assert E.verdict_from_text("YES, correct") and not E.verdict_from_text("No.")


def test_records_round_trip_and_missing_rule(tmp_path):
    r = record("q1", "S5_primary", verdicts={"cheap": True, "strong": False})
    m = E.missing_answer(qid="q2", corpus=E.LONGMEMEVAL, arm="chandan_live", reader="reader_a", budget=4000,
                         qtype="multi-session", abstention=False, question="Q", gold="G", judges=("cheap", "strong"))
    assert m.missing and m.answer == "" and m.verdicts == {"cheap": False, "strong": False}
    assert m.correct is False and m.second_verdict() is False
    E.set_primary([r, m], "cheap")
    assert r.correct is True and r.second_verdict() is False
    path = tmp_path / "answers.jsonl"
    E.write_records(path, [r, m])
    back = E.read_records(path)
    assert [x.to_dict() for x in back] == [r.to_dict(), m.to_dict()]
    # a missing record is never sent to a judge
    j = FakeJudge(lambda p: True)
    assert E.judge_record(m, j, "cheap") is False and j.calls == 0
    # an existing verdict is kept
    assert E.judge_record(r, j, "strong") is False and j.calls == 0


def test_judge_audit_decides_primary_and_second_judge_scores_wrong_answers():
    subsets = synthetic_subsets()
    order = subsets["ORDER"][E.LONGMEMEVAL]
    recs = []
    for arm in ("S5_primary", "chandan_live", "S4_static"):
        for q in order[:8]:
            recs.append(record(q, arm))
    recs.append(record(order[0], "S5_primary", budget=8000))
    cheap = FakeJudge(lambda p: "Model Response: 42" in p)
    strong = FakeJudge(lambda p: "Model Response: 42" in p)
    audit = E.run_judge_audit(recs, subsets, cheap, strong, "cheap", "strong")
    # 4 audit ids per cell, two cells present in the records (S4_static has no cell), 4k only
    assert audit.n == 8 and audit.n_agree == 8 and audit.agreement == 1.0
    assert audit.primary == "cheap" and audit.second == "strong"
    assert cheap.calls == 8 and strong.calls == 8
    assert {(c["arm"], c["n"]) for c in audit.cells} == {("S5_primary", 4), ("chandan_live", 4)}
    assert E.decide_primary(0.89, "cheap", "strong") == "strong"
    assert E.decide_primary(0.90, "cheap", "strong") == "cheap"
    E.set_primary(recs, audit.primary)
    # the primary judge scores everything else; make two head-to-head answers wrong
    E.judge_all(recs, cheap, "cheap")
    recs[13].verdicts["cheap"] = False         # chandan_live, order[5], not in the audit
    recs[21].verdicts["cheap"] = False         # S4_static, not head-to-head
    recs[13].answer = "not it"
    disagree = FakeJudge(lambda p: True)
    n = E.second_judge_on_wrong(recs, disagree, "strong")
    assert n == 1 and recs[13].verdicts["strong"] is True and "strong" not in recs[21].verdicts
    cells = E.accuracy_tables(recs)
    c = cells["reader_a"][E.LONGMEMEVAL]["chandan_live"]["4000"]
    assert c["n"] == 8 and c["acc_all"] == pytest.approx(7 / 8) and c["primary_judge"] == "cheap"
    assert c["second_judge"]["n"] == 5 and c["second_judge"]["n_disagree"] == 1
    assert "8000" in cells["reader_a"][E.LONGMEMEVAL]["S5_primary"]


# ----------------------------------------------------------------------------
# Failure buckets, in the stated order
# ----------------------------------------------------------------------------
def lme_case(**kw):
    t1, t2 = "s1#0", "s2#0"
    texts = {t1: "I moved to Leeds in May and the answer is 42 for sure.", t2: "Another turn about cats."}
    full = [unit(t1, texts[t1]), unit(t2, texts[t2])]
    base = dict(qid="q", arm="S5_primary", corpus=E.LONGMEMEVAL, qtype="multi-session", gold="42",
                rendered=context(full), candidate_units=full, verdict_primary=False, verdict_second=False,
                evidence=[t1, t2], lengths={t: len(x) for t, x in texts.items()}, texts=texts,
                dates={t1: "2023-05-20", t2: "2023-05-20"}, index_units=None)
    base.update(kw)
    return E.BucketCase(**base)


def test_buckets_in_order_on_longmemeval():
    t1, t2 = "s1#0", "s2#0"
    # not a wrong answer
    assert E.assign_bucket(lme_case(verdict_primary=True)).bucket == 0
    # bucket 5 first: the judges disagree, even though every unit is present
    r = E.assign_bucket(lme_case(verdict_second=True))
    assert r.bucket == 5 and r.decidable
    # undecidable when the second judge did not score it: falls through to 1 to 4
    r = E.assign_bucket(lme_case(verdict_second=None))
    assert r.bucket == 4 and not r.decidable
    # bucket 1: the exported index covers only one evidence turn (competitor arm)
    c = lme_case(arm="chandan_live", index_units=[unit("chunk1", "x" * 60, turn_chars={t1: 60}, kind="chunk")])
    assert E.assign_bucket(c).bucket == 1
    # a relation whose provenance chunk covers the turn represents it
    c = lme_case(arm="chandan_live", index_units=[unit("chunk1", "x" * 60, turn_chars={t1: 60}, kind="chunk"),
                                                  unit("rel1", "r", turn_chars={}, provenance={t2: 24}, kind="relation")])
    assert E.assign_bucket(c).bucket == 4
    # bucket 2: represented, but the candidate list lacks a turn
    c = lme_case(candidate_units=[unit(t1, lme_case().texts[t1])])
    assert E.assign_bucket(c).bucket == 2
    # bucket 3a: in the candidates, outside the rendered context
    only_t1 = [unit(t1, lme_case().texts[t1])]
    c = lme_case(rendered=context(only_t1))
    r = E.assign_bucket(c)
    assert r.bucket == 3 and "outside" in r.reason
    # bucket 3a also when the rendered part is under half of the turn
    half = [unit(t1, "I moved to Leeds", turn_chars={t1: 16}, truncated=True), unit(t2, lme_case().texts[t2])]
    assert E.assign_bucket(lme_case(rendered=context(half))).bucket == 3
    # bucket 3b: covered (over half) but the gold answer sits in the cut-off tail
    txt = lme_case().texts[t1]
    cut = txt[:36]
    assert "42" not in cut and len(cut) >= len(txt) / 2
    tail = [unit(t1, cut, turn_chars={t1: len(cut)}, truncated=True), unit(t2, lme_case().texts[t2])]
    r = E.assign_bucket(lme_case(rendered=context(tail)))
    assert r.bucket == 3 and "cut-off tail" in r.reason
    # the same truncation with the gold elsewhere counts as inside (bucket 4) and is counted
    r = E.assign_bucket(lme_case(rendered=context(tail), gold="Leeds"))
    assert r.bucket == 4 and r.n_inside_half_covered == 1
    # a turn rendered whole across two chunks (the rendered characters are summed, as the
    # coverage rule sums them) is inside: no cut-off tail, bucket 4
    first, second = txt[:20], txt[20:]
    two = [unit("chunk:1", first, turn_chars={t1: len(first)}, kind="chunk", container="s1"),
           unit("chunk:2", second, turn_chars={t1: len(second)}, kind="chunk", container="s1"),
           unit(t2, lme_case().texts[t2])]
    r = E.assign_bucket(lme_case(rendered=context(two)))
    assert r.bucket == 4 and r.n_inside_half_covered == 0
    assert E._rendered_chars(t1, two) == len(txt)
    # bucket 3c: knowledge-update, same date, superseding turn rendered first
    ku = lme_case(qtype="knowledge-update")
    r = E.assign_bucket(ku)
    assert r.bucket == 3 and "superseding" in r.reason
    # rendered in the right order: bucket 4
    texts = ku.texts
    right = context([unit(t2, texts[t2]), unit(t1, texts[t1])])
    r = E.assign_bucket(lme_case(qtype="knowledge-update", rendered=right))
    assert r.bucket == 4 and not r.ku_clause_skipped
    # the clause does not fire when both turns hold the gold; the case is counted
    both = dict(texts)
    both[t2] = "The answer was 42 before."
    r = E.assign_bucket(lme_case(qtype="knowledge-update", texts=both,
                                 rendered=context([unit(t1, both[t1]), unit(t2, both[t2])]),
                                 candidate_units=[unit(t1, both[t1]), unit(t2, both[t2])]))
    assert r.bucket == 4 and r.ku_clause_skipped
    # different dates: the clause does not apply
    r = E.assign_bucket(lme_case(qtype="knowledge-update", dates={t1: "2023-05-20", t2: "2023-05-21"}))
    assert r.bucket == 4


def test_buckets_session_level_and_multihoprag():
    # Graphiti: evidence at session level, edges cite sessions
    fact = unit("e1", "FACT one", turn_chars={}, sessions=("s1",), kind="fact", container="")
    other = unit("e2", "FACT two", turn_chars={}, sessions=("s2",), kind="fact", container="")
    base = dict(qid="q", arm="graphiti", corpus=E.LONGMEMEVAL, qtype="multi-session", gold="42",
                rendered=context([fact, other]), candidate_units=[fact, other], verdict_primary=False,
                verdict_second=False, session_level=True, evidence_sessions=["s1", "s2"], index_units=[fact, other])
    assert E.assign_bucket(E.BucketCase(**base)).bucket == 4
    assert E.assign_bucket(E.BucketCase(**{**base, "index_units": [fact]})).bucket == 1
    assert E.assign_bucket(E.BucketCase(**{**base, "candidate_units": [fact]})).bucket == 2
    assert E.assign_bucket(E.BucketCase(**{**base, "rendered": context([fact])})).bucket == 3
    # MultiHop-RAG: facts located to chunks
    full_a = "Alpha won the cup on Sunday. The final was played in Lyon."
    full_b = "Beta finished second. Ticket prices rose ten percent."
    ca = unit("d1#0", full_a, turn_chars={}, kind="chunk", container="d1", index=0)
    cb = unit("d2#0", full_b, turn_chars={}, kind="chunk", container="d2", index=0)
    locs = [FactLocation("q", "d1", "The final was played in Lyon", "d1#0", True, False),
            FactLocation("q", "d2", "Beta finished second", "d2#0", True, False)]
    mbase = dict(qid="q", arm="chandan_live", corpus=E.MULTIHOPRAG, qtype="inference_query", gold="Lyon",
                 rendered=context([ca, cb]), candidate_units=[ca, cb], verdict_primary=False, verdict_second=False,
                 locations=locs, texts={"d1#0": full_a, "d2#0": full_b}, index_units=[ca, cb])
    assert E.assign_bucket(E.BucketCase(**mbase)).bucket == 4
    assert E.assign_bucket(E.BucketCase(**{**mbase, "index_units": [cb]})).bucket == 1
    assert E.assign_bucket(E.BucketCase(**{**mbase, "candidate_units": [cb]})).bucket == 2
    # chunk rendered truncated with the fact in the cut-off tail
    cut = unit("d1#0", full_a[:20], turn_chars={}, kind="chunk", container="d1", index=0, truncated=True)
    r = E.assign_bucket(E.BucketCase(**{**mbase, "rendered": context([cut, cb])}))
    assert r.bucket == 3 and "cut-off tail" in r.reason
    r = E.assign_bucket(E.BucketCase(**{**mbase, "rendered": context([cb])}))
    assert r.bucket == 3 and "outside" in r.reason
    # a truncated chunk that still holds the fact is inside and counted
    cut2 = unit("d1#0", full_a[:57], turn_chars={}, kind="chunk", container="d1", index=0, truncated=True)
    r = E.assign_bucket(E.BucketCase(**{**mbase, "rendered": context([cut2, cb])}))
    assert r.bucket == 4 and r.n_inside_half_covered == 1
    # counts per arm, corpus and type
    rows = [E.assign_bucket(E.BucketCase(**mbase)), E.assign_bucket(E.BucketCase(**{**mbase, "index_units": [cb]})),
            E.assign_bucket(E.BucketCase(**{**mbase, "verdict_second": None}))]
    counts = E.bucket_counts(rows)
    tot = counts["chandan_live"][E.MULTIHOPRAG]["total"]
    assert tot["4"] == 2 and tot["1"] == 1 and tot["5"] == 0 and tot["n_wrong"] == 3 and tot["n_undecidable"] == 1
    assert counts["chandan_live"][E.MULTIHOPRAG]["by_type"]["inference_query"]["n_inside_half_covered"] == 0
    assert counts["chandan_live"][E.MULTIHOPRAG]["bucket5_decidable"] is True


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def test_mcnemar_exact_against_hand_computation():
    # a right and b wrong on 8, b right and a wrong on 2, both right on 3, both wrong on 2.
    a = [1] * 8 + [0] * 2 + [1] * 3 + [0] * 2
    b = [0] * 8 + [1] * 2 + [1] * 3 + [0] * 2
    r = mcnemar_exact(a, b)
    assert (r.n, r.n_discordant, r.a_only, r.b_only, r.both, r.neither) == (15, 10, 8, 2, 3, 2)
    # two-sided exact: 2 * P(X <= 2 | n=10, p=0.5) = 2 * (1 + 10 + 45) / 1024
    assert r.p_value == pytest.approx(2 * 56 / 1024)
    assert r.acc_a == pytest.approx(11 / 15) and r.acc_b == pytest.approx(5 / 15)
    assert r.delta == pytest.approx(6 / 15)
    # symmetric in the two arms, and 1.0 with no discordant pair
    assert mcnemar_exact(b, a).p_value == pytest.approx(r.p_value)
    assert mcnemar_exact([1, 0, 1], [1, 0, 1]).p_value == 1.0
    # equal discordant counts give p = 1 (2 * P(X <= k) with k = n/2 exceeds 1 and is capped)
    assert mcnemar_exact([1, 0], [0, 1]).p_value == 1.0
    # 1 versus 5: 2 * (1 + 6) / 64
    assert mcnemar_exact([1] + [0] * 5, [0] + [1] * 5).p_value == pytest.approx(14 / 64)
    with pytest.raises(ValueError):
        mcnemar_exact([1, 0], [1])


def test_holm_adjusted_matches_holm_verdicts():
    ps = {"T1": 0.010, "T2": 0.030, "T8a": 0.004, "T8b": 0.200}
    adj = holm_adjusted(ps)
    # sorted: T8a 0.004*4=0.016, T1 0.010*3=0.030, T2 0.030*2=0.060, T8b 0.200*1=0.200
    assert adj == pytest.approx({"T8a": 0.016, "T1": 0.030, "T2": 0.060, "T8b": 0.200})
    sig = holm(ps)
    assert {k: adj[k] <= 0.05 for k in ps} == sig == {"T8a": True, "T1": True, "T2": False, "T8b": False}
    # monotone: a later raw p that would fall below an earlier adjusted one is lifted
    adj2 = holm_adjusted({"a": 0.02, "b": 0.021, "c": 0.5})
    assert adj2["a"] == pytest.approx(0.06) and adj2["b"] == pytest.approx(0.06)
    assert holm_adjusted({}) == {}


def test_apply_holm_within_family_leaves_partial_out():
    def t(name, p, partial=False, ran=True):
        x = E.PairedTest(name, "A", "a", "b", "m", "c", "pop", ran=ran, partial=partial)
        x.p = p if ran else None
        x.delta = 0.05
        return x

    fam = [t("T1", 0.02), t("T2", 0.01, partial=True), t("T8a", 0.004), t("T8b", 0.3, ran=False)]
    sig = E.apply_holm(fam)
    assert sig == {"T1": True, "T8a": True}
    assert fam[0].holm_adjusted_p == pytest.approx(0.02) and fam[2].holm_adjusted_p == pytest.approx(0.008)
    assert fam[1].holm_significant is None and not fam[1].in_holm
    assert fam[3].holm_significant is None
    assert E.apply_holm([]) == {}


# ----------------------------------------------------------------------------
# Tests as data, the readings and the pass rule
# ----------------------------------------------------------------------------
def test_paired_values_apply_refused_and_missing_rules():
    a = {"q1": 1.0, "q2": 0.0, "q4": 1.0}
    b = {"q1": 1.0, "q2": 1.0, "q3": 1.0}
    kept, va, vb, dropped, ma, mb = E.paired_values(a, b, ["q1", "q2", "q3", "q4", "q5"], {"q5"})
    assert kept == ["q1", "q2", "q3", "q4"] and dropped == ["q5"]
    assert va == [1.0, 0.0, 0.0, 1.0] and vb == [1.0, 1.0, 1.0, 0.0]
    assert (ma, mb) == (1, 1)
    t = E.run_paired("T1", "A", "S5_primary", "chandan_live", a, b, ["q1", "q2", "q3", "q4", "q5"], {"q5"},
                     "joint_recall", E.LONGMEMEVAL, "pop")
    assert t.n == 4 and t.n_refused == 1 and t.refused_ids == ["q5"] and t.n_missing_b == 1
    assert (t.wins, t.ties, t.losses) == (1, 1, 2) and t.delta == pytest.approx(-0.25)
    short = E.run_paired("T1", "A", "a", "b", {"q1": 1.0}, {"q1": 0.0}, ["q1"], set(), "m", E.LONGMEMEVAL, "pop")
    assert not short.ran and "fewer" in short.reason
    m = E.run_mcnemar("C1", "C", "a", "b", {"q1": True, "q2": False}, {"q1": False, "q2": False}, ["q1", "q2", "q3"],
                      set(), "reader_a", E.LONGMEMEVAL, "pop")
    assert m.n == 3 and m.n_missing_a == 1 and m.n_missing_b == 1 and m.wins == 1 and m.losses == 0
    assert m.p == 1.0


def _paired(name, delta, lo, hi, sig, wins=10, losses=5, ran=True, partial=False):
    t = E.PairedTest(name, "A", "S5_primary", "x", "joint_recall", E.LONGMEMEVAL, "pop", ran=ran, partial=partial)
    if ran:
        t.delta, t.ci_low, t.ci_high, t.p = delta, lo, hi, 0.01
        t.wins, t.ties, t.losses = wins, 100, losses
        t.holm_significant = sig
        t.in_holm = not partial
    return t


def test_d1_d2_t5_t7_readings():
    assert E.d1_label(_paired("T1", 0.03, 0.01, 0.05, True)) == "shown"
    assert E.d1_label(_paired("T1", 0.03, -0.01, 0.05, False)) == "not shown"
    assert E.d1_label(_paired("T1", 0.0, -0.02, 0.02, False)) == "failed"
    assert E.d1_label(_paired("T1", -0.02, -0.04, 0.0, True)) == "failed"
    assert E.d1_label(_paired("T1", 0, 0, 0, False, ran=False)) == "not run"
    assert E.d1_label(_paired("T1", 0.05, 0.02, 0.08, True, partial=True)).startswith("partial")
    d2 = E.d2_reading(_paired("T2", 0.02, -0.03, 0.07, False), "run")
    assert d2["passes"] and not d2["strict"] and not d2["not_run"]
    d2 = E.d2_reading(_paired("T2", 0.04, 0.01, 0.07, True), "run")
    assert d2["passes"] and d2["strict"]
    d2 = E.d2_reading(_paired("T2", 0.04, 0.01, 0.07, True), "dropped")
    assert d2["not_run"] and not d2["passes"]
    assert not E.d2_reading(_paired("T2", -0.01, -0.03, 0.01, False), "run")["passes"]
    # T5 branches
    r3_r0 = _paired("T5a", 0.03, 0.01, 0.05, True)
    assert E.t5_branch(_paired("T5a", 0.01, -0.01, 0.03, False), _paired("T5b", 0.0, -0.02, 0.02, False),
                       _paired("P0", 0.0, -0.02, 0.02, False))["branch"] == "no measurable effect"
    assert E.t5_branch(r3_r0, _paired("T5b", 0.005, -0.01, 0.02, False),
                       _paired("P0", 0.025, 0.005, 0.045, True))["branch"] == "adds material"
    assert E.t5_branch(r3_r0, _paired("T5b", 0.03, 0.01, 0.05, True),
                       _paired("P0", 0.0, -0.02, 0.02, False))["branch"] == "reconciles"
    assert E.t5_branch(r3_r0, _paired("T5b", 0.03, -0.01, 0.05, False),
                       _paired("P0", 0.0, -0.02, 0.02, False))["branch"] == "not resolved"
    assert E.t5_branch(r3_r0, _paired("T5b", 0, 0, 0, False, ran=False), r3_r0)["branch"] == "not run"
    # T7: losses minus wins at most 3
    assert E.t7_rule(_paired("T7", -0.01, -0.03, 0.01, False, wins=1, losses=4))["holds"]
    assert not E.t7_rule(_paired("T7", -0.02, -0.04, 0.0, False, wins=1, losses=5))["holds"]
    assert not E.t7_rule(_paired("T7", 0, 0, 0, False, ran=False))["holds"]
    assert not E.t7_rule(_paired("T7", 0.0, 0, 0, False, wins=2, losses=2, partial=True))["holds"]


def test_pass_rule_is_one_boolean_over_named_quantities():
    t1 = _paired("T1", 0.03, 0.01, 0.05, True)
    t2 = _paired("T2", 0.02, -0.03, 0.07, False)
    t8a = _paired("T8a", 0.06, 0.02, 0.10, True)
    t7 = _paired("T7", -0.01, -0.03, 0.01, False, wins=2, losses=4)
    q = E.pass_rule(t1, t2, t8a, t7, "run")
    assert q["passed"] is True
    assert q["T1_label"] == "shown" and q["T2_passes"] and not q["T2_strict"] and q["T7_net_losses"] == 2
    assert not E.pass_rule(_paired("T1", 0.03, -0.01, 0.05, False), t2, t8a, t7, "run")["passed"]
    assert not E.pass_rule(t1, t2, _paired("T8a", 0.06, -0.02, 0.10, False), t7, "run")["passed"]
    assert not E.pass_rule(t1, _paired("T2", -0.02, -0.07, 0.03, False), t8a, t7, "run")["passed"]
    assert E.pass_rule(t1, _paired("T2", -0.02, -0.07, 0.03, False), t8a, t7, "dropped")["passed"]
    assert not E.pass_rule(t1, t2, t8a, _paired("T7", -0.03, -0.05, 0.0, False, wins=1, losses=5), "run")["passed"]
    assert not E.pass_rule(_paired("T1", 0.05, 0.02, 0.08, True, partial=True), t2, t8a, t7, "run")["passed"]


# ----------------------------------------------------------------------------
# End to end: synthetic scores, records, tests, payload and report
# ----------------------------------------------------------------------------
def synthetic_run():
    subsets = synthetic_subsets()
    rng = np.random.default_rng(13)
    answerable = E.answerable_ids(subsets)
    order_m = subsets["ORDER"][E.MULTIHOPRAG]
    non_null = E.non_null_ids(subsets)
    arms = ["ours_cheap", "S4_static", "S5_primary", "S5_overlay_R0", "S5_overlay_P0", "chandan_live"]
    rate = {"ours_cheap": 0.55, "S4_static": 0.65, "S5_primary": 0.85, "S5_overlay_R0": 0.75,
            "S5_overlay_P0": 0.75, "chandan_live": 0.45}
    lme = {}
    mhr = {}
    for arm in arms:
        lme[arm] = {4000: {}, 8000: {}}
        mhr[arm] = {4000: {}, 8000: {}}
        for q in answerable:
            if arm == "chandan_live" and q == answerable[3]:
                continue                       # no output: scored 0 and counted
            v = 1.0 if rng.random() < rate[arm] else 0.0
            lme[arm][4000][q] = qs(q, v)
            lme[arm][8000][q] = qs(q, max(v, 1.0 if rng.random() < 0.3 else 0.0))
        for i, q in enumerate(non_null):
            v = 1.0 if rng.random() < rate[arm] else 0.0
            mhr[arm][4000][q] = ms(q, v, all_located=(i % 5 != 0))
            mhr[arm][8000][q] = ms(q, v, all_located=(i % 5 != 0))
    g150 = subsets["GRAPHITI_150"]
    lme["graphiti"] = {4000: {q: qs(q, 1.0 if rng.random() < 0.5 else 0.0, session_level=True) for q in g150}}
    # answers under both readers, two judges named by model
    cheap, strong = "gemini-2.5-flash-lite", "gpt-5.4"
    records = []
    types = subsets["types"]
    lme_all = subsets["ORDER"][E.LONGMEMEVAL]
    abst = set(subsets["abstention"][E.LONGMEMEVAL])
    for reader in ("reader_a", "reader_b"):
        for arm in ["S5_primary", "chandan_live", "graphiti", "ours_cheap"]:
            ids = g150 if arm == "graphiti" else lme_all
            for q in ids:
                ok = rng.random() < rate.get(arm, 0.5)
                v = {cheap: ok, strong: ok if rng.random() < 0.9 else (not ok)}
                records.append(record(q, arm, reader, E.LONGMEMEVAL, types[E.LONGMEMEVAL][q], q in abst, v))
            if arm != "graphiti":
                ids_m = subsets["MHRAG_ANSWER"] if reader == "reader_a" else subsets["READER_B_MHRAG"]
                for q in ids_m:
                    ok = rng.random() < rate.get(arm, 0.5)
                    records.append(record(q, arm, reader, E.MULTIHOPRAG, types[E.MULTIHOPRAG][q],
                                          types[E.MULTIHOPRAG][q] == "null_query", {cheap: ok, strong: ok}))
    records.append(E.missing_answer(qid=lme_all[0], corpus=E.LONGMEMEVAL, arm="S4_static", reader="reader_a",
                                    budget=4000, qtype=types[E.LONGMEMEVAL][lme_all[0]], abstention=False,
                                    question="Q", gold="G", judges=(cheap, strong)))
    E.set_primary(records, cheap)
    audit = E.JudgeAudit(n=140, n_agree=130, agreement=130 / 140, threshold=0.9, cheap=cheap, strong=strong,
                         primary=cheap, second=strong,
                         cells=[{"arm": "S5_primary", "corpus": E.LONGMEMEVAL, "reader": "reader_a", "n": 50,
                                 "n_agree": 47, "agreement": 0.94}])
    buckets = [E.assign_bucket(lme_case(arm="chandan_live")), E.assign_bucket(lme_case(verdict_second=True))]
    costs = {
        "index": {E.LONGMEMEVAL: {"pgr_build": {"usd": 91.5, "calls": 50224, "tokens_in": 10, "tokens_out": 20, "seconds": 3600},
                                  "graphiti": {"usd": 22.0, "calls": 18000, "tokens_in": 5, "tokens_out": 1, "seconds": 900},
                                  "overlay": {"usd": 3.2, "calls": 5000, "tokens_in": 1, "tokens_out": 1, "seconds": 100,
                                              "arms": ["S5_primary", "S5_overlay_P0"]}},
                  E.MULTIHOPRAG: {"pgr_build": {"usd": 4.1, "calls": 1218, "tokens_in": 2, "tokens_out": 3, "seconds": 300}}},
        "query": {"S5_primary": {E.LONGMEMEVAL: {"calls": 500, "tokens_in": 100000, "tokens_out": 1000, "usd": 0.5,
                                                 "seconds_per_question": 0.8, "n_questions": 500}},
                  "chandan_live": {E.LONGMEMEVAL: {"calls": 1000, "tokens_in": 200000, "tokens_out": 5000, "usd": 2.0,
                                                   "seconds_per_question": 9.0, "n_questions": 500}}},
        "answering": {"reader_a": {"usd": 3.0}, "reader_b": {"usd": 40.0}},
        "judging": {"usd": 12.0},
        "proxy": {"pgr_build_lme_job_tags": ["pgr-lme-full"]},
        "caps": {"answering_and_judging": 100},
    }
    payload = E.build_metrics(
        lme=lme, mhr=mhr, records=records, audit=audit, buckets=buckets, subsets=subsets,
        subsets_sha256="0e2440828ec0e2f2568bb90742b21c684b73b2cc4db85738879703a5311f5082",
        commits={"design_and_env": "b8c1e20", "subsets": "deadbeef"}, costs=costs,
        setup={"encoder": "sentence-transformers/all-MiniLM-L6-v2", "graphiti_ingestion": "one message episode per turn",
               "long_evidence_turns": ["s_x#3"]},
        refused={"chandan_live": {answerable[5]}}, partial={}, graphiti_status="run",
        location={"n_queries": 20, "n_all_located": 16, "query_share": 0.8, "n_facts": 45, "n_located": 41,
                  "fact_share": 41 / 45, "n_fallback": 4, "n_straddle": 1, "n_unresolved": 0},
        context_hashes={"S5_primary": {q: "a" for q in answerable}, "S5_overlay_R0": {q: "b" for q in answerable[:10]},
                        "S5_overlay_P0": {q: "a" for q in answerable}},
        disclosures=["The Graphiti ingestion unit is one message episode per turn, Zep's own granularity."],
        models={"chat_model": cheap, "prices": {cheap: {"in": 0.1, "out": 0.4}}, "price_page_date": "2026-09-06",
                "locations": {cheap: "us-central1"}},
    )
    return subsets, lme, mhr, records, payload


def test_run_tests_families_populations_and_rules():
    subsets, lme, mhr, records, payload = synthetic_run()
    tests = payload["tests"]
    names_a = [t["name"] for t in tests["family_A"]]
    assert names_a == ["T1", "T2", "T8a", "T8b"]
    assert [t["name"] for t in tests["family_B"]] == ["T3", "T4", "T5a", "T5b", "T6"]
    assert [t["name"] for t in tests["family_C"]] == ["C1", "C2", "C3", "C4", "C5", "C6"]
    t1 = tests["family_A"][0]
    answerable = E.answerable_ids(subsets)
    # one refused id dropped and listed; the no-output question counted missing for chandan_live
    assert t1["n"] == len(answerable) - 1 and t1["refused_ids"] == [answerable[5]] and t1["n_missing_b"] == 1
    assert t1["in_holm"] and t1["holm_adjusted_p"] is not None and t1["label"].startswith("D1: ")
    assert tests["D1"] in ("shown", "not shown", "failed")
    t2 = tests["family_A"][1]
    assert t2["metric"] == "session_joint_recall" and t2["n"] == len(subsets["GRAPHITI_150"])
    t8a = tests["family_A"][2]
    assert t8a["population"].startswith("MultiHop-RAG") and t8a["n"] == len(E.all_located_ids(
        {a: d[4000] for a, d in mhr.items()}, subsets))
    assert set(tests["holm"]["A"]) == {"T1", "T2", "T8a", "T8b"}
    assert set(tests["holm"]["B"]) == {"T3", "T4", "T5a", "T5b", "T6"}
    assert set(tests["holm"]["C"]) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert tests["T5"]["branch"] in ("no measurable effect", "adds material", "reconciles", "not resolved")
    assert tests["T5"]["n_contexts_differ"] == len(answerable)      # R0 hash differs everywhere
    t7 = tests["T7"]
    assert t7["population"] == "LOCAL_120" and t7["n"] == len(subsets["derived"]["LOCAL_120"])
    assert t7["rule"]["net_losses"] == t7["losses"] - t7["wins"]
    c1 = tests["family_C"][0]
    assert c1["reader"] == "reader_a" and c1["n"] == len(subsets["ORDER"][E.LONGMEMEVAL]) - 1
    assert c1["robustness"] is not None and "outcome_differs" in c1["robustness"]
    c6 = tests["family_C"][5]
    assert c6["population"] == "READER_B_MHRAG" and c6["n"] == len(subsets["READER_B_MHRAG"])
    pr = tests["pass_rule"]
    assert isinstance(pr["passed"], bool)
    assert pr["passed"] == (pr["T1_shown"] and pr["T8a_positive_and_significant"] and pr["T2_ok"] and pr["T7_holds"])
    # graphiti dropped: T2 and the graphiti McNemar tests are not run, Holm A over three tests
    tests2 = E.run_tests({a: d[4000] for a, d in lme.items()}, {a: d[4000] for a, d in mhr.items()}, records,
                         subsets, graphiti_status="dropped")
    assert not tests2["family_A"][1]["ran"] and set(tests2["holm"]["A"]) == {"T1", "T8a", "T8b"}
    assert tests2["pass_rule"]["T2_not_run"] and tests2["pass_rule"]["T2_ok"]
    assert not tests2["family_C"][2]["ran"] and not tests2["family_C"][3]["ran"]
    # a partial arm: labelled, out of Holm and out of the pass rule
    tests3 = E.run_tests({a: d[4000] for a, d in lme.items()}, {a: d[4000] for a, d in mhr.items()}, records,
                         subsets, partial={"chandan_live": True})
    assert tests3["family_A"][0]["partial"] and not tests3["family_A"][0]["in_holm"]
    assert tests3["pass_rule"]["T1_label"].startswith("partial") and not tests3["pass_rule"]["passed"]
    # a subset override for T1 (section 13)
    tests4 = E.run_tests({a: d[4000] for a, d in lme.items()}, {a: d[4000] for a, d in mhr.items()}, records,
                         subsets, overrides={"T1": {"ids": subsets["GRAPHITI_150"], "label": "T1 on GRAPHITI_150"}})
    assert tests4["family_A"][0]["n"] == len(subsets["GRAPHITI_150"])
    assert "GRAPHITI_150" in tests4["family_A"][0]["label"]
    # a Reader B withdrawal under the answering cap: only the Family C tests under that reader are
    # partial; T1, T3, T8b and T7 on the arm stay complete and in Holm (section 5)
    lme4 = {a: d[4000] for a, d in lme.items()}
    mhr4 = {a: d[4000] for a, d in mhr.items()}
    tests5 = E.run_tests(lme4, mhr4, records, subsets, partial_answering={"reader_b": {"ours_cheap", "chandan_live"}})
    assert not tests5["family_A"][0]["partial"] and tests5["family_A"][0]["in_holm"]
    assert tests5["T7"]["rule"]["label"] in ("holds", "does not hold")
    c1, c2 = tests5["family_C"][0], tests5["family_C"][1]
    assert not c1["partial"] and c2["partial"] and "reader_b withdrawn" in c2["label"]
    assert tests5["partial_answering"] == {"reader_b": ["chandan_live", "ours_cheap"]}
    # an arm run on a subset of its population narrows every test with it and labels them
    g150 = subsets["GRAPHITI_150"]
    tests6 = E.run_tests(lme4, mhr4, records, subsets, arm_ids={"chandan_live": g150},
                         arm_ids_label="chandan arms restricted to GRAPHITI_150")
    assert tests6["family_A"][0]["n"] == len(g150) and "restricted to GRAPHITI_150" in tests6["family_A"][0]["label"]
    assert tests6["family_B"][0]["n"] == len(answerable)      # T3 has no chandan arm
    assert tests6["family_C"][0]["n"] == len(g150)
    # the second build: a robustness row outside Holm, the first build decides
    lme_second = dict(lme4)
    lme_second["chandan_live_second"] = {q: qs(q, 1.0 - s.joint_recall) for q, s in lme4["chandan_live"].items()}
    tests7 = E.run_tests(lme_second, mhr4, records, subsets)
    sb = tests7["second_build"]
    assert sb["ran"] and sb["T1_second_build"]["n"] == tests7["family_A"][0]["n"]
    assert sb["first_build_decides"] and not sb["T1_second_build"]["in_holm"]
    assert isinstance(sb["sign_differs"], bool) and isinstance(sb["significance_differs"], bool)
    assert not E.run_tests(lme4, mhr4, records, subsets)["second_build"]["ran"]


def test_predictions_are_read_against_the_tests():
    subsets, lme, mhr, records, payload = synthetic_run()
    rows = {r["name"]: r for r in payload["predictions"]}
    assert set(E.PREDICTION_LABELS) == {E.CONSISTENT, E.NOT_CONFIRMED, E.CONTRADICTED, E.UNTESTED}
    for name in ("T1_overall", "T1_multi_session", "T1_temporal_reasoning", "T1_local_set", "S4_over_ours_cheap",
                 "S5_gain_from_planner", "T5_branch", "largest_community_share", "graphiti_gap",
                 "mhrag_comparison", "mhrag_inference", "family_C_same_sign_LongMemEval",
                 "family_C_same_sign_MultiHop-RAG"):
        assert name in rows, name
        assert rows[name]["label"] in E.PREDICTION_LABELS
    t1 = payload["tests"]["family_A"][0]
    assert rows["T1_overall"]["measured"] == pytest.approx(t1["delta"]) and rows["T1_overall"]["band"] == [-0.01, 0.05]
    assert rows["T1_overall"]["label"] == E.band_label(t1["delta"], -0.01, 0.05, sign=1)
    answerable = E.answerable_ids(subsets)
    assert rows["T1_multi_session"]["n"] == sum(1 for q in answerable if q != answerable[5]
                                                and subsets["types"][E.LONGMEMEVAL][q] == "multi-session")
    assert rows["T1_local_set"]["n"] == len(subsets["derived"]["LOCAL_120"])
    assert rows["graphiti_gap"]["measured"] == pytest.approx(payload["tests"]["family_A"][1]["delta"])
    # S4_static and the overlay arms are in the synthetic run, so the planner row is measured
    assert rows["S5_gain_from_planner"]["measured"]["planner"] is not None
    assert rows["T5_branch"]["measured"] == payload["tests"]["T5"]["branch"]
    # no closed_book answers in the synthetic run: untested; no graph shares: untested
    assert rows["largest_community_share"]["label"] == E.UNTESTED
    assert any(r["name"].startswith("closed_book_floor") and r["label"] == E.UNTESTED for r in payload["predictions"])
    # the band rule
    assert E.band_label(0.02, -0.01, 0.05, 1) == E.CONSISTENT
    assert E.band_label(0.08, -0.01, 0.05, 1) == E.NOT_CONFIRMED
    assert E.band_label(-0.02, -0.01, 0.05, 1) == E.CONTRADICTED
    assert E.band_label(0.05, -0.02, 0.02, outside=E.CONTRADICTED) == E.CONTRADICTED
    assert E.band_label(None, 0, 1) == E.UNTESTED
    # majority-class rates from closed_book records
    recs = [record(f"m{i}", "closed_book", "reader_a", E.MULTIHOPRAG, "inference_query", False,
                   {"gemini-2.5-flash-lite": i % 2 == 0}) for i in range(4)]
    for i, r in enumerate(recs):
        r.gold = "Yes" if i < 3 else "No"
    E.set_primary(recs, "gemini-2.5-flash-lite")
    mc = E.majority_class_rates(recs)
    assert mc["inference_query"]["majority_class_rate"] == pytest.approx(0.75)
    assert mc["inference_query"]["accuracy"] == pytest.approx(0.5) and mc["inference_query"]["majority_class"] == "yes"


def test_metrics_payload_layout(tmp_path):
    subsets, lme, mhr, records, payload = synthetic_run()
    # JSON round trip with no unserialisable value
    path = E.write_metrics(payload, tmp_path / "metrics.json")
    m = json.loads(path.read_text())
    for key in ("design", "generated", "subsets_sha256", "commits", "setup", "arms", "retrieval", "answering",
                "judge_audit", "tests", "buckets", "cost", "multihoprag_location", "missing_output", "refused",
                "partial", "graphiti_status", "disclosures", "published", "predictions", "second_build",
                "partial_answering", "arm_populations"):
        assert key in m, key
    assert m["second_build"]["ran"] is False and m["second_build"]["first_build_usd"] == pytest.approx(91.5 + 4.1)
    assert m["retrieval"][E.LONGMEMEVAL]["S5_primary"]["4000"]["cut_evidence"] == 0
    r = m["retrieval"][E.LONGMEMEVAL]
    s5 = r["S5_primary"]["4000"]
    assert s5["n"] == len(E.answerable_ids(subsets)) and s5["n_missing"] == 0
    assert set(s5["by_type"]) == set(LME_TYPES)
    assert r["chandan_live"]["4000"]["n_missing"] == 1
    assert r["graphiti"]["4000"]["turn_r10"] is None and r["graphiti"]["4000"]["n"] == len(subsets["GRAPHITI_150"])
    assert "8000" in r["S5_primary"] and r["S5_primary"]["8000"]["joint_recall"] >= s5["joint_recall"]
    mh = m["retrieval"][E.MULTIHOPRAG]["S5_primary"]["4000"]
    n_non_null = len(E.non_null_ids(subsets))
    assert mh["n"] == n_non_null and mh["n_all_located"] == sum(1 for i in range(n_non_null) if i % 5 != 0)
    assert mh["fact_joint_recall_all_located"] is not None
    a = m["answering"]["reader_a"][E.LONGMEMEVAL]
    assert a["S4_static"]["4000"]["n_missing"] == 1 and a["S4_static"]["4000"]["acc_all"] == 0.0
    assert a["S5_primary"]["4000"]["n_abstention"] == 3 and a["S5_primary"]["4000"]["second_judge"]["n"] > 0
    assert m["answering"]["reader_b"][E.MULTIHOPRAG]["chandan_live"]["4000"]["n"] == len(subsets["READER_B_MHRAG"])
    # cost: his build charged to the arms that read his tables, overlay to the two named, graphiti to itself
    c = m["cost"]["arms"]
    assert c["S5_primary"][E.LONGMEMEVAL]["index_usd"] == pytest.approx(91.5 + 3.2)
    assert c["S4_static"][E.LONGMEMEVAL]["index_usd"] == pytest.approx(91.5)
    assert c["ours_cheap"][E.LONGMEMEVAL]["index_usd"] == 0.0
    assert c["graphiti"][E.LONGMEMEVAL]["index_usd"] == pytest.approx(22.0)
    assert c["chandan_live"][E.MULTIHOPRAG]["index_usd"] == pytest.approx(4.1)
    assert c["chandan_live"][E.LONGMEMEVAL]["query_calls_per_question"] == pytest.approx(2.0)
    assert m["multihoprag_location"]["partial_set"] is True
    assert m["missing_output"]["chandan_live"][E.LONGMEMEVAL]["4000"] == 1
    assert m["refused"]["chandan_live"] == [E.answerable_ids(subsets)[5]]
    assert m["published"]["zep_71.2"]["value"] == 0.712 and len(m["disclosures"]) == len(E.DISCLOSURES) + 1
    assert m["setup"]["reader_a"] == "gemini-2.5-flash-lite" and m["setup"]["budgets"] == [4000, 8000]
    assert m["buckets"]["chandan_live"][E.LONGMEMEVAL]["total"]["4"] == 1
    assert m["buckets"]["S5_primary"][E.LONGMEMEVAL]["total"]["5"] == 1


def test_report_renders_every_section_from_the_file(tmp_path):
    subsets, lme, mhr, records, payload = synthetic_run()
    path = E.write_metrics(payload, tmp_path / "metrics.json")
    out = RP.write_report(path, tmp_path / "REPORT.md")
    text = out.read_text()
    for heading in ("# Part 1:", "## The pass rule", "## Setup", "## Retrieval", "## By question type",
                    "## Pre-declared tests", "### Family A", "### Family B", "### Family C", "### T7",
                    "## Predictions (section 10)", "## Failure buckets", "## Judges", "## Answering accuracy",
                    "## Cost and time", "## MultiHop-RAG fact location", "## Missing outputs", "## Disclosures",
                    "## Published numbers"):
        assert heading in text, heading
    assert "Second post-graph-rag build (section 9): first build cost USD 95.60; second build ran: no." in text
    # the proxy cross-check is printed as pending until the payload carries a window total
    assert "cross-check against the proxy request log by time window and job tag is pending" in text
    assert "**consistent with, not confirmed**" in text or "**not confirmed**" in text or "**contradicted**" in text
    assert "of which cut at 2,000 characters" in text
    assert "0e2440828ec0e2f2568bb90742b21c684b73b2cc4db85738879703a5311f5082" in text
    assert "design_and_env: b8c1e20" in text
    assert "Part 1 passed: **" in text
    assert "T5 branch: **" in text
    assert "Primary judge decided before any test: **gemini-2.5-flash-lite**" in text
    assert "the gate ran on a partial set" in text
    assert "refused 1 ids" in text
    assert "zep_71.2" in text and "0.712" in text
    # every number came from the file: the T1 delta printed equals the payload value
    t1 = payload["tests"]["family_A"][0]
    assert f"{t1['delta']:+.3f}" in text
    assert f"{payload['retrieval'][E.LONGMEMEVAL]['S5_primary']['4000']['joint_recall']:.3f}" in text
    # no Python None, no raw dict outside the fenced JSON blocks, no banned character
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    assert "None" not in prose and "{" not in prose
    assert not BANNED.search(text)
    # the pass-rule table has a value for every named quantity
    block = text.split("## The pass rule")[1].split("## Setup")[0]
    for q in ("T1 label (D1)", "T8a positive and significant", "T2 satisfied", "T7 holds"):
        assert q in block
    assert block.count("| n/a |") == 0


def test_report_formatting_helpers():
    assert RP.f3(None) == "n/a" and RP.f3(0.12345) == "0.123" and RP.fs(0.02) == "+0.020"
    assert RP.fi(12345) == "12,345" and RP.fi(3.25) == "3.2" and RP.fi(None) == "n/a"
    assert RP.fci([-0.01, 0.02]) == "[-0.010, +0.020]" and RP.fci(None) == "n/a"
    assert RP.yes_no(True) == "yes" and RP.yes_no(None) == "n/a"
    assert RP.render_report({}).startswith("# Part 1:")


def test_source_files_have_no_banned_characters():
    import multicard.part1.evaluate as ev
    import multicard.part1.report as rp
    import multicard.metrics.stats as st
    from pathlib import Path

    for mod in (ev, rp, st):
        text = Path(mod.__file__).read_text()
        assert not BANNED.search(text), mod.__file__
    assert not BANNED.search(Path(__file__).read_text())


def test_index_charges_and_register_price_tiers():
    costs = {"index": {E.LONGMEMEVAL: {"pgr_build": {"usd": 10.0}, "graphiti": {"usd": 2.0},
                                       "planner": {"usd": 1.0, "arms": ["S5_primary"]}}}}
    ch = E.index_charges(costs, ["S5_primary", "chandan_live", "graphiti", "ours_cheap", "S5_noPGR", "S2_lazy"])
    assert ch["S5_primary"][E.LONGMEMEVAL]["usd"] == pytest.approx(11.0)
    assert ch["chandan_live"][E.LONGMEMEVAL]["usd"] == pytest.approx(10.0)
    assert ch["graphiti"][E.LONGMEMEVAL]["usd"] == pytest.approx(2.0)
    # section 1: his build cost is charged to every arm that reads his tables; S2_lazy fills its
    # budget with the fused ranking, whose relation and entity channels read them
    assert ch["S2_lazy"][E.LONGMEMEVAL]["usd"] == pytest.approx(10.0)
    assert "ours_cheap" not in ch and "S5_noPGR" not in ch
    from multicard.part1.retrieve import ARM_SPECS

    assert set(E.ARMS_READING_PGR_TABLES) == {n for n, s in ARM_SPECS.items() if s.pgr} | set(E.CHANDAN_ARMS)
    from multicard.llm.costmeter import PRICES_USD_PER_MTOK

    added = E.register_price_tiers({"prices": {"test-model-xyz": {"in": 0.1, "out": 0.4}, "vertex-flash": {"in": 9, "out": 9}}})
    assert added == {"test-model-xyz": {"in": 0.1, "out": 0.4}}
    assert PRICES_USD_PER_MTOK["vertex-flash"]["in"] != 9
    PRICES_USD_PER_MTOK.pop("test-model-xyz", None)
    assert E.generate.__doc__ and E.MIN_OUTPUT_TOKENS == 64

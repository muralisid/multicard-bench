"""Tests for the Part 1 subset draw (design section 2) on synthetic ids.

No data file is read. The synthetic LongMemEval list has the six real types
with 40 ids each and four abstention ids in four of the types; the synthetic
MultiHop-RAG list has the four real types with 160 ids each.
"""

import json

import pytest

from multicard.part1 import subsets as S

LME_TYPES = ("knowledge-update", "multi-session", "single-session-assistant",
             "single-session-preference", "single-session-user", "temporal-reasoning")
ABS_TYPES = ("knowledge-update", "multi-session", "single-session-user", "temporal-reasoning")
MH_TYPES = ("comparison_query", "inference_query", "null_query", "temporal_query")


def _lme(n_per_type=40, n_abs=4):
    recs = []
    for k, t in enumerate(LME_TYPES):
        for i in range(n_per_type):
            abst = t in ABS_TYPES and i < n_abs
            qid = f"lme{k}_{i:03d}" + ("_abs" if abst else "")
            recs.append((qid, t, abst))
    return recs


def _mh(n_per_type=160):
    return [(f"mhr_q{k * n_per_type + i:04d}", t)
            for k, t in enumerate(MH_TYPES) for i in range(n_per_type)]


@pytest.fixture(scope="module")
def payload():
    return S.build(_lme(), _mh())


def test_two_builds_are_byte_identical_and_seed_matters():
    a = S.serialise(S.build(_lme(), _mh()))
    b = S.serialise(S.build(_lme(), _mh()))
    assert a == b
    assert S.sha256(a) == S.sha256(b)
    # The same ids in a different input order give the same file.
    lme, mh = _lme(), _mh()
    c = S.serialise(S.build(list(reversed(lme)), list(reversed(mh))))
    assert c == a
    other = S.serialise(S.build(lme, mh, seed=14))
    assert other != a
    assert json.loads(other)["ORDER"]["longmemeval"] != json.loads(a)["ORDER"]["longmemeval"]


def test_order_is_a_permutation_of_every_id(payload):
    lme_ids = {q for q, _, _ in _lme()}
    mh_ids = {q for q, _ in _mh()}
    assert set(payload["ORDER"]["longmemeval"]) == lme_ids
    assert len(payload["ORDER"]["longmemeval"]) == len(lme_ids)
    assert set(payload["ORDER"]["multihoprag"]) == mh_ids
    assert payload["ORDER"]["multihoprag"] != sorted(mh_ids)


def test_graphiti_150_excludes_abstention_and_has_25_per_type(payload):
    ids = payload["GRAPHITI_150"]
    types = payload["types"]["longmemeval"]
    assert len(ids) == 150 and len(set(ids)) == 150
    assert not any(q.endswith("_abs") for q in ids)
    assert not set(ids) & set(payload["abstention"]["longmemeval"])
    per_type = {t: sum(1 for q in ids if types[q] == t) for t in LME_TYPES}
    assert per_type == {t: 25 for t in LME_TYPES}
    assert payload["derived"]["GRAPHITI_PILOT"] == ids[0]


def test_chandan_cal_18_is_3_per_type_inside_graphiti_150(payload):
    ids = payload["CHANDAN_CAL_18"]
    types = payload["types"]["longmemeval"]
    assert len(ids) == 18 and set(ids) <= set(payload["GRAPHITI_150"])
    assert {t: sum(1 for q in ids if types[q] == t) for t in LME_TYPES} == {t: 3 for t in LME_TYPES}


def test_mhrag_subsets_nest_and_balance(payload):
    types = payload["types"]["multihoprag"]
    ans, rb = payload["MHRAG_ANSWER"], payload["READER_B_MHRAG"]
    assert len(ans) == 600 and len(rb) == 200 and set(rb) <= set(ans)
    assert {t: sum(1 for q in ans if types[q] == t) for t in MH_TYPES} == {t: 150 for t in MH_TYPES}
    assert {t: sum(1 for q in rb if types[q] == t) for t in MH_TYPES} == {t: 50 for t in MH_TYPES}


def test_every_subset_is_listed_in_order(payload):
    pos = {c: {q: i for i, q in enumerate(payload["ORDER"][c])} for c in S.CORPORA}
    for key, corpus in (("GRAPHITI_150", "longmemeval"), ("CHANDAN_CAL_18", "longmemeval"),
                        ("MHRAG_ANSWER", "multihoprag"), ("READER_B_MHRAG", "multihoprag")):
        ranks = [pos[corpus][q] for q in payload[key]]
        assert ranks == sorted(ranks), key
    for cell in payload["JUDGE_AUDIT"]["cells"]:
        ranks = [pos[cell["corpus"]][q] for q in cell["ids"]]
        assert ranks == sorted(ranks)


def test_proportional_quota_uses_largest_remainder():
    counts = {"knowledge-update": 78, "multi-session": 133, "single-session-assistant": 56,
              "single-session-preference": 30, "single-session-user": 70,
              "temporal-reasoning": 133}
    q = S.proportional_quota(counts, 0.10)
    assert q == {"knowledge-update": 8, "multi-session": 13, "single-session-assistant": 6,
                 "single-session-preference": 3, "single-session-user": 7,
                 "temporal-reasoning": 13}
    assert sum(q.values()) == 50
    # 25 per type over six types: 2.5 each, the three first type names get the extra place.
    q = S.proportional_quota({t: 25 for t in LME_TYPES}, 0.10)
    assert sum(q.values()) == 15 and [q[t] for t in LME_TYPES] == [3, 3, 3, 2, 2, 2]


def test_judge_audit_cells_follow_the_arm_table(payload):
    audit = payload["JUDGE_AUDIT"]
    cells = audit["cells"]
    expected = sum(len(corpora) for _, arms in S.READERS for _, corpora in arms)
    assert len(cells) == expected
    assert len({(c["arm"], c["corpus"], c["reader"]) for c in cells}) == len(cells)
    assert audit["arms"]["reader_b"] == ["S5_primary", "chandan_live", "chandan_full", "graphiti",
                                         "ours_cheap", "oracle_full", "closed_book"]
    assert audit["arms"]["chandan_own"] == ["chandan_full_uncut"]
    assert "graphiti" in audit["arms"]["reader_a"] and "chandan_full_uncut" not in audit["arms"]["reader_a"]
    populations = {
        "LONGMEMEVAL_ALL": set(payload["ORDER"]["longmemeval"]),
        "GRAPHITI_150": set(payload["GRAPHITI_150"]),
        "MHRAG_ANSWER": set(payload["MHRAG_ANSWER"]),
        "READER_B_MHRAG": set(payload["READER_B_MHRAG"]),
    }
    for c in cells:
        pop = populations[c["population"]]
        assert c["n_population"] == len(pop)
        assert c["n"] == len(c["ids"]) == round(0.10 * len(pop))
        assert set(c["ids"]) <= pop and len(set(c["ids"])) == c["n"]
        if c["arm"] == "graphiti":
            assert c["corpus"] == "longmemeval" and c["population"] == "GRAPHITI_150"
        elif c["corpus"] == "longmemeval":
            assert c["population"] == "LONGMEMEVAL_ALL"
        elif c["reader"] == "reader_b":
            assert c["population"] == "READER_B_MHRAG"
        else:
            assert c["population"] == "MHRAG_ANSWER"
        # Stratified by type: each type is within one of its exact share.
        types = payload["types"][c["corpus"]]
        for t in set(types[q] for q in pop):
            n_t = sum(1 for q in pop if types[q] == t)
            got = sum(1 for q in c["ids"] if types[q] == t)
            assert abs(got - 0.10 * n_t) < 1, (c["arm"], c["corpus"], t)
    # Different cells over the same population draw different questions.
    a = next(c for c in cells if c["arm"] == "ours_cheap" and c["reader"] == "reader_a"
             and c["corpus"] == "longmemeval")
    b = next(c for c in cells if c["arm"] == "S4_static" and c["reader"] == "reader_a"
             and c["corpus"] == "longmemeval")
    assert a["ids"] != b["ids"]


def test_draw_by_type_raises_when_a_type_is_short():
    with pytest.raises(ValueError):
        S.draw_by_type([("a", "x"), ("b", "x")], {"x": 3}, S.rng(13))

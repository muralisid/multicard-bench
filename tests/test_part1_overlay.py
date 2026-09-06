"""Tests for part1.overlay on synthetic topics and communities with a mocked
chat client. No model is called and no cache is touched."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multicard.llm.costmeter import CostMeter
from multicard.llm.vertex import Response
from multicard.part1 import overlay as O
from multicard.part1 import units as U

REPO = Path(__file__).resolve().parents[1]


class Words:
    def count(self, text):
        return len(text.split())


class Chars:
    def count(self, text):
        return len(text)


def units(lo, hi):
    return {f"s#0/{i}" for i in range(lo, hi)}


# Topics and communities built by hand so every d is checkable.
TOPICS = {
    0: units(0, 20),
    1: units(20, 40),
    2: units(10, 15) | units(40, 50),
    -1: units(50, 60),           # the outlier topic, never a candidate
}
COMMUNITIES = {
    0: units(0, 15),             # shares 15 with topic 0, 5 with topic 2
    1: units(15, 30),            # shares 5 with topic 0, 10 with topic 1
    2: units(30, 34) | units(40, 50),   # shares 4 with topic 1 (below 5), 10 with topic 2
    3: units(50, 60),            # shares 10 with the outlier topic only
}
TERMS = {0: [f"t0w{i}" for i in range(12)], 1: ["cats", "pets"], 2: ["cars", "engines"]}
TEXTS = {f"s#0/{i}": f"unit {i} text about thing {i} with several more words here" for i in range(60)}


class FakeClient:
    """Replies keyed by (topic, community) read back from the prompt."""

    def __init__(self, replies, meter=None, cached=False):
        self.replies = replies
        self.meter = meter
        self.cached = cached
        self.calls = []
        self.model = "fake-model"

    def generate(self, prompt, max_output_tokens=800):
        self.calls.append((prompt, max_output_tokens))
        t = int(re.search(r"Topic (\S+), top terms", prompt).group(1))
        c = int(re.search(r"Community (\S+), member units", prompt).group(1))
        text = self.replies[(t, c)]
        if callable(text):
            text = text(prompt)
        if self.meter is not None:
            self.meter.record("vertex-flash", 100, 20)
        return Response(text, 100, 20, cached=self.cached)


def reply(same, conf, cited=()):
    return json.dumps({"same_subject": same, "confidence": conf, "cited_units": list(cited)})


# ----------------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------------
def test_topic_members_expands_owners_and_drops_the_outlier():
    rows = [{"unit_id": "s1#0", "topic": 3}, {"unit_id": "s1#1", "topic": -1},
            {"unit_id": "s2#0/1", "topic": 3}, {"unit_id": "s2#0", "topic": 4}]
    subs = {"s1#0": ["s1#0/0", "s1#0/1"], "s1#1": ["s1#1/0"], "s2#0": ["s2#0/0", "s2#0/1"]}
    m = O.topic_members(rows, subs)
    assert m == {3: {"s1#0/0", "s1#0/1", "s2#0/1"}, 4: {"s2#0/0", "s2#0/1"}}
    # a DataFrame with numpy ints works the same and the ids come back as python ints
    df = pd.DataFrame(rows)
    m2 = O.topic_members(df, subs)
    assert m2 == m and all(type(k) is int for k in m2)
    # an owner id without sub_units_of is an error, not a silent member
    with pytest.raises(ValueError):
        O.topic_members([{"unit_id": "s1#0", "topic": 1}])


def test_topic_terms_from_lists_names_and_pairs():
    rows = [{"topic": 0, "terms": [f"w{i}" for i in range(15)]},
            {"topic": 1, "name": "1_cat_dog_fish"},
            {"topic": 2, "terms": [("car", 0.5), ("bus", 0.2)]}]
    t = O.topic_terms(rows)
    assert t[0] == [f"w{i}" for i in range(10)]
    assert t[1] == ["cat", "dog", "fish"]
    assert t[2] == ["car", "bus"]
    assert O.topic_terms({5: ["a", "b"]}) == {5: ["a", "b"]}


def test_community_members_three_shapes():
    d = O.community_members({0: ["a/0", "a/1"], 1: ["b/0"]})
    assert d == {0: {"a/0", "a/1"}, 1: {"b/0"}}
    rows = [{"community": 0, "members": ["a/0", "a/1"]}, {"community": 1, "members": ["b/0"]}]
    assert O.community_members(rows) == d
    per_member = [{"community": 0, "unit_id": "a/0"}, {"community": 0, "unit_id": "a/1"},
                  {"community": 1, "unit_id": "b/0"}]
    assert O.community_members(pd.DataFrame(per_member)) == d


def test_sub_units_by_owner_and_unit_texts_from_tables():
    sessions = {"s1": {"date": "2023-05-20", "turns": [
        {"role": "user", "content": "I bought a red bike yesterday. It cost two hundred dollars."},
        {"role": "assistant", "content": "Great choice for the city.\nEnjoy the ride every day."}]}}
    tables = U.build_tables(sessions, ["s1"])
    subs = O.sub_units_by_owner(tables)
    assert subs["s1#0"] == ["s1#0/0", "s1#0/1"] and subs["s1#1"] == ["s1#1/0", "s1#1/1"]
    texts = O.unit_texts(tables)
    assert texts["s1#0/0"] == "I bought a red bike yesterday."


# ----------------------------------------------------------------------------
# Item 8: pairs and the flagged set
# ----------------------------------------------------------------------------
def test_candidate_pairs_match_hand_values():
    pairs = O.candidate_pairs(TOPICS, COMMUNITIES)
    by = {(p.topic_id, p.community_id): p for p in pairs}
    assert set(by) == {(0, 0), (0, 1), (1, 1), (2, 0), (2, 2)}      # (1, 2) shares 4, dropped
    assert by[(0, 0)].shared == 15 and by[(0, 0)].jaccard == pytest.approx(0.75)
    assert by[(0, 0)].d == pytest.approx(15 * 0.25)
    assert by[(1, 1)].d == pytest.approx(10 * (1 - 10 / 25))
    assert by[(2, 2)].d == pytest.approx(10 * (1 - 10 / 19))
    assert by[(0, 1)].d == pytest.approx(5 * (1 - 5 / 30))
    assert by[(2, 0)].d == pytest.approx(5 * (1 - 5 / 25))
    # sorted by d descending
    assert [(p.topic_id, p.community_id) for p in pairs] == [(1, 1), (2, 2), (0, 1), (2, 0), (0, 0)]
    # a lower min_shared admits the pair that shares 4
    assert (1, 2) in {(p.topic_id, p.community_id) for p in O.candidate_pairs(TOPICS, COMMUNITIES, min_shared=4)}


def test_flag_pairs_top_share_and_cap():
    pairs = O.candidate_pairs(TOPICS, COMMUNITIES)
    assert O.n_top_share(5) == 1 and O.n_top_share(10) == 2 and O.n_top_share(0) == 0
    top = O.flag_pairs(pairs, corpus="multihoprag")
    assert [(p.topic_id, p.community_id) for p in top] == [(1, 1)]
    assert [(p.topic_id, p.community_id) for p in O.flag_pairs(pairs, cap=500, share=0.6)] == [(1, 1), (2, 2), (0, 1)]
    assert len(O.flag_pairs(pairs, cap=2, share=0.6)) == 2
    assert O.CAPS == {"longmemeval": 5000, "multihoprag": 500}
    # ties in d break by topic id then community id, the same on every run
    tied = [O.Pair(2, 1, 5, 0.5, 2.5), O.Pair(1, 3, 5, 0.5, 2.5), O.Pair(1, 2, 5, 0.5, 2.5)]
    assert [(p.topic_id, p.community_id) for p in O.flag_pairs(tied, cap=10, share=1.0)] == [(1, 2), (1, 3), (2, 1)]


# ----------------------------------------------------------------------------
# Item 9: prompt, reply, proposals
# ----------------------------------------------------------------------------
def test_cut_to_tokens():
    text = " ".join(f"w{i}" for i in range(10))
    assert O.cut_to_tokens(text, Words(), 4) == "w0 w1 w2 w3"
    assert O.cut_to_tokens(text, Words(), 10) == text
    assert O.cut_to_tokens("a" * 50, Chars(), 5) == "aaaaa"
    assert O.cut_to_tokens("", Words(), 5) == ""


def test_shown_units_is_seeded_and_capped():
    members = {f"x/{i}" for i in range(30)}
    a = O.shown_units(members)
    assert len(a) == 8 and a == sorted(a) and set(a) <= members
    assert O.shown_units(members) == a
    assert O.shown_units(members, seed=14) != a
    assert O.shown_units({"x/1", "x/2"}) == ["x/1", "x/2"]


def test_build_prompt_and_parse_reply():
    p = O.build_prompt(3, [f"w{i}" for i in range(12)], 7, [("a/0", "first text"), ("a/1", "second text")])
    assert "Topic 3, top terms, most important first: w0, w1, w2, w3, w4, w5, w6, w7, w8, w9\n" in p
    assert "w10" not in p
    assert "Community 7, member units:\n\n[unit a/0] first text\n\n[unit a/1] second text\n" in p
    assert '{"same_subject": true, "confidence": 0.8, "cited_units": ["unit id", "unit id"]}' in p
    with pytest.raises(ValueError):
        O.build_prompt(3, [], 7, [])
    r = O.parse_reply('Sure. {"same_subject": true, "confidence": 0.85, "cited_units": ["a/0"]} done')
    assert r.ok and r.same_subject is True and r.confidence == 0.85 and r.cited == ["a/0"]
    r = O.parse_reply('{"same_subject": "no", "confidence": 1.7}')
    assert r.ok and r.same_subject is False and r.confidence == 1.0 and r.cited == []
    r = O.parse_reply('{"confidence": 0.3}')
    assert r.ok and r.same_subject is False
    assert not O.parse_reply("I cannot tell.").ok
    assert not O.parse_reply('{"same_subject": true}').ok
    assert not O.parse_reply('{"confidence": "high"}').ok
    assert not O.parse_reply("{not json}").ok


def test_propose_one_call_per_pair_with_mocked_client():
    pairs = O.candidate_pairs(TOPICS, COMMUNITIES)
    shown1 = O.shown_units(COMMUNITIES[1])
    hidden1 = [u for u in sorted(COMMUNITIES[1]) if u not in shown1][0]   # a member that is not shown
    replies = {
        (1, 1): reply(True, 0.9, [shown1[0], "not-shown"]),
        (2, 2): reply(True, 0.7),
        (0, 1): reply(False, 0.2, [hidden1]),
        (2, 0): "no json here",
        (0, 0): reply(False, 0.55),
    }
    meter = CostMeter(max_usd=1.0)
    client = FakeClient(replies, meter)
    props = O.propose(pairs, TERMS, COMMUNITIES, TEXTS, client, "fake-model", counter=Words(),
                      max_tokens=4, workers=2, log_every=0)
    assert len(client.calls) == 5 == meter.total_calls()
    assert all(m >= 64 for _, m in client.calls)
    # proposals come back in flagged order with the pair's ids
    assert [(p.topic_id, p.community_id) for p in props] == [(p.topic_id, p.community_id) for p in pairs]
    by = {(p.topic_id, p.community_id): p for p in props}
    p11 = by[(1, 1)]
    assert p11.parse_ok and p11.w == 0.9 and p11.same_subject is True and p11.model == "fake-model"
    assert p11.cited == [shown1[0]] and "not-shown" not in p11.cited
    assert p11.raw == replies[(1, 1)] and p11.tokens_in == 100 and p11.tokens_out == 20 and not p11.cached
    assert p11.shown == shown1 and len(shown1) == 8 and set(shown1) <= COMMUNITIES[1]
    assert by[(0, 1)].cited == []     # a cited member that was not shown is not kept
    bad = by[(2, 0)]
    assert not bad.parse_ok and bad.w == 0.0 and bad.confidence is None and bad.cited == [] and bad.error
    # the prompt shows the community's units cut at max_tokens, and the topic's terms
    prompt = client.calls[0][0]
    assert "cats, pets" in prompt
    shown_lines = re.findall(r"\[unit (\S+)\] (.*)", prompt)
    assert len(shown_lines) == 8 and all(len(t.split()) <= 4 for _, t in shown_lines)
    assert all(uid in COMMUNITIES[1] for uid, _ in shown_lines)
    # a topic without terms is refused rather than prompted blank
    with pytest.raises(ValueError):
        O.propose(pairs[:1], {}, COMMUNITIES, TEXTS, client, "m", counter=Words())


# ----------------------------------------------------------------------------
# Link sets and the placebo
# ----------------------------------------------------------------------------
def make_proposal(t, c, w, ok=True):
    return O.Proposal(t, c, w if ok else 0.0, w >= 0.5 if ok else None, w if ok else None,
                      [], [], "m", "", ok)


def test_link_sets_R0_R2_R3():
    props = [make_proposal(0, 0, 0.9), make_proposal(0, 1, 0.7), make_proposal(1, 1, 0.69),
             make_proposal(1, 2, 0.1), make_proposal(2, 2, 0.95, ok=False)]
    links, diag = O.link_sets(props)
    assert links["R0"] == []
    assert [(l.topic_id, l.community_id, l.w) for l in links["R2"]] == [(0, 0, 0.9), (0, 1, 0.7), (1, 1, 0.69), (1, 2, 0.1)]
    assert [(l.topic_id, l.community_id) for l in links["R3"]] == [(0, 0), (0, 1)]     # 0.7 is in, 0.69 is out
    # P0 is the placebo of R3, the primary set: the same count, degrees and weights
    assert len(links["P0"]) == 2 and diag["n"] == 2 and diag["base_set"] == "R3"
    assert sorted(l.w for l in links["P0"]) == [0.7, 0.9]
    assert O.degree_distribution(links["P0"]) == O.degree_distribution(links["R3"])


def test_placebo_keeps_degrees_and_avoids_real_pairs():
    gen = np.random.default_rng(1)
    real = set()
    while len(real) < 300:
        real.add((int(gen.integers(0, 40)), int(gen.integers(0, 60))))
    links = [O.Link(t, c, float(gen.random())) for t, c in sorted(real)]
    p0, diag = O.placebo(links)
    assert len(p0) == len(links)
    assert O.degree_distribution(p0) == O.degree_distribution(links)
    assert diag["n_coincident"] == 0 and diag["n_duplicate"] == 0
    assert len({(l.topic_id, l.community_id) for l in p0}) == len(p0)
    # the weight stays with the topic endpoint of the sorted position
    by_pos = sorted(links, key=lambda l: (l.topic_id, l.community_id))
    assert [l.w for l in p0] == [l.w for l in by_pos] and [l.topic_id for l in p0] == [l.topic_id for l in by_pos]
    # seeded: the same input gives the same shuffle, another seed another one
    again, _ = O.placebo(list(reversed(links)))
    assert [(l.topic_id, l.community_id) for l in again] == [(l.topic_id, l.community_id) for l in p0]
    other, _ = O.placebo(links, seed=14)
    assert [(l.topic_id, l.community_id) for l in other] != [(l.topic_id, l.community_id) for l in p0]
    assert O.degree_distribution(other) == O.degree_distribution(links)


def test_apply_links_reads_original_scores():
    st = {0: 0.5, 1: 0.2}
    sc = {10: 0.4, 11: 0.1}
    links = [O.Link(0, 10, 0.8), O.Link(0, 11, 0.5), O.Link(1, 10, 1.0), O.Link(7, 10, 1.0), O.Link(0, 99, 1.0)]
    st2, sc2 = O.apply_links(st, sc, links)
    # community 10 gets 0.8 * S(t0) + 1.0 * S(t1); topic 0 gets 0.8 * S(c10) + 0.5 * S(c11), from the original values
    assert sc2[10] == pytest.approx(0.4 + 0.8 * 0.5 + 1.0 * 0.2)
    assert sc2[11] == pytest.approx(0.1 + 0.5 * 0.5)
    assert st2[0] == pytest.approx(0.5 + 0.8 * 0.4 + 0.5 * 0.1)
    assert st2[1] == pytest.approx(0.2 + 1.0 * 0.4)
    # unknown endpoints add nothing and are not created; the inputs are untouched
    assert 7 not in st2 and 99 not in sc2 and st == {0: 0.5, 1: 0.2} and sc == {10: 0.4, 11: 0.1}
    assert O.apply_links(st, sc, []) == (st, sc)


def test_placebo_counts_unavoidable_coincidence():
    one, diag = O.placebo([O.Link(0, 0, 0.5)])
    assert (one[0].topic_id, one[0].community_id) == (0, 0) and diag["n_coincident"] == 1
    two, diag = O.placebo([O.Link(0, 0, 0.5), O.Link(0, 1, 0.6)])
    assert diag["n_coincident"] == 2 and diag["n_duplicate"] == 0
    empty, diag = O.placebo([])
    assert empty == [] and diag["n"] == 0


# ----------------------------------------------------------------------------
# Files and the orchestrator
# ----------------------------------------------------------------------------
def test_run_overlay_writes_and_loads_everything(tmp_path):
    replies = {(1, 1): reply(True, 0.9, ["s#0/20"]), (2, 2): reply(True, 0.75), (0, 1): reply(False, 0.2),
               (2, 0): "nothing", (0, 0): reply(True, 0.4)}
    meter = CostMeter(max_usd=1.0)
    client = FakeClient(replies, meter, cached=True)
    # share 1.0 so every pair is flagged in this small fixture
    links, diag = O.run_overlay("multihoprag", TOPICS, TERMS, COMMUNITIES, TEXTS, client=client,
                                meter=meter, counter=Words(), out_dir=tmp_path, workers=1, share=1.0)
    assert diag["corpus"] == "multihoprag" and diag["cap"] == 500 and diag["model"] == "fake-model"
    assert diag["flag_share"] == 1.0
    assert diag["n_topics"] == 3 and diag["n_communities"] == 4
    assert diag["n_candidate_pairs"] == 5 and diag["n_top_share"] == 5 and diag["n_flagged"] == 5
    assert diag["n_calls"] == 5 and diag["n_cached"] == 5 and diag["n_new"] == 0
    assert diag["n_parse_ok"] == 4 and diag["n_parse_failed"] == 1 and diag["n_inconsistent"] == 1
    assert diag["cost"]["total_calls"] == 5 and diag["cost"]["total_usd"] > 0
    hist = diag["confidence_histogram"]
    assert sum(hist.values()) == 4 and hist["0.9-1.0"] == 1 and hist["0.7-0.8"] == 1 and hist["0.2-0.3"] == 1
    assert diag["links"] == {"R0": 0, "R2": 4, "R3": 2, "P0": 2}
    assert diag["placebo"]["n"] == 2 and diag["placebo"]["base_set"] == "R3"
    for name in ("R0", "R2", "R3", "P0"):
        path = tmp_path / f"links_{name}.json"
        payload = json.loads(path.read_text())
        assert payload["set"] == name and payload["n"] == len(payload["links"]) == len(links[name])
        assert [(l.topic_id, l.community_id, l.w) for l in O.load_links(path)] == \
               [(l.topic_id, l.community_id, l.w) for l in links[name]]
    assert "shuffle" in json.loads((tmp_path / "links_P0.json").read_text())
    props = O.load_proposals(tmp_path / "proposals.jsonl")
    assert len(props) == 5 and sum(p.parse_ok for p in props) == 4
    assert props[0].raw == replies[(1, 1)] and props[0].cited == ["s#0/20"]
    assert json.loads((tmp_path / "diagnostics.json").read_text())["n_flagged"] == 5
    # the proposals file is one JSON object per line with the stored fields
    first = json.loads((tmp_path / "proposals.jsonl").read_text().splitlines()[0])
    assert {"topic_id", "community_id", "w", "cited", "model", "raw", "shown", "parse_ok"} <= set(first)


def test_chat_model_reads_models_json():
    model, location = O.chat_model(REPO / "docs/part1/env/models.json")
    assert model == "gemini-2.5-flash-lite" and location == "us-central1"
    assert O.MAX_USD == 15.0 and O.R3_THRESHOLD == 0.7 and O.MIN_SHARED == 5
    assert O.MAX_UNITS == 8 and O.MAX_UNIT_TOKENS == 300 and O.N_TERMS == 10 and O.SEED == 13


def test_style_no_banned_characters():
    banned = "\u2014\u2013\u2192\u2026\u2018\u2019\u201c\u201d"
    for path in (REPO / "src/multicard/part1/overlay.py", Path(__file__)):
        text = path.read_text()
        assert not any(ch in text for ch in banned), path

"""Tests for part1.retrieve and part1.planner on a tiny synthetic corpus with
hand-made vectors. No encoder, no tokenizer, no model call: token counting
uses a word counter and the chat client is a fake.

Covered: the fused score formula against a hand computation; the overlay
update reading the original S values; the planner table; every rules
pattern on a sample question and the rule order; the oracle map; the LLM
planner (prompt, memo, off-list reply); S5_noPGR never touching the relation
or entity inputs; ours_cheap equal to e5's own rankings; the speaker rule;
lazy expansion through the render budget; the S2_lazy walk with its caps;
the chunk-to-unit mapping on both corpora; the parquet adapters; and the
character lint on the new files.
"""

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multicard.data.multihoprag import Document
from multicard.experiments import e5_longmemeval as e5
from multicard.part1 import planner as P
from multicard.part1 import retrieve as R
from multicard.part1 import units as U


class Words:
    """One token per whitespace-separated word."""

    def count(self, text):
        return len(text.split())


def vec(*pairs):
    v = np.zeros(384, dtype=np.float64)
    for k, w in pairs:
        v[k] = w
    return v / np.linalg.norm(v)


E0, E1, E2, E3, E5 = vec((0, 1.0)), vec((1, 1.0)), vec((2, 1.0)), vec((3, 1.0)), vec((5, 1.0))

SESSIONS = {
    "s1": {"date": "2023-05-20", "turns": [
        {"role": "user", "content": "I bought a red bike yesterday. It cost two hundred dollars."},
        {"role": "assistant", "content": "Great choice for the city.\nEnjoy the ride every day."}]},
    "s2": {"date": "2023-05-18", "turns": [
        {"role": "user", "content": "My cat is called Tom and he is three years old."},
        {"role": "assistant", "content": "Tom is a fine name for a cat."}]},
    "s3": {"date": "2023-05-21", "turns": [
        {"role": "user", "content": "The bike shop in Leeds fixed my brakes last week."},
        {"role": "user", "content": "I paid forty dollars for the repair."}]},
}
SUB_VECS = {
    "s1#0/0": vec((0, 0.9), (1, math.sqrt(1 - 0.81))),   # cosine 0.9 to E0
    "s1#0/1": E0,                                        # 1.0
    "s1#1/0": E2, "s1#1/1": E2,                          # 0
    "s2#0/0": E3, "s2#1/0": E3,                          # 0
    "s3#0/0": vec((0, 0.6), (1, 0.8)),                   # 0.6
    "s3#1/0": vec((0, 0.8), (1, 0.6)),                   # 0.8
}
TURN_VECS = {"s1#0": E0, "s1#1": E2, "s2#0": E3, "s2#1": E3,
             "s3#0": vec((0, 0.6), (1, 0.8)), "s3#1": vec((0, 0.8), (1, 0.6))}
QUESTION = "How much did my bike cost?"


def lme_tables(order=("s1", "s2", "s3")):
    t = U.build_tables(SESSIONS, list(order))
    for table in t.values():
        table.turn_vecs = np.stack([TURN_VECS[x.unit_id] for x in table.turns]).astype(np.float32)
        table.sub_vecs = [np.stack([SUB_VECS[s.unit_id] for s in g]).astype(np.float32) for g in table.subs]
    return t


def topics():
    return R.Topics(ids=[0, 1, 2], vecs=np.stack([E0, E3, E2]),
                    names={0: "bike money", 1: "cat name", 2: "assistant chat"},
                    owner_topic={"s1#0": 0, "s3#0": 0, "s3#1": 0, "s2#0": 1, "s2#1": 1, "s1#1": 2})


def communities():
    return R.Communities(ids=[0, 1, 2, 3], vecs=np.stack([E3, E0, vec((0, 0.6), (1, 0.8)), E2]),
                         unit_community={"s2#0/0": 0, "s2#1/0": 0, "s1#0/0": 1, "s1#0/1": 1,
                                         "s3#0/0": 2, "s3#1/0": 2, "s1#1/0": 3, "s1#1/1": 3})


LINKS = {"R2": [R.Link(0, 1, 0.9), R.Link(2, 3, 0.5)], "P0": [R.Link(0, 3, 0.9), R.Link(2, 1, 0.5)]}


def pgr():
    ent = {"e1": {"entity_key": "e1", "name": "bike", "aliases": ["red bike"], "description": "", "entity_type": None},
           "e2": {"entity_key": "e2", "name": "shop", "aliases": ["bike shop"], "description": "", "entity_type": None},
           "e3": {"entity_key": "e3", "name": "Tom", "aliases": ["cat"], "description": "", "entity_type": "pet"}}
    chunks = {"c1": {"chunk_id": "c1", "content": "x", "doc_id": "s3", "doc_date": "2023-05-21",
                     "turn_start": 0, "turn_end": 1, "char_start": 0, "char_end": 1},
              "c2": {"chunk_id": "c2", "content": "x", "doc_id": "s1", "doc_date": "2023-05-20",
                     "turn_start": 0, "turn_end": 0, "char_start": 0, "char_end": 1},
              "c3": {"chunk_id": "c3", "content": "x", "doc_id": "s2", "doc_date": "2023-05-18",
                     "turn_start": None, "turn_end": None, "char_start": -1, "char_end": -1},
              "c4": {"chunk_id": "c4", "content": "x", "doc_id": "s9", "doc_date": "2023-05-18",
                     "turn_start": 0, "turn_end": 0, "char_start": 0, "char_end": 1}}
    rels = [{"edge_id": "r1", "src_key": "e1", "tgt_key": "e2", "relation_type": "bought_at",
             "negated": False, "sources": ["c1"]},
            {"edge_id": "r2", "src_key": "e3", "tgt_key": "e1", "relation_type": "likes",
             "negated": True, "sources": ["c3", "c4"]}]
    return R.PgrTables(ent, rels, chunks, {"e1": ["c2"], "e2": ["c1"], "e3": ["c3"]})


REL_VECS = np.stack([np.stack([E0, E1]), np.stack([E3, E3])])   # r1 canonical matches E0; r2 nothing


def lme_space(with_pgr=True, links=LINKS):
    return R.build_space(lme_tables(), "lme", topics=topics(), communities=communities(), links=links,
                         pgr=pgr() if with_pgr else None, rel_vecs=REL_VECS if with_pgr else None)


def question(text=QUESTION, qid="q1", qtype="single-session-user"):
    return R.Question(qid, text, E0, qtype=qtype)


class FakeGen:
    """Scripted replies; records prompts and the output limit asked for."""

    def __init__(self, replies=(), rule=None):
        self.replies = list(replies)
        self.rule = rule
        self.prompts = []
        self.limits = []

    def generate(self, prompt, max_output_tokens=800):
        self.prompts.append(prompt)
        self.limits.append(max_output_tokens)
        text = self.rule(prompt) if self.rule else self.replies.pop(0)
        return SimpleNamespace(text=text, tokens_in=10, tokens_out=2, cached=False)


class Poison:
    """Any use is an error: the input must not be touched."""

    def __getattr__(self, name):
        raise AssertionError(f"touched a post-graph-rag input ({name})")

    def __len__(self):
        raise AssertionError("touched a post-graph-rag input (len)")

    def __iter__(self):
        raise AssertionError("touched a post-graph-rag input (iter)")


# ----------------------------------------------------------------------------
# planner
# ----------------------------------------------------------------------------
def test_planner_table_lookup_matches_section_5():
    w, d = P.weights_for("local")
    assert w == {"dense": 2, "bm25": 2, "topic": 1, "community": 1, "relation": 1, "entity": 1} and d == 1
    assert P.weights_for("entity") == ({"dense": 1, "bm25": 1, "topic": 1, "community": 1, "relation": 2, "entity": 3}, 2)
    assert P.weights_for("thematic") == ({"dense": 1, "bm25": 1, "topic": 3, "community": 2, "relation": 1, "entity": 1}, 2)
    assert P.weights_for("cross-topic") == ({"dense": 1, "bm25": 1, "topic": 2, "community": 3, "relation": 1, "entity": 1}, 3)
    assert P.weights_for("multi-hop") == ({"dense": 1, "bm25": 1, "topic": 1, "community": 1, "relation": 3, "entity": 2}, 3)
    assert P.weights_for("temporal") == ({"dense": 2, "bm25": 1, "topic": 1, "community": 1, "relation": 2, "entity": 1}, 2)
    assert P.weights_for("lexical") == ({"dense": 1, "bm25": 3, "topic": 0, "community": 0, "relation": 0, "entity": 0}, 1)
    assert P.weights_for("unanswerable") == ({"dense": 1, "bm25": 1, "topic": 0, "community": 0, "relation": 0, "entity": 0}, 0)
    assert set(P.PLANNER_TABLE) == set(P.SHAPES)
    with pytest.raises(KeyError):
        P.weights_for("other")
    assert P.STATIC_WEIGHTS == {c: 1 for c in P.CHANNELS} and P.STATIC_DEPTH == 3


@pytest.mark.parametrize("q,shape", [
    # lexical: a quoted string, a token with letters and digits, four or more digits
    ('What did I mean by "the big plan"?', "lexical"),
    ("What is the code ab12 for?", "lexical"),
    ("What did I do in 2023?", "lexical"),
    # temporal
    ("How long did the course take?", "temporal"),
    ("How many days did the trip last?", "temporal"),
    ("How many weeks was I away?", "temporal"),
    ("How many months did it take?", "temporal"),
    ("How many years have I lived here?", "temporal"),
    ("How long ago did I move?", "temporal"),
    ("What have I done since the move?", "temporal"),
    ("What did I eat before the run?", "temporal"),
    ("What happened after the party?", "temporal"),
    ("When did I adopt the dog?", "temporal"),
    ("What date is the wedding?", "temporal"),
    ("What was the first time I ran?", "temporal"),
    ("What was the last time I called?", "temporal"),
    ("What is the most recent book I read?", "temporal"),
    ("What am I currently reading?", "temporal"),
    ("What do I drive now?", "temporal"),
    # cross-topic
    ("How much did I spend in total?", "cross-topic"),
    ("How many trips did I take altogether?", "cross-topic"),
    ("What is the combined cost?", "cross-topic"),
    ("What are all the places I visited?", "cross-topic"),
    ("How much did I spend across the trips?", "cross-topic"),
    ("Did I like both films?", "cross-topic"),
    ("Can we compare the two flats?", "cross-topic"),
    ("Did I buy the same brand twice?", "cross-topic"),
    ("Were the two meals different?", "cross-topic"),
    ("Did I choose either option?", "cross-topic"),
    ("Did I like neither film?", "cross-topic"),
    # multi-hop
    ("Which company makes the phone I bought?", "multi-hop"),
    ("Which organization did I join?", "multi-hop"),
    ("Which organisation did I join?", "multi-hop"),
    ("Which person fixed the bike?", "multi-hop"),
    ("Which team won the match?", "multi-hop"),
    ("Which country did I visit?", "multi-hop"),
    ("Who fixed the bike?", "multi-hop"),
    ("What was reported by the paper?", "multi-hop"),
    ("What is the figure according to the source?", "multi-hop"),
    # thematic
    ("Can you recommend a book?", "thematic"),
    ("Suggest a recipe for tonight.", "thematic"),
    ("Any ideas for dinner?", "thematic"),
    ("What should I cook tonight?", "thematic"),
    ("Give me advice on running.", "thematic"),
    ("What is the plan for Friday?", "thematic"),
    ("Any tips for sleeping better?", "thematic"),
    ("Help me choose a laptop.", "thematic"),
    # entity: the possessive list and a capitalised token that is not sentence-initial
    ("How old is my dog?", "entity"),
    ("How old is my cat?", "entity"),
    ("What does my wife do?", "entity"),
    ("What does my husband do?", "entity"),
    ("What does my partner do?", "entity"),
    ("What does my son study?", "entity"),
    ("What does my daughter study?", "entity"),
    ("What did my boss say?", "entity"),
    ("What did my friend say?", "entity"),
    ("What did my sister say?", "entity"),
    ("What did my brother say?", "entity"),
    ("What did my mother say?", "entity"),
    ("What did my father say?", "entity"),
    ("What colour is my car?", "entity"),
    ("How big is my house?", "entity"),
    ("What did I say about Leeds?", "entity"),
    ("Where is Leeds? Tell me.", "entity"),
    # the pronoun I is a capitalised token like any other (section 5 text)
    ("What did I say?", "entity"),
    # local: nothing matches; sentence-initial capitals do not count
    ("Leeds is where? Tell me.", "local"),
    ("I'm not sure. Tell me.", "local"),
    ("How much did my bike cost?", "local"),
    # order: earlier rules win
    ("When did my dog get sick?", "temporal"),
    ("Who is my sister?", "multi-hop"),
    ("What advice did my mother give?", "thematic"),
    ('Which company made the "X1" phone?', "lexical"),
    ("How long since I saw both of them?", "temporal"),
])
def test_rules_patterns_on_sample_questions(q, shape):
    assert P.rules_shape(q) == shape


def test_rules_never_assign_unanswerable_and_match_substrings():
    assert "unanswerable" not in [s for s, _ in P.RULES]
    # section 5 says "contains": a pattern fires inside a longer word too
    assert P.rules_shape("Is Chicago cold?") == "temporal"      # "ago" inside Chicago
    assert P.rules_shape("Do I know it?") == "temporal"         # "now" inside know
    assert P.rules_shape("What is the update?") == "temporal"   # "date" inside update
    assert P.rules_shape("What recommendations did you give me for my trip?") == "thematic"
    assert P.rules_shape("How did my salary change compared to last year?") == "cross-topic"
    assert P.rules_shape("Whose birthday did I say was coming up?") == "multi-hop"


def test_oracle_map():
    assert P.oracle_shape("single-session-user") == "local"
    assert P.oracle_shape("single-session-assistant") == "local"
    assert P.oracle_shape("single-session-preference") == "thematic"
    assert P.oracle_shape("multi-session") == "cross-topic"
    assert P.oracle_shape("temporal-reasoning") == "temporal"
    assert P.oracle_shape("knowledge-update") == "temporal"
    assert P.oracle_shape("single-session-user", abstention=True) == "unanswerable"
    assert P.oracle_shape("inference_query") == "multi-hop"
    assert P.oracle_shape("comparison_query") == "cross-topic"
    assert P.oracle_shape("temporal_query") == "temporal"
    assert P.oracle_shape("null_query") == "unanswerable"
    with pytest.raises(KeyError):
        P.oracle_shape("other")


def test_llm_planner_prompt_memo_and_off_list_reply():
    text = P.load_prompt()
    assert re.sub(r"\s+", " ", text).strip() == re.sub(r"\s+", " ", P.PROMPT_TEXT).strip()
    assert P.planner_prompt("Q?", "X\n") == "Q?\n\nX"
    # section 5: a reply that is not one of the eight labels is off-list (whitespace and case aside)
    assert P.parse_reply("Temporal") == ("temporal", False)
    assert P.parse_reply("  cross-topic\n") == ("cross-topic", False)
    assert P.parse_reply("Temporal.") == ("local", True)
    assert P.parse_reply("  cross-topic\nbecause") == ("local", True)
    assert P.parse_reply("I think local") == ("local", True)
    assert P.parse_reply("") == ("local", True)
    gen = FakeGen(["temporal", "banana"])
    pl = P.LLMPlanner(gen, prompt_text="PROMPT")
    d = pl.decide("q1", "When did it happen?")
    assert d.shape == "temporal" and d.calls == 1 and d.off_list is False and d.planner == "llm"
    assert gen.prompts == ["When did it happen?\n\nPROMPT"] and gen.limits == [64]
    again = pl.decide("q1", "When did it happen?")
    assert again.shape == "temporal" and again.calls == 0 and again.cached is True
    assert len(gen.prompts) == 1
    d2 = pl.decide("q2", "Anything?")
    assert d2.shape == "local" and d2.off_list is True and d2.reply == "banana"
    assert pl.off_list_log == [{"qid": "q2", "reply": "banana", "treated_as": "local"}]
    static = P.plan("static", "q", "x")
    assert P.weights_and_depth(static) == (P.STATIC_WEIGHTS, 3)
    assert P.plan("rules", "q", "Who fixed it?").shape == "multi-hop"
    assert P.plan("oracle", "q", "x", "multi-session").shape == "cross-topic"
    with pytest.raises(ValueError):
        P.plan("llm", "q", "x")


# ----------------------------------------------------------------------------
# fusion and overlay
# ----------------------------------------------------------------------------
def test_fused_score_formula_against_a_hand_computation():
    hits = {"dense": R.hits_from_ids(["a", "b"]), "bm25": R.hits_from_ids(["b", "c"])}
    out = R.fuse(hits, {"dense": 2, "bm25": 1})
    scores = {c.unit_id: c.score for c in out}
    assert scores["a"] == pytest.approx(2 / 61)
    assert scores["b"] == pytest.approx(2 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [c.unit_id for c in out] == ["b", "a", "c"]
    assert set(out[0].channels) == {"dense", "bm25"} and out[0].channels["dense"].rank == 2
    # a zero weight switches the channel off; ties break by unit id
    out = R.fuse(hits, {"dense": 1, "bm25": 0})
    assert [c.unit_id for c in out] == ["a", "b"] and "bm25" not in out[1].channels
    out = R.fuse({"dense": R.hits_from_ids(["z", "y"]), "bm25": R.hits_from_ids(["y", "z"])}, {"dense": 1, "bm25": 1})
    assert [c.unit_id for c in out] == ["y", "z"]
    assert len(R.fuse({"dense": R.hits_from_ids([str(i) for i in range(300)])}, {"dense": 1})) == 100


def test_overlay_update_reads_the_original_values():
    s_t, s_c = R.apply_overlay({0: 1.0, 1: 0.0}, {1: 1.0}, [R.Link(0, 1, 0.9), R.Link(1, 1, 0.3)])
    assert s_c == {1: pytest.approx(1.9)}
    assert s_t[0] == pytest.approx(1.9)
    assert s_t[1] == pytest.approx(0.3)      # 0.3 times the original S(c1) = 1.0, not the updated 1.9
    # a link to an unknown topic or community is skipped; no links means no change
    assert R.apply_overlay({0: 0.5}, {1: 0.5}, [R.Link(7, 1, 1.0)]) == ({0: 0.5}, {1: 0.5})
    assert R.apply_overlay({0: 0.5}, {1: 0.5}, []) == ({0: 0.5}, {1: 0.5})


def test_select_links_by_overlay():
    assert R.select_links(LINKS, "R0") == []
    assert R.select_links(LINKS, "R2") == LINKS["R2"]
    assert R.select_links(LINKS, "R3") == [R.Link(0, 1, 0.9)]
    assert R.select_links({"R2": [R.Link(0, 1, 0.7)], "P0": []}, "R3") == [R.Link(0, 1, 0.7)]
    assert R.select_links(LINKS, "P0") == LINKS["P0"]
    with pytest.raises(ValueError):
        R.select_links({"R2": []}, "P0")
    with pytest.raises(ValueError):
        R.select_links({}, "R3")


# ----------------------------------------------------------------------------
# channels on the synthetic space
# ----------------------------------------------------------------------------
def test_space_and_channels():
    sp = lme_space()
    assert sp.sub_ids == ["s1#0/0", "s1#0/1", "s1#1/0", "s1#1/1", "s2#0/0", "s2#1/0", "s3#0/0", "s3#1/0"]
    assert sp.owner_ids == ["s1#0", "s1#1", "s2#0", "s2#1", "s3#0", "s3#1"]
    assert sp.container_owners["s3"] == ["s3#0", "s3#1"]
    assert {t: list(v) for t, v in sp.topic_members.items()} == {0: [0, 1, 6, 7], 2: [2, 3], 1: [4, 5]}
    assert list(sp.sub_community) == [1, 1, 3, 3, 0, 0, 2, 2]
    assert sp.rel_units == [["s3#0", "s3#1"], []] and sp.n_unmapped_chunks == 2
    assert ("e1", "bike") in sp.entity_phrases and ("e1", "red bike") in sp.entity_phrases
    run = R.QueryRun(sp, question())
    assert [h.unit_id for h in run.dense()] == ["s1#0", "s3#1", "s3#0", "s1#1", "s2#0", "s2#1"]
    assert run.dense()[0].via == "s1#0/1" and run.dense()[0].score == pytest.approx(1.0)
    assert run.dense()[1].score == pytest.approx(0.8)
    # "bike" and "cost" hit s1#0, "bike" and "my" hit s3#0, "my" hits s2#0 ("my" is not a bm25s stopword)
    assert [h.unit_id for h in run.bm25()] == ["s1#0", "s3#0", "s2#0"] and all(h.score > 0 for h in run.bm25())
    assert [h.unit_id for h in run.topic("R0")] == ["s1#0", "s3#1", "s3#0", "s1#1", "s2#0", "s2#1"]
    assert run.topic("R0")[0].group == "0" and run.topic("R0")[3].group == "2"
    # community: the best three of four groups by prototype cosine (c1, c2, c0); c3 is out
    assert [h.unit_id for h in run.community("R0")] == ["s1#0", "s3#1", "s3#0", "s2#0", "s2#1"]
    assert run.community("R0")[0].group == "1"
    # relation: r1 matches through its canonical rendering and maps to chunk c1's turns
    rel = run.relation()
    assert [(h.unit_id, h.via, h.rank) for h in rel] == [("s3#0", "r1", 1), ("s3#1", "r1", 2)]
    assert rel[0].score == pytest.approx(1.0)
    # entity: "bike" matches e1 as a whole word; its mention chunk c2 maps to s1#0
    ent = run.entity()
    assert [(h.unit_id, h.via, h.group) for h in ent] == [("s1#0", "e1", "bike")]
    # a longer alias wins the match and the ordering; a substring inside a word does not match
    run2 = R.QueryRun(sp, question("Where is the bike shop and what about my cat?"))
    # one hit per provenance unit: e2's chunk covers two turns, e3's chunk is unmapped
    assert [(h.via, h.group) for h in run2.entity()] == [("e2", "bike shop"), ("e2", "bike shop"), ("e1", "bike")]
    assert [h.unit_id for h in run2.entity()] == ["s3#0", "s3#1", "s1#0"]
    assert R.QueryRun(sp, question("Do bikes cost much?")).entity() == []


def test_overlay_changes_which_groups_the_channel_pools():
    sp = lme_space()
    run = R.QueryRun(sp, question())
    s_t, s_c = run.group_scores("R0")
    assert s_t == {0: pytest.approx(1.0), 1: pytest.approx(0.0), 2: pytest.approx(0.0)}
    assert s_c[1] == pytest.approx(1.0) and s_c[2] == pytest.approx(0.6)
    s_t2, s_c2 = run.group_scores("P0")
    assert s_c2[3] == pytest.approx(0.9) and s_c2[1] == pytest.approx(1.0)
    assert run.links_used == {"R0": 0, "P0": 2}
    # under P0 the pooled communities are c1, c3, c2: s2's units drop out, s1#1 comes in
    assert [h.unit_id for h in run.community("P0")] == ["s1#0", "s3#1", "s3#0", "s1#1"]
    assert [h.unit_id for h in run.community("R0")] == ["s1#0", "s3#1", "s3#0", "s2#0", "s2#1"]
    run.group_scores("R3")
    assert run.links_used["R3"] == 1 and len(R.select_links(sp.links, "R3")) == 1


def test_group_scores_are_cosines_whatever_the_prototype_norm():
    # section 5: S(t) and S(c) are cosines; a prototype that is a plain mean is not unit length
    unit = R.QueryRun(lme_space(), question()).group_scores("R0")
    comm = communities()
    comm.vecs = comm.vecs * np.array([[0.5], [3.0], [0.25], [2.0]])
    top = topics()
    top.vecs = top.vecs * 0.7
    scaled = R.build_space(lme_tables(), "lme", topics=top, communities=comm, links=LINKS, pgr=None)
    s_t, s_c = R.QueryRun(scaled, question()).group_scores("R0")
    assert s_c == {c: pytest.approx(v) for c, v in unit[1].items()}
    assert s_t == {t: pytest.approx(v) for t, v in unit[0].items()}
    assert list(R.cosine_rows(np.zeros((1, 384)), E0)) == [0.0]
    assert R.cosine_rows(np.stack([E0 * 4.0]), E0)[0] == pytest.approx(1.0)


def test_s4_static_fusion_and_speaker_rule_by_hand():
    sp = lme_space()
    run = R.QueryRun(sp, question())
    out = R.run_arm("S4_static", run, 1000, counter=Words())
    scores = {c.unit_id: c.score for c in out.candidates}
    # every weight 1: dense, bm25, topic, community, relation, entity ranks as in test_space_and_channels
    assert scores["s1#0"] == pytest.approx(1 / 61 + 1 / 61 + 1 / 61 + 1 / 61 + 1 / 61)
    assert scores["s3#0"] == pytest.approx(1 / 63 + 1 / 62 + 1 / 63 + 1 / 63 + 1 / 61)
    assert scores["s3#1"] == pytest.approx(1 / 62 + 1 / 62 + 1 / 62 + 1 / 62)
    assert scores["s2#0"] == pytest.approx(1 / 65 + 1 / 63 + 1 / 65 + 1 / 64)
    assert scores["s2#1"] == pytest.approx(1 / 66 + 1 / 66 + 1 / 65)
    assert scores["s1#1"] == pytest.approx(1 / 64 + 1 / 64)
    assert [c.unit_id for c in out.candidates] == ["s1#0", "s3#0", "s3#1", "s2#0", "s2#1", "s1#1"]
    # the speaker rule drops the assistant turns (the question does not refer to the assistant)
    assert out.candidate_ids == ["s1#0", "s3#0", "s3#1", "s2#0"] and out.ranking == out.candidate_ids
    assert {c.unit_id: c.quarantined for c in out.candidates}["s1#1"] is True
    assert out.speaker_rule is True and out.expansion == [] and out.depth == 3
    assert out.weights == P.STATIC_WEIGHTS and out.shape == "" and out.model_calls == 0
    assert out.rendered.candidates == out.candidate_ids and out.rendered.unit_ids
    assert sorted(out.rendered.unit_ids) == sorted(out.ranking)
    d = out.to_dict()
    assert d["candidates"][0]["channels"]["entity"]["via"] == "e1" and d["arm"] == "S4_static"
    # the same question naming the assistant keeps the assistant turns
    out2 = R.run_arm("S4_static", R.QueryRun(sp, question("What did you say my bike cost?")), 1000, counter=Words())
    assert "s1#1" in out2.candidate_ids


def test_s5_arms_planner_overlay_and_no_pgr():
    sp = lme_space()
    run = R.QueryRun(sp, question(qtype="multi-session"))
    gen = FakeGen(["thematic"])
    llm = P.LLMPlanner(gen, prompt_text="P")
    primary = R.run_arm("S5_primary", run, 1000, llm=llm, counter=Words())
    assert primary.shape == "thematic" and primary.planner == "llm" and primary.model_calls == 1
    assert primary.weights == {"dense": 1, "bm25": 1, "topic": 3, "community": 2, "relation": 1, "entity": 1}
    assert primary.depth == 2 and primary.overlay == "R3" and primary.n_links == 1
    scores = {c.unit_id: c.score for c in primary.candidates}
    assert scores["s1#0"] == pytest.approx(1 / 61 + 1 / 61 + 3 / 61 + 2 / 61 + 1 / 61)
    # the ablations share the one planner call
    for arm in ("S5_overlay_R0", "S5_overlay_R2", "S5_overlay_P0", "S5_primary_norule", "S5_noPGR"):
        o = R.run_arm(arm, run, 1000, llm=llm, counter=Words())
        assert o.shape == "thematic" and o.model_calls == 0
    assert len(gen.prompts) == 1
    r0 = R.run_arm("S5_overlay_R0", run, 1000, llm=llm, counter=Words())
    p0 = R.run_arm("S5_overlay_P0", run, 1000, llm=llm, counter=Words())
    assert r0.n_links == 0 and p0.n_links == 2 and p0.overlay == "P0"
    assert "s2#0" in {c.unit_id for c in r0.candidates if "community" in c.channels}
    assert "s2#0" not in {c.unit_id for c in p0.candidates if "community" in c.channels}
    norule = R.run_arm("S5_primary_norule", run, 1000, llm=llm, counter=Words())
    assert norule.speaker_rule is False and "s1#1" in norule.candidate_ids
    # the rules and oracle planners
    rules = R.run_arm("S5_planner_rules", run, 1000, counter=Words())
    assert rules.shape == "local" and rules.planner == "rules" and rules.depth == 1
    oracle = R.run_arm("S5_planner_oracle", run, 1000, counter=Words())
    assert oracle.shape == "cross-topic" and oracle.weights["community"] == 3 and oracle.depth == 3
    # S5_noPGR: relation and entity weights 0 and the inputs never read
    nopgr = R.run_arm("S5_noPGR", run, 1000, llm=llm, counter=Words())
    assert nopgr.weights["relation"] == 0 and nopgr.weights["entity"] == 0
    assert not any("relation" in c.channels or "entity" in c.channels for c in nopgr.candidates)
    poisoned = lme_space(with_pgr=False)
    poisoned.pgr, poisoned.rel_vecs, poisoned.rel_units, poisoned.entity_phrases = Poison(), Poison(), Poison(), Poison()
    prun = R.QueryRun(poisoned, question())
    ok = R.run_arm("S5_noPGR", prun, 1000, llm=P.LLMPlanner(FakeGen(["entity"]), prompt_text="P"), counter=Words())
    assert ok.shape == "entity" and ok.candidates
    with pytest.raises(AssertionError):
        R.run_arm("S5_primary", prun, 1000, llm=P.LLMPlanner(FakeGen(["entity"]), prompt_text="P"), counter=Words())
    with pytest.raises(AssertionError):
        R.run_arm("S4_static", prun, 1000, counter=Words())
    with pytest.raises(ValueError):
        R.run_arm("S5_primary", run, 1000, counter=Words())     # no LLMPlanner given


def test_lazy_expansion_through_the_render_budget():
    sp = lme_space()
    run = R.QueryRun(sp, question())
    # depth 1: the first container is s1; its remaining unit s1#1 is appended when it fits
    exp, ctx = R.expand(run, ["s1#0"], 1, 1000, Words(), rule=False)
    assert exp == ["s1#1"] and ctx.candidates == ["s1#0"] and sorted(ctx.unit_ids) == ["s1#0", "s1#1"]
    exp, ctx = R.expand(run, ["s1#0"], 1, 13, Words(), rule=False)     # s1#0 alone fills 13 words
    assert exp == [] and ctx.unit_ids == ["s1#0"]
    exp, ctx = R.expand(run, ["s1#0"], 1, 20, Words(), rule=False)     # s1#1 fits truncated
    assert exp == ["s1#1"] and ctx.truncated["s1#1"] is True
    # the speaker rule also filters the appended units
    exp, _ = R.expand(run, ["s1#0"], 1, 1000, Words(), rule=True)
    assert exp == []
    # depth 2 with two containers: the remaining units are ordered by the owner vector cosine
    exp, _ = R.expand(run, ["s1#0", "s3#1"], 2, 1000, Words(), rule=False)
    assert exp == ["s3#0", "s1#1"]
    exp, ctx = R.expand(run, ["s1#0", "s3#1"], 0, 1000, Words(), rule=False)
    assert exp == [] and ctx.candidates == ["s1#0", "s3#1"]
    # inside an arm: a shallow channel depth leaves units for the expansion to add
    shallow = R.QueryRun(sp, question())
    shallow.depth = 2
    out = R.run_arm("S5_primary_norule", shallow, 1000, llm=P.LLMPlanner(FakeGen(["local"]), prompt_text="P"), counter=Words())
    assert out.candidate_ids == ["s1#0", "s3#1", "s3#0"] and out.expansion == ["s1#1"]
    assert out.ranking == ["s1#0", "s3#1", "s3#0", "s1#1"] and out.rendered.candidates == out.candidate_ids


def test_ours_cheap_equals_e5():
    sp = lme_space(with_pgr=False, links={})
    run = R.QueryRun(sp, question())
    units = {sid: R.session_units(t) for sid, t in sp.tables.items()}
    sids = list(sp.tables)
    per_turn = [(s, e5.user_units(units[s], "turn")) for s in sids] + [(s, e5.assistant_units(units[s], "turn")) for s in sids]
    per_sent = [(s, e5.user_units(units[s], "sentence")) for s in sids] + [(s, e5.assistant_units(units[s], "items")) for s in sids]
    turn = e5.dense_turn_ranking(E0, per_turn)
    sent = e5.dense_turn_ranking(E0, per_sent)
    bm = e5.bm25_turn_ranking(QUESTION, per_turn)
    rrf_turn = e5.rrf([bm[:100], turn[:100]], k=60)
    rrf_sentence = e5.rrf([bm[:100], sent[:100]], k=60)
    route = e5.quarantine(rrf_turn, QUESTION, units)
    cheap = run.cheap()
    assert cheap["turn"] == turn and cheap["sentence"] == sent and cheap["bm25"] == bm
    assert cheap["rrf_turn"] == rrf_turn and cheap["rrf_sentence"] == rrf_sentence and cheap["rrf_turn_route"] == route
    assert route != rrf_turn and "s1#1" not in route
    a = R.run_arm("ours_cheap", run, 1000, counter=Words())
    b = R.run_arm("ours_cheap_norule", run, 1000, counter=Words())
    c = R.run_arm("ours_sentence_norule", run, 1000, counter=Words())
    assert a.ranking == route and a.speaker_rule is True
    assert b.ranking == rrf_turn and b.speaker_rule is False
    assert c.ranking == rrf_sentence and c.speaker_rule is False
    # the candidate log orders as e5's fusion does, with scores and provenance
    assert [x.unit_id for x in b.candidates] == rrf_turn[:100]
    assert [x.unit_id for x in c.candidates] == rrf_sentence[:100]
    assert b.candidates[0].score == pytest.approx(2 / 61) and set(b.candidates[0].channels) == {"dense", "bm25"}
    assert a.candidate_ids == [u for u in route if u in set(rrf_turn[:100])]
    assert a.rendered.candidates == a.candidate_ids and a.model_calls == 0


def test_s2_lazy_walks_communities_with_the_caps():
    sp = lme_space()
    yes = FakeGen(rule=lambda p: "Yes." if ("bike" in p.split("Question:")[0] or "dollars" in p.split("Question:")[0]) else "No")
    run = R.QueryRun(sp, question())
    memo = {}
    out = R.run_arm("S2_lazy", run, 1000, lazy_client=yes, lazy_memo=memo, counter=Words())
    # communities by dense hits: c1 (best hit rank 0), c2, c3 (first zero hit), c0
    assert [c["community"] for c in out.lazy["communities"]] == [1, 2, 3, 0]
    assert [c["accepted"] for c in out.lazy["communities"]] == [2, 2, 0, 0]
    assert out.lazy["tests"] == 8 and out.model_calls == 8
    assert out.lazy["accepted"] == ["s1#0/1", "s1#0/0", "s3#1/0", "s3#0/0"]
    assert out.lazy["accepted_owners"] == ["s1#0", "s3#1", "s3#0"]
    assert out.ranking[:3] == ["s1#0", "s3#1", "s3#0"] and out.ranking == ["s1#0", "s3#1", "s3#0", "s2#0"]
    assert out.speaker_rule is True and len(memo) == 8
    p = yes.prompts[0]
    assert p.startswith("[2023-05-20, user] It cost two hundred dollars.")
    assert p.endswith(f"Question: {QUESTION}\n\n{R.LAZY_QUESTION}") and yes.limits[0] == 64
    # a rerun answers from the memo without a call
    again = R.run_arm("S2_lazy", R.QueryRun(sp, question()), 1000, lazy_client=FakeGen(), lazy_memo=memo, counter=Words())
    assert again.model_calls == 0 and again.ranking == out.ranking
    # always no: three communities in a row with zero accepted stop the walk
    no = FakeGen(rule=lambda p: "no")
    out = R.run_arm("S2_lazy", R.QueryRun(sp, question()), 1000, lazy_client=no, lazy_memo={}, counter=Words())
    assert [c["community"] for c in out.lazy["communities"]] == [1, 2, 3] and out.lazy["tests"] == 6
    assert out.lazy["accepted"] == [] and out.ranking == ["s1#0", "s3#0", "s3#1", "s2#0"]
    # the test cap
    acc, log, tests, calls = R.lazy_graph_rag(R.QueryRun(sp, question()), yes, {}, tests_cap=3)
    assert tests == 3 and calls.calls == 3 and acc == ["s1#0/1", "s1#0/0", "s3#1/0"]
    assert R.is_yes("YES, it does") and not R.is_yes("No. Yes.") and not R.is_yes("")
    with pytest.raises(ValueError):
        R.run_arm("S2_lazy", run, 1000, counter=Words())


def test_run_question_runs_every_arm_with_one_planner_call():
    sp = lme_space()
    gen = FakeGen(["temporal"])
    outs = R.run_question(question(), sp, 1000, llm=P.LLMPlanner(gen, prompt_text="P"),
                          lazy_client=FakeGen(rule=lambda p: "no"), counter=Words())
    assert list(outs) == list(R.ARMS) == R.arms_for("lme")
    assert len(gen.prompts) == 1
    assert sum(o.model_calls for o in outs.values()) == 1 + 6
    assert all(o.qid == "q1" for o in outs.values())
    assert R.arms_for("mhrag") == [a for a in R.ARMS if a not in ("ours_cheap_norule", "ours_sentence_norule", "S5_primary_norule")]
    with pytest.raises(ValueError):
        R.run_arm("ours_cheap_norule", R.QueryRun(mhrag_space(), R.Question("q", "x", E0)), 100, counter=Words())


# ----------------------------------------------------------------------------
# MultiHop-RAG
# ----------------------------------------------------------------------------
DOC_A = ("Alpha won the cup on Sunday. The final was played in Lyon.\n\n"
         "The coach said the team trained hard all year. Fans filled the square after the match.\n\n"
         "Ticket prices rose ten percent this season, according to the club.")
DOC_B = "Beta shipped a phone on Monday. Reviews were mixed.\n\nSales start Friday in Europe."


def mhrag_space(pgr_tables=None, rel_vecs=None):
    docs = [Document(doc_id="mhr_a", title="Alpha wins the cup", source="Sporting News", category="sports",
                     published_at="2023-10-01T10:00:00", body=DOC_A),
            Document(doc_id="mhr_b", title="Beta ships a phone", source="Tech Daily", category="technology",
                     published_at="2023-10-02T10:00:00", body=DOC_B)]
    t = U.build_doc_tables(docs, Words(), max_tokens=20)
    for table in t.values():
        vecs = [E0 if (table.doc_id == "mhr_a" and c.index == 0) else E5 for c in table.chunks]
        table.chunk_vecs = np.stack(vecs).astype(np.float32)
        table.sentence_vecs = [np.stack([v] * len(g)).astype(np.float32) for v, g in zip(vecs, table.sentences)]
    return R.build_space(t, "mhrag", pgr=pgr_tables, rel_vecs=rel_vecs)


def test_multihoprag_space_cheap_arm_and_no_speaker_rule():
    sp = mhrag_space()
    assert sp.e5_units is None and len(sp.tables["mhr_a"].chunks) >= 2
    q = R.Question("mq", "Who won the cup?", E0, qtype="inference_query")
    run = R.QueryRun(sp, q)
    a = R.run_arm("ours_cheap", run, 1000, counter=Words())
    assert a.ranking[0] == "mhr_a#0" and a.speaker_rule is False
    assert a.candidates[0].channels["dense"].via == "mhr_a#0"
    s4 = R.run_arm("S4_static", run, 1000, counter=Words())
    assert s4.candidate_ids[0] == "mhr_a#0" and s4.speaker_rule is False
    assert set(s4.candidates[0].channels) <= {"dense", "bm25"}
    oracle = R.run_arm("S5_planner_oracle", run, 1000, counter=Words())
    assert oracle.shape == "multi-hop" and oracle.depth == 3
    with pytest.raises(ValueError):
        run.cheap()


def test_chunk_units_on_both_corpora():
    lme = lme_tables()
    assert R.chunk_units({"doc_id": "s1", "turn_start": 0, "turn_end": 1}, lme, "lme") == ["s1#0", "s1#1"]
    assert R.chunk_units({"doc_id": "s1", "turn_start": 1, "turn_end": 9}, lme, "lme") == ["s1#1"]
    assert R.chunk_units({"doc_id": "s1", "turn_start": None, "turn_end": None}, lme, "lme") == []
    assert R.chunk_units({"doc_id": "s9", "turn_start": 0, "turn_end": 0}, lme, "lme") == []
    sp = mhrag_space()
    table = sp.tables["mhr_a"]
    h = R.article_header_len("2023-10-01")
    assert h == len("[Article published on 2023-10-01]\n")
    whole = {"doc_id": "mhr_a", "doc_date": "2023-10-01", "char_start": 0, "char_end": h + len(table.text)}
    assert R.chunk_units(whole, sp.tables, "mhrag") == [c.unit_id for c in table.chunks]
    first = table.chunks[0]
    part = {"doc_id": "mhr_a", "doc_date": "2023-10-01", "char_start": h + first.char_start, "char_end": h + first.char_end}
    assert R.chunk_units(part, sp.tables, "mhrag") == ["mhr_a#0"]
    assert R.chunk_units({"doc_id": "mhr_a", "doc_date": "2023-10-01", "char_start": -1, "char_end": -1}, sp.tables, "mhrag") == []
    assert R.chunk_units({"doc_id": "mhr_z", "char_start": 0, "char_end": 5}, sp.tables, "mhrag") == []


def test_relation_texts_and_encoding():
    sp = lme_space()
    canon, prefixed = R.relation_texts(sp.pgr, sp.topics, sp.rel_units)
    assert canon == ["bike bought at shop", "Tom not likes bike"]
    assert prefixed == ["bike money: bike bought at shop", "Tom not likes bike"]

    class Enc:
        def encode(self, texts):
            return np.stack([vec((len(t) % 384, 1.0)) for t in texts]).astype(np.float32)

    v = R.encode_relations(Enc(), canon, prefixed)
    assert v.shape == (2, 2, 384)
    assert np.allclose(v[0, 0], vec((len(canon[0]) % 384, 1.0))) and np.allclose(v[0, 1], vec((len(prefixed[0]) % 384, 1.0)))
    assert R.encode_relations(Enc(), [], []).shape == (0, 2, 384)
    built = R.build_space(lme_tables(), "lme", topics=topics(), pgr=pgr(), enc=Enc())
    assert built.rel_vecs.shape == (2, 2, 384)
    with pytest.raises(ValueError):
        R.build_space(lme_tables(), "lme", pgr=pgr())


# ----------------------------------------------------------------------------
# adapters
# ----------------------------------------------------------------------------
def test_parquet_adapters(tmp_path):
    # part1.topics.save layout: topics.parquet, unit_labels.parquet (the hard label) and
    # unit_topics.parquet (the sparse top 5, which must not be read as the label)
    tdir = tmp_path / "topics"
    tdir.mkdir()
    pq.write_table(pa.table({"topic_id": [-1, 0, 1], "terms": [["x"], ["bike", "money"], ["cat", "name"]],
                             "name": ["", "bike money", ""], "prototype": [list(E5), list(E0), list(E3)],
                             "size": [1, 3, 2], "is_outlier": [True, False, False]}), tdir / "topics.parquet")
    pq.write_table(pa.table({"unit_id": ["s1#0", "s1#1", "s2#0/0"], "label": [0, -1, 1],
                             "is_outlier": [False, True, False]}), tdir / "unit_labels.parquet")
    pq.write_table(pa.table({"unit_id": ["s1#0", "s1#0", "s1#1", "s1#1", "s2#0/0", "s2#0/0"],
                             "topic_id": [0, 1, -1, 0, 1, 0], "prob": [0.6, 0.4, 0.5, 0.5, 0.9, 0.1]}),
                   tdir / "unit_topics.parquet")
    for t in (R.load_topics(tdir), R.load_topics(tdir / "topics.parquet"),
              R.load_topics(tdir / "topics.parquet", tdir / "unit_labels.parquet")):
        assert t.ids == [0, 1] and t.names == {0: "bike money", 1: "cat name"} and t.vecs.shape == (2, 384)
        assert t.owner_topic == {"s1#0": 0, "s2#0": 1}
        assert np.allclose(t.vecs[1], E3)
    # part1.graph.write layout: variant plain or topic, member_unit_ids, prototype
    gdir = tmp_path / "graph"
    gdir.mkdir()
    pq.write_table(pa.table({"variant": ["plain", "topic", "topic"], "community_id": [0, 0, 1],
                             "phrase_ids": [[1], [2], [3, 4]],
                             "member_unit_ids": [["z"], ["s2#0/0"], ["s1#0/0", "s1#0/1"]],
                             "prototype": [list(E5), list(E3), list(E0)], "size": [1, 1, 2],
                             "n_members": [1, 1, 2]}), gdir / "communities.parquet")
    for c in (R.load_communities(gdir), R.load_communities(gdir / "communities.parquet", version="topic_weighted")):
        assert c.ids == [0, 1] and c.unit_community == {"s2#0/0": 0, "s1#0/0": 1, "s1#0/1": 1}
        assert np.allclose(c.vecs[0], E3)
    plain = R.load_communities(gdir, version="plain")
    assert plain.ids == [0] and plain.unit_community == {"z": 0}
    pq.write_table(pa.table({"community_id": [7], "embedding": [list(E1)]}), tmp_path / "c2.parquet")
    pq.write_table(pa.table({"unit_id": ["s3#0/0"], "community_id": [7]}), tmp_path / "uc.parquet")
    c2 = R.load_communities(tmp_path / "c2.parquet", unit_path=tmp_path / "uc.parquet")
    assert c2.ids == [7] and c2.unit_community == {"s3#0/0": 7}
    # part1.overlay.write_outputs layout: links_<set>.json; a parquet file also works
    odir = tmp_path / "overlay"
    odir.mkdir()
    (odir / "links_R2.json").write_text(json.dumps({"set": "R2", "n": 2, "links": [
        {"topic_id": 0, "community_id": 1, "w": 0.9}, {"topic_id": "2", "community_id": 3, "w": 0.5}]}))
    (odir / "links_R3.json").write_text(json.dumps({"set": "R3", "n": 1, "links": [
        {"topic_id": 0, "community_id": 1, "w": 0.9}]}))
    (odir / "links_P0.json").write_text(json.dumps({"set": "P0", "n": 2, "links": [
        {"topic_id": 0, "community_id": 3, "w": 0.9}, {"topic_id": 2, "community_id": 1, "w": 0.5}], "shuffle": {}}))
    sets = R.load_link_sets(odir)
    assert sets == {"R2": [R.Link(0, 1, 0.9), R.Link(2, 3, 0.5)], "R3": [R.Link(0, 1, 0.9)],
                    "P0": [R.Link(0, 3, 0.9), R.Link(2, 1, 0.5)]}
    assert R.select_links(sets, "R3") == [R.Link(0, 1, 0.9)]
    pq.write_table(pa.table({"topic_id": [0, 2], "community_id": [1, 3], "confidence": [0.9, 0.5]}), tmp_path / "links.parquet")
    assert R.load_links(tmp_path / "links.parquet") == [R.Link(0, 1, 0.9), R.Link(2, 3, 0.5)]
    # part1.overlay's own Link objects (topic_id, community_id, w) are accepted everywhere
    theirs = SimpleNamespace(topic_id=np.int32(0), community_id="1", w=0.9)
    assert R.as_link(theirs) == R.Link(0, 1, 0.9) and R.as_link({"topic": 2, "community": 3, "weight": 0.5}) == R.Link(2, 3, 0.5)
    assert R.select_links({"R2": [theirs], "P0": []}, "R2") == [R.Link(0, 1, 0.9)]
    sp = R.build_space(lme_tables(), "lme", topics=topics(), communities=communities(), links={"R2": [theirs], "P0": []})
    assert sp.links == {"R2": [R.Link(0, 1, 0.9)], "P0": []}
    with pytest.raises(TypeError):
        R.as_link(object())
    d = tmp_path / "pgr" / "lme" / "q1"
    d.mkdir(parents=True)
    assert R.pgr_space_dir("lme", "q1", tmp_path / "pgr") == d
    pq.write_table(pa.table({"entity_key": ["10"], "name": ["bike"], "aliases": [["red bike"]],
                             "description": ["d"], "entity_type": [None]}), d / "entities.parquet")
    pq.write_table(pa.table({"edge_id": ["5"], "src_key": ["10"], "tgt_key": ["11"], "relation_type": ["lives_in"],
                             "description": ["d"], "weight": [1.0], "negated": [False], "confidence": [None],
                             "valid_from": ["2023-05"], "valid_to": [None], "t_created": ["2026"], "t_expired": [None],
                             "superseded_by": [None], "asserted_at": [None], "sources": [["1"]],
                             "embedding": [[0.5, 0.25]]}), d / "relations.parquet")
    pq.write_table(pa.table({"chunk_id": ["1", "2"], "content": ["a", "b"], "doc_id": ["s1", "s1"],
                             "doc_date": ["2023-05-20"] * 2, "turn_start": [0, None], "turn_end": [1, None],
                             "char_start": [0, -1], "char_end": [5, -1]}), d / "chunks.parquet")
    pq.write_table(pa.table({"entity_key": ["10", "10", "11"], "chunk_id": ["2", "1", "1"]}), d / "doc_mentions.parquet")
    p = R.load_pgr(d)
    assert set(p.entities) == {"10"} and p.relations[0]["edge_id"] == "5" and p.relations[0]["sources"] == ["1"]
    assert set(p.chunks) == {"1", "2"} and p.mentions == {"10": ["1", "2"], "11": ["1"]}
    sp = R.build_space(lme_tables(), "lme", pgr=p, rel_vecs=np.zeros((1, 2, 384)))
    assert sp.rel_units == [["s1#0", "s1#1"]] and sp.n_unmapped_chunks == 1
    assert R.chat_model() == "gemini-2.5-flash-lite"


# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------
def test_no_dashes_arrows_or_curly_quotes_in_the_new_files():
    root = Path(__file__).resolve().parents[1]
    bad = re.compile("[\u2013\u2014\u2192\u2026\u2018\u2019\u201c\u201d]")
    for rel in ("src/multicard/part1/retrieve.py", "src/multicard/part1/planner.py", "tests/test_part1_retrieve.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert not bad.search(text), rel

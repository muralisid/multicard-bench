"""Tests for part1.units, part1.render and part1.score on tiny fixtures.

No model is loaded: token counting uses a word counter or a character counter,
and encoding uses a fake encoder that returns deterministic vectors.
"""

import hashlib

import numpy as np
import pytest

from multicard.data.multihoprag import Document, Evidence, Query
from multicard.part1 import render as R
from multicard.part1 import score as S
from multicard.part1 import units as U


class Words:
    """One token per whitespace-separated word."""

    def count(self, text):
        return len(text.split())


class Chars:
    """One token per character, so a budget is a character budget."""

    def count(self, text):
        return len(text)


class FakeEncoder:
    def encode(self, texts, normalise=True):
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out[i, :32] = np.frombuffer(h, dtype=np.uint8) / 255.0
        return out


SESSIONS = {
    "s1": {"date": "2023-05-20", "turns": [
        {"role": "user", "content": "I bought a red bike yesterday. It cost two hundred dollars. Love it."},
        {"role": "assistant", "content": "Great choice for the city.\nEnjoy the ride every day.\nStay safe out there."},
    ]},
    "s2": {"date": "2023-05-18", "turns": [
        {"role": "user", "content": "My cat is called Tom and he is three years old."},
        {"role": "assistant", "content": "Nice."},
    ]},
    "s3": {"date": "2023-05-20", "turns": [
        {"role": "user", "content": "one two three four"},
        {"role": "user", "content": "five six seven eight"},
    ]},
}


def tables(order=("s1", "s2", "s3")):
    return U.build_tables(SESSIONS, list(order))


# ----------------------------------------------------------------------------
# units: LongMemEval
# ----------------------------------------------------------------------------
def test_ids_and_sub_units_follow_the_e5_split():
    t = tables()
    assert list(t) == ["s1", "s2", "s3"]
    s1 = t["s1"]
    assert [x.unit_id for x in s1.turns] == ["s1#0", "s1#1"]
    assert s1.turns[0].role == "user" and s1.turns[0].date == "2023-05-20"
    subs = s1.subs[0]
    # "Love it." has two words, so the e5 split drops it
    assert [x.unit_id for x in subs] == ["s1#0/0", "s1#0/1"]
    assert [x.parent for x in subs] == ["s1#0"] * 2
    assert [x.position for x in subs] == [0, 1]
    assert [x.text for x in subs] == U.split_user(SESSIONS["s1"]["turns"][0]["content"])
    assert [x.text for x in s1.subs[1]] == U.split_assistant(SESSIONS["s1"]["turns"][1]["content"])
    # a short assistant turn falls back to the whole text as one item
    assert [x.text for x in t["s2"].subs[1]] == ["Nice."]
    assert U.owner_id("s1#0/2") == "s1#0" and U.owner_id("s1#0") == "s1#0"
    assert U.container_of("s1#0/2") == "s1" and U.index_of("s1#1") == 1
    ids, texts = U.sub_unit_texts(t)
    assert len(ids) == len(texts) == sum(len(x.sub_units) for x in t.values())


def test_encode_tables_attaches_vectors_from_e5_build_units():
    t = U.encode_tables(FakeEncoder(), SESSIONS, ["s1", "s2"])
    assert t["s1"].turn_vecs.shape == (2, 384)
    assert [v.shape[0] for v in t["s1"].sub_vecs] == [len(g) for g in t["s1"].subs]
    assert t["s1"].sub_matrix().shape == (len(t["s1"].sub_units), 384)
    # the vector of a turn is the encoding of its text
    assert np.allclose(t["s2"].turn_vecs[0], FakeEncoder().encode([t["s2"].turns[0].text])[0])


def test_turn_bm25_ranks_the_matching_turn_first():
    bm = U.turn_bm25(tables())
    assert bm.rank("red bike", k=3)[0] == "s1#0"
    assert bm.rank("cat Tom", k=3)[0] == "s2#0"


# ----------------------------------------------------------------------------
# units: MultiHop-RAG chunks and fact location
# ----------------------------------------------------------------------------
DOC_BODY = (
    "Alpha won the cup on Sunday. The final was played in Lyon.\n\n"
    "The coach said the team trained hard all year. Fans filled the square after the match.\n\n"
    "Beta finished second. Gamma finished third. Delta went home early.\n\n"
    "Ticket prices rose ten percent this season, according to the club."
)


def doc(doc_id="mhr_a", title="Alpha wins the cup", body=DOC_BODY):
    return Document(doc_id=doc_id, title=title, source="Sporting News", category="sports",
                    published_at="2023-10-01T10:00:00", body=body, url="https://example.com/a")


def test_chunk_spans_pack_paragraphs_and_split_long_ones_at_sentence_ends():
    text = doc().text
    spans = U.chunk_spans(text, Words(), max_tokens=20)
    # no overlap, in order, and every chunk within the limit unless it is one sentence
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    assert all(n <= 20 for _, _, n in spans)
    # the title paragraph and the first body paragraph pack together (3 + 12 words)
    assert text[spans[0][0]:spans[0][1]].startswith("Alpha wins the cup\n\nAlpha won the cup")
    # a paragraph over the limit is split at a sentence end
    long_para = "word " * 30 + "end here. " + "more " * 10 + "again."
    parts = U.chunk_spans(long_para.strip(), Words(), max_tokens=35)
    assert len(parts) == 2
    assert long_para.strip()[parts[0][0]:parts[0][1]].endswith("end here.")
    assert long_para.strip()[parts[1][0]:parts[1][1]].startswith("more")
    # every character of the text belongs to at most one chunk
    covered = sum(hi - lo for lo, hi, _ in parts)
    assert covered <= len(long_para.strip())


def test_doc_tables_ids_sentences_and_bm25():
    t = U.build_doc_tables([doc()], Words(), max_tokens=20)
    d = t["mhr_a"]
    assert d.date == "2023-10-01" and d.source == "Sporting News"
    assert [c.unit_id for c in d.chunks] == [f"mhr_a#{k}" for k in range(len(d.chunks))]
    assert d.chunks[1].text == d.text[d.chunks[1].char_start:d.chunks[1].char_end]
    assert d.sentences[0][0].unit_id == "mhr_a#0/0" and d.sentences[0][0].parent == "mhr_a#0"
    assert U.chunk_bm25(t).rank("ticket prices", k=2)[0] == d.chunks[-1].unit_id
    et = U.encode_doc_tables(FakeEncoder(), t)
    assert et["mhr_a"].chunk_vecs.shape == (len(d.chunks), 384)


def test_normalise_rules():
    assert U.normalise('  "Alpha   WON,"  ') == "alpha won"
    assert U.normalise("\u201cquoted\u201d.") == "quoted"     # curly quotes, written as escapes
    assert U.normalise("Ａlpha") == "alpha"     # NFKC folds the fullwidth letter


def test_fact_location_and_fallback_counting():
    t = U.build_doc_tables([doc()], Words(), max_tokens=20)
    q = Query(qid="q1", query="?", answer="a", question_type="inference_query", evidence=[
        Evidence(title="Alpha wins the cup", fact="The final was played in Lyon.", doc_id="mhr_a"),
        Evidence(title="Alpha wins the cup", fact='  "beta FINISHED   second"  ', doc_id="mhr_a"),
        Evidence(title="Alpha wins the cup", fact="This sentence is not in the article.", doc_id="mhr_a"),
    ])
    locs = U.locate_query(q, t)
    assert [l.located for l in locs] == [True, True, False]
    assert [l.fallback for l in locs] == [False, False, True]
    assert locs[0].chunk_id == "mhr_a#0"
    assert locs[1].chunk_id is not None and t["mhr_a"].chunks[U.index_of(locs[1].chunk_id)].text.find("Beta finished second") >= 0
    assert locs[2].chunk_id is None
    q2 = Query(qid="q2", query="?", answer="a", question_type="comparison_query", evidence=[
        Evidence(title="Alpha wins the cup", fact="Delta went home early", doc_id="mhr_a")])
    null = Query(qid="q3", query="?", answer="Insufficient information.", question_type="null_query")
    summary = U.location_summary([q, q2, null], {"q1": locs, "q2": U.locate_query(q2, t)})
    assert summary["n_queries"] == 2 and summary["n_all_located"] == 1
    assert summary["query_share"] == 0.5
    assert summary["n_facts"] == 4 and summary["n_located"] == 3 and summary["n_fallback"] == 1
    assert summary["fact_share"] == 0.75
    assert U.located_share([q, q2, null], {"q1": locs, "q2": U.locate_query(q2, t)}) == (0.5, 0.75)


def test_fact_across_a_chunk_boundary_maps_to_one_chunk():
    body = "First sentence here. Second sentence here. Third sentence here. Fourth one."
    # title "T" (1 word) packs with the first sentence; then one sentence per chunk
    t = U.build_doc_tables([doc(body=body, title="T")], Words(), max_tokens=4)
    d = t["mhr_a"]
    assert [c.text for c in d.chunks] == ["T\n\nFirst sentence here.", "Second sentence here.",
                                          "Third sentence here.", "Fourth one."]
    loc = U.locate_fact("q", "Second sentence here. Third sentence here.", "mhr_a", t)
    # the fact crosses chunks 1 and 2 and maps to the chunk holding more of it
    assert loc.located and loc.straddle and loc.chunk_id == "mhr_a#1"


# ----------------------------------------------------------------------------
# render: budget, truncation, tie-break, dedupe
# ----------------------------------------------------------------------------
def test_budget_rule_and_truncation_flag():
    t = tables(["s3", "s1", "s2"])
    # each s3 turn line is "[2023-05-20, user] " (2 words) + 4 words = 6 tokens
    ranked = ["s3#0", "s3#1", "s2#0"]
    ctx = R.render(ranked, 12, t, counter=Words())
    assert sorted(ctx.unit_ids) == ["s3#0", "s3#1"]
    assert ctx.tokens == 12 and ctx.n_truncated == 0
    assert not any(ctx.truncated.values())
    # 17 tokens: 5 remain for the third unit, its prefix takes 2, so 3 body words fit; flagged
    ctx = R.render(ranked, 17, t, counter=Words())
    assert ctx.tokens == 17 and ctx.n_truncated == 1
    assert ctx.truncated["s2#0"] is True and ctx.truncated["s3#0"] is False
    u = {x.unit_id: x for x in ctx.units}["s2#0"]
    assert u.body == "My cat is" and u.turn_chars == {"s2#0": len("My cat is")}
    assert u.line == "[2023-05-18, user] My cat is"
    # the reported token count equals the counter on the rendered text
    assert Words().count(ctx.text) == ctx.tokens
    assert ctx.candidates == ranked and ctx.n_candidates == 3
    # a budget that cannot hold any of the body drops the unit and stops
    for budget in (13, 14):
        ctx = R.render(ranked, budget, t, counter=Words())
        assert sorted(ctx.unit_ids) == ["s3#0", "s3#1"] and ctx.tokens == 12


def test_display_order_oldest_first_then_session_order_then_turn_index():
    t = tables(["s3", "s1", "s2"])     # s3 comes before s1 in session order, both dated 2023-05-20
    ranked = ["s1#1", "s3#1", "s2#0", "s3#0", "s1#0"]
    ctx = R.render(ranked, 1000, t, counter=Words())
    assert ctx.unit_ids == ["s2#0", "s3#0", "s3#1", "s1#0", "s1#1"]
    assert ctx.text.startswith("[2023-05-18, user] My cat")
    assert "\n\n[2023-05-20, assistant] Great choice" in ctx.text
    # a date_key override changes the sort
    ctx = R.render(ranked, 1000, t, date_key=lambda uid: "2024" if uid == "s2#0" else "2023",
                   counter=Words())
    assert ctx.unit_ids[-1] == "s2#0"


def test_hits_dedupe_to_their_owner_turn():
    t = tables()
    ctx = R.render(["s1#0/1", "s1#0", "s1#0/0", "s1#1/2"], 1000, t, counter=Words())
    assert ctx.unit_ids == ["s1#0", "s1#1"]
    assert ctx.duplicate_share == 0.0


def test_turns_are_cut_at_2000_characters():
    long = {"L": {"date": "2023-01-01", "turns": [{"role": "user", "content": "x" * 5000},
                                                 {"role": "user", "content": "y" * 100}]}}
    t = U.build_tables(long, ["L"])
    ctx = R.render(["L#0", "L#1"], 10 ** 6, t, counter=Chars())
    assert len(ctx.units[0].body) == 2000 and ctx.units[0].truncated is False
    # section 3: a truncated evidence turn is flagged and counted; the 2,000-character cut is
    # a second flag, separate from the budget truncation, and the score counts both
    assert ctx.units[0].cut is True and ctx.units[1].cut is False and ctx.n_truncated == 0
    sc = S.score_longmemeval("q", {"L#0", "L#1"}, {"L"}, ctx, t)
    assert sc.joint_recall == 1.0 and sc.truncated_evidence == 1 and sc.cut_evidence == 1
    agg = S.summarise([sc])
    assert agg["truncated_evidence"] == 1 and agg["cut_evidence"] == 1
    # the budget truncation alone leaves cut_evidence at zero
    ctx = R.render(["L#1"], len("[2023-01-01, user] ") + 60, t, counter=Chars())
    sc = S.score_longmemeval("q", {"L#1"}, {"L"}, ctx, t)
    assert ctx.n_truncated == 1 and sc.truncated_evidence == 1 and sc.cut_evidence == 0


def test_locate_fact_prefers_an_occurrence_inside_one_chunk():
    # the first occurrence straddles chunks 1 and 2; the second lies whole in chunk 3
    body = "x alpha beta\n\ngamma delta y\n\nalpha beta gamma delta"
    t = U.build_doc_tables([doc(body=body, title="T")], Words(), max_tokens=4)
    d = t["mhr_a"]
    assert [c.text for c in d.chunks] == ["T\n\nx alpha beta", "gamma delta y", "alpha beta gamma delta"]
    loc = U.locate_fact("q", "alpha beta gamma delta", "mhr_a", t)
    assert loc.located and not loc.straddle and loc.chunk_id == "mhr_a#2"
    # with the whole occurrence gone, the straddling one maps to the chunk holding more of it
    body2 = "x alpha beta\n\ngamma delta y\n\nnothing here"
    t2 = U.build_doc_tables([doc(body=body2, title="T")], Words(), max_tokens=4)
    loc2 = U.locate_fact("q", "alpha beta gamma delta", "mhr_a", t2)
    assert loc2.located and loc2.straddle and loc2.chunk_id == "mhr_a#1"


def test_native_tie_break_follows_session_order_then_turn_index():
    # two chunks on the same date from sessions b and a, native order b then a; the
    # question's session order is a then b (section 5: ties by session order then turn index)
    cb = R.NativeUnit("chunk:b", "[2023-05-20] b text", "chunk", date="2023-05-20", container="b",
                      turn_spans={"b#0": (13, 19)}, sessions=("b",))
    ca1 = R.NativeUnit("chunk:a1", "[2023-05-20] a text one", "chunk", date="2023-05-20", container="a",
                       turn_spans={"a#1": (13, 23)}, sessions=("a",))
    ca0 = R.NativeUnit("chunk:a0", "[2023-05-20] a text zero", "chunk", date="2023-05-20", container="a",
                       turn_spans={"a#0": (13, 24)}, sessions=("a",))
    ctx = R.render_native([cb, ca1, ca0], 1000, counter=Words(), order={"a": 0, "b": 1})
    assert ctx.unit_ids == ["chunk:a0", "chunk:a1", "chunk:b"]
    # without a session order the turn index then the native position decide
    ctx = R.render_native([cb, ca1, ca0], 1000, counter=Words())
    assert ctx.unit_ids == ["chunk:b", "chunk:a0", "chunk:a1"]
    assert R.render_native([cb, ca1, ca0], 1000, counter=Words(), keep_native_order=True).unit_ids == \
        ["chunk:b", "chunk:a1", "chunk:a0"]


# ----------------------------------------------------------------------------
# score: the symmetric coverage rule
# ----------------------------------------------------------------------------
def test_symmetric_coverage_rule_on_a_long_turn():
    long = {"L": {"date": "2023-01-01", "turns": [
        {"role": "user", "content": "a" * 5000}, {"role": "user", "content": "b" * 3000}]}}
    t = U.build_tables(long, ["L"])
    prefix = len("[2023-01-01, user] ")
    lengths = S.turn_lengths(t, ["L#0", "L#1"])
    assert lengths == {"L#0": 5000, "L#1": 3000}
    # 5,000-character turn: threshold is 2,000; the 2,000-character cut covers it
    ctx = R.render(["L#0"], prefix + 2000, t, counter=Chars())
    assert S.turn_coverage("L#0", ctx.units, 5000).covered is True
    # truncated to 1,999 characters: not covered, flagged truncated
    ctx = R.render(["L#0"], prefix + 1999, t, counter=Chars())
    cov = S.turn_coverage("L#0", ctx.units, 5000)
    assert cov.covered is False and cov.truncated is True and cov.chars == 1999
    # 3,000-character turn: threshold is 1,500
    ctx = R.render(["L#1"], prefix + 1500, t, counter=Chars())
    assert S.turn_coverage("L#1", ctx.units, 3000).covered is True
    ctx = R.render(["L#1"], prefix + 1499, t, counter=Chars())
    assert S.turn_coverage("L#1", ctx.units, 3000).covered is False
    # a competitor chunk covers the turn through its turn span
    chunk = R.NativeUnit("c1", "[2023-01-01] " + "a" * 5000 + " " + "b" * 3000, "chunk",
                         date="2023-01-01", container="L",
                         turn_spans={"L#0": (13, 5013), "L#1": (5014, 8014)}, sessions=("L",))
    ctx = R.render_native([chunk], 13 + 2000, counter=Chars())
    assert S.turn_coverage("L#0", ctx.units, 5000).covered is True
    assert S.turn_coverage("L#1", ctx.units, 3000).covered is False
    ctx = R.render_native([chunk], 13 + 5000 + 1 + 1500, counter=Chars())
    assert S.turn_coverage("L#1", ctx.units, 3000).covered is True
    # a relation covers the turn when its provenance chunk does
    rel = R.NativeUnit("r1", "(user) --[bought]--> (bike)", "relation", date="2023-01-01",
                       provenance_spans={"L#0": (13, 5013)}, sessions=("L",))
    ctx = R.render_native([rel], 1000, counter=Chars())
    cov = S.turn_coverage("L#0", ctx.units, 5000)
    assert cov.covered is True and cov.via_relation is True
    # the same rule at candidate level on full texts
    jr, _ = S.joint_recall({"L#0", "L#1"}, ctx.candidate_units, lengths)
    assert jr == 0.0
    jr, _ = S.joint_recall({"L#0", "L#1"}, R.full_native_units([chunk]), lengths)
    assert jr == 1.0


def test_native_date_key_rule_and_native_order():
    units = [
        R.NativeUnit("e1", "ENTITY Tom: a cat", "entity", sessions=()),
        R.NativeUnit("c2", "[2023-05-20] bike chunk", "chunk", date="2023-05-20", container="s1",
                     turn_spans={"s1#0": (13, 23)}, sessions=("s1",)),
        R.NativeUnit("r1", "(Tom) --[is]--> (cat) [from 2023-05-18]", "relation", date="2023-05-18",
                     provenance_spans={"s2#0": (0, 48)}, sessions=("s2",)),
        R.NativeUnit("e2", "ENTITY Alpha: a team", "entity"),
        R.NativeUnit("c2", "[2023-05-20] bike chunk", "chunk", date="2023-05-20"),   # duplicate id
    ]
    ctx = R.render_native(units, 1000, counter=Words())
    assert ctx.unit_ids == ["e1", "e2", "r1", "c2"]
    assert ctx.candidates == ["e1", "c2", "r1", "e2", "c2"] and ctx.n_taken == 4
    ctx = R.render_native(units, 1000, counter=Words(), keep_native_order=True)
    assert ctx.unit_ids == ["e1", "c2", "r1", "e2"]
    assert S.session_ranking_from_units(ctx.candidate_units) == ["s1", "s2"]
    assert S.session_joint_recall({"s1", "s2"}, ctx.units) == 1.0
    assert S.session_joint_recall({"s1", "s9"}, ctx.units) == 0.0


def test_duplicate_share_counts_units_whose_turns_were_already_covered():
    chunk = R.NativeUnit("c1", "one two three four", "chunk", date="2023-01-01",
                         turn_spans={"t#0": (0, 7), "t#1": (8, 18)}, sessions=("t",))
    rel = R.NativeUnit("r1", "a b", "relation", date="2023-01-01", provenance_spans={"t#0": (0, 7)})
    other = R.NativeUnit("r2", "c d e", "relation", date="2023-01-01", provenance_spans={"u#0": (0, 7)})
    ctx = R.render_native([chunk, rel, other], 1000, counter=Words())
    assert ctx.tokens == 9 and ctx.duplicate_share == pytest.approx(2 / 9)
    # order matters: the relation first, then the chunk, and the chunk is not a duplicate
    ctx = R.render_native([rel, chunk, other], 1000, counter=Words())
    assert ctx.duplicate_share == 0.0


# ----------------------------------------------------------------------------
# score: whole-question scoring
# ----------------------------------------------------------------------------
def test_all_evidence_in_the_top_scores_one():
    t = tables()
    evidence_turns = {"s1#0", "s2#0"}
    evidence_sessions = {"s1", "s2"}
    ranked = ["s1#0/1", "s2#0", "s3#0", "s1#1"]
    ctx = R.render(ranked, 1000, t, counter=Words())
    sc = S.score_longmemeval("q", evidence_turns, evidence_sessions, ctx, t)
    assert sc.joint_recall == 1.0 and sc.candidate_joint_recall == 1.0
    assert sc.session_joint_recall == 1.0
    assert sc.turn_r10 == 1.0 and sc.turn_ndcg10 == pytest.approx(1.0) and sc.sess_r5 == 1.0
    assert sc.uncovered == [] and sc.truncated_evidence == 0 and sc.missing is False
    assert sc.rendered_tokens == ctx.tokens and sc.n_candidates == 4
    # the evidence outside the budget is uncovered but still in the candidates
    ctx = R.render(ranked, 16, t, counter=Words())
    sc = S.score_longmemeval("q", evidence_turns, evidence_sessions, ctx, t)
    assert sc.joint_recall == 0.0 and sc.candidate_joint_recall == 1.0
    assert sc.uncovered == ["s2#0"]
    # the ranking can be given explicitly
    sc = S.score_longmemeval("q", evidence_turns, evidence_sessions, ctx, t, turn_ranking=["s3#0", "s1#0"])
    assert sc.turn_r10 == 0.5


def test_missing_output_scores_zero_and_is_counted():
    t = tables()
    ctx = R.render([], 1000, t, counter=Words())
    assert ctx.text == "" and ctx.tokens == 0
    sc = S.score_longmemeval("q", {"s1#0"}, {"s1"}, ctx, t)
    assert sc.missing is True and sc.joint_recall == 0.0 and sc.turn_r10 == 0.0 and sc.sess_r5 == 0.0
    good = S.score_longmemeval("q2", {"s1#0"}, {"s1"}, R.render(["s1#0"], 1000, t, counter=Words()), t)
    agg = S.summarise([sc, good])
    assert agg["n"] == 2 and agg["n_missing"] == 1 and agg["joint_recall"] == 0.5
    graphiti = S.score_longmemeval("q", {"s1#0"}, {"s1"}, ctx, t, session_level_only=True)
    assert graphiti.turn_r10 is None and graphiti.missing is True


def test_multihoprag_document_hit_and_fact_joint_recall():
    other = doc(doc_id="mhr_b", title="Beta ships a phone",
                body="Beta shipped a phone on Monday. Reviews were mixed.\n\nSales start Friday in Europe.")
    t = U.build_doc_tables([doc(), other], Words(), max_tokens=20)
    q = Query(qid="q1", query="?", answer="a", question_type="comparison_query", evidence=[
        Evidence(title="Alpha wins the cup", fact="Ticket prices rose ten percent this season", doc_id="mhr_a"),
        Evidence(title="Beta ships a phone", fact="Sales start Friday in Europe.", doc_id="mhr_b"),
    ])
    locs = U.locate_query(q, t)
    assert all(l.located for l in locs)
    a_chunk, b_chunk = locs[0].chunk_id, locs[1].chunk_id
    ctx = R.render([b_chunk, "mhr_a#0", a_chunk], 1000, t, counter=Words())
    sc = S.score_multihoprag("q1", locs, ctx, t)
    assert sc.fact_joint_recall == 1.0 and sc.doc_joint_recall == 1.0 and sc.all_located
    assert sc.doc_hits == {"mhr_a": True, "mhr_b": True} and sc.n_contained == 2
    # document b is one chunk of 18 words plus a 3-word prefix; a budget of 22 holds it and
    # nothing else, so the fact chunk of mhr_a is outside the budget: document a is not hit,
    # candidate level still is
    assert b_chunk == "mhr_b#0" and len(t["mhr_b"].chunks) == 1
    ctx = R.render([b_chunk, "mhr_a#0", a_chunk], 22, t, counter=Words())
    assert ctx.unit_ids == [b_chunk] and ctx.tokens == 21
    sc = S.score_multihoprag("q1", locs, ctx, t)
    assert sc.doc_hits == {"mhr_a": False, "mhr_b": True}
    assert sc.fact_joint_recall == 0.0 and sc.candidate_fact_joint_recall == 1.0
    # a fallback fact is credited by any rendered chunk of its document, a located one is not
    fb = U.locate_fact("q1", "not in the text at all", "mhr_a", t)
    assert fb.fallback
    ctx = R.render(["mhr_a#0", b_chunk], 1000, t, counter=Words())
    assert S.score_multihoprag("q1", locs, ctx, t).doc_hits == {"mhr_a": False, "mhr_b": True}
    sc = S.score_multihoprag("q1", [fb, locs[1]], ctx, t)
    assert sc.fact_joint_recall == 1.0 and sc.n_fallback == 1 and sc.all_located is False
    # a truncated chunk that cuts the fact off does not contain it
    ctx = R.render([a_chunk], 4, t, counter=Words())
    assert ctx.n_truncated == 1
    assert S.fact_contained(locs[0], ctx.units) is False
    # missing output
    sc = S.score_multihoprag("q1", locs, R.render([], 100, t, counter=Words()), t)
    assert sc.missing and sc.fact_joint_recall == 0.0
    agg = S.summarise([sc], S.MHR_FIELDS)
    assert agg["n_missing"] == 1 and agg["fact_joint_recall_all_located"] == 0.0

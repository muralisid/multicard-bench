"""Tests for the MultiHop-RAG loader, on a tiny fixture and without network.

What matters: a document id that depends on content and not on file order,
evidence that resolves to the right document by title with the two fallbacks
behaving as documented, distinct-document ground truth when a query cites one
article twice, a stratified sample that is balanced and seeded, and a first
download that lands in the JSONL cache with dates written as text.
"""

import json
from datetime import datetime

import pytest

from multicard.data import multihoprag as mh

DOCS = [
    {"title": "Alpha wins the cup", "author": "A. Writer", "source": "Sporting News",
     "published_at": "2023-10-01T10:00:00+00:00", "category": "sports",
     "url": "https://example.com/alpha", "body": "Alpha won the cup on Sunday. " * 5},
    {"title": "Beta ships a phone", "author": None, "source": "The Verge",
     "published_at": "2023-10-02T11:00:00+00:00", "category": "technology",
     "url": "https://example.com/beta", "body": "Beta shipped a phone on Monday. " * 5},
    {"title": "Gamma raises money", "author": "B. Writer", "source": "TechCrunch",
     "published_at": "2023-10-03T12:00:00+00:00", "category": "business",
     "url": "https://example.com/gamma", "body": "Gamma raised money on Tuesday. " * 5},
]


def _ev(title, url="", fact="a fact"):
    return {"title": title, "author": "", "url": url, "source": "", "category": "",
            "published_at": "2023-10-01T10:00:00+00:00", "fact": fact}


QUERIES = [
    # exact title on both entries
    {"query": "Who won and who shipped?", "answer": "Alpha and Beta",
     "question_type": "inference_query",
     "evidence_list": [_ev("Alpha wins the cup"), _ev("Beta ships a phone")]},
    # first entry falls back to url, second to normalised title
    {"query": "Did Beta ship before Gamma raised?", "answer": "Yes",
     "question_type": "temporal_query",
     "evidence_list": [_ev("Beta ships a phone (updated)", "https://example.com/beta"),
                       _ev("  gamma RAISES   money ")]},
    # the same article cited twice with two facts, plus one that resolves nowhere
    {"query": "Is Alpha the same as the winner?", "answer": "Yes",
     "question_type": "comparison_query",
     "evidence_list": [_ev("Alpha wins the cup", fact="fact one"),
                       _ev("Alpha wins the cup", fact="fact two"),
                       _ev("Delta does not exist", "https://example.com/delta")]},
    {"query": "What did Epsilon do?", "answer": "Insufficient information.",
     "question_type": "null_query", "evidence_list": []},
    {"query": "Which came first?", "answer": "Alpha", "question_type": "temporal_query",
     "evidence_list": [_ev("Alpha wins the cup"), _ev("Gamma raises money")]},
    {"query": "Are Alpha and Gamma alike?", "answer": "no",
     "question_type": "comparison_query",
     "evidence_list": [_ev("Alpha wins the cup"), _ev("Gamma raises money")]},
]


def _write(raw, docs=DOCS, queries=QUERIES):
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (raw / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
    return raw


@pytest.fixture
def raw(tmp_path):
    return _write(tmp_path / "multihoprag")


def test_corpus_fields_and_content_derived_ids(raw):
    docs = mh.load_corpus(raw)
    assert [d.title for d in docs] == [d["title"] for d in DOCS]
    d = docs[1]
    assert d.author == ""                      # None in the raw file becomes empty text
    assert d.published_at == "2023-10-02T11:00:00+00:00"
    assert d.source == "The Verge" and d.category == "technology"
    assert d.text.startswith("Beta ships a phone\n\nBeta shipped")
    assert d.doc_id == mh.doc_id_for(d.title, d.url)
    assert len({x.doc_id for x in docs}) == len(docs)

    # Reversing the file order does not change any id.
    other = _write(raw.parent / "reversed", docs=list(reversed(DOCS)))
    assert {x.doc_id for x in mh.load_corpus(other)} == {x.doc_id for x in docs}


def test_duplicate_title_and_url_still_get_distinct_ids(raw):
    other = _write(raw.parent / "dup", docs=DOCS + [DOCS[0]])
    ids = [d.doc_id for d in mh.load_corpus(other)]
    assert len(set(ids)) == 4


def test_evidence_resolution_paths(raw):
    docs = mh.load_corpus(raw)
    by_title = {d.title: d.doc_id for d in docs}
    qs = mh.load_queries(raw, docs=docs)
    assert [q.qid for q in qs][:2] == ["mhr_q0000", "mhr_q0001"]

    exact = qs[0].evidence
    assert [e.resolved_by for e in exact] == ["title", "title"]
    assert [e.doc_id for e in exact] == [by_title["Alpha wins the cup"], by_title["Beta ships a phone"]]

    fallback = qs[1].evidence
    assert fallback[0].resolved_by == "url"
    assert fallback[0].doc_id == by_title["Beta ships a phone"]
    assert fallback[1].resolved_by == "norm_title"
    assert fallback[1].doc_id == by_title["Gamma raises money"]

    repeated = qs[2]
    assert [e.resolved_by for e in repeated.evidence] == ["title", "title", ""]
    assert repeated.evidence[2].doc_id is None
    # Three entries, but one distinct resolved document.
    assert repeated.evidence_doc_ids == [by_title["Alpha wins the cup"]]
    assert [e.fact for e in repeated.evidence[:2]] == ["fact one", "fact two"]

    null = qs[3]
    assert null.question_type == "null_query" and null.evidence == []
    assert null.answer == mh.NULL_ANSWER


def test_summary_counts(raw):
    docs = mh.load_corpus(raw)
    s = mh.summary(docs, mh.load_queries(raw, docs=docs))
    assert s["n_docs"] == 3 and s["n_queries"] == 6
    assert s["per_type"] == {"comparison_query": 2, "inference_query": 1,
                             "null_query": 1, "temporal_query": 2}
    assert s["n_evidence_entries"] == 11
    assert s["unresolved_evidence"] == 1
    assert s["resolved_by"] == {"norm_title": 1, "title": 8, "unresolved": 1, "url": 1}
    assert s["evidence_docs_per_query"] == {"0": 1, "1": 1, "2": 4}
    assert s["evidence_entries_per_query"] == {"0": 1, "2": 4, "3": 1}
    assert s["queries_with_repeated_doc"] == 1
    assert s["queries_without_evidence"] == 1


def test_stratified_is_balanced_and_seeded(raw):
    qs = mh.load_queries(raw)
    picked = mh.stratified(qs, 4, seed=13)
    assert sorted(q.question_type for q in picked) == sorted(mh.QUESTION_TYPES)
    assert [q.qid for q in mh.stratified(qs, 4, seed=13)] == [q.qid for q in picked]
    # Asking for more than exists returns everything once.
    everything = mh.stratified(qs, 100, seed=13)
    assert sorted(q.qid for q in everything) == sorted(q.qid for q in qs)
    # Filtering first is how a type is left out.
    no_null = mh.stratified([q for q in qs if q.question_type != "null_query"], 3, seed=13)
    assert all(q.question_type != "null_query" for q in no_null)


def test_stratified_seed_changes_the_sample():
    qs = [mh.Query(qid=f"q{i}", query="", answer="", question_type=t)
          for i in range(60) for t in ("a", "b")]
    a = [q.qid for q in mh.stratified(qs, 10, seed=1)]
    b = [q.qid for q in mh.stratified(qs, 10, seed=2)]
    assert a != b
    assert a == [q.qid for q in mh.stratified(qs, 10, seed=1)]


def test_first_load_writes_jsonl_cache_with_dates_as_text(tmp_path, monkeypatch):
    calls = []

    def fake_load_dataset(name, config, split, revision, cache_dir):
        calls.append((name, config, split, revision))
        rows = DOCS if config == "corpus" else QUERIES
        out = []
        for r in rows:
            r = dict(r)
            r["published_at"] = datetime.fromisoformat(r["published_at"]) if "published_at" in r else None
            out.append(r)
        return out

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    raw = tmp_path / "fresh"
    docs = mh.load_corpus(raw)
    assert calls == [(mh.NAME, "corpus", "train", mh.REVISION)]
    assert (raw / "corpus.jsonl").exists()
    assert docs[0].published_at == "2023-10-01T10:00:00+00:00"
    # The second call reads the cache and does not download again.
    mh.load_corpus(raw)
    assert len(calls) == 1
    qs = mh.load_queries(raw, docs=docs)
    assert calls[-1] == (mh.NAME, "MultiHopRAG", "train", mh.REVISION)
    assert len(qs) == len(QUERIES)

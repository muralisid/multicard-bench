"""Dataset integrity tests: shared pools, exact evidence and pinned inputs."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from multicard.data import multihoprag as mhr
from multicard.version_a import document_data as data


def _musique():
    corpus = [{"title": "Same title", "text": "First paragraph."},
              {"title": "Same title", "text": "Second paragraph."},
              {"title": "Distractor", "text": "Unrelated paragraph."}]
    question = {
        "id": "2hop__fixture", "question": "Which two?", "answer": "A and B",
        "answer_aliases": ["B and A", "A and B"], "answerable": True,
        "paragraphs": [
            {"title": p["title"], "paragraph_text": p["text"], "is_supporting": i < 2}
            for i, p in enumerate(corpus)],
        "question_decomposition": [{"answer": "never put this in corpus"}],
    }
    return corpus, question


def test_musique_same_title_distinct_passages_and_no_gold_leakage():
    source, question = _musique()
    corpus, questions = data.convert_hipporag_rows("musique", source, [question])
    assert len(corpus) == 3
    assert len({p["id"] for p in corpus}) == 3
    assert {p["group"] for p in corpus} == {"global"}
    assert all(set(p) == {"id", "group", "text"} for p in corpus)
    assert "never put this in corpus" not in json.dumps(corpus)
    assert questions[0]["evidence_ids"] == [p["id"] for p in corpus[:2]]
    assert questions[0]["answers"] == ["A and B", "B and A"]
    assert questions[0]["type"] == "2hop"
    assert questions[0]["metric"] == "f1"
    reversed_corpus, _ = data.convert_hipporag_rows("musique", list(reversed(source)), [question])
    assert {p["id"] for p in reversed_corpus} == {p["id"] for p in corpus}


def test_missing_gold_passage_fails_instead_of_changing_denominator():
    corpus, question = _musique()
    with pytest.raises(ValueError, match="unresolved supporting"):
        data.convert_hipporag_rows("musique", corpus[1:], [question])
    question["answerable"] = False
    with pytest.raises(ValueError, match="must be answerable"):
        data.convert_hipporag_rows("musique", corpus, [question])


def test_2wiki_sentence_join_and_repeated_support_fact_are_exact():
    source = [{"title": "First", "text": "One. Two."},
              {"title": "Second", "text": "Three."}]
    sample = {
        "_id": "q1", "question": "Where?", "answer": "Here", "type": "compositional",
        "context": [["First", ["One.", "Two."]], ["Second", ["Three."]]],
        "supporting_facts": [["First", 0], ["First", 1]],
    }
    corpus, questions = data.convert_hipporag_rows("2wikimultihopqa", source, [sample])
    assert questions[0]["evidence_ids"] == [corpus[0]["id"]]
    assert questions[0]["answers"] == ["Here"]
    assert corpus[0]["text"] == "First\nOne. Two."


def test_duplicate_inputs_fail():
    source, question = _musique()
    with pytest.raises(ValueError, match="Duplicate question"):
        data.convert_hipporag_rows("musique", source, [question, question])
    with pytest.raises(ValueError, match="duplicate full passages"):
        data.convert_hipporag_rows("musique", source + [source[0]], [question])


def test_multihoprag_keeps_nulls_and_distinct_article_evidence():
    doc = mhr.Document("d1", "Title", "Outlet", "news", "2023-10-01", "Body")
    queries = [
        mhr.Query("q1", "Who?", "One", "inference_query", [
            mhr.Evidence("Title", "one fact", doc_id="d1"),
            mhr.Evidence("Title", "another fact", doc_id="d1")]),
        mhr.Query("q2", "Unknown?", mhr.NULL_ANSWER, "null_query"),
    ]
    counter = SimpleNamespace(count=lambda text: len(text.split()))
    corpus, questions = data.convert_multihoprag([doc], queries, counter=counter)
    assert corpus == [{"id": "d1#0", "parent_id": "d1", "group": "global", "text": "Title\n\nBody",
                       "date": "2023-10-01", "source": "Outlet", "char_start": 0, "char_end": 11}]
    assert questions[0]["evidence_ids"] == ["d1"]
    assert questions[1]["evidence_ids"] == []
    assert questions[1]["answers"] == [mhr.NULL_ANSWER]
    assert questions[1]["metric"] == "judge"
    queries[0].evidence[0].doc_id = None
    with pytest.raises(ValueError, match="unresolved evidence"):
        data.convert_multihoprag([doc], queries, counter=counter)


def test_multihoprag_returns_tail_chunks_and_evidence_stays_parent_level():
    body = "\n\n".join(" ".join(f"p{i}w{j}" for j in range(300)) for i in range(3))
    doc = mhr.Document("d1", "Title", "Outlet", "news", "2023-10-01", body)
    query = mhr.Query("q1", "What?", "Answer", "inference_query",
                      [mhr.Evidence("Title", "tail fact", doc_id="d1")])
    corpus, questions = data.convert_multihoprag(
        [doc], [query], counter=SimpleNamespace(count=lambda text: len(text.split())))
    assert len(corpus) == 3
    assert {d["parent_id"] for d in corpus} == {"d1"}
    assert "p2w299" in corpus[-1]["text"]
    assert all(d["text"] == doc.text[d["char_start"]:d["char_end"]] for d in corpus)
    assert questions[0]["evidence_ids"] == ["d1"]


def test_cached_download_is_verified_and_does_not_touch_network(tmp_path, monkeypatch):
    content = b"[]"
    path = tmp_path / "musique.json"
    path.write_bytes(content)
    monkeypatch.setitem(data.HIPPO_SHA256, path.name, hashlib.sha256(content).hexdigest())
    def no_network(*args, **kwargs):
        raise AssertionError("Cached input must not trigger a download")
    monkeypatch.setattr(data, "urlopen", no_network)
    assert data._download(path.name, tmp_path) == path
    path.write_bytes(b"[{}]")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data._download(path.name, tmp_path)


def test_manifest_recomputes_output_checksums_and_excludes_null_retrieval(tmp_path):
    corpus = [{"id": "d1", "group": "global", "text": "example"}]
    questions = [{"id": "q1", "type": "known", "evidence_ids": ["d1"]},
                 {"id": "q2", "type": "null", "evidence_ids": []}]
    manifest = data._write_dataset(tmp_path, corpus, questions, {"dataset": "fixture"})
    assert manifest["counts"] == {"corpus": 1, "questions": 2, "groups": 1,
                                  "evidence_questions": 1,
                                  "question_types": {"known": 1, "null": 1}}
    assert manifest["output_files"]["corpus.jsonl"]["sha256"] == data.sha256(tmp_path / "corpus.jsonl")
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest

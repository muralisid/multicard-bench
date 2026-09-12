"""Version A validity checks without downloads or model requests."""

import re
from types import SimpleNamespace

import numpy as np
import pytest

from multicard.version_a import core


class WordTokenizer:
    """Offset-preserving fixture; no silently truncated test tokenizer."""

    def __call__(self, text, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"offset_mapping": offsets, "input_ids": list(range(len(offsets)))}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]


class TinyEncoder:
    def __init__(self):
        self.model = SimpleNamespace(tokenizer=WordTokenizer())
        self._mem = {}
        self.calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        return np.asarray([[1.0, 0.0] if "needle" in text or text in {"", "the a an", "unseenquery"}
                           else [0.0, 1.0] for text in texts], dtype=np.float32)


def test_retrieval_scores_use_distinct_gold_and_exclude_partial_rendering():
    got = core.retrieval_metrics(
        ["a", "b", "c"], ["a", "b"], ["a", "b", "b"], fully_rendered=["a"])
    assert got["recall_at_5"] == 1.0
    assert got["evidence_recall_at_budget"] == 0.5
    assert got["all_evidence_at_budget"] == 0.0
    assert core.retrieval_metrics(["a"], ["a"], []) == {}


def test_answer_scoring_uses_alias_max_and_token_multiplicity():
    assert core.exact_match("The New York!", ["NYC", "new york"]) == 1
    assert core.answer_f1("red red blue", ["red blue green"]) == pytest.approx(2 / 3)
    assert core.answer_f1("NYC", ["new york", "NYC"]) == 1
    assert core.answer_f1("", ["present"]) == 0
    assert core.answer_f1("nonempty", []) == 0


def test_mcq_does_not_find_a_letter_inside_explanatory_prose():
    assert core.mcq_correct("Answer: (B).", ["B"]) == 1
    assert core.mcq_correct("A or B might both work", ["B"]) == 0
    assert core.mcq_correct("choice b because...", ["B"]) == 0
    assert core.mcq_correct("Second option", ["B"], {"A": "First option", "B": "Second option"}) == 1


def test_long_passage_tail_is_encoded_and_ranks_as_one_original_unit(tmp_path):
    encoder = TinyEncoder()
    long_text = " ".join([f"word{i}" for i in range(550)] + ["needle"])
    docs = [{"id": "target", "group": "global", "text": long_text},
            {"id": "distractor", "group": "global", "text": "Other unrelated stuff"}]
    index = core.HybridIndex(docs, encoder, tmp_path)
    encoded_words = set(" ".join(encoder.calls).split())
    assert set(long_text.split()) <= encoded_words
    assert all(len(encoder.model.tokenizer.encode(t)) <= core.CONFIG["window_tokens"] for t in encoder.calls)
    result = index.retrieve("needle", budget=4000)
    assert result["ranking"][0] == "target"
    assert len(result["ranking"]) == len(set(result["ranking"])) == 2
    assert "needle" in result["context"]


@pytest.mark.parametrize("query", ["", "the a an", "unseenquery"])
def test_no_lexical_overlap_cannot_outvote_dense_with_zero_score_ties(tmp_path, query):
    docs = [{"id": "target", "text": "needle passage"},
            {"id": "distractor", "text": "unrelated corpus"}]
    index = core.HybridIndex(docs, TinyEncoder(), tmp_path)
    assert index.retrieve(query)["ranking"][0] == "target"


def test_context_budget_counts_headers_and_does_not_credit_partial_gold(tmp_path):
    docs = [{"id": "target", "text": "needle " + "long " * 100,
             "speaker": "Some Speaker", "date": "2020-01-01"}]
    encoder = TinyEncoder()
    index = core.HybridIndex(docs, encoder, tmp_path)
    result = index.retrieve("needle", budget=35)
    assert len(encoder.model.tokenizer.encode(result["context"])) <= 35
    assert result["context_tokens"] <= 35
    assert result["selected"] == ["target"]
    assert result["fully_rendered"] == []
    assert core.retrieval_metrics(result["ranking"], result["selected"], ["target"],
                                  result["fully_rendered"])["evidence_recall_at_budget"] == 0


def test_embedding_cache_reuses_same_text_and_invalidates_changed_content(tmp_path):
    docs = [{"id": "target", "group": "one", "text": "needle original"}]
    first = core.HybridIndex(docs, TinyEncoder(), tmp_path)
    second_encoder = TinyEncoder()
    second = core.HybridIndex(docs, second_encoder, tmp_path)
    assert not first.cache_hit and second.cache_hit
    assert second_encoder.calls == []
    changed = core.HybridIndex([{**docs[0], "text": "needle changed"}], TinyEncoder(), tmp_path)
    assert not changed.cache_hit


def test_reader_prompt_has_no_gold_answers_or_rubric():
    q = {"question": "What happened?", "answers": ["secret reference"], "rubric": "secret rubric"}
    prompt = core.reader_prompt(q, "Retrieved evidence")
    assert "secret reference" not in prompt and "secret rubric" not in prompt
    assert "Retrieved evidence" in prompt


def test_runner_isolates_histories_even_when_document_ids_overlap(tmp_path, monkeypatch):
    from multicard.version_a import runner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "Encoder", lambda **kwargs: TinyEncoder())
    data_dir, out = tmp_path / "dataset", tmp_path / "result"
    docs = [{"id": "shared", "group": "alice", "text": "Alice private needle"},
            {"id": "shared", "group": "bob", "text": "Bob confidential needle"}]
    questions = [{"id": group, "group": group, "question": "needle", "type": "single",
                  "evidence_ids": ["shared"]} for group in ("alice", "bob")]
    for doc in docs:
        core.append_jsonl(data_dir / "corpus.jsonl", doc)
    for q in questions:
        core.append_jsonl(data_dir / "questions.jsonl", q)
    runner.retrieve(data_dir, out)
    rows = {r["id"]: r for r in core.read_jsonl(out / "retrieval.jsonl")}
    assert "Alice private" in rows["alice"]["context"]
    assert "Bob confidential" not in rows["alice"]["context"]
    assert "Bob confidential" in rows["bob"]["context"]
    assert "Alice private" not in rows["bob"]["context"]
    assert rows["alice"]["source_documents"] == rows["bob"]["source_documents"] == 1
    assert rows["alice"]["metrics"]["recall_at_5"] == 1
    def no_rebuild(*args, **kwargs):
        raise AssertionError("Completed histories must be skipped on resume")
    monkeypatch.setattr(runner, "HybridIndex", no_rebuild)
    runner.retrieve(data_dir, out)
    assert len(list(core.read_jsonl(out / "retrieval.jsonl"))) == 2


def test_runner_scores_chunk_hits_against_article_parents(tmp_path, monkeypatch):
    from multicard.version_a import runner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "Encoder", lambda **kwargs: TinyEncoder())
    data_dir, out = tmp_path / "dataset", tmp_path / "result"
    for i in range(2):
        core.append_jsonl(data_dir / "corpus.jsonl", {
            "id": f"article#{i}", "parent_id": "article", "group": "global", "text": f"needle chunk{i}"})
    core.append_jsonl(data_dir / "questions.jsonl", {
        "id": "q1", "group": "global", "question": "needle", "type": "known", "evidence_ids": ["article"]})
    runner.retrieve(data_dir, out)
    result = next(core.read_jsonl(out / "retrieval.jsonl"))
    assert result["metrics"]["recall_at_5"] == 1
    assert result["metrics"]["evidence_recall_at_budget"] == 1

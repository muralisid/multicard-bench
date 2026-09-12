"""Checks for leakage, isolated histories and reproducible benchmark labels."""
from pathlib import Path
import json

import pytest

from multicard.version_a.conversation_data import (
    _download,
    _write_dataset,
    locomo_official_score,
    locomo_records,
    persona_history_records,
    persona_question,
)


def locomo_sample(group="c1"):
    return {"sample_id": group,
            "conversation": {"speaker_a": "Sam", "speaker_b": "Alex",
                             "session_1_date_time": "1:56 pm on 8 May, 2023",
                             "session_1": [{"dia_id": "D1:1", "speaker": "Sam", "text": "I moved yesterday."},
                                           {"dia_id": "D1:2", "speaker": "Alex", "text": "Here is my dog.", "blip_caption": "a brown dog"}]},
            "observation": {"secret": "annotation should not enter corpus"},
            "qa": [{"question": "When?", "answer": "7 May", "category": 2, "evidence": ["D1:1; D1:2"]},
                   {"question": "Where did Sam go swimming?", "adversarial_answer": "pool", "category": 5, "evidence": ["D1:2"]}]}


def test_locomo_corpus_is_source_only_and_conversations_are_isolated():
    corpus, questions, diagnostics = locomo_records([locomo_sample(), locomo_sample("c2")])
    assert len({record["id"] for record in corpus}) == 4
    assert corpus[0]["date"] == "2023-05-08T13:56:00"
    assert "Image caption: a brown dog" in corpus[1]["text"]
    assert "annotation" not in json.dumps(corpus)
    assert "7 May" not in json.dumps(corpus)
    assert questions[0]["type"] == "temporal"
    assert questions[0]["evidence_ids"] == ["c1:D1:1", "c1:D1:2"]
    assert questions[1]["answers"] == ["No information available"]
    assert not questions[1]["headline_eligible"]
    assert questions[1]["evidence_ids"] == []
    assert diagnostics["caption_turns"] == 2


def test_missing_source_evidence_is_disclosed_not_fabricated():
    sample = locomo_sample()
    sample["qa"][0]["evidence"] = ["D1:1", "D9:99", "D"]
    _, questions, diagnostics = locomo_records([sample])
    assert questions[0]["evidence_ids"] == ["c1:D1:1"]
    assert not questions[0]["evidence_complete"]
    assert questions[0]["evidence_unresolved"] == ["D9:99", "D"]
    assert diagnostics["unresolved_evidence"][0]["question_id"] == "c1:q0000"


def persona_row():
    return {"persona_id": "7", "user_query": "{'role': 'user', 'content': 'Where should I eat?'}",
            "correct_answer": "Vegetarian cafe", "incorrect_answers": '["Steakhouse", "Seafood", "Barbecue"]',
            "pref_type": "neutral_preferences", "preference": "vegetarian", "related_conversation_snippet": "gold snippet",
            "updated": "True", "chat_history_32k_link": "data/chat_history_32k/persona7.json"}


def test_persona_options_reproducible_and_gold_metadata_not_corpus():
    question = persona_question(persona_row(), 6, "32k")
    again = persona_question(persona_row(), 6, "128k")
    assert question["options"] == again["options"]
    assert question["options"][question["answers"][0]] == "Vegetarian cafe"
    assert question["reference_answers"] == ["Vegetarian cafe"]
    assert question["metadata"]["related_conversation_snippet"] == "gold snippet"
    history = {"metadata": {"gold": "gold snippet"}, "chat_history": [
        {"role": "system", "content": "Provided source persona."},
        {"role": "user", "content": "I enjoy tofu."}]}
    corpus = persona_history_records(history, question["group"])
    assert [x["speaker"] for x in corpus] == ["system", "user"]
    assert "gold snippet" not in json.dumps(corpus)
    assert "Vegetarian cafe" not in json.dumps(corpus)


def test_corrupt_source_cache_rejected_before_network(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("wrong")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _download("https://unused.invalid", path, "0" * 64)


def test_dataset_integrity_checks(tmp_path):
    corpus, questions, _ = locomo_records([locomo_sample()])
    manifest = _write_dataset(tmp_path / "ok", corpus, questions, {})
    assert manifest["groups"] == 1
    assert manifest["headline_questions"] == 1
    assert set(manifest["output_sha256"]) == {"corpus.jsonl", "questions.jsonl"}
    with pytest.raises(ValueError, match="Duplicate corpus"):
        _write_dataset(tmp_path / "bad", corpus + corpus, questions, {})
    questions[0]["group"] = "absent"
    with pytest.raises(ValueError, match="Invalid/duplicate question"):
        _write_dataset(tmp_path / "bad_group", corpus, questions, {})


def test_official_locomo_category_specific_f1():
    pytest.importorskip("nltk")
    assert locomo_official_score("Dogs", "dog", 4) == 1.0
    assert locomo_official_score("apples", "apples, oranges", 1) == 0.5
    assert locomo_official_score("Yes", "Yes; extra annotation", 3) == 1.0
    assert locomo_official_score("Not mentioned", "anything", 5) == 1.0
    assert locomo_official_score("At the pool", "anything", 5) == 0.0

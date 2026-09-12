import copy
import json

import pytest

from multicard.version_a.scale_data import beam_records, fact_records, word_chunks, write_dataset
from multicard.version_a.scale_scoring import (
    beam_event_ordering,
    beam_rubric_mean,
    beam_rubric_prompts,
    evaluate_beam,
    parse_beam_rubric,
    substring_exact_match,
)


def beam_row():
    return {
        "conversation_id": "7",
        "chat": [[
            {"id": 0, "role": "user", "content": "I live in Delhi.", "time_anchor": "January-01-2024"},
            {"id": 1, "role": "assistant", "content": "Understood.", "time_anchor": None},
            {"id": 2, "role": "user", "content": "I moved to Chennai.", "time_anchor": "February-01-2024"},
        ]],
        "user_profile": "UNSEEN PROFILE MUST NOT ENTER MEMORY",
        "conversation_plan": "UNSEEN PLAN MUST NOT ENTER MEMORY",
        "probing_questions": repr({"knowledge_update": [{
            "question": "Where do I live?", "answer": "Chennai", "rubric": ["Must state Chennai"],
            "source_chat_ids": {"original_info": [0], "updated_info": [2, 999]},
        }]}),
    }


def test_beam_keeps_chronology_and_never_indexes_evaluation_material():
    row = beam_row()
    corpus, questions = beam_records(row, "beam_100k")
    assert [x["text"] for x in corpus] == ["I live in Delhi.", "Understood.", "I moved to Chennai."]
    assert corpus[1]["date"] == "January-01-2024"
    assert corpus[2]["date"] == "February-01-2024"
    assert questions[0]["evidence_ids"] == [corpus[0]["id"], corpus[2]["id"]]
    assert questions[0]["unresolved_source_ids"] == ["999"]
    changed = copy.deepcopy(row)
    changed["probing_questions"] = repr({"abstention": [{"question": "Different?", "ideal_response": "GOLD SECRET",
                                                        "rubric": ["DIFFERENT GOLD"]}]})
    assert beam_records(changed, "beam_100k")[0] == corpus


def test_beam_10m_nullable_plan_struct_preserves_numeric_order():
    row = beam_row()
    messages = row["chat"][0]
    row["chat"] = [{"plan-10": [{"turns": [[messages[2]]]}], "plan-2": [{"turns": [[messages[1]]]}],
                    "plan-1": [{"turns": [[messages[0]]]}], "plan-3": None}]
    corpus, _ = beam_records(row, "beam_10m")
    assert [x["source_id"] for x in corpus] == ["0", "1", "2"]
    assert all(x["group"] == "beam_10m:7" for x in corpus)


def test_fact_consolidation_retains_old_and_counterfactual_facts():
    row = {"context": "Here is a list of facts:\n0. Alice works in London.\n100. Alice works in Mars.",
           "questions": ["Where does Alice work?"], "answers": [["Mars"]],
           "metadata": {"source": "factconsolidation_sh_32k", "qa_pair_ids": ["fact-question-1"]}}
    corpus, questions = fact_records(row)
    assert corpus[0]["text"] == row["context"]
    assert "real-world" in questions[0]["reader_instruction"]
    assert questions[0]["metric"] == "substring_exact_match"
    assert questions[0]["evidence_ids"] == []
    row["answers"] = [["A changed gold answer"]]
    assert fact_records(row)[0] == corpus


def test_chunk_boundaries_preserve_all_words_and_line_breaks():
    text = "\n".join(f"{i}. word{i}" for i in range(350))
    chunks = list(word_chunks(text))
    assert max(len(x.split()) for x in chunks) <= 200
    assert all(f"word{i}" in " ".join(chunks) for i in range(350))
    assert "\n" in chunks[0]
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]
    assert list(word_chunks("")) == []
    with pytest.raises(ValueError):
        list(word_chunks("text", size=20, overlap=20))


def test_manifest_counts_every_question_and_tracks_missing_evidence(tmp_path):
    manifest = write_dataset("beam_test", [beam_row()], lambda r: beam_records(r, "beam_test"),
                             tmp_path, [], ["fixture"])
    assert manifest["counts"]["questions"] == 1
    assert manifest["counts"]["corpus_records"] == 3
    assert manifest["counts"]["unresolved_source_ids"] == 1
    loaded = json.loads((tmp_path / "beam_test" / "manifest.json").read_text())
    assert loaded["output_sha256"] == manifest["output_sha256"]


def test_official_substring_scoring_differs_from_f1():
    assert substring_exact_match("The final answer is: London!", ["Paris", "London"]) == 1
    assert substring_exact_match("The city is not specified", ["London"]) == 0
    assert substring_exact_match("A New-York office", ["newyork"]) == 1


def test_rubric_average_requires_all_valid_items():
    question = {"rubric": ["Must mention March 29", "Must decline unknown claim"]}
    prompts = beam_rubric_prompts(question, "March 29. I do not know the other detail.")
    assert len(prompts) == 2
    assert "Must mention March 29" in prompts[0]
    assert "<llm_response>" not in prompts[0]
    parsed = parse_beam_rubric('```json\n{"score": 0.5, "reason": "partial"}\n```')
    assert beam_rubric_mean([parsed, {"score": 1}], 2) == 0.75
    with pytest.raises(ValueError):
        beam_rubric_mean([parsed], 2)
    for invalid in ["true", "0.7", '"1"', "NaN"]:
        with pytest.raises(ValueError):
            parse_beam_rubric('{"score":' + invalid + '}')


def test_event_ordering_penalizes_reverse_and_extra_events():
    reference = ["first", "second", "third"]
    assert beam_event_ordering(reference, reference)["final_score"] == 1
    assert beam_event_ordering(reference, list(reversed(reference)))["final_score"] == 0
    assert beam_event_ordering(reference, reference + ["hallucinated"])["final_score"] < 1
    assert beam_event_ordering(reference, ["wrong"])["f1"] == 0


class FakeJudge:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, prompt, max_output_tokens, role):
        assert role == "judge"
        assert max_output_tokens in (64, 512)
        self.calls.append(prompt)
        return {"text": next(self.outputs), "usd": 0.01, "cached": False,
                "tokens_in": 10, "tokens_out": 2}


def test_beam_evaluation_keeps_graded_score_without_binary_threshold():
    judge = FakeJudge(['{"score": 0.5}', '{"score": 1}'])
    result = evaluate_beam({"type": "knowledge_update", "rubric": ["one", "two"]}, "answer", judge)
    assert result["score"] == 0.75
    assert len(result["calls"]) == 2
    assert result["supplementary"] == {}
    assert "correct" not in result


def test_beam_event_alignment_uses_first_unmatched_reference_and_accounts_all_calls():
    # Candidate C matches reference second. Candidate A then matches first.
    judge = FakeJudge(['{"score": 1}', '{"score": 1}', "NO", "YES", "YES"])
    result = evaluate_beam({"type": "event_ordering", "rubric": ["first", "second"]},
                           "C\nA", judge)
    assert result["score"] == 1
    assert result["supplementary"]["aligned_prediction"] == ["second", "first"]
    assert result["supplementary"]["final_score"] == 0
    assert len(result["calls"]) == 5
    assert sum(item["usd"] for item in result["calls"]) == pytest.approx(0.05)


def test_beam_event_malformed_equivalence_cannot_be_silently_scored():
    judge = FakeJudge(['{"score": 1}', "MAYBE"])
    with pytest.raises(ValueError, match="equivalence"):
        evaluate_beam({"type": "event_ordering", "rubric": ["first"]}, "answer", judge)


def test_reader_prompt_includes_counterfactual_instruction():
    from multicard.version_a.core import reader_prompt
    from multicard.version_a.scale_data import FACT_INSTRUCTION
    prompt = reader_prompt({"question": "Where?", "metric": "substring_exact_match",
                            "reader_instruction": FACT_INSTRUCTION}, "0. A lives in London.")
    assert FACT_INSTRUCTION in prompt
    assert "larger serial number" in prompt

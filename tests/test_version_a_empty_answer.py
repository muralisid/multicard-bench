"""An empty billed response stays in the quality and cost denominators."""
from types import SimpleNamespace

import pytest

from multicard.version_a.runner import evaluate_one


@pytest.mark.parametrize("metric", ["f1", "mcq", "substring_exact_match", "judge", "beam_rubric_mean"])
def test_empty_answer_is_zero_without_retry_or_judge(metric):
    calls = []

    def generate(prompt, allowance, role):
        calls.append((allowance, role))
        return {"text": "", "key": "original-call", "cached": True,
                "tokens_in": 300, "tokens_out": 768, "usd": .004, "seconds": 0}

    def forbidden(*args, **kwargs):
        raise AssertionError("An empty answer must not consume a judge call")

    q = {"id": "q", "type": "test", "question": "Where?", "answers": ["Rome"], "metric": metric}
    result = evaluate_one(q, {"context": "Evidence", "context_tokens": 10},
                          SimpleNamespace(model="reader", generate=generate),
                          SimpleNamespace(model="judge", generate=forbidden))
    assert calls == [(768, "answer")]
    assert result["score"] == 0
    assert result["correct"] == (None if metric == "beam_rubric_mean" else 0)
    assert result["supplementary"] == {"empty_answer": True}
    assert result["answer_usd"] == .004
    assert result["incremental_answer_usd"] == 0
    assert result["answer_key"] == "original-call"
    assert result["judge_usd"] == 0

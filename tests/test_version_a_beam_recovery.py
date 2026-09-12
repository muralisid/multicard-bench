import json

import pytest

from multicard.version_a.recover_beam import repair_json_escapes
from multicard.version_a.scale_scoring import parse_beam_rubric


def test_illegal_reason_escape_preserves_score_and_literal_content():
    text = r'''{"score": 0.5, "reason": "The phrase \'missing fact\' is incomplete."}'''
    with pytest.raises(json.JSONDecodeError):
        parse_beam_rubric(text)
    result = parse_beam_rubric(repair_json_escapes(text))
    assert result["score"] == .5
    assert "missing fact" in result["reason"]


def test_valid_json_is_byte_identical_and_invalid_scores_still_fail():
    text = json.dumps({"score": 1, "reason": 'A quote " and a newline\n and slash \\.'})
    assert repair_json_escapes(text) == text
    with pytest.raises(ValueError):
        parse_beam_rubric(repair_json_escapes(r'''{"score": 7, "reason": "bad \'score\'"}'''))
    with pytest.raises(json.JSONDecodeError):
        parse_beam_rubric(repair_json_escapes('{"score": 1, "reason": "truncated'))

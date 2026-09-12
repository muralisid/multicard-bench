"""Resume frozen BCD requests with strict recovery of a malformed reason string."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

from multicard.version_a import scale_scoring
from multicard.version_a.core import append_jsonl
from multicard.version_a.recover_beam import repair_json_escapes
from multicard.version_abcd import runner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/version_abcd"
ORIGINAL = scale_scoring.parse_beam_rubric


def recover(response):
    try:
        return ORIGINAL(repair_json_escapes(response)), False
    except json.JSONDecodeError:
        text = response.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence:
            text = fence[1]
        # The score must be the sole leading numeric field. Only reason is
        # treated as opaque text. Never infer a score from prose or resubmit it.
        match = re.fullmatch(r'\{\s*"score"\s*:\s*(0(?:\.0)?|0\.5|1(?:\.0)?)\s*,\s*"reason"\s*:\s*"(.*)"\s*\}', text, re.DOTALL)
        if not match or re.search(r'"score"\s*:', match[2]):
            raise
        payload = {"score": float(match[1]), "reason": match[2]}
        return ORIGINAL(json.dumps(payload)), True


def self_test():
    for score in ("0.0", "0.5", "1.0"):
        raw = '{"score": ' + score + ', "reason": "The response says "quoted words" here."}'
        parsed, changed = recover(raw)
        assert parsed["score"] == float(score) and changed
    valid = '{"score": 0.5, "reason": "ordinary reason"}'
    assert recover(valid) == (ORIGINAL(valid), False)
    for bad in ('{"score": 2, "reason": "bad "quote""}',
                '{"score": "0.5", "reason": "bad "quote""}',
                '{"score": 0, "reason": "bad "quote", "score": 1}',
                'The score should be 1.0', 'NO', '{"reason":"text", "score":true}'):
        try:
            recover(bad)
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError("Ambiguous or invalid score accepted")
    print("Recovery checks passed", flush=True)


def main():
    self_test()
    if "--self-test" in sys.argv:
        return
    if (ROOT / "FREEZE").exists():
        raise RuntimeError("Repository FREEZE kill switch")
    protocol = {
        "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_run_sha256": hashlib.sha256((OUT / "run.json").read_bytes()).hexdigest(),
        "recovery": "For malformed JSON reason text only, preserve the sole leading score in {0,0.5,1}. Reject duplicate score fields and ambiguous forms.",
        "unchanged": "Frozen package code, request prompts, model parameters, samples, contexts, raw cached responses and usage ledgers.",
        "quality_retry": False,
    }
    path = OUT / "JUDGE-JSON-RECOVERY.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2) + "\n")
        (OUT / "campaign_before_judge_recovery.json").write_bytes((OUT / "campaign.json").read_bytes())

    def parse(response):
        parsed, changed = recover(response)
        if changed:
            append_jsonl(OUT / "judge_reason_repairs.jsonl", {
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "score": parsed["score"], "recovery": "preserve_numeric_score_and_opaque_reason"})
        return parsed

    scale_scoring.parse_beam_rubric = parse
    runner.main()


if __name__ == "__main__":
    main()

"""Bounded BEAM transport recovery without changing frozen model requests.

The overlay is recorded separately from the original runner identity. Original
cached response text, usage and scores are never edited. Only illegal JSON
backslash escapes are made literal; score validation remains unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from . import client, runner, scale_scoring


def repair_json_escapes(text):
    pieces, index = [], 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            following = text[index + 1]
            if following in '"\\/bfnrtu':
                pieces.append(text[index:index + 2])
                index += 2
                continue
            pieces.append("\\\\")
        else:
            pieces.append(char)
        index += 1
    return "".join(pieces)


def install_overlay(output, dataset):
    original_generate = client.ProxyClient.generate
    original_parse = scale_scoring.parse_beam_rubric

    def generate(self, *args, **kwargs):
        deadline = time.monotonic() + 150
        while True:
            try:
                return original_generate(self, *args, **kwargs)
            except RuntimeError as exc:
                prefix = "Unresolved prior API attempt; reserved cost retained: "
                if not str(exc).startswith(prefix):
                    raise
                key = str(exc)[len(prefix):]
                with self.connect() as connection:
                    row = connection.execute("SELECT status FROM calls WHERE key=?", (key,)).fetchone()
                if not row or row[0] not in {"pending", "done"} or time.monotonic() >= deadline:
                    raise
                # The original request owns its reservation. Wait for its cache
                # entry instead of issuing a second request or releasing a hold.
                time.sleep(.25)

    def parse(text):
        try:
            return original_parse(text)
        except json.JSONDecodeError:
            repaired = repair_json_escapes(text)
            if repaired == text:
                raise
            result = original_parse(repaired)
            runner.append_jsonl(output / dataset / "json_escape_repairs.jsonl", {
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "score": result["score"], "repair": "make_illegal_backslash_escapes_literal"})
            return result

    client.ProxyClient.generate = generate
    scale_scoring.parse_beam_rubric = parse


def main():
    # Use the runner's existing command line and identity checks unchanged.
    args = sys.argv[1:]
    dataset = args[args.index("--datasets") + 1]
    if not dataset.startswith("beam_") or "," in dataset:
        raise ValueError("Recovery requires exactly one BEAM dataset")
    output = Path(args[args.index("--output") + 1]) if "--output" in args else Path("results/version_a")
    protocol = {
        "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_code_sha256": {name: runner.sha256_file(Path(__file__).parent / name)
                             for name in ("runner.py", "client.py", "scale_scoring.py", "core.py")},
        "changes": ["Wait up to 150 seconds for an identical in-flight call; never submit a duplicate.",
                    "Parse illegal backslash escapes as literal characters; preserve numeric rubric scores."],
        "unchanged": "Model requests, prompts, token caps, datasets, retrieval, cached responses and cost accounting",
    }
    path = output / dataset / "recovery_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Recovery overlay changed; preserve and review the previous protocol first")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    install_overlay(output, dataset)
    runner.main()


if __name__ == "__main__":
    main()

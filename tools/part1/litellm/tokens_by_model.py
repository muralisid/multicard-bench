"""Sum token counts and cost per model from the proxy request log.
Usage: .venv/bin/python tokens_by_model.py [path/to/requests.jsonl] [--by-tag]
                                          [--since 2026-09-06T12:00:00]

--by-tag groups by job_tag (rows without one land under "untagged").
--since keeps rows whose ts is at or after the given local time prefix
(string comparison on the ts field, so any prefix of the ts format works).
"""
import json
import sys
from collections import defaultdict


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    by_tag = "--by-tag" in argv
    since = None
    if "--since" in argv:
        since = argv[argv.index("--since") + 1]
        args = [a for a in args if a != since]
    path = args[0] if args else \
        "/Users/muralisid/github_other/part1-tools/env/requests.jsonl"
    agg = defaultdict(lambda: {"calls": 0, "failures": 0, "prompt_tokens": 0,
                               "completion_tokens": 0, "cost_usd": 0.0})
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if since and (r.get("ts") or "") < since:
                continue
            if by_tag:
                key = r.get("job_tag") or "untagged"
            else:
                key = r.get("model") or "?"
            a = agg[key]
            a["calls"] += 1
            if r.get("status") != "success":
                a["failures"] += 1
            a["prompt_tokens"] += r.get("prompt_tokens") or 0
            a["completion_tokens"] += r.get("completion_tokens") or 0
            a["cost_usd"] += float(r.get("response_cost") or 0.0)
    print(json.dumps(agg, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])

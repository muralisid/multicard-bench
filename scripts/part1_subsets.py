#!/usr/bin/env python3
"""Write the fixed Part 1 subsets (docs/PART1-DESIGN.md version 4, section 2).

Reads the question ids and types through the bench loaders, draws every
subset with multicard.part1.subsets at seed 13, builds the payload twice and
checks the two serialisations are byte-identical, then writes
docs/part1/subsets.json and its sha256 to docs/part1/subsets.sha256 and
prints the sha256 and the counts. No model is called.

Run from the repo:
  uv run python scripts/part1_subsets.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multicard.part1 import subsets  # noqa: E402
from multicard.part1.subsets import (  # noqa: E402
    CHANDAN_OWN_ARMS, READER_A_ARMS, READER_B_ARMS, SEED,
)

ARMS = {"reader_a": READER_A_ARMS, "reader_b": READER_B_ARMS, "chandan_own": CHANDAN_OWN_ARMS}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--longmemeval",
                    default=os.environ.get("LONGMEMEVAL_S", str(ROOT / "data/raw/longmemeval_s.json")))
    ap.add_argument("--multihoprag", default=str(ROOT / "data/raw/multihoprag"))
    ap.add_argument("--out", default=str(ROOT / "docs/part1/subsets.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    lme = subsets.longmemeval_records(args.longmemeval)
    mh = subsets.multihoprag_records(args.multihoprag)

    first = subsets.serialise(subsets.build(lme, mh, args.seed))
    second = subsets.serialise(subsets.build(lme, mh, args.seed))
    if first != second:
        raise SystemExit("two builds of the subsets differ; the draw is not deterministic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(first)
    digest = subsets.sha256(first)
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n")

    payload = json.loads(first)
    print(f"sha256 {digest}")
    print(f"wrote {out} ({len(first)} bytes) and {out.with_suffix('.sha256')}")
    print(json.dumps(subsets.counts(payload), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

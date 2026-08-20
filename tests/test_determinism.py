"""The reproducibility statement claims two seeded runs produce identical
output. That claim was previously true in practice but unchecked in the
repository, which the review correctly called out. This is the check.

It runs a small experiment twice in separate processes and compares the recorded
output byte for byte. Separate processes matter: in-process reuse would hide
exactly the cross-run float variation that made this hard to achieve.
"""

import subprocess
import sys
from pathlib import Path


def test_two_seeded_runs_are_byte_identical(tmp_path):
    outputs = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        code = (
            "from multicard.experiments.e0_dilution import run, KS;"
            "KS.clear(); KS.extend([2, 3]);"
            f"run(n_docs=40, queries_per_k=15, seed=13, out_dir={str(d)!r})"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, f"run {i} failed: {r.stderr[-800:]}"
        outputs.append((d / "per_query.csv").read_bytes())

    assert outputs[0] == outputs[1], (
        "two seeded runs produced different per-query output; determinism is the "
        "property the reproducibility statement rests on"
    )
    assert len(outputs[0]) > 0

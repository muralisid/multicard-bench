"""Single entrypoint: uv run mcb run <experiment_id>.

Threading is pinned to one BLAS thread before numpy or torch is imported.
Multithreaded reductions vary their summation order between runs, which moves
float32 similarities by about 1e-8 and makes recorded outputs differ even under
a fixed seed. The matmuls here are small, so the cost is negligible and the
reward is bitwise reproducibility. Override with MCB_THREADS if speed matters
more than an identical rerun.
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("MCB_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

REGISTRY = {
    "e0_dilution": ("multicard.experiments.e0_dilution", "run"),
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "nogit"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mcb")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a registered experiment")
    r.add_argument("experiment", choices=sorted(REGISTRY))
    r.add_argument("--seed", type=int, default=13)
    r.add_argument("--n-docs", type=int, default=500)
    r.add_argument("--queries-per-k", type=int, default=200)
    r.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = p.parse_args(argv)

    mod_name, fn_name = REGISTRY[a.experiment]
    mod = __import__(mod_name, fromlist=[fn_name])
    try:
        import torch
        torch.set_num_threads(int(_THREADS))
    except ImportError:
        pass
    print(f"[mcb] {a.experiment} at {git_sha()} seed={a.seed} threads={_THREADS}")
    fn = getattr(mod, fn_name)
    fn(n_docs=a.n_docs, queries_per_k=a.queries_per_k, seed=a.seed, model=a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

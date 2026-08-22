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
import sys

# Python randomises string hashing per process, so any iteration over a set of
# strings comes out in a different order each run. Setting PYTHONHASHSEED from
# inside the process (as utils.seeds does) cannot affect the interpreter that is
# already running; it has to be in the environment before start-up. Re-exec once
# with it fixed so that seeded sampling over such orders is reproducible.
if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "13"
    os.execv(sys.executable, [sys.executable, "-m", "multicard.run", *sys.argv[1:]])

_THREADS = os.environ.get("MCB_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse  # noqa: E402
import subprocess  # noqa: E402

REGISTRY = {
    "e0_dilution": ("multicard.experiments.e0_dilution", "run"),
    "e1_limit": ("multicard.experiments.e1_limit", "run"),
    "e1_scifact": ("multicard.experiments.e1_beir", "run"),
    "e1_nfcorpus": ("multicard.experiments.e1_beir", "run"),
    "e2_economics": ("multicard.experiments.e2_economics", "run"),
    "e3_diversity": ("multicard.experiments.e3_diversity", "run"),
    "e2_gate": ("multicard.experiments.e2_gate", "run"),
    "e1_anchor_sensitivity": ("multicard.experiments.e1_anchor_sensitivity", "run"),
    "e3b_consumer": ("multicard.experiments.e3b_consumer", "run"),
    "e1_llm_taxonomy": ("multicard.experiments.e1_llm_taxonomy", "run"),
    "e_dyn_all": ("multicard.experiments.e_dyn", "run"),
    "e_dyn2": ("multicard.experiments.e_dyn2", "run"),
    "e_dyn3": ("multicard.experiments.e_dyn3", "run"),
}


# experiment id -> BEIR collection name
BEIR_COLLECTIONS = {
    "e1_scifact": "scifact",
    "e1_nfcorpus": "nfcorpus",
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
    # The encoder may use every core. Verified on this project: transformer
    # encoding is bit-identical regardless of thread count, so threading it costs
    # nothing in reproducibility and saves most of the wall clock. Determinism of
    # the scoring step is handled where it actually breaks, by accumulating the
    # similarity products in double precision.
    try:
        import torch
        torch.set_num_threads(int(os.environ.get(
            "MCB_TORCH_THREADS", max(1, (os.cpu_count() or 2) // 2))))
    except ImportError:
        pass
    print(f"[mcb] {a.experiment} at {git_sha()} seed={a.seed} threads={_THREADS}")
    fn = getattr(mod, fn_name)
    kwargs = dict(n_docs=a.n_docs, queries_per_k=a.queries_per_k, seed=a.seed,
                  model=a.model)
    # BEIR collections share one entrypoint and take the collection name from the
    # experiment id. Listing them explicitly rather than inferring from the "e1_"
    # prefix, which silently passed a bogus collection to any new e1 experiment.
    if a.experiment in BEIR_COLLECTIONS:
        kwargs["collection"] = BEIR_COLLECTIONS[a.experiment]
    fn(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

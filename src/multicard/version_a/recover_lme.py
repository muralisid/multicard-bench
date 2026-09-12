"""One-off recovery of the repaired LME data, sharing the campaign budget."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/version_a"
STATE = OUT / "longmemeval_recovery.json"


def write(status, **fields):
    STATE.write_text(json.dumps({"pid": os.getpid(), "time": time.time(), "status": status, **fields}, indent=2))


def campaign():
    if (ROOT / "FREEZE").exists():
        raise RuntimeError("FREEZE: recovery stopped")
    x = json.loads((OUT / "campaign.json").read_text())
    if x.get("status") == "stopped":
        raise RuntimeError("Main campaign was stopped; recovery stopped")
    return x


def run(stage):
    write("running", stage=stage)
    args = [sys.executable, "-u", "-m", "multicard.version_a.runner", "--datasets", "longmemeval_s",
            "--stage", stage, "--output", str(OUT), "--workers", "4", "--max-usd", "150", "--run-max-usd", "40"]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false")
    with (OUT / "logs" / f"longmemeval_s_recovery_{stage}.log").open("a") as log:
        subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    write("waiting_for_indexer")
    while True:
        x = campaign()
        other = [v for k, v in x["datasets"].items() if k != "longmemeval_s"]
        if all(v.get("retrieve", {}).get("status") in ("complete", "failed", "skipped") for v in other):
            break
        if x["status"] != "running":
            raise RuntimeError("Main campaign ended before remaining indexes finished")
        time.sleep(30)
    run("retrieve")
    write("waiting_for_qa")
    while campaign()["status"] == "running":
        time.sleep(30)
    run("qa")
    run("report")
    x = campaign()
    for stage in ("retrieve", "qa"):
        x["datasets"]["longmemeval_s"][stage] = {"status": "complete", "exit_code": 0, "recovery": str(STATE)}
    if all(v.get("qa", {}).get("status") == "complete" for v in x["datasets"].values()):
        x["status"] = "complete"
    temp = OUT / "campaign.recovery.tmp"
    temp.write_text(json.dumps(x, indent=2))
    temp.replace(OUT / "campaign.json")
    write("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write("failed", error=str(exc))
        raise

"""Finish failed BEAM QA while the original local indexing campaign continues."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .suite import REPO


def main():
    output = REPO / "results/version_a"
    path = output / "beam_recovery.json"
    with (output / "beam_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "status": "running", "datasets": {}}

        def save():
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(path)

        attempted = set()
        save()
        while not (REPO / "FREEZE").exists():
            campaign = json.loads((output / "campaign.json").read_text())
            for name, stages in campaign["datasets"].items():
                if not name.startswith("beam_") or name in attempted:
                    continue
                if stages.get("retrieve", {}).get("status") != "complete":
                    continue
                if stages.get("qa", {}).get("status") != "failed":
                    continue
                attempted.add(name)
                command = [sys.executable, "-m", "multicard.version_a.recover_beam",
                           "--datasets", name, "--stage", "qa", "--workers", "16",
                           "--output", str(output), "--data", str(REPO / "data/version_a"),
                           "--max-usd", "150", "--run-max-usd", "40"]
                log = output / "logs" / f"{name}_recovery.log"
                state["datasets"][name] = {"status": "running", "command": command, "log": str(log)}
                save()
                with log.open("a") as handle:
                    code = subprocess.call(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                state["datasets"][name].update(status="complete" if code == 0 else "failed", exit_code=code)
                save()
            if campaign["status"] != "running":
                # All original jobs have stopped; reconcile campaign status by
                # rerunning its normal entrypoint only if every metric is full.
                complete = all((output / name / "metrics.json").exists() and
                               json.loads((output / name / "metrics.json").read_text()).get("status") == "complete"
                               for name in campaign["order"])
                if complete:
                    with (output / "campaign.lock").open("a") as main_lock:
                        fcntl.flock(main_lock, fcntl.LOCK_EX)
                    with (output / "logs/recovery_reconciliation.log").open("a") as handle:
                        code = subprocess.call([sys.executable, "-m", "multicard.version_a.suite",
                                                "--workers", "16", "--index-workers", "2",
                                                "--datasets", ",".join(campaign["order"])],
                                               cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                    state["status"] = "complete" if code == 0 else "incomplete"
                else:
                    state["status"] = "incomplete"
                save()
                return
            time.sleep(30)
        state["status"] = "stopped_by_FREEZE"
        save()


if __name__ == "__main__":
    main()

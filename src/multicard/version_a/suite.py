"""Run Version A with bounded local indexers and one metered QA job.

No scheduler is installed. The process can run under nohup, and rerunning it
resumes the runner's per-question artifacts and persistent shared cost meter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


REPO = Path(__file__).resolve().parents[3]
DATASETS = (
    "locomo", "musique", "2wikimultihopqa", "longmemeval_s",
    "longmemeval_oracle", "multihoprag",
    *(f"factconsolidation_{hop}_{size}" for size in ("6k", "32k", "64k", "262k")
      for hop in ("sh", "mh")),
    "personamem_v2_32k", "personamem_v2_128k",
    "beam_100k", "beam_500k", "beam_1m", "beam_10m",
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def command(args, dataset, stage):
    result = [sys.executable, "-m", "multicard.version_a.runner", "--datasets", dataset,
              "--data", str(args.data), "--output", str(args.output), "--stage", stage,
              "--reader", args.reader, "--judge", args.judge,
              "--workers", str(args.workers), "--max-usd", str(args.max_usd),
              "--run-max-usd", str(args.run_max_usd)]
    if args.limit is not None:
        result += ["--limit", str(args.limit)]
    return result


def budget_failure(log_text):
    return "BudgetExceeded" in log_text or "Reservation would exceed budget" in log_text


class Campaign:
    def __init__(self, args, names):
        self.args, self.names = args, names
        self.lock = threading.RLock()
        self.paid_stopped = threading.Event()
        self.stopped = threading.Event()
        self.processes = {}
        self.path = args.output / "campaign.json"
        self.state = {"schema_version": 1, "pid": os.getpid(), "started_at": now(),
                      "updated_at": now(), "status": "running", "paid_jobs_blocked": False,
                      "budget_usd": args.max_usd, "per_variant_budget_usd": args.run_max_usd,
                      "index_workers": args.index_workers, "qa_workers": args.workers,
                      "data": str(args.data), "output": str(args.output),
                      "order": names, "datasets": {name: {} for name in names}}

    def save(self):
        with self.lock:
            self.state["updated_at"] = now()
            temporary = self.path.with_suffix(".json.partial")
            temporary.write_text(json.dumps(self.state, indent=2) + "\n")
            temporary.replace(self.path)

    def mark(self, name, stage, **fields):
        with self.lock:
            self.state["datasets"][name].setdefault(stage, {}).update(fields)
            self.save()

    def stop(self):
        self.stopped.set()
        # RLock allows a signal interrupting the main thread to call stop.
        with self.lock:
            processes = list(self.processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def stage(self, name, stage):
        if self.stopped.is_set() or (stage == "qa" and self.paid_stopped.is_set()):
            self.mark(name, stage, status="skipped", reason="campaign_stopped" if self.stopped.is_set()
                      else "paid_budget_blocked", finished_at=now(), exit_code=None)
            return None
        if (REPO / "FREEZE").exists():
            self.stop()
            self.mark(name, stage, status="skipped", reason="FREEZE", finished_at=now(), exit_code=None)
            return None
        logfile = self.args.output / "logs" / f"{name}_{stage}.log"
        cmd = command(self.args, name, stage)
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        self.mark(name, stage, status="running", started_at=now(), command=cmd, log=str(logfile))
        process, code = None, 127
        with logfile.open("a+") as handle:
            handle.seek(0, os.SEEK_END)
            offset = handle.tell()
            handle.write(json.dumps({"started_at": now(), "command": cmd}) + "\n")
            handle.flush()
            try:
                process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
                with self.lock:
                    self.processes[(name, stage)] = process
                self.mark(name, stage, child_pid=process.pid)
                if self.stopped.is_set():
                    process.terminate()
                code = process.wait()
            except OSError as exc:
                handle.write(f"Launch failed: {exc}\n")
            finally:
                with self.lock:
                    self.processes.pop((name, stage), None)
            handle.flush()
            handle.seek(offset)
            failed_budget = code != 0 and budget_failure(handle.read())
        if failed_budget:
            self.paid_stopped.set()
            with self.lock:
                self.state["paid_jobs_blocked"] = True
                self.state["paid_stop_reason"] = {"dataset": name, "stage": stage, "log": str(logfile)}
        self.mark(name, stage, status="complete" if code == 0 else "failed",
                  finished_at=now(), exit_code=code, budget_failure=failed_budget)
        print(json.dumps({"dataset": name, "stage": stage, "exit_code": code,
                          "paid_jobs_blocked": self.paid_stopped.is_set()}), flush=True)
        return code

    def run(self):
        self.save()
        futures = []
        try:
            # Submit no more index futures than there are local worker slots.
            # QA starts as soon as any index finishes, including out of order.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="version-a-qa") as pool, \
                 ThreadPoolExecutor(max_workers=self.args.index_workers,
                                    thread_name_prefix="version-a-index") as index_pool:
                remaining = iter(self.names)
                pending = {}

                def fill_slots():
                    while not self.stopped.is_set() and len(pending) < self.args.index_workers:
                        name = next(remaining, None)
                        if name is None:
                            break
                        self.mark(name, "retrieve", status="queued", queued_at=now())
                        pending[index_pool.submit(self.stage, name, "retrieve")] = name

                fill_slots()
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        name = pending.pop(future)
                        try:
                            code = future.result()
                        except Exception as exc:
                            code = 1
                            self.mark(name, "retrieve", status="failed", finished_at=now(),
                                      exit_code=code, error=str(exc))
                        if code == 0:
                            self.mark(name, "qa", status="queued", queued_at=now())
                            futures.append(pool.submit(self.stage, name, "qa"))
                        else:
                            self.mark(name, "qa", status="skipped", reason="campaign_stopped"
                                      if self.stopped.is_set() else "retrieval_failed", exit_code=None)
                    fill_slots()
                for future in futures:
                    future.result()
                if self.stopped.is_set():
                    for name in remaining:
                        for stage in ("retrieve", "qa"):
                            self.mark(name, stage, status="skipped", reason="campaign_stopped",
                                      finished_at=now(), exit_code=None)
        finally:
            # Produce the combined table once more after concurrent jobs finish.
            # Each runner invocation already emits per-dataset metrics and tables.
            if not self.stopped.is_set() and self.names:
                self.stage(self.names[-1], "report")
            with self.lock:
                complete = all(self.state["datasets"][name].get("qa", {}).get("status") == "complete"
                               for name in self.names)
                complete = complete and (not self.names or
                    self.state["datasets"][self.names[-1]].get("report", {}).get("status") == "complete")
                self.state["status"] = "stopped" if self.stopped.is_set() else "complete" if complete else "incomplete"
                self.state["finished_at"] = now()
                self.save()
        return 0 if self.state["status"] == "complete" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/version_a")
    parser.add_argument("--output", type=Path, default=REPO / "results/version_a")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--reader", default="gemini-3.6-flash")
    parser.add_argument("--judge", default="gemini-2.5-flash-lite")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 33))
    parser.add_argument("--index-workers", type=int, default=1, choices=(1, 2),
                        help="Concurrent index subprocesses, each limited to four CPU threads")
    parser.add_argument("--max-usd", type=float, default=150)
    parser.add_argument("--run-max-usd", type=float, default=40)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.max_usd <= 0 or args.run_max_usd <= 0:
        parser.error("Budgets must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("Limit must be positive")
    selected = list(DATASETS) if args.datasets == "all" else args.datasets.split(",")
    if len(selected) != len(set(selected)) or any(name not in DATASETS for name in selected):
        parser.error("Datasets must be unique names from the prepared Version A suite")
    return args, selected


def main(argv=None):
    args, names = parse_args(argv)
    if args.dry_run:
        for name in names:
            for stage in ("retrieve", "qa"):
                print(json.dumps({"dataset": name, "stage": stage, "command": command(args, name, stage)}))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    with (args.output / "campaign.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A Version A campaign already owns this output directory") from None
        campaign = Campaign(args, names)
        previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: campaign.stop())
        try:
            return campaign.run()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())

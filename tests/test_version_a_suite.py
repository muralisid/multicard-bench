import json
import threading
import time

from multicard.version_a import suite


class ProcessFactory:
    def __init__(self, failures=None):
        self.failures = failures or {}
        self.started = []
        self.active = {"retrieve": 0, "qa": 0, "report": 0}
        self.maximum = dict(self.active)
        self.lock = threading.Lock()

    def __call__(self, cmd, cwd, env, stdout, stderr):
        stage = cmd[cmd.index("--stage") + 1]
        name = cmd[cmd.index("--datasets") + 1]
        assert env["OMP_NUM_THREADS"] == "4"
        assert env["TOKENIZERS_PARALLELISM"] == "false"
        with self.lock:
            self.started.append((name, stage))
            self.active[stage] += 1
            self.maximum[stage] = max(self.maximum[stage], self.active[stage])
        factory = self

        class FakeProcess:
            pid = 12345
            returncode = None

            def wait(self):
                time.sleep(0.002)
                failure = factory.failures.get((name, stage))
                if failure:
                    stdout.write(failure + "\n")
                self.returncode = int(bool(failure))
                with factory.lock:
                    factory.active[stage] -= 1
                return self.returncode

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

        return FakeProcess()


def test_suite_dry_run_lists_full_scope_without_mutating_output(tmp_path, capsys):
    output = tmp_path / "absent"
    assert suite.main(["--output", str(output), "--dry-run"]) == 0
    commands = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert len(commands) == 40
    assert len({x["dataset"] for x in commands}) == 20
    assert commands[0]["dataset"] == "locomo"
    assert commands[-1]["dataset"] == "beam_10m"
    assert not output.exists()
    command = commands[0]["command"]
    assert command[command.index("--max-usd") + 1] == "150"
    assert command[command.index("--run-max-usd") + 1] == "40"


def test_suite_limits_concurrency_and_finishes_report(tmp_path, monkeypatch):
    factory = ProcessFactory()
    monkeypatch.setattr(suite.subprocess, "Popen", factory)
    assert suite.main(["--output", str(tmp_path), "--datasets", "locomo,musique,2wikimultihopqa"]) == 0
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert state["status"] == "complete"
    assert factory.maximum["retrieve"] == factory.maximum["qa"] == 1
    assert factory.started[-1] == ("2wikimultihopqa", "report")
    assert all(state["datasets"][x]["qa"]["exit_code"] == 0 for x in state["order"])
    assert (tmp_path / "logs/locomo_qa.log").exists()


def test_suite_budget_failure_blocks_later_paid_jobs_but_not_retrieval(tmp_path, monkeypatch):
    factory = ProcessFactory({("locomo", "qa"): "BudgetExceeded: Reservation would exceed budget"})
    monkeypatch.setattr(suite.subprocess, "Popen", factory)
    assert suite.main(["--output", str(tmp_path), "--datasets", "locomo,musique,2wikimultihopqa"]) == 1
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert state["paid_jobs_blocked"] is True
    assert [x for x in factory.started if x[1] == "qa"] == [("locomo", "qa")]
    assert sum(x[1] == "retrieve" for x in factory.started) == 3
    assert state["datasets"]["musique"]["qa"]["reason"] == "paid_budget_blocked"


def test_suite_retrieval_failure_skips_only_that_dataset(tmp_path, monkeypatch):
    factory = ProcessFactory({("locomo", "retrieve"): "ValueError: missing corpus"})
    monkeypatch.setattr(suite.subprocess, "Popen", factory)
    assert suite.main(["--output", str(tmp_path), "--datasets", "locomo,musique"]) == 1
    assert ("locomo", "qa") not in factory.started
    assert ("musique", "qa") in factory.started
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert state["paid_jobs_blocked"] is False


def test_suite_does_not_call_success_when_final_report_fails(tmp_path, monkeypatch):
    factory = ProcessFactory({("locomo", "report"): "Report failure"})
    monkeypatch.setattr(suite.subprocess, "Popen", factory)
    assert suite.main(["--output", str(tmp_path), "--datasets", "locomo"]) == 1
    assert json.loads((tmp_path / "campaign.json").read_text())["status"] == "incomplete"


def test_suite_two_indexers_start_qa_before_all_indexes_finish(tmp_path, monkeypatch):
    factory = ProcessFactory()
    indexes_started = threading.Event()
    qa_started = threading.Event()
    index_finished = set()
    first_qa_index_count = []

    def controlled_process(cmd, **kwargs):
        name = cmd[cmd.index("--datasets") + 1]
        stage = cmd[cmd.index("--stage") + 1]
        process = factory(cmd, **kwargs)
        original_wait = process.wait
        if stage == "retrieve" and name == "musique":
            indexes_started.set()

        def controlled_wait():
            if stage == "retrieve" and name == "locomo":
                assert indexes_started.wait(2), "Second index never launched concurrently"
            if stage == "retrieve" and name == "musique":
                assert qa_started.wait(2), "QA waited for all indexes instead of first completion"
            if stage == "qa":
                if not qa_started.is_set():
                    first_qa_index_count.append(len(index_finished))
                qa_started.set()
                assert cmd[cmd.index("--workers") + 1] == "16"
            result = original_wait()
            if stage == "retrieve":
                index_finished.add(name)
            return result

        process.wait = controlled_wait
        return process

    monkeypatch.setattr(suite.subprocess, "Popen", controlled_process)
    assert suite.main(["--output", str(tmp_path), "--datasets", "locomo,musique,2wikimultihopqa",
                       "--index-workers", "2", "--workers", "16"]) == 0
    assert factory.maximum["retrieve"] == 2
    assert factory.maximum["qa"] == 1
    assert first_qa_index_count == [1]
    assert len(index_finished) == 3
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert state["index_workers"] == 2
    assert state["qa_workers"] == 16
    assert state["status"] == "complete"

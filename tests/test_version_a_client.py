"""Pre-request spending reservations and immutable call-cache accounting."""

import io
import json
import sqlite3
import urllib.error

import pytest

from multicard.llm.costmeter import BudgetExceeded
from multicard.version_a import client


@pytest.fixture
def setup(tmp_path, monkeypatch):
    key = tmp_path / "test-key"
    key.write_text("fixture-secret")
    monkeypatch.setenv("MCB_PROXY_KEY_FILE", str(key))
    monkeypatch.setenv("MCB_PROXY_URL", "http://fixture.invalid/v1")
    monkeypatch.chdir(tmp_path)
    requests = []
    payload = {"usage": {"prompt_tokens": 100, "completion_tokens": 12},
               "choices": [{"message": {"content": "an answer"}}]}
    def fake_urlopen(request, **kwargs):
        requests.append(request)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    return tmp_path / "ledger", requests, payload


def _ledger(root):
    with sqlite3.connect(root / "calls.sqlite") as connection:
        return connection.execute("SELECT status,usd FROM calls").fetchall()


def test_budget_cap_is_enforced_before_http_request(setup):
    root, requests, _ = setup
    api = client.ProxyClient(root, "one", "gemini-3.6-flash", max_usd=0.000001)
    with pytest.raises(BudgetExceeded):
        api.generate("A prompt")
    assert requests == []
    assert _ledger(root) == []


def test_repeat_and_cross_run_cache_hits_are_charged_only_once(setup):
    root, requests, _ = setup
    api = client.ProxyClient(root, "one", "gemini-3.6-flash")
    first = api.generate("A prompt")
    second = api.generate("A prompt")
    later = client.ProxyClient(root, "two", "gemini-3.6-flash")
    third = later.generate("A prompt")
    assert len(requests) == 1
    assert not first["cached"] and second["cached"] and third["cached"]
    assert len(_ledger(root)) == 1
    assert _ledger(root)[0][1] == pytest.approx(first["usd"])
    assert api.meter.total_calls() == 1 and later.meter.total_calls() == 0
    assert api.meter.total_usd() == pytest.approx(first["usd"])
    assert later.meter.total_usd() == 0


def test_persistent_cap_survives_new_client_instance(setup):
    root, requests, _ = setup
    api = client.ProxyClient(root, "one", "gemini-3.6-flash")
    first = api.generate("First prompt", max_output_tokens=20)
    restarted = client.ProxyClient(root, "one", "gemini-3.6-flash", run_max_usd=first["usd"])
    with pytest.raises(BudgetExceeded):
        restarted.generate("Second prompt", max_output_tokens=20)
    assert len(requests) == 1


def test_uncertain_request_keeps_reservation_and_never_blindly_retries(setup, monkeypatch):
    root, requests, _ = setup
    def disconnect(request, **kwargs):
        requests.append(request)
        raise urllib.error.URLError("Connection lost after sending request")
    monkeypatch.setattr(client.urllib.request, "urlopen", disconnect)
    api = client.ProxyClient(root, "one", "gemini-3.6-flash")
    with pytest.raises(urllib.error.URLError):
        api.generate("A prompt")
    rows = _ledger(root)
    assert rows[0][0] == "pending" and rows[0][1] > 0
    with pytest.raises(RuntimeError, match="Unresolved prior API attempt"):
        api.generate("A prompt")
    assert len(requests) == 1


def test_missing_usage_does_not_create_free_cached_success(setup):
    root, requests, payload = setup
    payload.pop("usage")
    api = client.ProxyClient(root, "one", "gemini-3.6-flash")
    with pytest.raises(RuntimeError, match="Missing usage"):
        api.generate("A prompt")
    assert _ledger(root)[0][0] == "pending"
    with pytest.raises(RuntimeError, match="Unresolved prior"):
        api.generate("A prompt")
    assert len(requests) == 1


def test_freeze_stops_even_a_cached_response(setup):
    root, requests, _ = setup
    api = client.ProxyClient(root, "one", "gemini-3.6-flash")
    api.generate("A prompt")
    (root.parent / "FREEZE").touch()
    with pytest.raises(RuntimeError, match="FREEZE"):
        api.generate("A prompt")
    assert len(requests) == 1


def test_evaluation_distinguishes_cached_spend_and_partial_f1(setup):
    from multicard.version_a import runner
    root, requests, payload = setup
    payload["choices"][0]["message"]["content"] = "red blue"
    reader = client.ProxyClient(root, "one", "gemini-3.6-flash")
    judge = client.ProxyClient(root, "one", "gemini-2.5-flash-lite")
    q = {"id": "q1", "type": "twohop", "question": "Which colors?",
         "answers": ["red blue green"], "metric": "f1"}
    context = {"context": "Source information", "context_tokens": 2}
    first = runner.evaluate_one(q, context, reader, judge)
    repeated = runner.evaluate_one(q, context, reader, judge)
    assert len(requests) == 1
    assert first["score"] == pytest.approx(0.8)
    assert first["correct"] == 0
    assert first["incremental_answer_usd"] > 0
    assert repeated["answer_usd"] == first["answer_usd"]
    assert repeated["incremental_answer_usd"] == repeated["cash_usd"] == 0
    assert repeated["judge_usd"] == 0

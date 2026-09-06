"""Offline checks for the Part 1 infrastructure records.

Covers: the two added dependencies import; the committed planner prompt
(docs/part1/prompts/planner.txt) is the design section 5 text, one shape per
line, ASCII only; docs/part1/env/models.json carries the added models with the
prices the survey recorded; and the proxy request-log tag rule (job_tag from
metadata.tags first, else the body "user" field) as implemented in
part1-tools/litellm/request_log.py, loaded here with a stub for litellm so
no proxy dependency is needed. No network, no model call.
"""
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs" / "PART1-DESIGN.md"
PROMPT = ROOT / "docs" / "part1" / "prompts" / "planner.txt"
MODELS = ROOT / "docs" / "part1" / "env" / "models.json"
REQUEST_LOG_PY = Path(
    "/Users/muralisid/github_other/part1-tools/litellm/request_log.py")

SHAPES = ["local", "entity", "thematic", "cross-topic", "multi-hop",
          "temporal", "lexical", "unanswerable"]


def test_dependencies_import():
    import psycopg
    import pyarrow
    assert pyarrow.__version__
    assert psycopg.__version__


def test_planner_prompt_matches_design_section_5():
    design = re.sub(r"\s+", " ", DESIGN.read_text())
    start = design.index("Label the question with exactly one shape")
    end = design.index("unlikely to contain.", start) + len("unlikely to contain.")
    expected = design[start:end]
    prompt = PROMPT.read_text()
    assert re.sub(r"\s+", " ", prompt).strip() == expected


def test_planner_prompt_layout():
    lines = PROMPT.read_text().splitlines()
    assert len(lines) == 9
    assert lines[0].startswith("Label the question with exactly one shape")
    assert [ln.split(":")[0] for ln in lines[1:]] == SHAPES
    assert all(ord(ch) < 128 for ch in PROMPT.read_text())


def test_models_json_has_added_models():
    m = json.loads(MODELS.read_text())
    for name in ("gemini-3.7-flash", "gemini-3.8-flash"):
        assert name in m["working_models"]
        assert m["locations"][name] == "global"
        assert m["prices"][name] == {"in": 0.75, "out": 3.75}
    assert m["proxy_base_url"] == "http://127.0.0.1:4000/v1"
    assert m["chat_model"] == "gemini-2.5-flash-lite"


def _load_request_log_module():
    if not REQUEST_LOG_PY.exists():
        pytest.skip("part1-tools is not on this machine")
    stub_pkg = types.ModuleType("litellm")
    stub_int = types.ModuleType("litellm.integrations")
    stub_cl = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:
        pass

    stub_cl.CustomLogger = CustomLogger
    saved = {k: sys.modules.get(k) for k in
             ("litellm", "litellm.integrations", "litellm.integrations.custom_logger")}
    sys.modules["litellm"] = stub_pkg
    sys.modules["litellm.integrations"] = stub_int
    sys.modules["litellm.integrations.custom_logger"] = stub_cl
    try:
        spec = importlib.util.spec_from_file_location("part1_request_log", REQUEST_LOG_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def test_request_log_tag_rule():
    mod = _load_request_log_module()
    # metadata.tags wins over user
    kwargs = {"user": "u1", "litellm_params": {"metadata": {"tags": ["t1", "t2"]}}}
    assert mod._client_tags(kwargs, {}) == ["t1", "t2"]
    assert mod._client_user(kwargs, {}) == "u1"
    # user only
    kwargs = {"user": "u2", "litellm_params": {"metadata": {}}}
    assert mod._client_tags(kwargs, {}) == []
    assert mod._client_user(kwargs, {}) == "u2"
    # neither: fall back to the standard logging object, dropping litellm's
    # User-Agent entries
    slo = {"request_tags": ["User-Agent: OpenAI/Python 3.8.0"], "metadata": {}}
    assert mod._client_tags({}, slo) == []
    assert mod._client_user({}, slo) is None
    slo = {"request_tags": ["job-x", "User-Agent: x"],
           "metadata": {"user_api_key_end_user_id": "u3"}}
    assert mod._client_tags({}, slo) == ["job-x"]
    assert mod._client_user({}, slo) == "u3"

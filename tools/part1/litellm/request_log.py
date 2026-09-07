"""Per-request log for the litellm proxy.

One JSON line per request goes to LITELLM_REQUEST_LOG (default
part1-tools/env/requests.jsonl). Fields: ts, status, model, call_type,
prompt_tokens, completion_tokens, total_tokens, response_cost, latency_s,
job_tag, tags, user, and error on failures. Read it back with
tokens_by_model.py.

Job tags. A client can label its requests so the shared log can be split by
caller (design section 11: proxy spend is cross-checked by time window and
job tag). Two request body fields are read:

- "user": the OpenAI "user" string. The OpenAI python client sends it with
  the user= keyword on chat.completions.create and embeddings.create.
- "metadata": {"tags": [...]}: litellm's own tag list. The OpenAI python
  client sends it with extra_body={"metadata": {"tags": ["..."]}}.

job_tag is the first metadata tag when one is sent, else the "user" string,
else null. tags holds the full metadata tag list as sent by the client, and
user holds the raw "user" string, so nothing is lost when both are sent.
"""
import json
import os
import time

from litellm.integrations.custom_logger import CustomLogger

LOG_PATH = os.environ.get(
    "LITELLM_REQUEST_LOG",
    "/Users/muralisid/github_other/part1-tools/env/requests.jsonl")


def _num(x):
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _client_tags(kwargs, slo):
    """Return the tag list the client sent under metadata.tags, or [].

    The list is read from litellm_params.metadata first, which is the client's
    own list. standard_logging_object.request_tags is the fallback; litellm
    appends "User-Agent: ..." entries there, and those are dropped.
    """
    lp = kwargs.get("litellm_params") or {}
    for key in ("metadata", "litellm_metadata"):
        md = lp.get(key) or {}
        tags = md.get("tags") if isinstance(md, dict) else None
        if isinstance(tags, list) and tags:
            return [str(t) for t in tags]
    tags = slo.get("request_tags") or []
    return [str(t) for t in tags if not str(t).startswith("User-Agent")]


def _client_user(kwargs, slo):
    """Return the request body "user" string, or None."""
    user = kwargs.get("user")
    if user is None:
        md = slo.get("metadata") or {}
        user = md.get("user_api_key_end_user_id") if isinstance(md, dict) else None
    if user is None:
        user = slo.get("end_user")
    if user is None or user == "":
        return None
    return str(user)


class RequestLogger(CustomLogger):
    def _record(self, kwargs, response_obj, start_time, end_time, status):
        slo = kwargs.get("standard_logging_object") or {}
        usage = getattr(response_obj, "usage", None)
        tags = _client_tags(kwargs, slo)
        user = _client_user(kwargs, slo)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": status,
            "model": slo.get("model") or kwargs.get("model"),
            "call_type": slo.get("call_type") or kwargs.get("call_type"),
            "prompt_tokens": _num(slo.get("prompt_tokens")
                                  or getattr(usage, "prompt_tokens", None)),
            "completion_tokens": _num(slo.get("completion_tokens")
                                      or getattr(usage, "completion_tokens", None)),
            "total_tokens": _num(slo.get("total_tokens")
                                 or getattr(usage, "total_tokens", None)),
            "response_cost": slo.get("response_cost", kwargs.get("response_cost")),
            "job_tag": tags[0] if tags else user,
            "tags": tags,
            "user": user,
        }
        try:
            rec["latency_s"] = round((end_time - start_time).total_seconds(), 3)
        except Exception:
            rec["latency_s"] = None
        if status != "success":
            exc = kwargs.get("exception") or slo.get("error_str")
            rec["error"] = str(exc)[:300] if exc else None
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time, "success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time, "failure")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time, "success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time, "failure")


proxy_handler_instance = RequestLogger()

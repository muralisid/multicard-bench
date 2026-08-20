"""A thin, provider-agnostic generative client.

Reads credentials from the environment and writes none of them anywhere. Every
call goes through the cost meter, so a run cannot quietly exceed its budget, and
responses are cached on disk by a hash of (model, prompt) so that a rerun costs
nothing and produces identical text. That caching is what makes an experiment
involving a generative model reproducible at all: without it, the same script
run twice gives different outputs and no downstream number can be checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

CACHE_DB = Path(os.environ.get("MCB_LLM_CACHE", "data/cache/llm.sqlite"))

# Model ids are supplied by configuration, never hard-coded to one vendor's
# naming. These are the defaults the study used.
DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class Response:
    text: str
    tokens_in: int
    tokens_out: int
    cached: bool


class GenerativeClient:
    """Vertex AI via application default credentials or an inline service account."""

    def __init__(self, model: str = DEFAULT_MODEL, location: str = "us-central1",
                 meter=None, tier: str = "vertex-flash", cache: bool = True,
                 temperature: float = 0.0):
        self.model = model
        self.location = location
        self.meter = meter
        self.tier = tier
        self.cache = cache
        self.temperature = temperature
        self._client = None
        self._cred_path = None
        self._init_cache()

    def _init_cache(self) -> None:
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(CACHE_DB) as c:
            c.execute("CREATE TABLE IF NOT EXISTS responses ("
                      "key TEXT PRIMARY KEY, text TEXT, tin INT, tout INT)")

    @property
    def client(self):
        if self._client is None:
            from google import genai

            sa_json = os.environ.get("VERTEX_AI_SERVICE_ACCOUNT_JSON")
            project = os.environ.get("VERTEX_AI_PROJECT")
            if sa_json:
                sa = json.loads(sa_json)
                project = project or sa.get("project_id")
                # Written to a private temporary file because the client library
                # wants a path; never into the repository or any logged location.
                fd, path = tempfile.mkstemp(suffix=".json")
                with os.fdopen(fd, "w") as f:
                    json.dump(sa, f)
                os.chmod(path, 0o600)
                self._cred_path = path
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
            if not project:
                raise RuntimeError(
                    "no Google Cloud project: set VERTEX_AI_PROJECT or "
                    "VERTEX_AI_SERVICE_ACCOUNT_JSON")
            self._client = genai.Client(vertexai=True, project=project,
                                        location=self.location)
        return self._client

    def close(self) -> None:
        if self._cred_path and os.path.exists(self._cred_path):
            os.unlink(self._cred_path)
            self._cred_path = None

    # Bump when the request configuration changes in a way that alters output.
    # The cache previously keyed only on model, temperature and prompt, so a
    # change to the token budget silently served the old, truncated responses
    # and a fixed bug looked unfixed. Every parameter that can change the text
    # now belongs in the key.
    CONFIG_VERSION = 2

    def _key(self, prompt: str, max_output_tokens: int) -> str:
        h = hashlib.sha256()
        for part in (self.model, str(self.temperature), str(max_output_tokens),
                     str(self.CONFIG_VERSION)):
            h.update(part.encode())
            h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def generate(self, prompt: str, max_output_tokens: int = 800) -> Response:
        key = self._key(prompt, max_output_tokens)
        if self.cache:
            with sqlite3.connect(CACHE_DB) as c:
                row = c.execute("SELECT text, tin, tout FROM responses WHERE key=?",
                                (key,)).fetchone()
            if row:
                return Response(row[0], row[1], row[2], cached=True)

        from google.genai import types

        r = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=max_output_tokens,
                # Reasoning models spend the output allowance on hidden thinking
                # before writing anything, which returned ten-word stubs where a
                # two-hundred-word summary was asked for. The task here is
                # summarisation, not reasoning, so the budget goes to the answer.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (r.text or "").strip()
        u = getattr(r, "usage_metadata", None)
        tin = int(getattr(u, "prompt_token_count", 0) or 0)
        tout = int(getattr(u, "candidates_token_count", 0) or 0)

        if self.meter is not None:
            self.meter.record(self.tier, tin, tout)   # raises if over budget
        if self.cache:
            with sqlite3.connect(CACHE_DB) as c:
                c.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
                          (key, text, tin, tout))
        return Response(text, tin, tout, cached=False)

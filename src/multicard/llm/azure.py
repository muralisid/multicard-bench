"""Azure OpenAI client, used for cross-family judging.

Judge diversity is the weakest point of any study that scores generated text with
a generative model. Two judges from one family sharing a training lineage can
agree for reasons that have nothing to do with the text. This client exists so
that at least one judge comes from a different family and a different vendor,
which is the check a reviewer will ask for first.

Interface and caching match the Vertex client deliberately, so an experiment can
swap judges by name without any other change.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE_DB = Path(os.environ.get("MCB_LLM_CACHE", "data/cache/llm.sqlite"))


@dataclass
class Response:
    text: str
    tokens_in: int
    tokens_out: int
    cached: bool


class AzureClient:
    CONFIG_VERSION = 1

    def __init__(self, model: str | None = None, meter=None,
                 tier: str = "azure-gpt54", cache: bool = True,
                 temperature: float = 0.0, timeout: int = 120):
        self.endpoint = os.environ.get("AZURE_GPT54_ENDPOINT", "").rstrip("/")
        self.api_key = os.environ.get("AZURE_GPT54_API_KEY", "")
        self.model = model or os.environ.get("AZURE_GPT54_DEPLOYMENT", "gpt-5.4")
        self.meter = meter
        self.tier = tier
        self.cache = cache
        self.temperature = temperature
        self.timeout = timeout
        if not self.endpoint or not self.api_key:
            raise RuntimeError(
                "Azure is not configured: set AZURE_GPT54_ENDPOINT and "
                "AZURE_GPT54_API_KEY")
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(CACHE_DB) as c:
            c.execute("CREATE TABLE IF NOT EXISTS responses ("
                      "key TEXT PRIMARY KEY, text TEXT, tin INT, tout INT)")

    def close(self) -> None:  # symmetry with the Vertex client
        return None

    def _key(self, prompt: str, max_output_tokens: int) -> str:
        h = hashlib.sha256()
        for part in ("azure", self.model, str(self.temperature),
                     str(max_output_tokens), str(self.CONFIG_VERSION)):
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

        body = json.dumps({
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "api-key": self.api_key,
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Azure request failed ({e.code}): "
                f"{e.read().decode()[:300]}") from None

        text = ""
        for item in payload.get("output", []):
            for part in item.get("content", []) or []:
                if part.get("type") in ("output_text", "text"):
                    text += part.get("text", "")
        text = (text or payload.get("output_text", "") or "").strip()

        u = payload.get("usage", {}) or {}
        tin = int(u.get("input_tokens", 0) or 0)
        tout = int(u.get("output_tokens", 0) or 0)

        if self.meter is not None:
            self.meter.record(self.tier, tin, tout)
        if self.cache:
            with sqlite3.connect(CACHE_DB) as c:
                c.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
                          (key, text, tin, tout))
        return Response(text, tin, tout, cached=False)

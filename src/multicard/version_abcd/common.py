from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

from multicard.version_a.client import ProxyClient
from multicard.version_a.core import digest

CONFIG = {"version": "bcd-screen-1", "seed": 13, "sample_size": 100,
          "context_tokens": 4000, "piece_tokens": 500, "max_unit_tokens": 1000,
          "detect_depth": 30, "extract_units_per_query": 8, "extract_pieces_per_query": 12,
          "extractor": "gemini-2.5-flash-lite", "extract_output_tokens": 2048,
          "reader": "gemini-3.6-flash", "judge": "gemini-2.5-flash-lite",
          "rrf_k": 60, "weights": {"base": 1.0, "topics": .2, "communities": .2, "relations": .4},
          "max_usd": 60, "job_max_usd": 10}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Client(ProxyClient):
    """Reuse A's cache read-only and wait for shared in-flight requests."""
    def __init__(self, output, run, model, baseline_meter=None):
        super().__init__(Path(output) / "_meter", run, model, CONFIG["max_usd"], CONFIG["job_max_usd"])
        self.baseline_meter = Path(baseline_meter) if baseline_meter else None

    def generate(self, prompt, max_output_tokens=512, role="answer"):
        request = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_output_tokens, "temperature": 0}
        if self.model == "gemini-2.5-flash-lite":
            request["reasoning_effort"] = "none"
        key = digest([self.url, request])
        if self.baseline_meter and self.baseline_meter.exists():
            with sqlite3.connect(self.baseline_meter.resolve().as_uri() + "?mode=ro", uri=True) as c:
                row = c.execute("SELECT status,text,tin,tout,usd FROM calls WHERE key=?", (key,)).fetchone()
            if row and row[0] == "done":
                return dict(text=row[1], tokens_in=row[2], tokens_out=row[3], usd=row[4],
                            seconds=0., cached=True, key=key)
        deadline = time.monotonic() + 150
        while True:
            try:
                return super().generate(prompt, max_output_tokens, role)
            except RuntimeError as exc:
                if not str(exc).startswith("Unresolved prior API attempt;"):
                    raise
                with self.connect() as c:
                    row = c.execute("SELECT status FROM calls WHERE key=?", (key,)).fetchone()
                if not row or row[0] not in {"done", "pending"} or time.monotonic() >= deadline:
                    raise
                time.sleep(.25)

"""Cached proxy calls with persistent, pre-request budget reservation.

The service is already configured outside this repository. Credentials are read
from its existing private key file and never stored in run artifacts.
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

from multicard.llm.costmeter import BudgetExceeded, CostMeter, PRICES_USD_PER_MTOK
from .core import digest

PRICES = {"gemini-3.6-flash": {"in": 0.75, "out": 3.75},
          "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40}}


class ProxyClient:
    def __init__(self, root, run, model, max_usd=100, run_max_usd=40):
        self.root, self.run, self.model = Path(root), run, model
        self.max_usd, self.run_max_usd = max_usd, run_max_usd
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "calls.sqlite"
        self.url = os.environ.get("MCB_PROXY_URL", "http://127.0.0.1:4000/v1").rstrip("/")
        key_path = os.environ.get("MCB_PROXY_KEY_FILE", "/Users/muralisid/github_other/part1-tools/env/.proxy_key")
        self.key = Path(key_path).read_text().strip()
        self.price = PRICES[model]
        PRICES_USD_PER_MTOK[model] = self.price
        self.meter = CostMeter(max_usd=run_max_usd)
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, run TEXT, model TEXT, "
                      "status TEXT, text TEXT, tin INT, tout INT, usd REAL, seconds REAL)")
            if "role" not in {r[1] for r in c.execute("PRAGMA table_info(calls)")}:
                c.execute("ALTER TABLE calls ADD COLUMN role TEXT")

    def connect(self):
        c = sqlite3.connect(self.db, timeout=60)
        c.execute("PRAGMA busy_timeout=60000")
        return c

    def generate(self, prompt, max_output_tokens=512, role="answer"):
        if Path("FREEZE").exists():
            raise RuntimeError("Repository FREEZE kill switch")
        request = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_output_tokens, "temperature": 0,
                   "user": "version-a-" + self.run + "-" + role}
        if self.model == "gemini-2.5-flash-lite":
            request["reasoning_effort"] = "none"
        key = digest([self.url, {k: v for k, v in request.items() if k != "user"}])
        # UTF-8 bytes conservatively bound text tokens for these tokenizers;
        # output bound includes hidden reasoning within max_tokens.
        reserve = ((len(prompt.encode()) + 1024) * self.price["in"]
                   + max_output_tokens * self.price["out"]) / 1e6
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT status,text,tin,tout,usd,seconds FROM calls WHERE key=?", (key,)).fetchone()
            if row and row[0] == "done":
                return dict(text=row[1], tokens_in=row[2], tokens_out=row[3], usd=row[4],
                            seconds=0.0, cached=True, key=key)
            if row:
                raise RuntimeError("Unresolved prior API attempt; reserved cost retained: " + key)
            total = c.execute("SELECT COALESCE(SUM(usd),0) FROM calls").fetchone()[0]
            run_total = c.execute("SELECT COALESCE(SUM(usd),0) FROM calls WHERE run=?", (self.run,)).fetchone()[0]
            if total + reserve > self.max_usd or run_total + reserve > self.run_max_usd:
                raise BudgetExceeded(f"Reservation would exceed budget: total={total:.2f}, run={run_total:.2f}")
            c.execute("INSERT INTO calls (key,run,model,status,text,tin,tout,usd,seconds,role) "
                      "VALUES (?,?,?,'pending','',0,0,?,0,?)",
                      (key, self.run, self.model, reserve, role))
        started = time.perf_counter()
        payload = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url + "/chat/completions", json.dumps(request).encode(),
                                             {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    payload = json.load(r)
                break
            except urllib.error.HTTPError as exc:
                # Rate limits reject before generation. Other failures may have
                # incurred usage, so retain their reservation and stop.
                if exc.code == 429 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                with self.connect() as c:
                    c.execute("UPDATE calls SET status='error' WHERE key=?", (key,))
                raise RuntimeError(f"Model API HTTP {exc.code}; no response body logged") from None
        usage = payload.get("usage", {})
        if "prompt_tokens" not in usage or "completion_tokens" not in usage:
            raise RuntimeError("Missing usage; reservation retained, refusing unmetered result")
        tin, tout = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        cost = (tin * self.price["in"] + tout * self.price["out"]) / 1e6
        elapsed = time.perf_counter() - started
        text = (payload.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        with self.connect() as c:
            c.execute("UPDATE calls SET status='done',text=?,tin=?,tout=?,usd=?,seconds=? WHERE key=?",
                      (text, tin, tout, cost, elapsed, key))
        self.meter.record(self.model, tin, tout)
        return dict(text=text, tokens_in=tin, tokens_out=tout, usd=cost, seconds=elapsed, cached=False, key=key)

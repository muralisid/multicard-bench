"""post-graph-rag runner for Part 1: one space per question, every query variant saved, tables exported.

Implements PART1-DESIGN.md version 4, section 4 item 6 (Chandan's tables, built
the way his LongMemEval harness builds them), the inputs of section 5 for the
chandan_live, chandan_full and chandan_full_uncut rows, and section 13 (one
study model, metered spend, projection after the first shard, cap).

What one question costs and produces, LongMemEval:

  1. A space named by the question id in a realm named by the corpus.
  2. One document per haystack session in session-date order. The document is
     one line per turn, "role: text" with internal newlines replaced by
     spaces, under the header line his run.py writes. A turn-span table
     (turn index to char_start, char_end) is kept.
  3. index_document per session (never chunked), the network half of the call
     run 8 sessions at a time and the write half in session-date order. A
     session his extractor refuses is retried with the library's document
     prompt the way his harness does; a second refusal is recorded and the
     question continues. His harness calls index_document one session at a
     time; the concurrency here is recorded in meta.json.
  4. query_data in mix mode at the shipped top_k (--shipped-top-k, see
     SHIPPED_TOP_K), then with top_k raised 16, 32, 64, 128 until the units
     the bench renders for chandan_live (his chunk lines, entity lines and
     relation lines, deduplicated) count at least 4,000 tokens (raised_4k)
     and 8,000 tokens (raised_8k) by the bench TokenCounter (design section
     5, B counted by the bench TokenCounter). The character count is written
     beside it as a side figure. One query() call at the shipped
     configuration for full_uncut with his synthesis.
  5. The space's entities, relations (with sources and embeddings), chunks
     (with turn spans) and doc_mentions exported to parquet, meta.json
     written, the space dropped.

MultiHop-RAG: one space "corpus", one document per article in
publication-date order, his default extraction prompt, the same query
variants for every non-null query in ORDER, one export at the end.

Every model call goes through a metered client. Usage is read from each
response, priced from models.json, summed per call window, per question, per
shard and per run, and written into every query file and meta.json together
with the wall-clock window of the run so the proxy log can be cross-checked.

Usage:
    .venv/bin/python run_spaces.py --corpus lme --ids docs/part1/subsets.json ...
    .venv/bin/python run_spaces.py --corpus lme --smoke --out-root out-smoke/lme --cap-usd 0.60

The proxy key is read from a file and never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.util
import json
import pathlib
import sys
import time
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parent
ENV = TOOLS / "env"
BENCH = pathlib.Path("/Users/muralisid/github_other/multicard-bench")
CLONE = TOOLS / "post-graph-rag-src"

# The result limit of the shipped variant. Three candidates exist in his
# repository and the design names none of them by number: the package default
# QueryParam.top_k = 5 (post_graph_rag/models.py line 123), the published
# 94.0 run's reader_sweep.py line 244 (top_k=8), and his original harness
# run.py line 393 (top_k=32). 8 is used here because pgr.md section 2.1
# records the published run's configuration; it is not the package default.
# The value is a command line option (--shipped-top-k) and is written into
# every query file and meta.json, so the main session's written decision in
# pgr.md can be applied without a code change.
SHIPPED_TOP_K = 8
# design gap: the design says the limit is raised until the rendered context
# reaches B and fixes no top step; the ladder stops at 128 and a question
# whose top step did not reach B is counted in the run log (raised_not_reached)
# and reported beside the chandan_live rows.
RAISE_STEPS = (16, 32, 64, 128)      # design section 5, chandan_live raised variant
BUDGET_TOKENS = {"raised_4k": 4000, "raised_8k": 8000}   # design section 5: B by the bench TokenCounter
VARIANTS = ("shipped", "raised_4k", "raised_8k", "full_uncut")
# Design section 11 caps, USD: the study-model build and queries on
# LongMemEval 130, the calibration row 70, MultiHop-RAG 10. --cap-usd
# overrides them and the override is logged.
CAP_USD = {"lme": 130.0, "calibration": 70.0, "mhrag": 10.0}
# How the question reaches the query call. reader_sweep.py lines 220-221 (the
# published run) send "Today is {date}. {question}"; run.py lines 381-392
# (his original harness) prepend a three-rule answer-side block to that text.
# The text sent reaches the keyword-extraction call and the question
# embedding, so the form changes retrieval. The form used is a command line
# option and is recorded in every query file and meta.json (question_form);
# the main session records its decision in pgr.md.
QUESTION_FORMS = ("reader_sweep", "run_py")
RUN_PY_RULES = (
    "Answer from the facts and conversation excerpts provided.\n"
    "- When records conflict about the same thing, the most recent "
    "statement is the current truth. Give the current value only; "
    "do not present the conflict or hedge between versions.\n"
    "- For questions about durations or counts of days/weeks: find "
    "the dates of the events, then compute the difference "
    "yourself. Use the stated 'today' for phrases like 'how long "
    "ago'. Prefer a specific number over a refusal.\n"
    "- If an event's date is only implied (e.g. said during a "
    "session), use that session's date.\n\n"
)
# Design section 13: gemini-3.7-flash serves at the same price as 3.6. Used only
# when a models.json predates the 3.7 price row.
PRICE_ALIASES = {"gemini-3.7-flash": "gemini-3.6-flash"}
GRAPH_TABLES = ("documents", "entities", "communities", "relations",
                "doc_mentions", "community_members", "community_children")


def now_iso() -> str:
    """Local time with offset, the same shape as the proxy log's ts field."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def question_sent(question: str, question_date: Optional[str], form: str = "reader_sweep") -> str:
    """The text the query call receives under one of QUESTION_FORMS."""
    asked = f"Today is {question_date}. {question}" if question_date else question
    if form == "reader_sweep":
        return asked
    if form == "run_py":
        return RUN_PY_RULES + asked
    raise ValueError(f"unknown question form {form!r}; choose from {QUESTION_FORMS}")


# ----------------------------------------------------------------------------
# The bench TokenCounter without transformers (design section 5: B in tokens)
# ----------------------------------------------------------------------------
class BenchTokenCounter:
    """The count of multicard.llm.costmeter.TokenCounter (the all-MiniLM-L6-v2
    tokenizer, no special tokens) through the tokenizers library over the
    same tokenizer.json in the Hugging Face cache, padding and truncation
    off. This venv has no transformers; tests/test_part1_pgr.py checks the
    two counters agree in the bench venv."""

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, path: Optional[pathlib.Path] = None):
        from tokenizers import Tokenizer

        self.path = pathlib.Path(path) if path else self.tokenizer_path()
        self._tok = Tokenizer.from_file(str(self.path))
        self._tok.no_padding()
        self._tok.no_truncation()

    @classmethod
    def tokenizer_path(cls) -> pathlib.Path:
        import os

        cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        root = pathlib.Path(cache) / ("models--" + cls.MODEL.replace("/", "--"))
        ref = root / "refs" / "main"
        if ref.exists():
            p = root / "snapshots" / ref.read_text().strip() / "tokenizer.json"
            if p.exists():
                return p
        found = sorted((root / "snapshots").glob("*/tokenizer.json")) if (root / "snapshots").exists() else []
        if found:
            return found[0]
        raise FileNotFoundError(f"no tokenizer.json for {cls.MODEL} under {root}; the bench encodes it into the cache")

    def count(self, text: str) -> int:
        return len(self._tok.encode(text or "", add_special_tokens=False).ids)


# ----------------------------------------------------------------------------
# Prices and the usage meter
# ----------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


def load_prices(models_json: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(models_json).read_text())["prices"]


def price_for(prices: dict, model: str) -> dict:
    """USD per million tokens for a model, through the section 13 alias table."""
    name = model if model in prices else PRICE_ALIASES.get(model, model)
    if name not in prices:
        raise KeyError(f"no price for model {model!r} in models.json")
    return prices[name]


class UsageMeter:
    """Sums usage from every response and prices it. Section 11 and 13.

    record() appends one row per call. mark() returns the row count, and
    summary(mark) sums the rows written since that mark, so one meter serves
    the per-call, per-question, per-shard and per-run windows.
    """

    def __init__(self, prices: dict, cap_usd: Optional[float] = None):
        self.prices = prices
        self.cap_usd = cap_usd
        self.records: list[dict] = []
        self.total_usd = 0.0
        self.exceeded = False
        self.usage_missing = 0

    def record(self, kind: str, model: str, tokens_in: int, tokens_out: int,
               reasoning: int = 0) -> None:
        p = price_for(self.prices, model)
        usd = tokens_in * p["in"] / 1e6 + tokens_out * p["out"] / 1e6
        self.records.append({"ts": now_iso(), "kind": kind, "model": model,
                             "tokens_in": int(tokens_in), "tokens_out": int(tokens_out),
                             "reasoning_tokens": int(reasoning), "usd": usd})
        self.total_usd += usd
        if self.cap_usd is not None and self.total_usd > self.cap_usd:
            self.exceeded = True

    def check(self) -> None:
        if self.exceeded:
            raise BudgetExceeded(
                f"spend {self.total_usd:.4f} USD exceeds the cap of {self.cap_usd:.2f}")

    def mark(self) -> int:
        return len(self.records)

    def summary(self, start: int = 0, end: Optional[int] = None) -> dict:
        rows = self.records[start:end]
        by_model: dict[str, dict] = {}
        out = {"calls": 0, "chat_calls": 0, "embed_calls": 0, "tokens_in": 0,
               "tokens_out": 0, "reasoning_tokens": 0, "usd": 0.0}
        for r in rows:
            out["calls"] += 1
            out["chat_calls" if r["kind"] == "chat" else "embed_calls"] += 1
            out["tokens_in"] += r["tokens_in"]
            out["tokens_out"] += r["tokens_out"]
            out["reasoning_tokens"] += r["reasoning_tokens"]
            out["usd"] += r["usd"]
            m = by_model.setdefault(r["model"], {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0})
            m["calls"] += 1
            m["tokens_in"] += r["tokens_in"]
            m["tokens_out"] += r["tokens_out"]
            m["usd"] += r["usd"]
        out["usd"] = round(out["usd"], 6)
        for m in by_model.values():
            m["usd"] = round(m["usd"], 6)
        out["by_model"] = by_model
        return out


def _usage_ints(usage: Any) -> tuple[int, int, int]:
    tin = int(getattr(usage, "prompt_tokens", 0) or 0)
    tout = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    return tin, tout, reasoning


class _MeteredCompletions:
    """chat.completions and beta.chat.completions with usage capture and the job tag."""

    def __init__(self, inner: Any, meter: UsageMeter, tag: str, capture: list):
        self._inner, self._meter, self._tag, self._capture = inner, meter, tag, capture

    def _tagged(self, kw: dict) -> dict:
        kw = dict(kw)
        # proxy.md documents neither field, so both are sent (task rule).
        kw.setdefault("user", self._tag)
        body = dict(kw.get("extra_body") or {})
        body.setdefault("metadata", {"job_tag": self._tag, "tags": [self._tag]})
        kw["extra_body"] = body
        return kw

    def _after(self, kw: dict, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self._meter.usage_missing += 1
            tin = tout = reasoning = 0
        else:
            tin, tout, reasoning = _usage_ints(usage)
        self._meter.record("chat", kw["model"], tin, tout, reasoning)
        self._capture.append({"model": kw["model"], "messages": kw.get("messages"),
                              "tokens_in": tin, "tokens_out": tout})

    async def create(self, **kw):
        self._meter.check()
        kw = self._tagged(kw)
        response = await self._inner.create(**kw)
        if not kw.get("stream"):
            self._after(kw, response)
        return response

    async def parse(self, **kw):
        self._meter.check()
        kw = self._tagged(kw)
        response = await self._inner.parse(**kw)
        self._after(kw, response)
        return response


class _MeteredEmbeddings:
    def __init__(self, inner: Any, meter: UsageMeter, tag: str):
        self._inner, self._meter, self._tag = inner, meter, tag

    async def create(self, **kw):
        self._meter.check()
        kw = dict(kw)
        kw.setdefault("user", self._tag)
        body = dict(kw.get("extra_body") or {})
        body.setdefault("metadata", {"job_tag": self._tag, "tags": [self._tag]})
        kw["extra_body"] = body
        response = await self._inner.create(**kw)
        usage = getattr(response, "usage", None)
        if usage is None:
            self._meter.usage_missing += 1
            tin = 0
        else:
            tin = int(getattr(usage, "prompt_tokens", 0) or 0)
        self._meter.record("embed", kw["model"], tin, 0)
        return response


class MeteredOpenAI:
    """The three endpoints post-graph-rag's LLMService uses, wrapped."""

    def __init__(self, inner: Any, meter: UsageMeter, tag: str):
        self.captured: list[dict] = []
        self.chat = SimpleNamespace(completions=_MeteredCompletions(
            inner.chat.completions, meter, tag, self.captured))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=_MeteredCompletions(
            inner.beta.chat.completions, meter, tag, self.captured)))
        self.embeddings = _MeteredEmbeddings(inner.embeddings, meter, tag)


def make_llm_service(config: Any, meter: UsageMeter, tag: str) -> Any:
    """post-graph-rag LLMService with 1536-dim embeddings and metering.

    The dimensions parameter is the DimLLMService approach of smoke.py
    (pgr.md section 3): gemini-embedding-001 returns 3072 floats unless asked,
    and pgvector will not index an HNSW column above 2000.
    """
    from post_graph_rag.llm import LLMService

    class RunnerLLMService(LLMService):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.client = MeteredOpenAI(self.client, meter, tag)

        def _encoding_format_kwargs(self):
            kwargs = super()._encoding_format_kwargs()
            kwargs["dimensions"] = self.config.embedding_dim
            return kwargs

    return RunnerLLMService(config)


# ----------------------------------------------------------------------------
# Documents and turn spans (section 4 item 6)
# ----------------------------------------------------------------------------
def oneline(text: str) -> str:
    """Internal newlines replaced by spaces; nothing else changes."""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")


def build_session_document(turns: list[dict], header: str) -> tuple[str, list[tuple[int, int]]]:
    """One line per turn under his header line; returns the text and the turn spans.

    spans[i] is (char_start, char_end) of turn i's whole line "role: text",
    end exclusive, in the returned text.
    """
    lines, spans = [], []
    pos = len(header)
    for t in turns:
        line = f"{t.get('role', '')}: {oneline(t.get('content', ''))}"
        lines.append(line)
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return header + "\n".join(lines), spans


def map_chunk(doc_text: str, spans: list[tuple[int, int]], content: str) -> Optional[dict]:
    """Locate a returned chunk as an exact substring and map it to turns.

    Returns char_start, char_end (exclusive), turn_start, turn_end (inclusive)
    or None when the content is not an exact substring. A chunk that overlaps
    no turn line (header only) keeps null turns.
    """
    if not content:
        return None
    pos = doc_text.find(content)
    if pos < 0:
        return None
    a, b = pos, pos + len(content)
    hit = [i for i, (s, e) in enumerate(spans) if e > a and s < b]
    return {"char_start": a, "char_end": b,
            "turn_start": hit[0] if hit else None, "turn_end": hit[-1] if hit else None}


# ----------------------------------------------------------------------------
# Query variants (section 5)
# ----------------------------------------------------------------------------
def relation_line(r: dict, render_validity: bool = True) -> str:
    """One relation the way his query() renders it into the synthesis prompt."""
    vf, vt = r.get("valid_from"), r.get("valid_to")
    validity = ""
    if render_validity:
        if vf and vt:
            validity = f" [valid {vf} to {vt}]"
        elif vf:
            validity = f" [from {vf}]"
        elif vt:
            validity = f" [until {vt}]"
    return (f"- ({r['src_id']}) --[{'NOT ' if r.get('negated') else ''}{r['relation_type']}"
            f" (weight={r['weight']})]--> ({r['tgt_id']}){validity}: {r['description']}")


def context_chars(data: dict) -> dict:
    """Characters of the returned chunks plus the relation lines (a side figure)."""
    d = data.get("data", data)
    chunks = sum(len(c.get("content") or "") for c in d.get("chunks", []))
    rels = sum(len(relation_line(r)) for r in d.get("relationships", []))
    return {"chunks": chunks, "relation_lines": rels, "total": chunks + rels,
            "n_chunks": len(d.get("chunks", [])), "n_relations": len(d.get("relationships", []))}


def chunk_line(k: int, document: str, content: str) -> str:
    """One passage the way the bench renders it for chandan_live (multicard.part1.competitors.chunk_line)."""
    return f"Chunk [{k}] ({document}): {content}"


def entity_line(e: dict) -> str:
    """One entity the way the bench renders it (multicard.part1.competitors.entity_line)."""
    return f"- Entity {e.get('entity_name')} ({e.get('entity_type')}): {e.get('description') or ''}"


def rendered_units(data: dict) -> list[tuple[str, str]]:
    """(unit id, line) for every unit the bench renders from one query_data
    output, in his block order, deduplicated by id as the bench's render
    does: the chunk lines, the entity lines, the relation lines."""
    d = data.get("data", data)
    out, seen = [], set()
    for k, c in enumerate(d.get("chunks", []), start=1):
        meta = c.get("metadata") or {}
        uid = f"chunk:{c.get('chunk_id')}"
        if uid not in seen:
            seen.add(uid)
            out.append((uid, chunk_line(k, str(meta.get("document") or ""), c.get("content") or "")))
    for e in d.get("entities", []):
        uid = f"entity:{e.get('entity_name')}"
        if uid not in seen:
            seen.add(uid)
            out.append((uid, entity_line(e)))
    for r in d.get("relationships", []):
        uid = f"rel:{r.get('edge_id')}"
        if uid not in seen:
            seen.add(uid)
            out.append((uid, relation_line(r)))
    return out


def rendered_tokens(data: dict, counter) -> int:
    """Tokens of the units as the bench renders them, summed per unit as the
    bench's render_native counts them (design section 5: B counted by the
    bench TokenCounter over the rendered units)."""
    return sum(counter.count(line) for _, line in rendered_units(data))


async def raise_until(run_k: Callable[[int], Awaitable[dict]], shipped: dict, counter,
                      targets: dict = None, steps: tuple = RAISE_STEPS,
                      shipped_k: int = SHIPPED_TOP_K) -> dict:
    """Raise top_k stepwise until the rendered units count at least B tokens.

    run_k(top_k) returns a query_data output; counter is the bench token
    counter. Returns, per variant name, the top_k reached, the output, the
    rendered token count, whether B was reached, and the variant or "shipped"
    whose call is reused when no new call was needed. design gap: the ladder
    ends at the last step (RAISE_STEPS); a variant that did not reach B there
    is returned with reached False and counted by the caller.
    """
    targets = targets or dict(BUDGET_TOKENS)
    outputs = {shipped_k: ("shipped", shipped)}
    tokens = {shipped_k: rendered_tokens(shipped, counter)}
    k, current = shipped_k, shipped
    out = {}
    for name, target in targets.items():
        while tokens[k] < target:
            nxt = [s for s in steps if s > k]
            if not nxt:
                break
            k = nxt[0]
            current = await run_k(k)
            outputs[k] = (name, current)
            tokens[k] = rendered_tokens(current, counter)
        owner, _ = outputs[k]
        out[name] = {"top_k": k, "data": current, "rendered_tokens": tokens[k],
                     "reached": tokens[k] >= target, "target_tokens": target,
                     "reused_from": None if owner == name else owner}
    return out


# ----------------------------------------------------------------------------
def _scrub_nul(obj: Any, _seen: Optional[set] = None) -> Any:
    """Remove NUL characters from every string reachable inside obj, in place
    where possible. Handles dicts, lists, tuples, sets, dataclasses, pydantic
    models and plain objects with a __dict__. Returns the (possibly new) value
    for immutable containers and strings."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, str):
        return obj.replace("\x00", "") if "\x00" in obj else obj
    if isinstance(obj, (bytes, int, float, bool, type(None))):
        return obj
    oid = id(obj)
    if oid in _seen:
        return obj
    _seen.add(oid)
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _scrub_nul(obj[k], _seen)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _scrub_nul(v, _seen)
        return obj
    if isinstance(obj, tuple):
        return tuple(_scrub_nul(v, _seen) for v in obj)
    if isinstance(obj, set):
        return {_scrub_nul(v, _seen) for v in obj}
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for k in list(d.keys()):
            try:
                v = d[k]
                nv = _scrub_nul(v, _seen)
                if nv is not v:
                    setattr(obj, k, nv)
            except Exception:  # noqa: BLE001
                pass
    return obj


# Shards and projection (section 2 and section 4 item 6)
# ----------------------------------------------------------------------------
def plan_shards(counts: list[tuple[str, int]], shard_sessions: int) -> list[list[str]]:
    """Cut the question list into shards of about shard_sessions sessions, on question boundaries."""
    shards, cur, n = [], [], 0
    for qid, k in counts:
        cur.append(qid)
        n += k
        if n >= shard_sessions:
            shards.append(cur)
            cur, n = [], 0
    if cur:
        shards.append(cur)
    return shards


def projection(usd: float, sessions: int, seconds: float, run_sessions: int,
               project_sessions: int) -> dict:
    per_usd = usd / sessions if sessions else None
    per_sec = seconds / sessions if sessions else None
    return {
        "sessions_measured": sessions, "usd_measured": round(usd, 6),
        "seconds_measured": round(seconds, 1),
        "usd_per_session": per_usd, "seconds_per_session": per_sec,
        "run_sessions": run_sessions,
        "projected_usd_run": per_usd * run_sessions if per_usd is not None else None,
        "projected_hours_run": per_sec * run_sessions / 3600 if per_sec is not None else None,
        "project_sessions": project_sessions,
        "projected_usd_full": per_usd * project_sessions if per_usd is not None else None,
        "projected_hours_full": per_sec * project_sessions / 3600 if per_sec is not None else None,
    }


# ----------------------------------------------------------------------------
# Question id lists
# ----------------------------------------------------------------------------
_CORPUS_KEYS = {"lme": ("lme", "longmemeval", "longmemeval_s"),
                "mhrag": ("mhrag", "multihoprag", "multihop-rag")}


def _pick_corpus(d: dict, corpus: str):
    for k in _CORPUS_KEYS[corpus]:
        if k in d:
            return d[k]
    raise KeyError(f"no key for corpus {corpus} among {sorted(d)}")


def read_id_list(path: pathlib.Path, corpus: str, key: str = "ORDER") -> list[str]:
    """A JSON list of ids, or a dict holding the list under key (then under the corpus)."""
    obj = json.loads(pathlib.Path(path).read_text())
    if isinstance(obj, dict):
        obj = obj[key] if key in obj else _pick_corpus(obj, corpus)
        if isinstance(obj, dict):
            obj = _pick_corpus(obj, corpus)
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected a list of ids under {key}")
    return [str(x) for x in obj]


def restrict(order: list[str], subset: Optional[list[str]], limit: int = 0) -> list[str]:
    """ORDER restricted to a subset, keeping ORDER, then cut to limit."""
    ids = order if subset is None else [q for q in order if q in set(subset)]
    return ids[:limit] if limit else ids


# ----------------------------------------------------------------------------
# Parquet export (the shared schema)
# ----------------------------------------------------------------------------
def write_tables(out_dir: pathlib.Path, entities: list[dict], relations: list[dict],
                 chunks: list[dict], mentions: list[dict]) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "entities": pa.schema([("entity_key", pa.string()), ("name", pa.string()),
                               ("aliases", pa.list_(pa.string())), ("description", pa.string()),
                               ("entity_type", pa.string())]),
        "relations": pa.schema([("edge_id", pa.string()), ("src_key", pa.string()),
                                ("tgt_key", pa.string()), ("relation_type", pa.string()),
                                ("description", pa.string()), ("weight", pa.float64()),
                                ("negated", pa.bool_()), ("confidence", pa.float64()),
                                ("valid_from", pa.string()), ("valid_to", pa.string()),
                                ("t_created", pa.string()), ("t_expired", pa.string()),
                                ("superseded_by", pa.string()), ("asserted_at", pa.string()),
                                ("sources", pa.list_(pa.string())),
                                ("embedding", pa.list_(pa.float32()))]),
        "chunks": pa.schema([("chunk_id", pa.string()), ("content", pa.string()),
                             ("doc_id", pa.string()), ("doc_date", pa.string()),
                             ("turn_start", pa.int32()), ("turn_end", pa.int32()),
                             ("char_start", pa.int32()), ("char_end", pa.int32())]),
        "doc_mentions": pa.schema([("entity_key", pa.string()), ("chunk_id", pa.string())]),
    }
    rows = {"entities": entities, "relations": relations, "chunks": chunks, "doc_mentions": mentions}
    counts = {}
    for name, schema in schemas.items():
        table = pa.Table.from_pylist(rows[name], schema=schema)
        pq.write_table(table, out_dir / f"{name}.parquet")
        counts[name] = table.num_rows
    return counts


def _payload(row: Any) -> dict:
    p = row["payload"]
    return json.loads(p) if isinstance(p, str) else (p or {})


def _vec(text: Optional[str]) -> Optional[list[float]]:
    if not text:
        return None
    return [float(x) for x in text.strip("[]").split(",") if x.strip()]


async def export_space(rag: Any, realm: str, space: str, out_dir: pathlib.Path,
                       doc_texts: dict[str, str], doc_spans: dict[str, list],
                       doc_dates: dict[str, str]) -> dict:
    """Read the space's tables and write the four parquet files.

    Relation provenance comes from the relations table (payload sources by
    edge id). Chunks are mapped to turns by exact substring location in the
    document text this runner built; unmapped chunk ids are returned.
    """
    client = rag.store.client
    ref = {t: client._get_table_ref(t, realm) for t in ("documents", "entities", "relations", "doc_mentions")}
    ents = await client._fetch(
        f"SELECT id, payload FROM {ref['entities']} WHERE realm = $1 AND space = $2 ORDER BY id", realm, space)
    rels = await client._fetch(
        f"SELECT id, from_id, to_id, relation_type, payload, created_at, updated_at, embedding::text AS emb "
        f"FROM {ref['relations']} WHERE realm = $1 AND space = $2 ORDER BY id", realm, space)
    docs = await client._fetch(
        f"SELECT id, payload FROM {ref['documents']} WHERE realm = $1 AND space = $2 ORDER BY id", realm, space)
    ments = await client._fetch(
        f"SELECT from_id, to_id FROM {ref['doc_mentions']} WHERE realm = $1 AND space = $2 ORDER BY id", realm, space)

    entities = []
    for r in ents:
        p = _payload(r)
        entities.append({"entity_key": str(r["id"]), "name": p.get("name") or "",
                         "aliases": [str(a) for a in (p.get("aliases") or [])],
                         "description": p.get("description") or "", "entity_type": p.get("type")})
    relations = []
    for r in rels:
        p = _payload(r)
        asserted = r["updated_at"] or r["created_at"]
        conf = p.get("confidence")
        relations.append({
            "edge_id": str(r["id"]), "src_key": str(r["from_id"]), "tgt_key": str(r["to_id"]),
            "relation_type": r["relation_type"], "description": p.get("description") or "",
            "weight": float(p.get("weight", 1) or 0), "negated": bool(p.get("negated", False)),
            "confidence": float(conf) if conf is not None else None,
            "valid_from": p.get("valid_from"), "valid_to": p.get("valid_to"),
            "t_created": p.get("t_created"), "t_expired": p.get("t_expired"),
            "superseded_by": (str(p["superseded_by"]) if p.get("superseded_by") is not None else None),
            "asserted_at": asserted.isoformat() if asserted else None,
            "sources": [str(s) for s in (p.get("sources") or [])],
            "embedding": _vec(r["emb"]),
        })
    chunks, unmapped = [], []
    for r in docs:
        p = _payload(r)
        doc_id = p.get("document") or ""
        content = p.get("text") or ""
        loc = map_chunk(doc_texts.get(doc_id, ""), doc_spans.get(doc_id, []), content)
        if loc is None:
            unmapped.append(str(r["id"]))
            loc = {"char_start": -1, "char_end": -1, "turn_start": None, "turn_end": None}
        chunks.append({"chunk_id": str(r["id"]), "content": content, "doc_id": doc_id,
                       "doc_date": doc_dates.get(doc_id, p.get("session_date") or p.get("doc_date") or ""),
                       **loc})
    mentions = [{"entity_key": str(r["to_id"]), "chunk_id": str(r["from_id"])} for r in ments]
    counts = write_tables(out_dir, entities, relations, chunks, mentions)
    return {"rows": counts, "unmapped_chunks": unmapped,
            "chunks_mapped": len(chunks) - len(unmapped),
            "relations_with_embedding": sum(1 for x in relations if x["embedding"]),
            "relations_with_sources": sum(1 for x in relations if x["sources"])}


async def drop_space(rag: Any, realm: str, space: str, vacuum: bool = True) -> None:
    """Delete every row of the space, its audit rows, and vacuum the realm's tables.

    Deleting documents and entities cascades to the edge tables. The audit
    tables carry a copy of every row (embedding included), so they are cleared
    per space to keep the database small. VACUUM removes the dead tuples from
    the shared HNSW indexes before the next space is queried.
    """
    client = rag.store.client
    for t in ("documents", "entities", "communities", "relations", "doc_mentions",
              "community_members", "community_children"):
        await client._execute(
            f"DELETE FROM {client._get_table_ref(t, realm)} WHERE realm = $1 AND space = $2", realm, space)
    for t in GRAPH_TABLES:
        await client._execute(
            f"DELETE FROM {client._get_table_ref(t + '_audit', realm)} WHERE realm = $1 AND space = $2",
            realm, space)
    if vacuum:
        for t in ("documents", "entities", "relations", "doc_mentions"):
            await client._execute(f"VACUUM {client._get_table_ref(t, realm)}")


async def space_row_count(rag: Any, realm: str, space: str, table: str = "documents") -> int:
    client = rag.store.client
    row = await client._fetchrow(
        f"SELECT count(*) AS n FROM {client._get_table_ref(table, realm)} WHERE realm = $1 AND space = $2",
        realm, space)
    return int(row["n"])


# ----------------------------------------------------------------------------
# Turn-span check (used by the smoke and by --verify)
# ----------------------------------------------------------------------------
def check_turn_spans(space_dir: pathlib.Path, sessions: dict) -> dict:
    """Share of exported chunks whose turn span reproduces the bench's session turns.

    A chunk passes when, for every turn index in its span, the bench turn
    rendered as "role: text" (newlines replaced) is a substring of the chunk
    content, and the chunk content is a substring of the document rebuilt
    from the bench turns.
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(pathlib.Path(space_dir) / "chunks.parquet").to_pylist()
    ok, failed = 0, []
    for r in rows:
        s = sessions.get(r["doc_id"])
        if s is None or r["turn_start"] is None:
            failed.append(r["chunk_id"])
            continue
        turns = s["turns"]
        good = True
        for i in range(r["turn_start"], r["turn_end"] + 1):
            if i >= len(turns) or f"{turns[i]['role']}: {oneline(turns[i]['content'])}" not in r["content"]:
                good = False
                break
        if good:
            ok += 1
        else:
            failed.append(r["chunk_id"])
    return {"chunks": len(rows), "mapped_to_bench_turns": ok,
            "share": ok / len(rows) if rows else None, "failed_chunk_ids": failed}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_harness(clone_root: pathlib.Path) -> Any:
    """His evaluation/longmemeval/run.py as a module: CONVERSATIONAL_PROMPT, render_session, parse_date.

    post_graph_rag is imported first so the clone's copy on sys.path (run.py
    inserts its own root) cannot shadow the PyPI package.
    """
    import post_graph_rag  # noqa: F401

    path = pathlib.Path(clone_root) / "evaluation" / "longmemeval" / "run.py"
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location("pgr_lme_harness_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = before
    return mod


def load_lme(bench_root: pathlib.Path, data: pathlib.Path) -> dict:
    sys.path.insert(0, str(pathlib.Path(bench_root) / "src"))
    from multicard.data.longmemeval import load_longmemeval

    return load_longmemeval(data)


def load_mhrag(bench_root: pathlib.Path, raw: pathlib.Path) -> tuple[list, list]:
    sys.path.insert(0, str(pathlib.Path(bench_root) / "src"))
    from multicard.data.multihoprag import load_corpus, load_queries

    docs = load_corpus(pathlib.Path(raw))
    return docs, load_queries(pathlib.Path(raw), docs)


def build_article_document(doc: Any) -> tuple[str, str]:
    """An article as one document with its publication date in the body. Returns text and ISO date."""
    date = (doc.published_at or "")[:10]
    # design gap: the design fixes the date line for sessions only; articles use this line.
    header = f"[Article published on {date}]\n" if date else ""
    return header + doc.text, date


# ----------------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------------
def _atomic_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _config_dict(config: Any) -> dict:
    out = {}
    for k, v in vars(config).items():
        if k == "api_key":
            continue
        out[k] = sorted(v) if isinstance(v, set) else ([sorted(s) for s in v] if k == "exclusive_predicate_groups" else v)
    return out


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.prices = load_prices(args.models_json)
        for m in (args.index_model, args.answer_model):
            price_for(self.prices, m)
        self.meter = UsageMeter(self.prices, cap_usd=args.cap_usd)
        self.job_tag = args.job_tag
        self.out_root = pathlib.Path(args.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.started = now_iso()
        self.t_start = time.time()
        self.counter = BenchTokenCounter()
        self.run_log: dict = {
            "corpus": args.corpus, "job_tag": self.job_tag, "started": self.started, "ended": None,
            "args": {k: v for k, v in vars(args).items() if k != "key_file"} | {"key_file": str(args.key_file)},
            "index_model": args.index_model, "answer_model": args.answer_model,
            "embedding_model": args.embedding_model, "embedding_dim": args.embedding_dim,
            "cap_usd": args.cap_usd, "cap_usd_source": args.cap_usd_source,
            "shipped_top_k": args.shipped_top_k, "question_form": args.question_form,
            "budget_tokens": dict(BUDGET_TOKENS), "token_counter": f"bench TokenCounter ({BenchTokenCounter.MODEL}) "
                                                                    f"via tokenizers, {self.counter.path}",
            "raise_steps": list(RAISE_STEPS),
            "shards": [], "projection": None, "stopped": None,
            "questions_done": [], "questions_skipped_resume": [], "meter": None,
            # design gap: questions whose raised variant did not reach B at the top step
            "raised_not_reached": {name: [] for name in BUDGET_TOKENS},
        }
        self.run_log_path = self.out_root / f"run_{self.job_tag}.json"
        self.rag = None
        self.harness = None
        self.realm = args.realm or args.corpus

    # ------------------------------------------------------------ plumbing
    def log(self, msg: str) -> None:
        print(f"[{time.time() - self.t_start:8.0f}s usd {self.meter.total_usd:8.4f}] {msg}", flush=True)

    def save_run_log(self) -> None:
        self.run_log["meter"] = self.meter.summary()
        self.run_log["usage_missing"] = self.meter.usage_missing
        _atomic_json(self.run_log_path, self.run_log)

    def make_config(self, extraction_prompt: Optional[str]) -> Any:
        from post_graph_rag import RAGConfig

        a = self.args
        return RAGConfig(
            api_base=a.api_base, api_key=pathlib.Path(a.key_file).read_text().strip(),
            model=a.index_model, embedding_model=a.embedding_model, embedding_dim=a.embedding_dim,
            db_uri=a.db_uri, realm=self.realm, schema_per_realm=True,
            embed_relations=True, merge_strategy="rrf",
            max_concurrent_chunks=a.max_concurrent_chunks,
            pool_min_size=1, pool_max_size=4,
            extraction_prompt=extraction_prompt,
            max_retries=40, retry_deadline_secs=1800,
        )

    async def open(self, extraction_prompt: Optional[str]) -> None:
        from post_graph_rag import GraphRAG

        config = self.make_config(extraction_prompt)
        self.config = config
        self.rag = GraphRAG(config, llm=make_llm_service(config, self.meter, self.job_tag))
        await self.rag.initialize()

    async def close(self) -> None:
        if self.rag is not None:
            await self.rag.close()

    # ------------------------------------------------------------ indexing
    async def index_documents(self, space: str, docs: list[tuple[str, str, Any]]) -> dict:
        """Index (doc_id, text, metadata) triples into the space.

        The network half of index_document (_prepare_document) runs
        max_concurrent_chunks at a time; the write half (_write_document) runs
        in the given order. A refusal under the configured prompt is retried
        with the library's document prompt, as his harness does, with no other
        extraction in flight; a second refusal is recorded.
        """
        from post_graph_rag import DocumentMetadata  # noqa: F401

        rag = self.rag
        window = max(1, self.args.max_concurrent_chunks)
        results, refused, fallback_used = [], [], []
        t0 = time.time()
        for start in range(0, len(docs), window):
            batch = docs[start:start + window]
            t_batch = time.time()
            outcomes = await asyncio.gather(
                *[rag._prepare_document(text, meta, space, None) for _, text, meta in batch],
                return_exceptions=True)
            if self.meter.exceeded:
                self.meter.check()
            prepared = []
            for (doc_id, text, meta), outcome in zip(batch, outcomes):
                if isinstance(outcome, BaseException):
                    first = f"{type(outcome).__name__}: {str(outcome)[:300]}"
                    if self.meter.exceeded:
                        self.meter.check()
                    if self.args.no_prompt_fallback or rag.extractor._system_prompt is None:
                        refused.append({"doc_id": doc_id, "error": first, "fallback_error": None})
                        prepared.append((doc_id, text, None))
                        continue
                    saved = rag.extractor._system_prompt
                    rag.extractor._system_prompt = None
                    try:
                        outcome = await rag._prepare_document(text, meta, space, None)
                        fallback_used.append({"doc_id": doc_id, "error": first})
                    except BaseException as second:  # noqa: BLE001
                        if self.meter.exceeded:
                            self.meter.check()
                        refused.append({"doc_id": doc_id, "error": first,
                                        "fallback_error": f"{type(second).__name__}: {str(second)[:300]}"})
                        prepared.append((doc_id, text, None))
                        continue
                    finally:
                        rag.extractor._system_prompt = saved
                prepared.append((doc_id, text, outcome))
            for doc_id, text, item in prepared:
                if item is None:
                    continue
                # A model output can carry a NUL character, which Postgres
                # refuses ("invalid byte sequence for encoding UTF8: 0x00").
                # Strip it from every string in the prepared item before the
                # write, and treat any write failure as a refusal of that
                # document instead of a crash of the run (design section 5,
                # refused documents are recorded and the question continues).
                _scrub_nul(item)
                try:
                    written = await rag._write_document(item)
                except Exception as werr:  # noqa: BLE001
                    refused.append({"doc_id": doc_id, "error": f"write: {type(werr).__name__}: {str(werr)[:300]}",
                                    "fallback_error": None})
                    self.log(f"  {space}: write refused for {doc_id}: {type(werr).__name__}: {str(werr)[:120]}")
                    continue
                results.append({"doc_id": doc_id, "chars": len(text),
                                "entities": written["entities_extracted"], "triples": written["triples_extracted"],
                                "relations_added": written["relations_added"],
                                "superseded": written["relations_superseded"],
                                "mentions": written["mentions_added"], "document_id": written["document_id"]})
            self.log(f"  {space}: indexed {min(start + window, len(docs))}/{len(docs)} "
                     f"({time.time() - t_batch:.0f}s this window)")
        return {"documents": results, "refused": refused, "fallback_used": fallback_used,
                "seconds": time.time() - t0}

    # ------------------------------------------------------------ queries
    async def query_variants(self, space: str, question_sent: str, qid: str, qdir: pathlib.Path,
                             done: Optional[set] = None) -> dict:
        """The four variants of section 5 for one question, each saved to its own file."""
        from post_graph_rag import QueryParam

        rag = self.rag
        done = done or set()
        rag.config.model = self.args.answer_model
        summary = {}
        shipped_k = self.args.shipped_top_k

        def base(variant: str, top_k: int) -> dict:
            return {"corpus": self.args.corpus, "space_id": space, "question_id": qid,
                    "variant": variant, "mode": "mix", "top_k": top_k, "shipped_top_k": shipped_k,
                    "question_sent": question_sent, "question_form": self.args.question_form,
                    "index_model": self.args.index_model,
                    "answer_model": self.args.answer_model, "job_tag": self.job_tag}

        async def run_k(top_k: int) -> dict:
            m0, t0, s0 = self.meter.mark(), time.time(), now_iso()
            data = await rag.query_data(question_sent, param=QueryParam(mode="mix", top_k=top_k, space=space))
            data["_call"] = {"wall_seconds": round(time.time() - t0, 3), "started": s0, "ended": now_iso(),
                             "meter": self.meter.summary(m0)}
            return data

        if "shipped" in done and "raised_4k" in done and "raised_8k" in done:
            shipped = None
        else:
            shipped = await run_k(shipped_k)
            self.meter.check()
            raised = await raise_until(run_k, shipped, self.counter, shipped_k=shipped_k)
            self.meter.check()
            variants = {"shipped": {"top_k": shipped_k, "data": shipped, "reached": None, "reused_from": None,
                                    "rendered_tokens": rendered_tokens(shipped, self.counter), "target_tokens": None}}
            variants.update(raised)
            for name, v in variants.items():
                data = v["data"]
                call = data["_call"]
                rec = base(name, v["top_k"]) | {
                    "reused_from": v["reused_from"], "budget_reached": v["reached"],
                    "target_tokens": v["target_tokens"], "rendered_tokens": v["rendered_tokens"],
                    "context_chars": context_chars(data),
                    "wall_seconds": 0.0 if v["reused_from"] else call["wall_seconds"],
                    "started": call["started"], "ended": call["ended"],
                    "meter": self.meter.summary(0, 0) if v["reused_from"] else call["meter"],
                    "query_data": {k: val for k, val in data.items() if k != "_call"},
                }
                _atomic_json(qdir / f"{name}.json", rec)
                summary[name] = {"top_k": v["top_k"], "reused_from": v["reused_from"],
                                 "rendered_tokens": v["rendered_tokens"], "budget_reached": v["reached"],
                                 "context_chars": rec["context_chars"]["total"], "usd": rec["meter"]["usd"]}
                if name in BUDGET_TOKENS and v["reached"] is False:
                    self.run_log["raised_not_reached"][name].append(qid)

        if "full_uncut" not in done:
            captured: dict = {}
            orig = rag.query_data

            async def capturing(question, param=None):
                res = await orig(question, param=param)
                captured["data"] = res
                return res

            rag.query_data = capturing
            n_before = len(rag.llm.client.captured)
            m0, t0, s0 = self.meter.mark(), time.time(), now_iso()
            try:
                out = await rag.query(question_sent, param=QueryParam(mode="mix", top_k=shipped_k, space=space))
            finally:
                rag.query_data = orig
            self.meter.check()
            chats = rag.llm.client.captured[n_before:]
            synth = chats[-1] if chats else None
            context = None
            if synth and synth["messages"]:
                context = synth["messages"][-1]["content"]
            rec = base("full_uncut", shipped_k) | {
                "reused_from": None, "wall_seconds": round(time.time() - t0, 3),
                "started": s0, "ended": now_iso(), "meter": self.meter.summary(m0),
                "context_chars": context_chars(captured.get("data", {})),
                "rendered_tokens": rendered_tokens(captured.get("data", {}), self.counter),
                "query_data": captured.get("data"), "query": out,
                "context": context, "context_model": synth["model"] if synth else None,
                "answer": out.get("answer") if isinstance(out, dict) else str(out),
            }
            _atomic_json(qdir / "full_uncut.json", rec)
            summary["full_uncut"] = {"top_k": shipped_k, "context_chars": rec["context_chars"]["total"],
                                     "usd": rec["meter"]["usd"], "answer_chars": len(rec["answer"] or "")}
            rag.llm.client.captured.clear()
        rag.config.model = self.args.index_model
        return summary

    # ------------------------------------------------------------ lme
    async def run_lme(self) -> int:
        a = self.args
        ds = load_lme(a.bench_root, a.data)
        instances = {x.qid: x for x in ds["instances"]}
        sessions = ds["sessions"]
        if a.smoke:
            ids = sorted(q for q in instances if not instances[q].abstention)[:2]
        else:
            order = read_id_list(a.ids, "lme", a.ids_key)
            subset = None
            if a.subset:
                p = pathlib.Path(a.subset)
                subset = read_id_list(p, "lme", a.ids_key) if p.exists() else read_id_list(a.ids, "lme", a.subset)
            ids = restrict(order, subset, a.limit)
        missing = [q for q in ids if q not in instances]
        if missing:
            raise SystemExit(f"ids not in the data file: {missing[:5]}")
        counts = [(q, len(instances[q].session_ids)) for q in ids]
        shards = plan_shards(counts, a.shard_sessions)
        run_sessions = sum(k for _, k in counts)
        self.run_log["questions"] = ids
        self.run_log["run_sessions"] = run_sessions
        self.run_log["shard_plan"] = [{"shard": i + 1, "questions": len(s),
                                       "sessions": sum(dict(counts)[q] for q in s)} for i, s in enumerate(shards)]
        self.log(f"lme: {len(ids)} questions, {run_sessions} sessions, {len(shards)} shards of about "
                 f"{a.shard_sessions} sessions; index {a.index_model}, answer {a.answer_model}; realm {self.realm}")

        self.harness = load_harness(a.clone_root)
        await self.open(self.harness.CONVERSATIONAL_PROMPT)
        n_done = 0
        try:
            for si, shard in enumerate(shards, start=1):
                m_shard, t_shard, s_shard = self.meter.mark(), time.time(), 0
                for qid in shard:
                    qdir = self.out_root / qid
                    if (qdir / "meta.json").exists():
                        self.run_log["questions_skipped_resume"].append(qid)
                        self.log(f"{qid}: meta.json exists, skipped (resume)")
                        continue
                    n_done += 1
                    s_shard += await self.run_lme_question(instances[qid], sessions, qdir,
                                                           f"{n_done}/{len(ids)}")
                    self.run_log["questions_done"].append(qid)
                    self.save_run_log()
                shard_rec = {"shard": si, "questions": shard, "sessions_indexed": s_shard,
                             "seconds": round(time.time() - t_shard, 1),
                             "meter": self.meter.summary(m_shard), "ended": now_iso()}
                self.run_log["shards"].append(shard_rec)
                if si == 1:
                    proj = projection(shard_rec["meter"]["usd"], s_shard, shard_rec["seconds"],
                                      run_sessions, a.project_sessions)
                    self.run_log["projection"] = proj
                    self.save_run_log()
                    self.log(f"shard 1 done: {s_shard} sessions, USD {shard_rec['meter']['usd']:.4f}, "
                             f"{shard_rec['seconds']:.0f}s; per session USD "
                             f"{proj['usd_per_session'] if proj['usd_per_session'] is not None else 'n/a'} "
                             f"and {proj['seconds_per_session'] if proj['seconds_per_session'] is not None else 'n/a'}s; "
                             f"projected USD {proj['projected_usd_run']} over this run's {run_sessions} sessions, "
                             f"USD {proj['projected_usd_full']} over {a.project_sessions} sessions")
                    if a.cap_usd is not None and proj["projected_usd_run"] is not None \
                            and proj["projected_usd_run"] > a.cap_usd and len(shards) > 1:
                        self.run_log["stopped"] = (f"projection {proj['projected_usd_run']:.2f} USD over the "
                                                   f"cap {a.cap_usd:.2f}; stopped before shard 2")
                        self.log(self.run_log["stopped"])
                        break
                self.save_run_log()
        except BudgetExceeded as e:
            self.run_log["stopped"] = f"cap: {e}"
            self.log(f"stopped: {e}")
        finally:
            self.run_log["ended"] = now_iso()
            self.save_run_log()
            await self.close()
        return 0 if not self.run_log["stopped"] else 2

    async def run_lme_question(self, inst: Any, sessions: dict, qdir: pathlib.Path, label: str) -> int:
        from post_graph_rag import DocumentMetadata

        a = self.args
        qid, space = inst.qid, inst.qid
        m_q, t_q, s_q = self.meter.mark(), time.time(), now_iso()
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "queries").mkdir(exist_ok=True)
        # A crashed earlier run may have left rows in this space.
        if await space_row_count(self.rag, self.realm, space):
            self.log(f"{qid}: leftover rows in the space, dropping them first")
            await drop_space(self.rag, self.realm, space, vacuum=False)

        sids = sorted(inst.session_ids, key=lambda s: sessions[s]["date"])
        docs, texts, spans, dates, empty = [], {}, {}, {}, []
        for sid in sids:
            s = sessions[sid]
            date = self.harness.parse_date(s["date"]) or s["date"]
            header = self.harness.render_session([], date)
            text, span = build_session_document(s["turns"], header)
            if not s["turns"] or not text.strip():
                empty.append(sid)
                continue
            texts[sid], spans[sid], dates[sid] = text, span, date
            meta = DocumentMetadata(document=sid, source=f"session://{sid}", category="chat_session",
                                    extra={"session_date": date})
            docs.append((sid, text, meta))
        self.log(f"[{label}] {qid} ({inst.qtype}): {len(docs)} sessions, "
                 f"{sum(len(t) for t in texts.values())} chars")
        idx = await self.index_documents(space, docs)
        index_meter = self.meter.summary(m_q)
        self.log(f"[{label}] {qid}: indexed {len(idx['documents'])} sessions, {len(idx['refused'])} refused, "
                 f"{len(idx['fallback_used'])} on the document prompt, {idx['seconds']:.0f}s, "
                 f"USD {index_meter['usd']:.4f}")

        sent = question_sent(inst.question, inst.question_date, a.question_form)
        m_query = self.meter.mark()
        t_query = time.time()
        qsummary = await self.query_variants(space, sent, qid, qdir / "queries")
        query_meter = self.meter.summary(m_query)

        export = await export_space(self.rag, self.realm, space, qdir, texts, spans, dates)
        meta = {
            "corpus": "lme", "space_id": space, "realm": self.realm, "question_id": qid,
            "question_type": inst.qtype, "question": inst.question, "question_date": inst.question_date,
            "question_sent": sent, "question_form": a.question_form,
            "question_form_note": ("reader_sweep: the published run's form, reader_sweep.py lines 220-221, "
                                   "'Today is {date}. {question}'; run_py: his original harness, run.py lines "
                                   "381-392, a three-rule answer-side block before that text"),
            "abstention": inst.abstention,
            "package": self.package_info(), "index_model": a.index_model, "answer_model": a.answer_model,
            "embedding_model": a.embedding_model, "embedding_dim": a.embedding_dim,
            "extraction_prompt": "CONVERSATIONAL_PROMPT from his evaluation/longmemeval/run.py",
            "prompt_fallback": "library document prompt on refusal, as in his run.py",
            "shipped_top_k": a.shipped_top_k,
            "shipped_top_k_note": ("8 from the published run (reader_sweep.py line 244); the package default "
                                   "QueryParam.top_k is 5 (models.py line 123) and run.py uses 32 (line 393); "
                                   "the value is chosen by --shipped-top-k, decided in pgr.md"),
            "budget_tokens": dict(BUDGET_TOKENS), "raise_steps": list(RAISE_STEPS),
            "token_counter": self.run_log["token_counter"],
            "max_concurrent_chunks": a.max_concurrent_chunks,
            "concurrency_note": (f"extraction (the network half of index_document) ran {a.max_concurrent_chunks} "
                                 "sessions at a time with the writes in session-date order; his harness calls "
                                 "index_document one session at a time (run.py line 164)"),
            "cap_usd": a.cap_usd, "cap_usd_source": a.cap_usd_source,
            "config": _config_dict(self.config), "job_tag": self.job_tag,
            "build_timestamp": a.build_timestamp, "started": s_q, "ended": now_iso(),
            "document_count": len(docs), "documents_indexed": len(idx["documents"]),
            "documents_refused": len(idx["refused"]), "documents_empty_skipped": empty,
            "refused": idx["refused"], "fallback_used": idx["fallback_used"], "documents": idx["documents"],
            "index_wall_seconds": round(idx["seconds"], 1), "index_meter": index_meter,
            "query_wall_seconds": round(time.time() - t_query, 1), "query_meter": query_meter,
            "queries": qsummary, "export": export, "question_meter": self.meter.summary(m_q),
            "question_wall_seconds": round(time.time() - t_q, 1),
            "session_order": [sid for sid, _, _ in docs],
        }
        if a.verify:
            meta["turn_span_check"] = check_turn_spans(qdir, sessions)
            self.log(f"[{label}] {qid}: turn-span check {meta['turn_span_check']['mapped_to_bench_turns']}/"
                     f"{meta['turn_span_check']['chunks']}")
        await drop_space(self.rag, self.realm, space)
        _atomic_json(qdir / "meta.json", meta)
        self.log(f"[{label}] {qid}: done in {meta['question_wall_seconds']:.0f}s, USD "
                 f"{meta['question_meter']['usd']:.4f} (index {index_meter['usd']:.4f}, "
                 f"queries {query_meter['usd']:.4f}); export {export['rows']}, "
                 f"unmapped chunks {len(export['unmapped_chunks'])}")
        return len(idx["documents"])

    # ------------------------------------------------------------ mhrag
    async def run_mhrag(self) -> int:
        from post_graph_rag import DocumentMetadata

        a = self.args
        docs, queries = load_mhrag(a.bench_root, a.mhrag_raw)
        by_qid = {q.qid: q for q in queries}
        if a.smoke:
            ids = [q.qid for q in queries if q.question_type != "null_query"][:2]
        else:
            order = read_id_list(a.ids, "mhrag", a.ids_key)
            subset = None
            if a.subset:
                p = pathlib.Path(a.subset)
                subset = read_id_list(p, "mhrag", a.ids_key) if p.exists() else read_id_list(a.ids, "mhrag", a.subset)
            ids = restrict(order, subset, a.limit)
        missing = [q for q in ids if q not in by_qid]
        if missing:
            raise SystemExit(f"query ids not in the corpus cache: {missing[:5]}")
        ids = [q for q in ids if by_qid[q].question_type != "null_query"]
        if a.smoke:
            docs = docs[:a.smoke_docs]
        space = "corpus"
        sdir = self.out_root / space
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "queries").mkdir(exist_ok=True)
        if (sdir / "meta.json").exists():
            self.log("mhrag: meta.json exists, nothing to do (resume)")
            return 0
        self.run_log["questions"] = ids
        self.run_log["run_sessions"] = len(docs)
        self.log(f"mhrag: {len(docs)} documents, {len(ids)} non-null queries; index {a.index_model}, "
                 f"answer {a.answer_model}; realm {self.realm}")
        await self.open(None)
        m_run, s_run = self.meter.mark(), now_iso()
        try:
            ordered = sorted(docs, key=lambda d: d.published_at)
            texts, dates, items = {}, {}, []
            for d in ordered:
                text, date = build_article_document(d)
                texts[d.doc_id], dates[d.doc_id] = text, date
                items.append((d.doc_id, text, DocumentMetadata(
                    document=d.doc_id, source=f"article://{d.doc_id}", category=d.category or None,
                    extra={"published_at": d.published_at, "doc_date": date})))
            index_marker = sdir / "index.json"
            have = await space_row_count(self.rag, self.realm, space)
            if index_marker.exists() and have == json.loads(index_marker.read_text())["documents_indexed"]:
                idx = json.loads(index_marker.read_text())
                self.log(f"mhrag: index marker matches {have} documents in the space, indexing skipped (resume)")
            else:
                if have:
                    self.log(f"mhrag: {have} leftover rows in the space, dropping them first")
                    await drop_space(self.rag, self.realm, space, vacuum=False)
                m_idx = self.meter.mark()
                idx = await self.index_documents(space, items)
                idx["documents_indexed"] = len(idx["documents"])
                idx["index_meter"] = self.meter.summary(m_idx)
                idx["document_order"] = [d.doc_id for d in ordered]
                idx["started"], idx["ended"] = s_run, now_iso()
                _atomic_json(index_marker, idx)
                proj = projection(idx["index_meter"]["usd"], len(idx["documents"]), idx["seconds"],
                                  len(docs), a.project_sessions)
                self.run_log["projection"] = proj
                self.log(f"mhrag: indexed {len(idx['documents'])} documents, {len(idx['refused'])} refused, "
                         f"{idx['seconds']:.0f}s, USD {idx['index_meter']['usd']:.4f}")
            self.save_run_log()
            for n, qid in enumerate(ids, start=1):
                q = by_qid[qid]
                qdir = sdir / "queries" / qid
                done = {v for v in VARIANTS if (qdir / f"{v}.json").exists()}
                if len(done) == len(VARIANTS):
                    self.run_log["questions_skipped_resume"].append(qid)
                    continue
                qdir.mkdir(parents=True, exist_ok=True)
                m_q, t_q = self.meter.mark(), time.time()
                # MultiHop-RAG queries carry no date; the question form adds nothing for reader_sweep
                summary = await self.query_variants(space, question_sent(q.query, None, a.question_form), qid, qdir, done)
                self.run_log["questions_done"].append(qid)
                if n % 10 == 0 or n == len(ids):
                    self.save_run_log()
                self.log(f"[{n}/{len(ids)}] {qid} ({q.question_type}): {time.time() - t_q:.0f}s, "
                         f"USD {self.meter.summary(m_q)['usd']:.4f}, "
                         + ", ".join(f"{k} k={v['top_k']}" for k, v in summary.items()))
            export = await export_space(self.rag, self.realm, space, sdir, texts, {}, dates)
            meta = {
                "corpus": "mhrag", "space_id": space, "realm": self.realm,
                "package": self.package_info(), "index_model": a.index_model, "answer_model": a.answer_model,
                "embedding_model": a.embedding_model, "embedding_dim": a.embedding_dim,
                "extraction_prompt": "library default", "config": _config_dict(self.config),
                "shipped_top_k": a.shipped_top_k, "question_form": a.question_form,
                "budget_tokens": dict(BUDGET_TOKENS), "raise_steps": list(RAISE_STEPS),
                "token_counter": self.run_log["token_counter"],
                "max_concurrent_chunks": a.max_concurrent_chunks,
                "concurrency_note": (f"extraction ran {a.max_concurrent_chunks} articles at a time with the writes "
                                     "in publication-date order; his harness indexes one document at a time"),
                "cap_usd": a.cap_usd, "cap_usd_source": a.cap_usd_source,
                "job_tag": self.job_tag, "build_timestamp": a.build_timestamp,
                "started": s_run, "ended": now_iso(),
                "document_count": len(docs), "documents_indexed": idx["documents_indexed"],
                "documents_refused": len(idx["refused"]), "refused": idx["refused"],
                "fallback_used": idx["fallback_used"], "documents": idx["documents"],
                "document_order": idx["document_order"],
                "index_wall_seconds": round(idx["seconds"], 1), "index_meter": idx["index_meter"],
                "queries": ids, "query_meter": self.meter.summary(m_run), "export": export,
            }
            if not a.keep_space:
                await drop_space(self.rag, self.realm, space)
            _atomic_json(sdir / "meta.json", meta)
            self.log(f"mhrag: done, export {export['rows']}, unmapped chunks {len(export['unmapped_chunks'])}")
        except BudgetExceeded as e:
            self.run_log["stopped"] = f"cap: {e}"
            self.log(f"stopped: {e}")
        finally:
            self.run_log["ended"] = now_iso()
            self.save_run_log()
            await self.close()
        return 0 if not self.run_log["stopped"] else 2

    def package_info(self) -> dict:
        import post_graph_rag

        return {"name": "post-graph-rag", "version": post_graph_rag.__version__,
                "file": post_graph_rag.__file__, "python": sys.version.split()[0]}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    models_json = BENCH / "docs" / "part1" / "env" / "models.json"
    defaults = json.loads(models_json.read_text()) if models_json.exists() else {}
    chat_model = defaults.get("chat_model", "gemini-2.5-flash-lite")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", choices=("lme", "mhrag"), required=True)
    ap.add_argument("--data", default=str(BENCH / "data" / "raw" / "longmemeval_s.json"),
                    help="LongMemEval_S json (lme)")
    ap.add_argument("--mhrag-raw", default=str(BENCH / "data" / "raw" / "multihoprag"),
                    help="MultiHop-RAG JSONL cache directory (mhrag)")
    ap.add_argument("--ids", help="JSON id list, or subsets.json (ORDER under --ids-key, then the corpus)")
    ap.add_argument("--ids-key", default="ORDER")
    ap.add_argument("--subset", help="a key in the ids file (e.g. GRAPHITI_150) or a JSON file; restricts ORDER")
    ap.add_argument("--limit", type=int, default=0, help="first N ids after the restriction")
    ap.add_argument("--smoke", action="store_true",
                    help="lme: the first two answerable questions in sorted id order; mhrag: first two non-null queries")
    ap.add_argument("--smoke-docs", type=int, default=20, help="mhrag smoke: documents indexed")
    ap.add_argument("--index-model", default=chat_model)
    ap.add_argument("--answer-model", default=None, help="default: the index model")
    ap.add_argument("--calibration", action="store_true",
                    help="section 13 calibration row: index chandan_index_model, answer chandan_answer_model "
                         "from models.json (gemini-3.7-flash and gemini-3.6-flash)")
    ap.set_defaults(calibration_models=(defaults.get("chandan_index_model", "gemini-3.7-flash"),
                                        defaults.get("chandan_answer_model", "gemini-3.6-flash")))
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--embedding-dim", type=int, default=1536)
    ap.add_argument("--api-base", default="http://127.0.0.1:4000/v1")
    ap.add_argument("--key-file", default=str(ENV / ".proxy_key"))
    ap.add_argument("--db-uri", default="postgresql://pgr:pgr@127.0.0.1:5433/pgr")
    ap.add_argument("--realm", default=None, help="Postgres schema; default: the corpus name")
    ap.add_argument("--out-root", default=None, help="default: bench data/part1/pgr/<corpus>")
    ap.add_argument("--job-tag", default=None, help="sent with every request; default pgr-<corpus>-<timestamp>")
    ap.add_argument("--shard-sessions", type=int, default=2000)
    ap.add_argument("--cap-usd", type=float, default=None,
                    help="hard stop when the metered spend passes this; projection check after shard 1. "
                         "Default: the design section 11 cap for the run (lme 130, calibration 70, mhrag 10)")
    ap.add_argument("--shipped-top-k", type=int, default=SHIPPED_TOP_K,
                    help="the shipped variant's result limit (see SHIPPED_TOP_K for the three candidates)")
    ap.add_argument("--question-form", choices=QUESTION_FORMS, default="reader_sweep",
                    help="how the question reaches the query call (see QUESTION_FORMS)")
    ap.add_argument("--project-sessions", type=int, default=25112,
                    help="sessions the projection after shard 1 is printed for (25,112 haystack slots)")
    ap.add_argument("--max-concurrent-chunks", type=int, default=8)
    ap.add_argument("--models-json", default=str(models_json))
    ap.add_argument("--build-timestamp", default=None)
    ap.add_argument("--bench-root", default=str(BENCH))
    ap.add_argument("--clone-root", default=str(CLONE))
    ap.add_argument("--verify", action="store_true", help="lme: run the turn-span check per question")
    ap.add_argument("--no-prompt-fallback", action="store_true",
                    help="do not retry a refused session with the library document prompt")
    ap.add_argument("--keep-space", action="store_true", help="mhrag: do not drop the space after export")
    return ap


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    a = build_parser().parse_args(argv)
    if a.calibration:
        a.index_model, a.answer_model = a.calibration_models
    if a.answer_model is None:
        a.answer_model = a.index_model
    if a.cap_usd is None:
        key = "calibration" if a.calibration else a.corpus
        a.cap_usd = CAP_USD[key]
        a.cap_usd_source = f"design section 11 default for {key}"
    else:
        a.cap_usd_source = "--cap-usd override"
    if a.out_root is None:
        a.out_root = str(BENCH / "data" / "part1" / "pgr" / a.corpus)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if a.job_tag is None:
        a.job_tag = f"pgr-{a.corpus}-{stamp}"
    if a.build_timestamp is None:
        a.build_timestamp = now_iso()
    if not a.smoke and not a.ids:
        raise SystemExit("--ids is required unless --smoke is given")
    return a


def main(argv: Optional[list[str]] = None) -> int:
    a = parse_args(argv)
    runner = Runner(a)
    if a.corpus == "lme":
        code = asyncio.run(runner.run_lme())
    else:
        code = asyncio.run(runner.run_mhrag())
    runner.log(f"run log: {runner.run_log_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())

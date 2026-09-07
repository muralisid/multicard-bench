"""Graphiti runner for Part 1 (design sections 2, 5, 11 and 13; graphiti arm).

Three commands, run from part1-tools/graphiti with its own venv:

  .venv/bin/python run_groups.py pilot --qid <qid> [--n 20] [--max-usd 60]
  .venv/bin/python run_groups.py run --variant per_turn [--k 3] [--max-usd 60]
  .venv/bin/python run_groups.py cleanup --prefix pilot-

pilot (design section 2, "The Graphiti pilot", and the pilot paragraph of
section 5): one question, its first N haystack sessions in session-date order,
ingested three ways into three separate groups. Per variant it records LLM
calls, tokens in and out, embedding calls, wall seconds, requests per minute,
USD at the models.json price of the study model, edge and node counts, and
evidence presence per marked evidence turn. It projects the cost of each
variant over GRAPHITI_150 as USD per rendered character in the pilot times
the rendered characters of every haystack slot of the 150 questions (one
group per question holds that question's haystack sessions, section 2, so a
session shared by two questions is ingested twice), read from
docs/part1/subsets.json and the bench loader at pilot time. It applies the
choice rule (highest evidence presence whose projection fits USD 60, ties to
per-turn), prints the table, writes the table and the choice into
part1-tools/env/graphiti.md under a "Pilot" heading, copies graphiti.md to
the bench docs/part1/env/, and deletes the pilot groups.

run (design section 5, graphiti arm): the GRAPHITI_150 questions in ORDER
(docs/part1/subsets.json), the chosen variant, one group at a time in that
order (section 2: a capped or partial run is a defined prefix of ORDER; --k
above 1 ingests several groups at once, is not a prefix, and is recorded as
such). Each group is searched with the library's COMBINED_HYBRID_SEARCH_RRF
recipe at the shipped limit (20 edges, 20 nodes) and at raised limits (40,
then 80) until the rendered FACT and ENTITY lines count 4,000 and 8,000
tokens by the bench TokenCounter (section 5: B in tokens), rendered with Zep's
template, exported to parquet, and deleted. A question whose directory
already holds meta.json is skipped. The pilot's spend counts against the
one 60 USD cap of section 11 (--pilot-json or --spent-usd).

Ingestion variants (design section 5):
  text_session     one text episode per session, turns as "role: text" lines
  message_session  one message episode per session, same rendering
  per_turn         one message episode per turn, in turn order (Zep's own
                   granularity; the other two are labelled as deviations)
All with group_id equal to the question id and reference_time equal to the
session date at 00:00 UTC. The bench loader keeps the date only, so the time
of day printed in the raw file is dropped; meta.json records this.

Output layout (OUT is data/part1/graphiti for run, data/part1/graphiti_pilot
for pilot):
  run:   OUT/<qid>/edges.parquet, nodes.parquet, episodes.parquet,
         search/shipped.json, search/raised_4k.json, search/raised_8k.json,
         meta.json; OUT/run_summary.json; OUT/run.log
  pilot: OUT/<qid>/<variant>/ the same files (search/shipped.json only) plus
         OUT/<qid>/pilot.json with the table, the evidence detail and the choice

Model calls go through the litellm proxy (models.json proxy_base_url) with the
OpenAI generic client and json_schema structured output. Every request carries
the job tag as the OpenAI "user" field and as litellm metadata tags, so the
shared proxy log can be split by caller (design section 11). Usage is read
from every response by the Meter below, per episode, per session and per
question.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import importlib.metadata
import json
import logging
import os
import re
import shutil
import string
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

try:
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.edges import EntityEdge
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
    from openai import AsyncOpenAI

    HAVE_GRAPHITI = True
except ImportError:  # the bench venv runs the unit tests without graphiti_core
    HAVE_GRAPHITI = False
    CrossEncoderClient = object

ENV = Path("/Users/muralisid/github_other/part1-tools/env")
BENCH = Path("/Users/muralisid/github_other/multicard-bench")
DATA = BENCH / "data/raw/longmemeval_s.json"
SUBSETS = BENCH / "docs/part1/subsets.json"
OUT_RUN = BENCH / "data/part1/graphiti"
OUT_PILOT = BENCH / "data/part1/graphiti_pilot"
GRAPHITI_MD = ENV / "graphiti.md"
DOCS_ENV = BENCH / "docs/part1/env"
NEO4J_CONF = Path("/opt/homebrew/Cellar/neo4j/2026.07.1/libexec/conf/neo4j.conf")

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphiti"  # local dev only, as in graphiti.md

VARIANTS = ("text_session", "message_session", "per_turn")
VARIANT_LABEL = {
    "text_session": "one text episode per session",
    "message_session": "one message episode per session",
    "per_turn": "one message episode per turn",
}
PILOT_N = 20                 # design section 2
CAP_USD = 60.0               # design section 11, Graphiti on the fixed subset, pilot and run together
# design gap: the design says the limit is raised until the rendered context
# reaches B and fixes no top step; the ladder stops at 80 and a question whose
# top step did not reach B is counted in meta.json and run_summary.json
# (raised_not_reached) and reported beside the graphiti rows.
SEARCH_LIMITS = (20, 40, 80)  # shipped, then raised (design section 5)
BUDGET_TOKENS = {"raised_4k": 4000, "raised_8k": 8000}   # design section 5: B by the bench TokenCounter
CONCURRENCY_DEFAULT = 1       # design section 2: one question at a time in ORDER
EPISODE_ATTEMPTS = 3
EPISODE_BACKOFF_S = (5.0, 20.0)

# What the library's COMBINED_HYBRID_SEARCH_RRF recipe contains, read from
# graphiti_core/search/search_config_recipes.py (graphiti-core 0.30.1). The
# design text says "BM25, cosine and BFS"; the recipe as shipped has no BFS
# method in any scope and no cross-encoder. It is used as shipped.
RECIPE = {
    "name": "COMBINED_HYBRID_SEARCH_RRF",
    "edge": {"methods": ["bm25", "cosine_similarity"], "reranker": "rrf"},
    "node": {"methods": ["bm25", "cosine_similarity"], "reranker": "rrf"},
    "episode": {"methods": ["bm25"], "reranker": "rrf"},
    "community": {"methods": ["bm25", "cosine_similarity"], "reranker": "rrf"},
    "bfs": False,
    "cross_encoder": False,
    "candidates_per_method": "2 x limit",
    "sim_min_score": 0.6,
    "rrf": "score = sum over methods of 1 / (rank + 1), rank from 0",
    "limit_note": "SearchConfig.limit applies to every scope; communities are never built here, so that scope returns nothing",
}

# Zep's context string template, quoted from the Zep paper (Rasmussen et al.,
# arXiv:2501.13956, section 3 "Memory Retrieval", "Sample context string
# template"). The paper was read through a text fetch on 2026-09-06 that keeps
# the words and drops the line structure; the line form below, with "#"
# comment lines and one "  - " item per fact and per entity, is the one Zep's
# documentation shows for memory.context:
#
#   FACTS and ENTITIES represent relevant context to the current conversation.
#
#   # These are the most relevant facts and their valid date ranges
#   # If the fact is about an event, the event takes place during this time.
#   # format: FACT (Date range: from - to)
#   <FACTS>
#     - <fact> (<valid_at> - <invalid_at or present>)
#   </FACTS>
#
#   # These are the most relevant entities
#   # ENTITY_NAME: entity summary
#   <ENTITIES>
#     - <entity name>: <entity summary>
#   </ENTITIES>
#
# The words "date unknown" and "present" for missing dates are the ones
# graphiti_core.search.search_helpers.format_edge_date_range uses.
ZEP_TEMPLATE = (
    "FACTS and ENTITIES represent relevant context to the current conversation.\n"
    "\n"
    "# These are the most relevant facts and their valid date ranges\n"
    "# If the fact is about an event, the event takes place during this time.\n"
    "# format: FACT (Date range: from - to)\n"
    "<FACTS>\n"
    "{facts}\n"
    "</FACTS>\n"
    "\n"
    "# These are the most relevant entities\n"
    "# ENTITY_NAME: entity summary\n"
    "<ENTITIES>\n"
    "{entities}\n"
    "</ENTITIES>\n"
)

EDGE_SCHEMA = pa.schema([
    ("uuid", pa.string()), ("fact", pa.string()), ("name", pa.string()),
    ("source_node_uuid", pa.string()), ("target_node_uuid", pa.string()),
    ("episodes", pa.list_(pa.string())), ("valid_at", pa.string()),
    ("invalid_at", pa.string()), ("created_at", pa.string()), ("expired_at", pa.string()),
])
NODE_SCHEMA = pa.schema([
    ("uuid", pa.string()), ("name", pa.string()), ("summary", pa.string()),
    ("labels", pa.list_(pa.string())), ("created_at", pa.string()),
])
EPISODE_SCHEMA = pa.schema([
    ("uuid", pa.string()), ("name", pa.string()), ("source_description", pa.string()),
    ("reference_time", pa.string()), ("session_id", pa.string()), ("turn_index", pa.int64()),
])

log = logging.getLogger("run_groups")


class CapExceeded(RuntimeError):
    """Raised when the metered spend reaches the run's USD cap."""


# ----------------------------------------------------------------------------
# Data: sessions, order, episodes (design sections 2 and 5)
# ----------------------------------------------------------------------------
def load_lme(path: Path = DATA) -> dict:
    """Load LongMemEval_S with the bench loader (imported, not copied)."""
    src = str(BENCH / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from multicard.data.longmemeval import load_longmemeval

    return load_longmemeval(path)


class BenchTokenCounter:
    """The count of multicard.llm.costmeter.TokenCounter (the all-MiniLM-L6-v2
    tokenizer, no special tokens) through the tokenizers library over the
    same tokenizer.json in the Hugging Face cache, padding and truncation
    off. This venv has no transformers; tests/test_part1_graphiti.py checks
    the two counters agree in the bench venv."""

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, path: Path | None = None):
        from tokenizers import Tokenizer

        self.path = Path(path) if path else self.tokenizer_path()
        self._tok = Tokenizer.from_file(str(self.path))
        self._tok.no_padding()
        self._tok.no_truncation()

    @classmethod
    def tokenizer_path(cls) -> Path:
        cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        root = Path(cache) / ("models--" + cls.MODEL.replace("/", "--"))
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


def render_turn(role: str, text: str) -> str:
    """One "role: text" line, the message form Graphiti's EpisodeType.message expects."""
    return f"{role}: {text}"


def render_session(turns: list[dict]) -> str:
    """The turns of one session as "role: text" lines, one per line."""
    return "\n".join(render_turn(t["role"], t["content"]) for t in turns)


def session_order(session_ids: list[str], sessions: dict) -> list[str]:
    """Haystack sessions in session-date order; ties keep the haystack position."""
    return sorted(session_ids, key=lambda s: (sessions[s]["date"], session_ids.index(s)))


def pilot_sessions(session_ids: list[str], sessions: dict, n: int = PILOT_N) -> list[str]:
    """The first n haystack sessions in session-date order (design section 2)."""
    return session_order(session_ids, sessions)[:n]


def reference_time(date: str) -> datetime:
    """The session date at 00:00 UTC. The time of day is not in the loader's output."""
    return datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)


@dataclass
class Episode:
    name: str
    body: str
    source: str            # EpisodeType name: "text" or "message"
    reference_time: datetime
    session_id: str
    turn_index: int | None  # None when one episode is a whole session


def episode_plan(variant: str, sid: str, session: dict) -> list[Episode]:
    """The episodes one session becomes under a variant (design section 5).

    A session with no turns gives no episode. Under per_turn a turn with blank
    text gives no episode either (design gap: the design does not say what to
    do with empty sessions or blank turns; ingesting them would spend a model
    call on nothing).
    """
    turns = session["turns"]
    if not turns:
        return []
    ref = reference_time(session["date"])
    desc = f"LongMemEval_S session {sid}"
    if variant == "text_session":
        return [Episode(sid, render_session(turns), "text", ref, sid, None)]
    if variant == "message_session":
        return [Episode(sid, render_session(turns), "message", ref, sid, None)]
    if variant == "per_turn":
        out = []
        for i, t in enumerate(turns):
            if not (t.get("content") or "").strip():
                continue  # design gap: blank turn skipped
            out.append(Episode(f"{sid}#{i}", render_turn(t["role"], t["content"]),
                               "message", ref, sid, i))
        return out
    raise ValueError(variant)


def plan_question(sids: list[str], sessions: dict, variant: str) -> tuple[list[Episode], list[dict]]:
    """Episodes for the given sessions in the given order, plus one record per session."""
    plan: list[Episode] = []
    records = []
    for sid in sids:
        s = sessions[sid]
        eps = episode_plan(variant, sid, s)
        records.append({"session_id": sid, "date": s["date"], "turns": len(s["turns"]),
                        "chars": len(render_session(s["turns"])) if s["turns"] else 0,
                        "episodes": len(eps), "empty": not s["turns"]})
        plan.extend(eps)
    return plan, records


# ----------------------------------------------------------------------------
# Normalisation and evidence presence (design sections 3 and 5)
# ----------------------------------------------------------------------------
_STRIP = string.punctuation + "\u2018\u2019\u201c\u201d\u00ab\u00bb"  # curly quotes and guillemets, escaped


def normalise(text: str) -> str:
    """Design section 3: NFKC, lowercase, whitespace collapsed, leading and
    trailing punctuation and quotes stripped."""
    s = unicodedata.normalize("NFKC", text or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(_STRIP).strip()


def evidence_presence(inst, edges: list[dict], episodes: list[dict]) -> dict:
    """Per marked evidence turn: whether any exported edge cites an episode of
    the evidence session, and whether any fact text contains the normalised
    gold answer (design section 5, pilot paragraph). present is either.

    edges: dicts with fact and episodes (episode uuids). episodes: dicts with
    uuid, session_id, turn_index. presence is over every marked evidence turn
    of the question, as the design says; presence_over_ingested counts only
    the turns whose session was in the pilot, and is reported beside it.
    """
    by_session: dict[str, set] = defaultdict(set)
    by_turn: dict[tuple, str] = {}
    for ep in episodes:
        by_session[ep["session_id"]].add(ep["uuid"])
        if ep.get("turn_index") is not None:
            by_turn[(ep["session_id"], ep["turn_index"])] = ep["uuid"]
    cited_uuids = {u for e in edges for u in (e.get("episodes") or [])}
    gold = normalise(inst.answer)
    answer_in_fact = bool(gold) and any(gold in normalise(e["fact"]) for e in edges)
    rows = []
    for t in sorted(inst.evidence_turns):
        sid, i = t.rsplit("#", 1)
        i = int(i)
        ingested = sid in by_session
        cited = bool(cited_uuids & by_session[sid]) if ingested else False
        turn_uuid = by_turn.get((sid, i))
        cited_turn = (turn_uuid in cited_uuids) if turn_uuid else None
        rows.append({"turn": t, "session_id": sid, "turn_index": i, "session_in_pilot": ingested,
                     "cited": cited, "cited_turn_episode": cited_turn,
                     "answer_in_fact": answer_in_fact, "present": cited or answer_in_fact})
    n = len(rows)
    n_present = sum(1 for r in rows if r["present"])
    ing = [r for r in rows if r["session_in_pilot"]]
    return {"turns": rows, "n_turns": n, "n_present": n_present,
            "presence": (n_present / n) if n else None,
            "n_ingested": len(ing),
            "presence_over_ingested": (sum(1 for r in ing if r["present"]) / len(ing)) if ing else None,
            "answer_in_any_fact": answer_in_fact, "gold_normalised": gold}


# ----------------------------------------------------------------------------
# Meter: usage from every response, attributed by a context label
# ----------------------------------------------------------------------------
USAGE_KEYS = ("llm_calls", "llm_tokens_in", "llm_tokens_out", "embed_calls",
              "embed_tokens_in", "embed_inputs", "errors")


def fresh_usage() -> dict:
    return {k: 0 for k in USAGE_KEYS}


def cost_usd(models: dict, usage: dict) -> float:
    """USD at the models.json prices (per 1M tokens) of the study model and embedder."""
    p_chat = models["prices"][models["chat_model"]]
    p_emb = models["prices"][models["embed_model"]]
    return (usage["llm_tokens_in"] * p_chat["in"] / 1e6
            + usage["llm_tokens_out"] * p_chat["out"] / 1e6
            + usage["embed_tokens_in"] * p_emb["in"] / 1e6)


class Meter:
    """Counts this process's LLM and embedding calls from the OpenAI responses.

    The proxy log is shared by every caller on the machine, so the runner reads
    usage itself (design section 11). Calls are attributed to the label held in
    a ContextVar, a tuple such as (group_id, session_id, episode_name); asyncio
    tasks inherit the label of the task that set it. Every request also gets
    the job tag as the OpenAI "user" field and as litellm metadata tags
    [job_tag, group_id], which the proxy's request_log.py records.
    """

    def __init__(self, models: dict, cap_usd: float | None = None, job_tag: str = "",
                 llm_log: Path | None = None):
        self.models = models
        self.cap_usd = cap_usd
        self.job_tag = job_tag
        self.llm_log = llm_log
        self.label: contextvars.ContextVar = contextvars.ContextVar("run_groups_label", default=())
        self.buckets: dict[tuple, dict] = {}
        self.times: list[tuple[float, tuple]] = []

    def _bucket(self) -> dict:
        return self.buckets.setdefault(tuple(self.label.get()), fresh_usage())

    def total(self, prefix: tuple = ()) -> dict:
        out = fresh_usage()
        for key, b in self.buckets.items():
            if key[:len(prefix)] == tuple(prefix):
                for k in USAGE_KEYS:
                    out[k] += b[k]
        return out

    def usd(self, prefix: tuple = ()) -> float:
        return cost_usd(self.models, self.total(prefix))

    def over_cap(self) -> bool:
        return self.cap_usd is not None and self.usd() >= self.cap_usd

    def _tags(self) -> dict:
        label = self.label.get()
        tags = [self.job_tag] + ([str(label[0])] if label else [])
        return {"user": self.job_tag, "extra_body": {"metadata": {"tags": tags}}}

    def wrap(self, client) -> None:
        chat_create = client.chat.completions.create
        emb_create = client.embeddings.create

        async def chat(*a, **kw):
            self.times.append((time.time(), tuple(self.label.get())))
            kw.update(self._tags())
            try:
                r = await chat_create(*a, **kw)
            except Exception:
                self._bucket()["errors"] += 1
                raise
            b = self._bucket()
            u = getattr(r, "usage", None)
            b["llm_calls"] += 1
            b["llm_tokens_in"] += getattr(u, "prompt_tokens", 0) or 0
            b["llm_tokens_out"] += getattr(u, "completion_tokens", 0) or 0
            if self.llm_log is not None:
                with open(self.llm_log, "a") as f:
                    f.write(json.dumps({
                        "ts": time.strftime("%H:%M:%S"), "label": list(self.label.get()),
                        "model": kw.get("model"),
                        "prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "finish_reason": r.choices[0].finish_reason if r.choices else None,
                        "system_head": (kw.get("messages") or [{}])[0].get("content", "")[:160],
                        "response": (r.choices[0].message.content or "")[:3000] if r.choices else None,
                    }) + "\n")
            return r

        async def emb(*a, **kw):
            self.times.append((time.time(), tuple(self.label.get())))
            kw.update(self._tags())
            try:
                r = await emb_create(*a, **kw)
            except Exception:
                self._bucket()["errors"] += 1
                raise
            b = self._bucket()
            u = getattr(r, "usage", None)
            b["embed_calls"] += 1
            b["embed_tokens_in"] += getattr(u, "prompt_tokens", 0) or 0
            inp = kw.get("input")
            b["embed_inputs"] += len(inp) if isinstance(inp, list) else 1
            return r

        client.chat.completions.create = chat
        client.embeddings.create = emb


def rpm_stats(times: list[tuple[float, tuple]], t0: float, t1: float, prefix: tuple = ()) -> dict:
    """Requests per minute in [t0, t1]: the mean over the window and the peak
    count in any sliding 60 second window."""
    sel = sorted(t for t, key in times if t0 <= t <= t1 and key[:len(prefix)] == tuple(prefix))
    n = len(sel)
    minutes = max((t1 - t0) / 60.0, 1e-9)
    peak = 0
    j = 0
    for i, t in enumerate(sel):
        while sel[j] < t - 60.0:
            j += 1
        peak = max(peak, i - j + 1)
    return {"requests": n, "mean_rpm": round(n / minutes, 1), "peak_rpm_60s": peak,
            "window_seconds": round(t1 - t0, 1)}


# ----------------------------------------------------------------------------
# Projection and choice (design sections 2 and 5)
# ----------------------------------------------------------------------------
def project(value: float, chars: int, total_chars: int) -> float | None:
    """Scale a measured total (USD or seconds) by length: value per rendered
    character in the pilot times the rendered characters of every haystack
    slot of GRAPHITI_150 (graphiti_basis)."""
    if chars <= 0 or total_chars <= 0:
        return None
    return value / chars * total_chars


def graphiti_basis(subsets_path: Path, data: dict) -> dict:
    """The projection basis of the pilot (design sections 2 and 5): the
    GRAPHITI_150 questions, their haystack slots (one group per question
    holds that question's sessions, so a shared session counts once per
    question), the unique sessions, and the rendered characters over the
    slots, from subsets.json and the bench loader."""
    qids = load_subsets(subsets_path)
    by_qid = {x.qid: x for x in data["instances"]}
    sessions = data["sessions"]
    slots = 0
    chars = 0
    unique: set = set()
    for q in qids:
        inst = by_qid[q]
        for sid in inst.session_ids:
            slots += 1
            unique.add(sid)
            s = sessions[sid]
            chars += len(render_session(s["turns"])) if s["turns"] else 0
    return {"n_questions": len(qids), "slots": slots, "unique_sessions": len(unique), "chars": chars,
            "mean_chars_per_slot": (chars / slots) if slots else None, "subsets": str(subsets_path)}


def choose_variant(rows: list[dict], cap: float = CAP_USD) -> tuple[str | None, str]:
    """The design's choice rule: the variant with the highest evidence presence
    whose projected cost fits the cap; ties go to per_turn. A tie without
    per_turn goes to message_session, the closer of the two to Zep's message
    ingestion (design gap: the design names per-turn as the tie-break only).
    """
    fits = [r for r in rows if r.get("projected_usd") is not None and r["projected_usd"] <= cap
            and not r.get("partial")]
    if not fits:
        return None, f"no variant's projection fits USD {cap:.0f}; Graphiti is dropped unless the owner decides otherwise"
    best = max((r["evidence_presence"] or 0.0) for r in fits)
    tied = [r["variant"] for r in fits if (r["evidence_presence"] or 0.0) == best]
    for v in ("per_turn", "message_session", "text_session"):
        if v in tied:
            why = "highest evidence presence" if len(tied) == 1 else f"tie on evidence presence among {', '.join(sorted(tied))}, tie-break"
            return v, f"{why}; evidence presence {best:.3f}; projected USD within {cap:.0f}"
    return None, "no candidate"


def _fmt(x, nd=0):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def pilot_table_md(rows: list[dict]) -> str:
    """The pilot table as markdown, one row per variant."""
    head = ("| Variant | Sessions | Episodes | Chars | LLM calls | Tokens in | Tokens out | "
            "Embed calls | Embed tokens | Wall s | RPM mean | RPM peak | USD | Edges | Nodes | "
            "Evidence presence | Projected USD (150) | Fits 60 |")
    sep = "|" + "---|" * 18
    lines = [head, sep]
    for r in rows:
        pres = "n/a" if r["evidence_presence"] is None else f"{r['evidence_n_present']}/{r['evidence_n']} ({r['evidence_presence']:.2f})"
        fits = "n/a" if r["projected_usd"] is None else ("yes" if r["fits_cap"] else "no")
        tag = " (partial)" if r.get("partial") else ""
        lines.append("| " + " | ".join([
            r["variant"] + tag, _fmt(r["sessions"]), _fmt(r["episodes"]), _fmt(r["chars"]),
            _fmt(r["llm_calls"]), _fmt(r["tokens_in"]), _fmt(r["tokens_out"]),
            _fmt(r["embed_calls"]), _fmt(r["embed_tokens"]), _fmt(r["wall_s"], 1),
            _fmt(r["rpm_mean"], 1), _fmt(r["rpm_peak"]), _fmt(r["usd"], 4), _fmt(r["edges"]),
            _fmt(r["nodes"]), pres, _fmt(r["projected_usd"], 2), fits]) + " |")
    return "\n".join(lines)


def pilot_md(ctx: dict, rows: list[dict], choice: str | None, reason: str, evidence: dict) -> str:
    """The body written under the Pilot heading of graphiti.md."""
    out = []
    out.append(f"Date: {ctx['date']}. Question {ctx['qid']} ({ctx['qtype']}): {ctx['question']} "
               f"Gold answer: {ctx['answer']}. Sessions: the first {ctx['n']} haystack sessions in "
               f"session-date order ({ctx['n_ingested']} ingested, {ctx['n_empty']} empty and skipped). "
               f"Evidence sessions in the pilot set: {ctx['evidence_in_pilot']} of {ctx['evidence_sessions']}.")
    if ctx.get("session_override"):
        out.append("")
        out.append("The session list was given on the command line (smoke only), not taken by the "
                   "first-N rule.")
    out.append("")
    out.append(f"Chat model {ctx['chat_model']} through the proxy, json_schema output, temperature "
               f"{ctx['temperature']} and max_tokens {ctx['max_tokens']} (library defaults). Embeddings "
               f"{ctx['embed_model']} at {ctx['embed_dim']} dims. graphiti-core {ctx['graphiti_core']}. "
               f"Previous episodes per prompt: {ctx['previous_episodes']}. reference_time is the session "
               f"date at 00:00 UTC; the time of day in the raw file is dropped. Job tag {ctx['job_tag']}.")
    out.append("")
    out.append("Variants: text_session is one text episode per session with the turns as \"role: text\" "
               "lines; message_session is one message episode per session with the same rendering; "
               "per_turn is one message episode per turn, Zep's own granularity. The first two are "
               "deviations from Zep's ingestion.")
    out.append("")
    out.append(pilot_table_md(rows))
    out.append("")
    basis = ctx.get("basis") or {}
    out.append(f"Projection: USD per rendered character measured in the pilot, times the rendered characters of "
               f"every haystack slot of GRAPHITI_150 read from subsets.json at pilot time: {_fmt(basis.get('slots'))} "
               f"slots ({_fmt(basis.get('unique_sessions'))} unique sessions) over {_fmt(basis.get('n_questions'))} "
               f"questions, {_fmt(basis.get('chars'))} characters, {_fmt(basis.get('mean_chars_per_slot'), 1)} per slot. "
               "Wall time projected the same way, one group at a time: "
               + "; ".join(f"{r['variant']} {_fmt(r['projected_wall_h'], 1)} h" for r in rows) + ".")
    out.append("")
    out.append("Evidence presence per marked evidence turn (cited: an exported edge cites an episode of "
               "the turn's session; answer: some fact text contains the normalised gold answer):")
    out.append("")
    out.append("| Variant | Evidence turn | Session in pilot | Cited | Answer in a fact | Present |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        for t in evidence.get(r["variant"], {}).get("turns", []):
            out.append(f"| {r['variant']} | {t['turn']} | {'yes' if t['session_in_pilot'] else 'no'} | "
                       f"{'yes' if t['cited'] else 'no'} | {'yes' if t['answer_in_fact'] else 'no'} | "
                       f"{'yes' if t['present'] else 'no'} |")
    out.append("")
    if choice:
        out.append(f"Choice: {choice} ({VARIANT_LABEL[choice]}). Rule: highest evidence presence whose "
                   f"projection fits USD {CAP_USD:.0f}, ties to per-turn. Reason: {reason}."
                   + ("" if choice == "per_turn" else " This variant is a deviation from Zep's per-message ingestion and is labelled as such."))
    else:
        out.append(f"Choice: none. {reason}.")
    if ctx.get("partial"):
        out.append("")
        out.append("The pilot stopped at the USD cap before every variant finished; rows marked partial "
                   "are incomplete and were not eligible for the choice.")
    return "\n".join(out) + "\n"


def write_md_section(path: Path, heading: str, body: str) -> None:
    """Replace the "## <heading>" section of a markdown file, or append it."""
    text = path.read_text() if path.exists() else ""
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if l.strip() == f"## {heading}"), None)
    block = [f"## {heading}", ""] + body.rstrip("\n").split("\n") + [""]
    if start is None:
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(block)
    else:
        end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
        lines[start:end] = block
    path.write_text("\n".join(lines).rstrip("\n") + "\n")


# ----------------------------------------------------------------------------
# Search rendering and export
# ----------------------------------------------------------------------------
def iso(dt) -> str | None:
    return None if dt is None else (dt.isoformat() if hasattr(dt, "isoformat") else str(dt))


def date_range(valid_at: str | None, invalid_at: str | None) -> str:
    """FACT date range in Zep's form: "<from> - <to>", "date unknown" and "present" when missing."""
    a = valid_at.replace("T", " ") if valid_at else "date unknown"
    b = invalid_at.replace("T", " ") if invalid_at else "present"
    return f"{a} - {b}"


def fact_line(e: dict) -> str:
    """One FACT line as the bench renders it (multicard.part1.competitors.fact_line)."""
    return f"  - {e['fact']} ({date_range(e.get('valid_at'), e.get('invalid_at'))})"


def entity_summary_line(n: dict) -> str:
    """One ENTITY line as the bench renders it (multicard.part1.competitors.entity_summary_line)."""
    return f"  - {n['name']}: {n.get('summary') or ''}"


def render_context(edges: list[dict], nodes: list[dict]) -> str:
    """Zep's template over ranked edges then ranked nodes, in that order."""
    facts = "\n".join(fact_line(e) for e in edges)
    ents = "\n".join(entity_summary_line(n) for n in nodes)
    return ZEP_TEMPLATE.format(facts=facts, entities=ents)


def rendered_tokens(edges: list[dict], nodes: list[dict], counter) -> int:
    """Tokens of the FACT and ENTITY lines as the bench renders them for the
    graphiti arm, summed per unit, deduplicated by uuid (design section 5: B
    counted by the bench TokenCounter over the rendered units)."""
    seen: set = set()
    total = 0
    for k, e in enumerate(edges):
        uid = f"fact:{e.get('uuid', k)}"
        if uid not in seen:
            seen.add(uid)
            total += counter.count(fact_line(e))
    for k, n in enumerate(nodes):
        uid = f"entity:{n.get('uuid', k)}"
        if uid not in seen:
            seen.add(uid)
            total += counter.count(entity_summary_line(n))
    return total


def pick_ladder(value_by_limit: dict[int, int], target: int, limits=SEARCH_LIMITS) -> tuple[int, bool]:
    """The first limit whose rendered context reaches the target (tokens),
    else the largest limit tried (reached False; design gap, see SEARCH_LIMITS)."""
    last = None
    for lim in limits:
        if lim not in value_by_limit:
            continue
        last = lim
        if value_by_limit[lim] >= target:
            return lim, True
    return last, False


def edge_row(e) -> dict:
    return {"uuid": e.uuid, "fact": e.fact, "name": e.name,
            "source_node_uuid": e.source_node_uuid, "target_node_uuid": e.target_node_uuid,
            "episodes": list(e.episodes or []), "valid_at": iso(e.valid_at),
            "invalid_at": iso(e.invalid_at), "created_at": iso(e.created_at),
            "expired_at": iso(e.expired_at)}


def node_row(n) -> dict:
    return {"uuid": n.uuid, "name": n.name, "summary": n.summary or "",
            "labels": list(n.labels or []), "created_at": iso(n.created_at)}


def episode_row(ep, episode_map: dict) -> dict:
    sid, turn = episode_map.get(ep.uuid, (None, None))
    return {"uuid": ep.uuid, "name": ep.name, "source_description": ep.source_description,
            "reference_time": iso(ep.valid_at), "session_id": sid, "turn_index": turn}


def write_parquet(rows: list[dict], schema: pa.Schema, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)
    return table.num_rows


def load_subsets(path: Path = SUBSETS) -> list[str]:
    """GRAPHITI_150 in ORDER from docs/part1/subsets.json.

    design gap: the file's exact shape is produced by another agent. Accepted
    here: GRAPHITI_150 as a list of ids, or a dict holding one (key longmemeval
    or ids); ORDER as a list, or a dict keyed by corpus name.
    """
    data = json.loads(Path(path).read_text())
    sub = data["GRAPHITI_150"]
    if isinstance(sub, dict):
        sub = sub.get("longmemeval") or sub.get("ids") or next(v for v in sub.values() if isinstance(v, list))
    order = data.get("ORDER")
    if isinstance(order, dict):
        order = next((order[k] for k in ("longmemeval", "LongMemEval", "longmemeval_s", "LongMemEval_S", "lme") if k in order),
                     next((v for v in order.values() if isinstance(v, list)), None))
    sub = [str(q) for q in sub]
    if order:
        pos = {str(q): i for i, q in enumerate(order)}
        sub = sorted(sub, key=lambda q: (pos.get(q, len(pos)), sub.index(q)))
    return sub


# ----------------------------------------------------------------------------
# Graphiti client, ingestion, search, export, delete
# ----------------------------------------------------------------------------
if HAVE_GRAPHITI:
    class NoRerank(CrossEncoderClient):
        """Cross encoder that keeps the input order. The library default asks
        for OpenAI logprobs, which the Gemini proxy does not return, and the
        RRF recipe never calls it (design section 5)."""

        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            return [(p, 1.0) for p in passages]


def build_graphiti(models: dict, key: str, meter: Meter):
    """Graphiti over local Neo4j with the study model through the proxy."""
    client = AsyncOpenAI(api_key=key, base_url=models["proxy_base_url"], max_retries=6, timeout=300.0)
    meter.wrap(client)
    llm = OpenAIGenericClient(
        config=LLMConfig(api_key=key, base_url=models["proxy_base_url"],
                         model=models["chat_model"], small_model=models["chat_model"]),
        client=client, structured_output_mode="json_schema")
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(api_key=key, base_url=models["proxy_base_url"],
                                    embedding_model=models["embed_model"],
                                    embedding_dim=models["embed_dim"]),
        client=client)
    g = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, llm_client=llm, embedder=embedder,
                 cross_encoder=NoRerank())
    return g, llm


async def ready(g) -> None:
    """Await the driver's own index build (scheduled at construction)."""
    init_task = getattr(g.driver, "_init_task", None)
    if init_task is not None:
        await init_task
    else:
        await g.build_indices_and_constraints()


async def delete_group(g, gid: str) -> int:
    records, _, _ = await g.driver.execute_query(
        "MATCH (n {group_id: $g}) DETACH DELETE n RETURN count(n) AS n", params={"g": gid})
    return records[0]["n"] if records else 0


async def list_groups(g, prefix: str) -> list[str]:
    records, _, _ = await g.driver.execute_query(
        "MATCH (n) WHERE n.group_id STARTS WITH $p RETURN DISTINCT n.group_id AS g", params={"p": prefix})
    return sorted(r["g"] for r in records)


async def neo4j_version(g) -> str:
    records, _, _ = await g.driver.execute_query(
        "CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS v, edition")
    return "; ".join(f"{r['name']} {r['v']} {r['edition']}" for r in records)


def neo4j_heap() -> str | None:
    if not NEO4J_CONF.exists():
        return None
    m = re.search(r"^server\.memory\.heap\.max_size=(\S+)", NEO4J_CONF.read_text(), re.M)
    return m.group(1) if m else None


@dataclass
class IngestResult:
    episode_map: dict            # episode uuid -> (session_id, turn_index)
    episode_uuids: list[str]     # in ingestion order
    refused: list[dict]
    per_episode: list[dict]
    wall_seconds: float
    t0: float
    t1: float


async def ingest_group(g, gid: str, plan: list[Episode], meter: Meter,
                       previous_window: int | None = None) -> IngestResult:
    """Ingest the planned episodes into one group, one after the other.

    previous_window None keeps the library default: add_episode pulls the last
    RELEVANT_SCHEMA_LIMIT (10) episodes of the same source type in the group
    with valid_at at or before reference_time into every prompt. An integer
    passes the last that many ingested episode uuids instead (0 passes none).
    An episode that still fails after EPISODE_ATTEMPTS is recorded as refused
    and the group goes on. The USD cap is checked before every episode.
    """
    episode_map: dict = {}
    uuids: list[str] = []
    refused: list[dict] = []
    per_episode: list[dict] = []
    t0 = time.time()
    for ep in plan:
        if meter.over_cap():
            raise CapExceeded(f"spend {meter.usd():.4f} USD reached the cap {meter.cap_usd}")
        meter.label.set((gid, ep.session_id, ep.name))
        kwargs = dict(name=ep.name, episode_body=ep.body,
                      source_description=f"LongMemEval_S session {ep.session_id}"
                      + (f" turn {ep.turn_index}" if ep.turn_index is not None else ""),
                      reference_time=ep.reference_time, source=EpisodeType[ep.source], group_id=gid)
        if previous_window is not None:
            kwargs["previous_episode_uuids"] = uuids[-previous_window:] if previous_window > 0 else []
        before = meter.total((gid, ep.session_id, ep.name))
        te = time.time()
        res = None
        err = None
        for attempt in range(1, EPISODE_ATTEMPTS + 1):
            try:
                res = await g.add_episode(**kwargs)
                break
            except Exception as e:  # 429 and 5xx are retried inside the client first
                err = f"{type(e).__name__}: {str(e)[:300]}"
                log.warning("group %s episode %s attempt %s failed: %s", gid, ep.name, attempt, err)
                if attempt < EPISODE_ATTEMPTS:
                    await asyncio.sleep(EPISODE_BACKOFF_S[min(attempt - 1, len(EPISODE_BACKOFF_S) - 1)])
        wall = time.time() - te
        after = meter.total((gid, ep.session_id, ep.name))
        usage = {k: after[k] - before[k] for k in USAGE_KEYS}
        rec = {"name": ep.name, "session_id": ep.session_id, "turn_index": ep.turn_index,
               "chars": len(ep.body), "wall_seconds": round(wall, 2), **usage,
               "usd": round(cost_usd(meter.models, usage), 6)}
        if res is None:
            refused.append({"name": ep.name, "session_id": ep.session_id, "turn_index": ep.turn_index,
                            "error": err})
            rec["refused"] = True
        else:
            uuids.append(res.episode.uuid)
            episode_map[res.episode.uuid] = (ep.session_id, ep.turn_index)
            rec.update({"uuid": res.episode.uuid, "nodes": len(res.nodes), "edges": len(res.edges)})
            log.info("group %s episode %s: %s chars, %s nodes, %s edges, %.1fs, %s llm calls, %s embed calls, %.5f USD",
                     gid, ep.name, len(ep.body), len(res.nodes), len(res.edges), wall,
                     usage["llm_calls"], usage["embed_calls"], rec["usd"])
        per_episode.append(rec)
    t1 = time.time()
    return IngestResult(episode_map, uuids, refused, per_episode, round(t1 - t0, 2), t0, t1)


async def search_group(g, gid: str, question: str, limit: int, meter: Meter, episode_map: dict,
                       counter=None) -> dict:
    """One COMBINED_HYBRID_SEARCH_RRF search at one limit, rendered with Zep's template."""
    cfg = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
    cfg.limit = limit
    meter.label.set((gid, "search", str(limit)))
    before = meter.total((gid, "search", str(limit)))
    t0 = time.time()
    res = await g.search_(question, config=cfg, group_ids=[gid])
    wall = time.time() - t0
    after = meter.total((gid, "search", str(limit)))
    edges = [dict(edge_row(e), score=s) for e, s in zip(res.edges, res.edge_reranker_scores)] \
        if res.edge_reranker_scores else [edge_row(e) for e in res.edges]
    nodes = [dict(node_row(n), score=s) for n, s in zip(res.nodes, res.node_reranker_scores)] \
        if res.node_reranker_scores else [node_row(n) for n in res.nodes]
    episodes = [episode_row(ep, episode_map) for ep in res.episodes]
    context = render_context(edges, nodes)
    return {"limit": limit, "limits": {"edges": limit, "nodes": limit, "episodes": limit, "communities": limit},
            "recipe": RECIPE, "wall_seconds": round(wall, 3),
            "usage": {k: after[k] - before[k] for k in USAGE_KEYS},
            "edges": edges, "nodes": nodes, "episodes": episodes,
            "communities": [node_row(c) for c in res.communities],
            "context": context, "context_chars": len(context),
            "rendered_tokens": rendered_tokens(edges, nodes, counter) if counter is not None else None}


async def search_ladder(g, gid: str, question: str, meter: Meter, episode_map: dict, counter,
                        limits=SEARCH_LIMITS, targets=BUDGET_TOKENS) -> dict[str, dict]:
    """shipped at the first limit; raised_4k and raised_8k at the first limit
    whose rendered FACT and ENTITY lines count 4,000 and 8,000 tokens by the
    bench TokenCounter (design section 5, "run twice as for chandan_live", B
    in tokens). The character count is kept beside it as a side figure."""
    by_limit: dict[int, dict] = {}
    tokens: dict[int, int] = {}
    for lim in limits:
        by_limit[lim] = await search_group(g, gid, question, lim, meter, episode_map, counter)
        tokens[lim] = by_limit[lim]["rendered_tokens"]
        if tokens[lim] >= max(targets.values()):
            break
    out = {"shipped": dict(by_limit[limits[0]], variant="shipped", target_tokens=None, reached=None)}
    for name, target in targets.items():
        lim, reached = pick_ladder(tokens, target, limits)
        out[name] = dict(by_limit[lim], variant=name, target_tokens=target, reached=reached)
    return out


async def export_group(g, gid: str, episode_map: dict, out_dir: Path) -> dict:
    """Every edge, node and episode of the group to parquet (shared schema)."""
    edges = await EntityEdge.get_by_group_ids(g.driver, [gid])
    nodes = await EntityNode.get_by_group_ids(g.driver, [gid])
    eps = await EpisodicNode.get_by_group_ids(g.driver, [gid])
    e_rows = [edge_row(e) for e in edges]
    n_rows = [node_row(n) for n in nodes]
    p_rows = [episode_row(ep, episode_map) for ep in eps]
    write_parquet(e_rows, EDGE_SCHEMA, out_dir / "edges.parquet")
    write_parquet(n_rows, NODE_SCHEMA, out_dir / "nodes.parquet")
    write_parquet(p_rows, EPISODE_SCHEMA, out_dir / "episodes.parquet")
    return {"edges": e_rows, "nodes": n_rows, "episodes": p_rows}


def session_usage(meter: Meter, gid: str, records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        u = meter.total((gid, r["session_id"]))
        out.append({**r, **u, "usd": round(cost_usd(meter.models, u), 6)})
    return out


def base_meta(models: dict, llm, job_tag: str, previous_window: int | None) -> dict:
    return {
        "graphiti_core": importlib.metadata.version("graphiti-core"),
        "neo4j_driver": importlib.metadata.version("neo4j"),
        "openai_python": importlib.metadata.version("openai"),
        "chat_model": models["chat_model"], "embed_model": models["embed_model"],
        "embed_dim": models["embed_dim"],
        "temperature": llm.temperature, "max_tokens": llm.max_tokens,
        "structured_output_mode": llm.structured_output_mode,
        "cross_encoder": "NoRerank (identity); the recipe never calls it",
        "semaphore_limit": int(os.getenv("SEMAPHORE_LIMIT", 20)),
        "previous_episodes": ("library default: add_episode retrieves the last 10 (RELEVANT_SCHEMA_LIMIT) "
                              "episodes of the same source type in the group with valid_at at or before "
                              "reference_time" if previous_window is None
                              else f"explicit: the last {previous_window} ingested episodes of the group"),
        "reference_time_rule": "session date at 00:00 UTC; the time of day printed in the raw file is dropped (the bench loader keeps the date only)",
        "session_order": "session date ascending; ties keep the haystack position",
        "prices": {models["chat_model"]: models["prices"][models["chat_model"]],
                   models["embed_model"]: models["prices"][models["embed_model"]]},
        "job_tag": job_tag,
        "recipe": RECIPE,
        "neo4j_heap_max": neo4j_heap(),
        "budget_tokens": dict(BUDGET_TOKENS), "search_limits": list(SEARCH_LIMITS),
        "token_counter": f"bench TokenCounter ({BenchTokenCounter.MODEL}) via tokenizers",
    }


# ----------------------------------------------------------------------------
# One question (run mode) and the pilot
# ----------------------------------------------------------------------------
async def process_question(g, meter: Meter, inst, sessions: dict, variant: str, out: Path,
                           common: dict, previous_window: int | None, counter=None) -> dict:
    qid = inst.qid
    gid = qid
    qdir = out / qid
    if (qdir / "meta.json").exists():
        log.info("question %s: meta.json exists, skipped", qid)
        return {"qid": qid, "status": "skipped"}
    counter = counter or BenchTokenCounter()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    removed = await delete_group(g, gid)  # a partial group from an earlier attempt
    if removed:
        log.info("question %s: removed %s nodes left by an earlier attempt", qid, removed)
    sids = session_order(inst.session_ids, sessions)
    plan, records = plan_question(sids, sessions, variant)
    log.info("question %s (%s): %s sessions, %s episodes, variant %s", qid, inst.qtype, len(sids), len(plan), variant)
    ingest = await ingest_group(g, gid, plan, meter, previous_window)
    searches = await search_ladder(g, gid, inst.question, meter, ingest.episode_map, counter)
    (qdir / "search").mkdir(parents=True, exist_ok=True)
    for name, s in searches.items():
        (qdir / "search" / f"{name}.json").write_text(json.dumps(
            {"qid": qid, "group_id": gid, "question": inst.question, "question_date": inst.question_date, **s}, indent=1))
    export = await export_group(g, gid, ingest.episode_map, qdir)
    deleted = await delete_group(g, gid)
    usage = meter.total((gid,))
    meta = {
        **common, "qid": qid, "group_id": gid, "qtype": inst.qtype, "question": inst.question,
        "question_date": inst.question_date, "gold_answer": inst.answer,
        "ingestion_variant": variant, "variant_label": VARIANT_LABEL[variant],
        "deviation_from_zep_ingestion": variant != "per_turn",
        "sessions": session_usage(meter, gid, records),
        "session_count": len(sids), "empty_sessions_skipped": sum(1 for r in records if r["empty"]),
        "episode_count": len(ingest.episode_uuids), "episodes_planned": len(plan),
        "refused_episodes": ingest.refused, "ingest_wall_seconds": ingest.wall_seconds,
        "usage": usage, "usd": round(cost_usd(meter.models, usage), 6),
        "rpm": rpm_stats(meter.times, ingest.t0, ingest.t1, (gid,)),
        "search": {name: {k: s[k] for k in ("variant", "limit", "wall_seconds", "usage", "context_chars",
                                             "rendered_tokens", "target_tokens", "reached")}
                   for name, s in searches.items()},
        "raised_not_reached": [name for name, s in searches.items() if s.get("reached") is False],
        "counts": {k: len(v) for k, v in export.items()},
        "deleted_nodes": deleted, "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (qdir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info("question %s done: %s edges, %s nodes, %s episodes, %.4f USD, %.0fs", qid,
             meta["counts"]["edges"], meta["counts"]["nodes"], meta["counts"]["episodes"], meta["usd"], ingest.wall_seconds)
    return {"qid": qid, "status": "done", "usd": meta["usd"], "refused": len(ingest.refused),
            "raised_not_reached": meta["raised_not_reached"]}


async def cmd_run(args) -> None:
    models = json.loads((ENV / "models.json").read_text())
    key = Path(models["proxy_key_file"]).read_text().strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "run.log")
    job_tag = args.job_tag or f"graphiti-run-{time.strftime('%Y%m%d-%H%M%S')}"
    # Section 11: one cap of 60 USD for Graphiti, pilot and run together. The
    # pilot's spend is read from pilot.json (or --spent-usd) and taken off the
    # run's cap; both are written into run_summary.json.
    spent = pilot_spend(args)
    run_cap = max(0.0, args.max_usd - spent)
    meter = Meter(models, cap_usd=run_cap, job_tag=job_tag,
                  llm_log=Path(args.llm_log) if args.llm_log else None)
    if args.qids:
        qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    else:
        qids = load_subsets(Path(args.subsets))
    if args.limit_questions:
        qids = qids[:args.limit_questions]
    d = load_lme()
    by_qid = {x.qid: x for x in d["instances"]}
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"unknown question ids: {missing[:5]}")
    counter = BenchTokenCounter()
    g, llm = build_graphiti(models, key, meter)
    results = []
    stop = {"cap": False}
    try:
        await ready(g)
        common = base_meta(models, llm, job_tag, args.previous_window)
        common["neo4j_server"] = await neo4j_version(g)
        common["concurrency"] = args.k
        common["cap_usd_run"] = run_cap
        common["cap_usd_pilot_spent"] = spent

        async def one(qid):
            if stop["cap"]:
                return {"qid": qid, "status": "not started (cap)"}
            try:
                return await process_question(g, meter, by_qid[qid], d["sessions"], args.variant, out,
                                              common, args.previous_window, counter)
            except CapExceeded as e:
                stop["cap"] = True
                log.error("question %s stopped: %s", qid, e)
                return {"qid": qid, "status": "stopped at cap", "error": str(e)}
            except Exception as e:
                log.exception("question %s failed", qid)
                return {"qid": qid, "status": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}"}

        if args.k <= 1:
            # design section 2: one question at a time in ORDER, so a capped or
            # failed run leaves a prefix of ORDER
            for q in qids:
                results.append(await one(q))
        else:
            sem = asyncio.Semaphore(args.k)

            async def worker(qid):
                async with sem:
                    return await one(qid)

            results = list(await asyncio.gather(*[worker(q) for q in qids]))
    finally:
        await g.close()
    summary = {"job_tag": job_tag, "variant": args.variant, "k": args.k,
               "prefix_of_order": args.k <= 1,
               "prefix_note": ("questions processed one at a time in ORDER (section 2)" if args.k <= 1 else
                               f"{args.k} groups in flight: a capped or failed run is not a prefix of ORDER "
                               "(a deviation from section 2 the report must state)"),
               "max_usd": args.max_usd, "pilot_spent_usd": spent, "run_cap_usd": run_cap,
               "questions": results, "usd": round(meter.usd(), 4), "usage": meter.total(),
               "total_usd_with_pilot": round(meter.usd() + spent, 4),
               "partial": stop["cap"] or any(r["status"] == "failed" for r in results),
               "done": sum(1 for r in results if r["status"] == "done"),
               "skipped": sum(1 for r in results if r["status"] == "skipped"),
               # design gap: questions whose raised variant did not reach B at the top search limit
               "raised_not_reached": {name: [r["qid"] for r in results if name in (r.get("raised_not_reached") or [])]
                                      for name in BUDGET_TOKENS},
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("done", "skipped", "usd", "partial", "run_cap_usd")}))


def pilot_spend(args) -> float:
    """The pilot's USD to charge against the run's cap: --spent-usd, else the
    usd_total of --pilot-json, else the pilot.json of the pilot question
    (derived.GRAPHITI_PILOT in subsets.json) under OUT_PILOT when it exists."""
    if getattr(args, "spent_usd", None) is not None:
        return float(args.spent_usd)
    path = Path(args.pilot_json) if getattr(args, "pilot_json", None) else None
    if path is None:
        try:
            data = json.loads(Path(args.subsets).read_text())
            pilot_qid = (data.get("derived") or {}).get("GRAPHITI_PILOT")
        except (OSError, ValueError):
            pilot_qid = None
        if pilot_qid:
            path = OUT_PILOT / pilot_qid / "pilot.json"
    if path is not None and path.exists():
        return float(json.loads(path.read_text()).get("usd_total") or 0.0)
    return 0.0


async def cmd_pilot(args) -> None:
    models = json.loads((ENV / "models.json").read_text())
    key = Path(models["proxy_key_file"]).read_text().strip()
    out = Path(args.out) / args.qid
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "pilot.log")
    job_tag = args.job_tag or f"graphiti-pilot-{time.strftime('%Y%m%d-%H%M%S')}"
    meter = Meter(models, cap_usd=args.max_usd, job_tag=job_tag,
                  llm_log=Path(args.llm_log) if args.llm_log else None)
    variants = [v.strip() for v in args.variants.split(",")] if args.variants else list(VARIANTS)
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {VARIANTS}")
    d = load_lme()
    inst = next((x for x in d["instances"] if x.qid == args.qid), None)
    if inst is None:
        raise SystemExit(f"unknown question id {args.qid}")
    sessions = d["sessions"]
    if args.session_ids:
        sids = [s.strip() for s in args.session_ids.split(",") if s.strip()]
        unknown = [s for s in sids if s not in inst.session_ids]
        if unknown:
            raise SystemExit(f"sessions not in the haystack of {args.qid}: {unknown}")
        sids = session_order(sids, sessions)
    else:
        sids = pilot_sessions(inst.session_ids, sessions, args.n)
    basis = graphiti_basis(Path(args.subsets), d)
    log.info("pilot %s (%s): %s; gold %s; sessions %s; projection basis %s", inst.qid, inst.qtype, inst.question,
             inst.answer, sids, basis)
    counter = BenchTokenCounter()
    g, llm = build_graphiti(models, key, meter)
    rows: list[dict] = []
    evidence: dict = {}
    partial = False
    gids = [f"pilot-{args.qid}-{v}" for v in variants]
    try:
        await ready(g)
        common = base_meta(models, llm, job_tag, args.previous_window)
        common["neo4j_server"] = await neo4j_version(g)
        for variant, gid in zip(variants, gids):
            vdir = out / variant
            vdir.mkdir(parents=True, exist_ok=True)
            await delete_group(g, gid)
            plan, records = plan_question(sids, sessions, variant)
            log.info("variant %s: %s episodes in group %s", variant, len(plan), gid)
            ingest = None
            stopped = None
            try:
                ingest = await ingest_group(g, gid, plan, meter, args.previous_window)
            except CapExceeded as e:
                stopped = str(e)
                partial = True
                log.error("variant %s stopped: %s", variant, e)
            if ingest is None:
                ingest = IngestResult({}, [], [], [], 0.0, time.time(), time.time())
            episode_map = ingest.episode_map
            searches = {}
            if not stopped:
                searches["shipped"] = dict(await search_group(g, gid, inst.question, SEARCH_LIMITS[0], meter,
                                                              episode_map, counter),
                                           variant="shipped", target_tokens=None, reached=None)
                (vdir / "search").mkdir(exist_ok=True)
                (vdir / "search" / "shipped.json").write_text(json.dumps(
                    {"qid": inst.qid, "group_id": gid, "question": inst.question, **searches["shipped"]}, indent=1))
            export = await export_group(g, gid, episode_map, vdir)
            ev = evidence_presence(inst, export["edges"], export["episodes"])
            evidence[variant] = ev
            usage = meter.total((gid,))
            usd = cost_usd(meter.models, usage)
            chars = sum(len(e.body) for e in plan[:len(ingest.per_episode)])
            rpm = rpm_stats(meter.times, ingest.t0, ingest.t1, (gid,))
            proj_usd = project(usd, chars, basis["chars"])
            row = {"variant": variant, "label": VARIANT_LABEL[variant], "group_id": gid,
                   "deviation_from_zep": variant != "per_turn",
                   "sessions": sum(1 for r in records if not r["empty"]),
                   "empty_skipped": sum(1 for r in records if r["empty"]),
                   "episodes": len(ingest.episode_uuids), "episodes_planned": len(plan), "chars": chars,
                   "llm_calls": usage["llm_calls"], "tokens_in": usage["llm_tokens_in"],
                   "tokens_out": usage["llm_tokens_out"], "embed_calls": usage["embed_calls"],
                   "embed_tokens": usage["embed_tokens_in"], "errors": usage["errors"],
                   "wall_s": ingest.wall_seconds, "rpm_mean": rpm["mean_rpm"], "rpm_peak": rpm["peak_rpm_60s"],
                   "usd": round(usd, 6), "edges": len(export["edges"]), "nodes": len(export["nodes"]),
                   "evidence_presence": ev["presence"], "evidence_n_present": ev["n_present"],
                   "evidence_n": ev["n_turns"], "evidence_presence_over_ingested": ev["presence_over_ingested"],
                   "projected_usd": None if proj_usd is None else round(proj_usd, 2),
                   "projected_wall_h": (None if not chars or not basis["chars"]
                                        else round(project(ingest.wall_seconds, chars, basis["chars"]) / 3600, 1)),
                   "projection_basis": basis,
                   "fits_cap": (proj_usd is not None and proj_usd <= CAP_USD),
                   "refused": len(ingest.refused), "partial": bool(stopped), "stopped": stopped}
            rows.append(row)
            meta = {**common, "qid": inst.qid, "group_id": gid, "qtype": inst.qtype, "question": inst.question,
                    "gold_answer": inst.answer, "ingestion_variant": variant, "variant_label": VARIANT_LABEL[variant],
                    "deviation_from_zep_ingestion": variant != "per_turn", "pilot_n": args.n,
                    "session_ids": sids, "sessions": session_usage(meter, gid, records),
                    "episode_count": len(ingest.episode_uuids), "episodes_planned": len(plan),
                    "per_episode": ingest.per_episode, "refused_episodes": ingest.refused,
                    "ingest_wall_seconds": ingest.wall_seconds, "usage": usage, "usd": round(usd, 6), "rpm": rpm,
                    "evidence": ev, "counts": {k: len(v) for k, v in export.items()},
                    "search": {k: {x: s[x] for x in ("limit", "wall_seconds", "usage", "context_chars", "rendered_tokens")}
                               for k, s in searches.items()},
                    "row": row, "partial": bool(stopped)}
            (vdir / "meta.json").write_text(json.dumps(meta, indent=1))
            log.info("variant %s: %s", variant, json.dumps({k: row[k] for k in ("episodes", "llm_calls", "tokens_in", "tokens_out", "embed_calls", "wall_s", "usd", "edges", "nodes", "evidence_presence", "projected_usd")}))
            if stopped:
                break
        choice, reason = choose_variant(rows, CAP_USD)
        ev_sessions = sorted(inst.evidence_sessions)
        ctx = {"date": time.strftime("%Y-%m-%d"), "qid": inst.qid, "qtype": inst.qtype, "question": inst.question,
               "answer": inst.answer, "n": len(sids), "n_ingested": sum(1 for s in sids if sessions[s]["turns"]),
               "n_empty": sum(1 for s in sids if not sessions[s]["turns"]),
               "evidence_sessions": len(ev_sessions), "evidence_in_pilot": sum(1 for s in ev_sessions if s in sids),
               "session_override": bool(args.session_ids), "chat_model": models["chat_model"],
               "embed_model": models["embed_model"], "embed_dim": models["embed_dim"],
               "temperature": llm.temperature, "max_tokens": llm.max_tokens,
               "graphiti_core": common["graphiti_core"], "previous_episodes": common["previous_episodes"],
               "job_tag": job_tag, "partial": partial, "basis": basis}
        body = pilot_md(ctx, rows, choice, reason, evidence)
        print("\n" + pilot_table_md(rows))
        print(f"\nchoice: {choice} ({reason})")
        result = {"qid": inst.qid, "question": inst.question, "answer": inst.answer, "session_ids": sids,
                  "rows": rows, "evidence": evidence, "choice": choice, "reason": reason, "cap_usd": CAP_USD,
                  "projection_basis": basis,
                  "usd_total": round(meter.usd(), 6), "usage_total": meter.total(), "job_tag": job_tag,
                  "partial": partial, "md_heading": None if args.no_md else args.md_heading,
                  "context": ctx}
        (out / "pilot.json").write_text(json.dumps(result, indent=1))
        if not args.no_md:
            write_md_section(GRAPHITI_MD, args.md_heading, body)
            DOCS_ENV.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(GRAPHITI_MD, DOCS_ENV / "graphiti.md")
            log.info("wrote section '%s' to %s and copied to %s", args.md_heading, GRAPHITI_MD, DOCS_ENV / "graphiti.md")
        if not args.keep_groups:
            for gid in gids:
                n = await delete_group(g, gid)
                log.info("deleted %s nodes under %s", n, gid)
    finally:
        await g.close()
    print(f"spend {meter.usd():.4f} USD; outputs under {out}")


async def cmd_cleanup(args) -> None:
    models = json.loads((ENV / "models.json").read_text())
    key = Path(models["proxy_key_file"]).read_text().strip()
    setup_logging(None)
    g, _ = build_graphiti(models, key, Meter(models))
    try:
        await ready(g)
        gids = [x.strip() for x in args.group_ids.split(",") if x.strip()] if args.group_ids else []
        if args.prefix:
            gids += await list_groups(g, args.prefix)
        for gid in sorted(set(gids)):
            n = await delete_group(g, gid)
            print(f"deleted {n} nodes under {gid}")
        if not gids:
            print("nothing to delete")
    finally:
        await g.close()


def setup_logging(path: Path | None) -> None:
    handlers = [logging.StreamHandler()]
    if path is not None:
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "neo4j", "graphiti_core", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def window(s):
        return None if s == "library" else int(s)

    pp = sub.add_parser("pilot", help="the 20-session pilot on one question, three variants")
    pp.add_argument("--qid", required=True)
    pp.add_argument("--n", type=int, default=PILOT_N, help="first N haystack sessions in session-date order")
    pp.add_argument("--variants", default=None, help="comma list; default all three")
    pp.add_argument("--max-usd", type=float, default=CAP_USD, help="stop when metered spend reaches this")
    pp.add_argument("--subsets", default=str(SUBSETS), help="subsets.json: GRAPHITI_150 for the projection basis")
    pp.add_argument("--out", default=str(OUT_PILOT))
    pp.add_argument("--session-ids", default=None, help="smoke only: explicit haystack sessions instead of the first N")
    pp.add_argument("--md-heading", default="Pilot", help="heading written into graphiti.md")
    pp.add_argument("--no-md", action="store_true", help="do not write graphiti.md")
    pp.add_argument("--keep-groups", action="store_true", help="leave the pilot groups in Neo4j")
    pp.add_argument("--previous-window", type=window, default=None,
                    help="'library' (default) or an integer count of previous episodes to pass")
    pp.add_argument("--job-tag", default=None)
    pp.add_argument("--llm-log", default=None, help="append every LLM exchange to this jsonl")

    pr = sub.add_parser("run", help="GRAPHITI_150 in ORDER with the chosen variant")
    pr.add_argument("--variant", required=True, choices=VARIANTS)
    pr.add_argument("--subsets", default=str(SUBSETS))
    pr.add_argument("--qids", default=None, help="comma list overriding the subset (tests)")
    pr.add_argument("--limit-questions", type=int, default=0)
    pr.add_argument("--k", type=int, default=CONCURRENCY_DEFAULT,
                    help="groups ingested at a time; 1 (default) processes ORDER one question at a time (section 2)")
    pr.add_argument("--max-usd", type=float, default=CAP_USD, help="the Graphiti cap, pilot and run together")
    pr.add_argument("--spent-usd", type=float, default=None, help="USD already spent by the pilot (taken off the cap)")
    pr.add_argument("--pilot-json", default=None, help="pilot.json whose usd_total is taken off the cap "
                                                        "(default: the pilot question's file under the pilot out dir)")
    pr.add_argument("--out", default=str(OUT_RUN))
    pr.add_argument("--previous-window", type=window, default=None)
    pr.add_argument("--job-tag", default=None)
    pr.add_argument("--llm-log", default=None)

    pc = sub.add_parser("cleanup", help="delete groups by id or prefix")
    pc.add_argument("--group-ids", default=None, help="comma list")
    pc.add_argument("--prefix", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not HAVE_GRAPHITI:
        raise SystemExit("graphiti_core is not installed in this interpreter; run from part1-tools/graphiti/.venv")
    if args.cmd == "pilot":
        asyncio.run(cmd_pilot(args))
    elif args.cmd == "run":
        asyncio.run(cmd_run(args))
    else:
        asyncio.run(cmd_cleanup(args))


if __name__ == "__main__":
    main()

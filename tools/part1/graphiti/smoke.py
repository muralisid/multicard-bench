"""Graphiti smoke test on two LongMemEval sessions through the local litellm proxy.

Ingests each evidence session of one LongMemEval S instance as one Graphiti
episode, then searches the graph with the instance question and prints the
returned edges (facts) with valid_at, invalid_at and the episode ids they cite.
LLM and embedding calls go through the proxy at part1-tools/env/models.json
proxy_base_url, so the per-request log (requests.jsonl) gives call counts,
tokens and litellm's cost estimate per session.

Usage (from part1-tools/graphiti):
    .venv/bin/python smoke.py                 # ingest + search, leave graph in place
    .venv/bin/python smoke.py --cleanup-only  # delete everything under the smoke group_id
    .venv/bin/python smoke.py --instance <qid>

Outputs:
    part1-tools/env/graphiti-smoke.json   per-session numbers and search results
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402

ENV = Path("/Users/muralisid/github_other/part1-tools/env")
BENCH = Path("/Users/muralisid/github_other/multicard-bench")
DATA = BENCH / "data/raw/longmemeval_s.json"
LOADER = BENCH / "src/multicard/data/longmemeval.py"
REQ_LOG = ENV / "requests.jsonl"
OUT = ENV / "graphiti-smoke.json"

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphiti"  # local dev only
GROUP_ID = "smoke-longmemeval-2026-09-06"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
log = logging.getLogger("smoke")


class NoRerank(CrossEncoderClient):
    """Cross encoder that keeps the input order. The default OpenAIRerankerClient
    needs logprobs, which the Gemini proxy does not return. The default search
    recipe (EDGE_HYBRID_SEARCH_RRF) never calls the cross encoder anyway."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0) for p in passages]


def load_loader():
    spec = importlib.util.spec_from_file_location("longmemeval", LOADER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["longmemeval"] = mod  # dataclasses looks the module up by name
    spec.loader.exec_module(mod)
    return mod


def render_session(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def log_offset() -> int:
    return REQ_LOG.stat().st_size if REQ_LOG.exists() else 0


def log_since(offset: int) -> dict:
    """Aggregate the proxy request log lines written after byte offset."""
    agg = {"calls": 0, "llm_calls": 0, "embed_calls": 0, "failures": 0,
           "llm_tokens_in": 0, "llm_tokens_out": 0, "embed_tokens_in": 0,
           "litellm_cost_usd": 0.0, "by_model": {}}
    if not REQ_LOG.exists():
        return agg
    with open(REQ_LOG) as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            agg["calls"] += 1
            if r.get("status") != "success":
                agg["failures"] += 1
            model = r.get("model") or "?"
            bm = agg["by_model"].setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
            bm["calls"] += 1
            bm["tokens_in"] += r.get("prompt_tokens") or 0
            bm["tokens_out"] += r.get("completion_tokens") or 0
            if "embedding" in (r.get("call_type") or ""):
                agg["embed_calls"] += 1
                agg["embed_tokens_in"] += r.get("prompt_tokens") or 0
            else:
                agg["llm_calls"] += 1
                agg["llm_tokens_in"] += r.get("prompt_tokens") or 0
                agg["llm_tokens_out"] += r.get("completion_tokens") or 0
            agg["litellm_cost_usd"] += float(r.get("response_cost") or 0.0)
    return agg


class Meter:
    """Counts this process's own LLM and embedding calls from the OpenAI responses.

    The proxy request log (requests.jsonl) is shared with other processes on this
    machine, so it cannot attribute calls to this run. The meter wraps the two
    AsyncOpenAI methods Graphiti uses and reads response.usage directly. It also
    appends every LLM exchange to DEBUG_LOG so the extraction can be inspected.
    """

    def __init__(self):
        self.llm_calls = 0
        self.llm_tokens_in = 0
        self.llm_tokens_out = 0
        self.embed_calls = 0
        self.embed_tokens_in = 0
        self.embed_inputs = 0
        self.errors = 0

    def snapshot(self) -> dict:
        return dict(vars(self))

    def delta(self, before: dict) -> dict:
        now = self.snapshot()
        return {k: now[k] - before[k] for k in now}

    def wrap(self, llm_client, embed_client) -> None:
        chat_create = llm_client.chat.completions.create
        emb_create = embed_client.embeddings.create

        async def chat(*a, **kw):
            try:
                r = await chat_create(*a, **kw)
            except Exception:
                self.errors += 1
                raise
            self.llm_calls += 1
            u = getattr(r, "usage", None)
            self.llm_tokens_in += getattr(u, "prompt_tokens", 0) or 0
            self.llm_tokens_out += getattr(u, "completion_tokens", 0) or 0
            with open(DEBUG_LOG, "a") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%H:%M:%S"),
                    "model": kw.get("model"),
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "finish_reason": r.choices[0].finish_reason if r.choices else None,
                    "system_head": (kw.get("messages") or [{}])[0].get("content", "")[:160],
                    "user_tail": (kw.get("messages") or [{}])[-1].get("content", "")[-400:],
                    "response": (r.choices[0].message.content or "")[:3000] if r.choices else None,
                }) + "\n")
            return r

        async def emb(*a, **kw):
            try:
                r = await emb_create(*a, **kw)
            except Exception:
                self.errors += 1
                raise
            self.embed_calls += 1
            u = getattr(r, "usage", None)
            self.embed_tokens_in += getattr(u, "prompt_tokens", 0) or 0
            inp = kw.get("input")
            self.embed_inputs += len(inp) if isinstance(inp, list) else 1
            return r

        llm_client.chat.completions.create = chat
        embed_client.embeddings.create = emb


DEBUG_LOG = ENV / "graphiti-smoke-llm.jsonl"
METER = Meter()


def cost_usd(models: dict, usage: dict) -> float:
    """USD at the models.json prices (per 1M tokens) for one meter delta."""
    p_chat = models["prices"][models["chat_model"]]
    p_emb = models["prices"][models["embed_model"]]
    return round(
        usage["llm_tokens_in"] * p_chat["in"] / 1e6
        + usage["llm_tokens_out"] * p_chat["out"] / 1e6
        + usage["embed_tokens_in"] * p_emb["in"] / 1e6, 6)


def build_graphiti(models: dict, key: str, mode: str = "json_schema") -> Graphiti:
    base_url = models["proxy_base_url"]
    chat_model = models["chat_model"]
    llm = OpenAIGenericClient(
        config=LLMConfig(api_key=key, base_url=base_url, model=chat_model, small_model=chat_model),
        structured_output_mode=mode,
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=key, base_url=base_url,
            embedding_model=models["embed_model"], embedding_dim=models["embed_dim"],
        )
    )
    METER.wrap(llm.client, embedder.client)
    return Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, llm_client=llm,
                    embedder=embedder, cross_encoder=NoRerank())


async def cleanup(g: Graphiti) -> int:
    records, _, _ = await g.driver.execute_query(
        "MATCH (n {group_id: $g}) DETACH DELETE n RETURN count(n) AS n", params={"g": GROUP_ID})
    n = records[0]["n"] if records else 0
    log.info("deleted %s nodes under group_id %s", n, GROUP_ID)
    return n


async def count_group(g: Graphiti) -> dict:
    records, _, _ = await g.driver.execute_query(
        "MATCH (n {group_id: $g}) RETURN labels(n)[0] AS label, count(n) AS n", params={"g": GROUP_ID})
    return {r["label"]: r["n"] for r in records}


async def main(args) -> None:
    models = json.loads((ENV / "models.json").read_text())
    if args.chat_model:
        models["chat_model"] = args.chat_model  # comparison runs only; default is models.json
    key = Path(models["proxy_key_file"]).read_text().strip()
    g = build_graphiti(models, key, args.mode)
    out_path = OUT if not args.tag else ENV / f"graphiti-smoke-{args.tag}.json"
    try:
        if args.cleanup_only:
            await cleanup(g)
            return
        if args.clean_first:
            await cleanup(g)
        # Neo4jDriver schedules build_indices_and_constraints() itself at construction
        # (neo4j_driver.py, _init_task). Await that task rather than issuing the same
        # CREATE INDEX statements a second time in parallel.
        init_task = getattr(g.driver, "_init_task", None)
        if init_task is not None:
            await init_task
        else:
            await g.build_indices_and_constraints()

        lme = load_loader()
        t = time.time()
        d = lme.load_longmemeval(DATA)
        log.info("loaded %s instances, %s sessions in %.1fs", len(d["instances"]), len(d["sessions"]), time.time() - t)
        inst = next(x for x in d["instances"] if x.qid == args.instance)
        sids = sorted(inst.evidence_sessions, key=lambda s: d["sessions"][s]["date"])
        log.info("instance %s (%s): %s", inst.qid, inst.qtype, inst.question)
        log.info("evidence sessions %s, gold answer %s", sids, inst.answer)

        summary = {"instance": inst.qid, "question": inst.question, "answer": inst.answer,
                   "question_date": inst.question_date, "group_id": GROUP_ID,
                   "chat_model": models["chat_model"], "embed_model": models["embed_model"],
                   "structured_output_mode": args.mode, "tag": args.tag,
                   "sessions": [], "search": None}

        prev_uuid = None
        for sid in sids:
            s = d["sessions"][sid]
            body = render_session(s["turns"])
            ref = datetime.strptime(s["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            off = log_offset()
            before = METER.snapshot()
            t0 = time.time()
            res = await g.add_episode(
                name=f"longmemeval session {sid}",
                episode_body=body,
                source_description=f"longmemeval session {sid}",
                reference_time=ref,
                source=EpisodeType.message,
                group_id=GROUP_ID,
                previous_episode_uuids=[prev_uuid] if prev_uuid else None,
            )
            wall = time.time() - t0
            prev_uuid = res.episode.uuid
            usage = METER.delta(before)
            usage["cost_usd"] = cost_usd(models, usage)
            rec = {"session_id": sid, "date": s["date"], "turns": len(s["turns"]),
                   "chars": len(body), "episode_uuid": res.episode.uuid,
                   "nodes": len(res.nodes), "edges": len(res.edges),
                   "wall_seconds": round(wall, 1), **usage,
                   "proxy_log_window": log_since(off)}
            summary["sessions"].append(rec)
            log.info("session %s: %s chars, %s nodes, %s edges, %.1fs, %s llm calls (%s in / %s out), %s embed calls (%s inputs, %s tokens), %s errors, cost %.5f USD",
                     sid, len(body), len(res.nodes), len(res.edges), wall, usage["llm_calls"],
                     usage["llm_tokens_in"], usage["llm_tokens_out"], usage["embed_calls"],
                     usage["embed_inputs"], usage["embed_tokens_in"], usage["errors"], usage["cost_usd"])
            for n in res.nodes:
                log.info("   node %s | %s", n.name, (n.summary or "")[:120])
            for e in res.edges:
                log.info("   edge %s | %s | valid_at=%s invalid_at=%s", e.name, e.fact, e.valid_at, e.invalid_at)

        off = log_offset()
        before = METER.snapshot()
        t0 = time.time()
        edges = await g.search(inst.question, group_ids=[GROUP_ID], num_results=10)
        wall = time.time() - t0
        usage = METER.delta(before)
        usage["cost_usd"] = cost_usd(models, usage)
        usage["proxy_log_window"] = log_since(off)
        uuid_to_sid = {r["episode_uuid"]: r["session_id"] for r in summary["sessions"]}
        results = []
        print("\n=== SEARCH:", inst.question)
        print("=== gold answer:", inst.answer)
        for i, e in enumerate(edges, 1):
            cited = [uuid_to_sid.get(u, u) for u in e.episodes]
            print(f"{i:2d}. [{e.name}] {e.fact}")
            print(f"    valid_at={e.valid_at} invalid_at={e.invalid_at} expired_at={e.expired_at}")
            print(f"    episodes={e.episodes} -> {cited}")
            results.append({"rank": i, "name": e.name, "fact": e.fact, "uuid": e.uuid,
                            "valid_at": str(e.valid_at), "invalid_at": str(e.invalid_at),
                            "expired_at": str(e.expired_at), "episodes": e.episodes,
                            "episode_sessions": cited})
        summary["search"] = {"num_results": len(edges), "wall_seconds": round(wall, 2),
                             **usage, "edges": results}
        summary["graph_counts"] = await count_group(g)
        out_path.write_text(json.dumps(summary, indent=2))
        log.info("wrote %s", out_path)
    finally:
        await g.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="9aaed6a3")
    p.add_argument("--cleanup-only", action="store_true", help="delete the smoke group_id and exit")
    p.add_argument("--clean-first", action="store_true", help="delete the smoke group_id before ingesting")
    p.add_argument("--mode", default="json_schema", choices=["json_schema", "json_object"],
                   help="OpenAIGenericClient structured_output_mode")
    p.add_argument("--chat-model", default=None, help="override models.json chat_model (comparison runs)")
    p.add_argument("--tag", default=None, help="suffix for the output JSON name")
    asyncio.run(main(p.parse_args()))

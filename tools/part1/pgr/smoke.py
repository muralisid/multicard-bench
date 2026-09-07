"""Smoke test for post-graph-rag 1.12.0 through the local litellm proxy into local Postgres.

Indexes three short synthetic documents in date order (one chunk each, the way
evaluation/longmemeval/run.py indexes a session), then runs one query in mix
mode and one in naive mode and prints the retrieved context and the answer.

Everything goes into a throwaway database (default pgr_smoke) with
schema_per_realm=True, realm "smoke", so dropping that database removes all
trace of the run.

Usage:
    .venv/bin/python smoke.py [--db-uri ...] [--model gemini-3.6-flash]

The proxy master key is read from part1-tools/env/.proxy_key and never printed.
"""
import argparse
import asyncio
import json
import pathlib
import sys
import time

from post_graph_rag import DocumentMetadata, GraphRAG, QueryParam, RAGConfig
from post_graph_rag.llm import LLMService

HERE = pathlib.Path(__file__).resolve().parent
ENV = HERE.parent / "env"
DOCS = HERE / "smoke_docs"
REQ_LOG = ENV / "requests.jsonl"


class DimLLMService(LLMService):
    """LLMService that asks the embedding endpoint for config.embedding_dim dims.

    gemini-embedding-001 returns 3072 floats by default. pgvector 0.8.6 refuses
    an HNSW index above 2000 dims and post-graph always builds one, so the width
    has to be reduced at the API. The OpenAI embeddings API has a `dimensions`
    parameter and the litellm proxy maps it to Vertex outputDimensionality.
    post-graph-rag does not send it, so this subclass adds it.
    """

    def _encoding_format_kwargs(self):
        kwargs = super()._encoding_format_kwargs()
        kwargs["dimensions"] = self.config.embedding_dim
        return kwargs


def log_lines() -> int:
    try:
        return sum(1 for _ in REQ_LOG.open())
    except FileNotFoundError:
        return 0


def dump(label, obj):
    print(f"\n===== {label} =====")
    print(json.dumps(obj, indent=2, default=str))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-uri", default="postgresql://pgr:pgr@127.0.0.1:5433/pgr_smoke")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--embedding-dim", type=int, default=1536)
    ap.add_argument("--realm", default="smoke")
    ap.add_argument("--api-base", default="http://127.0.0.1:4000/v1")
    args = ap.parse_args()

    api_key = (ENV / ".proxy_key").read_text().strip()

    config = RAGConfig(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        db_uri=args.db_uri,
        realm=args.realm,
        schema_per_realm=True,
        # Everything else stays at the library default (the LongMemEval harness
        # also leaves these at default): embed_relations=True, merge rrf,
        # gleaning_passes=1, max_hops=2, lexical_search=True, mmr off,
        # node_distance off, contradiction_detection off, no exclusive groups.
    )
    print("config (key withheld):", {k: v for k, v in vars(config).items() if k != "api_key"})

    rag = GraphRAG(config, llm=DimLLMService(config))
    phases = {}

    t0 = time.time()
    await rag.initialize()
    print(f"initialize() ok in {time.time() - t0:.1f}s")

    # ---- indexing, one chunk per document, in date order ----------------------
    before = log_lines()
    results = []
    for path in sorted(DOCS.glob("*.txt")):
        text = path.read_text()
        date = path.name[:10]
        meta = DocumentMetadata(document=path.stem, source=f"file://{path.name}",
                                category="synthetic", extra={"report_date": date})
        t = time.time()
        res = await rag.index_document(text, metadata=meta)
        res["_secs"] = round(time.time() - t, 1)
        res["_chars"] = len(text)
        results.append(res)
        print(f"indexed {path.name}: {len(text)} chars, "
              f"entities={res['entities_extracted']} triples={res['triples_extracted']} "
              f"relations_added={res['relations_added']} superseded={res['relations_superseded']} "
              f"mentions={res['mentions_added']} negated={res['negated_relations']} "
              f"in {res['_secs']}s")
    await asyncio.sleep(2)  # let the proxy callback flush
    phases["index"] = (before, log_lines())
    dump("index_document results", results)

    question = ("Who is the chief financial officer of Harbourline Robotics now, "
                "and who held the job before? Where is the company headquartered?")

    # ---- raw retrieval structure, no synthesis --------------------------------
    before = log_lines()
    data = await rag.query_data(question, param=QueryParam(mode="mix", top_k=8))
    await asyncio.sleep(2)
    phases["query_data_mix"] = (before, log_lines())
    dump("query_data(mode=mix, top_k=8) raw structure", data)

    # ---- mix mode query with synthesis ----------------------------------------
    before = log_lines()
    t = time.time()
    out_mix = await rag.query(question, param=QueryParam(mode="mix", top_k=8))
    secs_mix = time.time() - t
    await asyncio.sleep(2)
    phases["query_mix"] = (before, log_lines())
    dump(f"query(mode=mix, top_k=8) in {secs_mix:.1f}s", out_mix)

    # ---- naive mode query with synthesis --------------------------------------
    before = log_lines()
    t = time.time()
    out_naive = await rag.query(question, param=QueryParam(mode="naive", top_k=8))
    secs_naive = time.time() - t
    await asyncio.sleep(2)
    phases["query_naive"] = (before, log_lines())
    dump(f"query(mode=naive, top_k=8) in {secs_naive:.1f}s", out_naive)

    # ---- what the retrieval output exposes ------------------------------------
    rel_fields = sorted(data["data"]["relationships"][0].keys()) if data["data"]["relationships"] else []
    chunk_fields = sorted(data["data"]["chunks"][0].keys()) if data["data"]["chunks"] else []
    chunk_meta_fields = sorted(data["data"]["chunks"][0]["metadata"].keys()) if data["data"]["chunks"] else []
    ent_fields = sorted(data["data"]["entities"][0].keys()) if data["data"]["entities"] else []
    dump("retrieval output fields", {
        "query() keys": sorted(out_mix.keys()),
        "query_data() keys": sorted(data.keys()),
        "query_data()['data'] keys": sorted(data["data"].keys()),
        "query_data()['metadata'] keys": sorted(data["metadata"].keys()),
        "relationship fields": rel_fields,
        "chunk fields": chunk_fields,
        "chunk metadata fields": chunk_meta_fields,
        "entity fields": ent_fields,
    })

    dump("proxy request log line ranges per phase (start, end)", phases)
    (ENV / "pgr-smoke-phases.json").write_text(json.dumps(phases, indent=2))
    await rag.close()


if __name__ == "__main__":
    asyncio.run(main())

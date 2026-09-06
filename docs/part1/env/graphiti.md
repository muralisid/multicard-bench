# Neo4j plus Graphiti: local install, smoke on two LongMemEval sessions, cost projection

Date: 2026-09-06. Apple Silicon Mac, macOS, 8 cores, 16 GB. Everything runs as the current user, no sudo. Nothing was committed and nothing went public.

## 1. Versions

| Piece | Version | Where |
|---|---|---|
| Neo4j | 2026.07.1, community edition, Cypher 5 and 25 | brew, /opt/homebrew/Cellar/neo4j/2026.07.1 |
| cypher-shell | 2026.07.1 | brew, /opt/homebrew/bin/cypher-shell |
| OpenJDK | 21.0.12.1 (openjdk@21, keg-only) | brew, /opt/homebrew/opt/openjdk@21 |
| graphiti-core | 0.30.1 (latest on PyPI on 2026-09-06) | /Users/muralisid/github_other/part1-tools/graphiti/.venv |
| openai (Python) | 3.8.0 | same venv |
| neo4j (Python driver) | 6.3.0 | same venv |
| httpx | 0.28.1, installed by hand | same venv (openai 3.8.0 no longer depends on it; graphiti_core imports it) |
| Python | 3.11.16 (uv) | same venv |

Install log: /Users/muralisid/github_other/part1-tools/env/neo4j-install.log and graphiti-install.log.

## 2. Neo4j as a user process

The brew wrapper /opt/homebrew/opt/neo4j/bin/neo4j sets JAVA_HOME to the openjdk@21 keg and NEO4J_HOME to /opt/homebrew/Cellar/neo4j/2026.07.1/libexec. The config there is owned by the user, so no copy was needed.

- Config: /opt/homebrew/Cellar/neo4j/2026.07.1/libexec/conf/neo4j.conf
- Data: /opt/homebrew/var/neo4j/data
- Server logs: /opt/homebrew/var/log/neo4j
- Console output: /Users/muralisid/github_other/part1-tools/env/neo4j.log
- Pid file: /Users/muralisid/github_other/part1-tools/env/neo4j.pid
- Bolt: bolt://127.0.0.1:7687. HTTP: http://127.0.0.1:7474 (answers 200).
- User neo4j, password graphiti (local dev only). Set once with `neo4j-admin dbms set-initial-password graphiti` before the first start.

One config line was appended to neo4j.conf: `server.fleet_discovery.enabled=false`. Without it the first start logged "Fleet discovery broadcasts have been enabled for 1 network" and sent UDP broadcasts on the LAN. After the restart that line is gone from the log.

Start (leaves it running in the background):

    cd /Users/muralisid/github_other/part1-tools/neo4j
    nohup /opt/homebrew/opt/neo4j/bin/neo4j console > /Users/muralisid/github_other/part1-tools/env/neo4j.log 2>&1 &
    echo $! > /Users/muralisid/github_other/part1-tools/env/neo4j.pid

It prints "Started." in neo4j.log after about 6 seconds.

Stop:

    kill "$(cat /Users/muralisid/github_other/part1-tools/env/neo4j.pid)"
    rm -f /Users/muralisid/github_other/part1-tools/env/neo4j.pid

Shutdown takes about 11 seconds. Check:

    /opt/homebrew/bin/cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p graphiti "RETURN 1;"

Both checks passed: `RETURN 1` gave 1, and `CALL dbms.components()` gave Neo4j Kernel 2026.07.1 community.

Neo4j was left RUNNING at the end of this stage (pid in neo4j.pid). Graphiti created 31 indexes in the neo4j database (27 range, 4 fulltext), all online. They stay.

## 3. Graphiti configuration

Script: /Users/muralisid/github_other/part1-tools/graphiti/smoke.py. Run from that directory with `.venv/bin/python smoke.py`.

- LLM: `graphiti_core.llm_client.openai_generic_client.OpenAIGenericClient` with `LLMConfig(api_key=<proxy key>, base_url="http://127.0.0.1:4000/v1", model="gemini-2.5-flash-lite", small_model="gemini-2.5-flash-lite")`. The model comes from chat_model in models.json, the key from part1-tools/env/.proxy_key. Library defaults kept: temperature 1, max_tokens 16384, structured_output_mode "json_schema" (the client sends response_format type json_schema without strict; the proxy passed it through and every reply parsed).
- Embedder: `graphiti_core.embedder.openai.OpenAIEmbedder` with `OpenAIEmbedderConfig(api_key, base_url, embedding_model="gemini-embedding-001", embedding_dim=3072)`. Graphiti stores the vectors as node and edge properties and scores with `vector.similarity.cosine` at query time, so no vector index is needed for 3072 dims.
- Cross encoder: replaced with a no-op class (NoRerank in smoke.py) that returns the passages in the given order. The library default, OpenAIRerankerClient, asks for logprobs, which the Gemini proxy does not return. The default search recipe never calls it.
- Search: `Graphiti.search(question, group_ids=[...], num_results=10)`, which is the library's EDGE_HYBRID_SEARCH_RRF recipe: BM25 over the fulltext index plus cosine over fact embeddings, merged with RRF. No LLM call in search, one embedding call for the query.
- Telemetry: GRAPHITI_TELEMETRY_ENABLED=false set in the script.
- Graph driver: Graphiti(uri, user, password) builds its own Neo4jDriver, database neo4j. The driver schedules build_indices_and_constraints() itself at construction; the script awaits that task instead of calling it again (calling both in parallel logs "equivalent index already exists" errors, harmless).
- Concurrency: Graphiti's SEMAPHORE_LIMIT default 20 inside one episode. Sessions were ingested one after the other.
- Per-run accounting: the proxy log requests.jsonl is shared with other processes on this Mac (a gemini-3.6-flash call from another process landed inside the first run's window). The script therefore wraps the two AsyncOpenAI methods Graphiti uses and reads `usage` from each response. Those numbers are the ones below. Every LLM exchange is also written to part1-tools/env/graphiti-smoke-llm-<tag>.jsonl.

## 4. Ingest and search code path

1. Load /Users/muralisid/github_other/multicard-bench/data/raw/longmemeval_s.json with the bench loader src/multicard/data/longmemeval.py (imported by file path; 500 instances, 19,829 unique sessions, 0.8 s).
2. Instance 9aaed6a3 (type multi-session). Question: "How much cashback did I earn at SaveMart last Thursday?" Gold answer: $0.75. Question date 2023-05-30. Evidence sessions: answer_353d3c6d_2 (2023-05-22, 12 turns, 6,222 chars rendered) and answer_353d3c6d_1 (2023-05-23, 12 turns, 2,735 chars). Both are below the corpus mean of 10,069 rendered chars per session (median 10,308, mean 10.1 turns).
3. Each session becomes ONE episode: `add_episode(name="longmemeval session <sid>", episode_body="role: text" lines joined with newlines, source_description="longmemeval session <sid>", reference_time=<session date at 00:00 UTC>, source=EpisodeType.message, group_id="smoke-longmemeval-2026-09-06", previous_episode_uuids=[<the earlier episode>])`. Ingested in date order. Without previous_episode_uuids Graphiti pulls the last EPISODE_WINDOW_LEN = 3 episodes of the group into every prompt.
4. `search(question, group_ids=[group_id], num_results=10)` and print each edge: name, fact, valid_at, invalid_at, expired_at, the episode uuids it cites, mapped back to session ids.
5. Results JSON: part1-tools/env/graphiti-smoke-<tag>.json. Logs: graphiti-smoke-<tag>.log.

## 5. Per-session numbers

Main run, the configuration above (tag flashlite-json_schema):

| Session | Chars | LLM calls | Tokens in | Tokens out | Embed calls | Embed tokens | Wall s | Nodes | Edges | USD |
|---|---|---|---|---|---|---|---|---|---|---|
| answer_353d3c6d_2 | 6,222 | 2 | 6,946 | 210 | 6 | 92 | 7.8 | 2 | 1 | 0.00079 |
| answer_353d3c6d_1 | 2,735 | 4 | 16,064 | 1,232 | 2 | 20 | 9.3 | 5 | 0 | 0.00210 |
| mean per session | 4,479 | 3 | 11,505 | 721 | 4 | 56 | 8.55 | 3.5 | 0.5 | 0.00145 |

USD is at the models.json prices: 0.10 in and 0.40 out per 1M for gemini-2.5-flash-lite, 0.15 per 1M for gemini-embedding-001. The LLM calls were: extract entities, extract edges, (second session only) dedupe entities and summarise entities. No failures, no retries, no 429 from Vertex (0 failure lines in requests.jsonl for the whole day, 0 errors counted by the script).

Search: 1.0 s, 0 LLM calls, 1 embedding call (12 tokens). It returned the only edge in the graph:

    1. [HAS_SALE] SaveMart has a sale on ground beef this week, Buy One Get One 50% Off.
       valid_at=2023-05-22 00:00:00+00:00 invalid_at=None expired_at=None
       episodes=['46c7438a-...'] -> ['answer_353d3c6d_2']

valid_at came from the session date, so the temporal fields work through the proxy. The answer fact ($75 at SaveMart last Thursday, 1% cashback) was not in the graph. The raw replies (graphiti-smoke-llm-flashlite-json_schema.jsonl) show why: gemini-2.5-flash-lite never extracted the speaker "user" although the prompt asks for it, found only SaveMart and ground beef in the 12-turn grocery session, and in the expenses session wrote self-loop edges (SaveMart -> SaveMart) that Graphiti drops. The JSON itself was always well formed.

Two comparison runs on the same two sessions, same code, graph cleared in between:

| Tag | Chat model | Mode | LLM calls | Tokens in | Tokens out | Embed calls | Wall s | Nodes | Edges | Search hits | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flashlite-json_schema (main) | gemini-2.5-flash-lite | json_schema | 6 | 23,010 | 1,442 | 8 | 17.1 | 7 | 1 | 1 | 0.0029 |
| flashlite-json_object | gemini-2.5-flash-lite | json_object | 11 | 27,567 | 2,232 | 16 | 23.4 | 8 | 5 | 5 | 0.0037 |
| flash-json_schema | gemini-2.5-flash | json_schema | 15 | 26,035 | 13,896 | 26 | 75.1 | 10 | 9 | 9 | 0.0427 |

Totals are for both sessions. gemini-2.5-flash extracted "user" and dated facts (for example "user spent $50 at Sephora", valid_at 2023-05-18) but also missed the $75 SaveMart purchase and the 1% cashback. Its output tokens are high because the model thinks by default and thinking tokens are billed as output. None of the three runs put the answer fact into the graph.

Spend for this stage, all runs including one early run before the in-script meter existed: 0.052 USD.

## 6. Projections

Basis: the main run, mean per session 11,505 tokens in, 721 tokens out, 56 embedding tokens, 8.55 s wall, 0.001447 USD. Two views, because the two smoke sessions average 4,479 chars and the corpus averages 10,069 (ratio 2.25):

| | Sessions | As measured | Scaled by length (x2.25) | Wall, one at a time |
|---|---|---|---|---|
| (a) full S set | 19,829 | 28.7 USD | 64.5 USD | 47.1 h |
| (b) 150-question subset | 6,000 | 8.7 USD | 19.5 USD | 14.3 h |

(a) fits under 200 USD in both views with gemini-2.5-flash-lite. So does (b).

The same projection for the comparison models, as measured: flash-lite json_object 36.8 USD full and 11.1 USD subset; gemini-2.5-flash 422.8 USD full (over 200) and 127.9 USD subset (under 200 as measured, 288 USD scaled by length, over). Wall for gemini-2.5-flash: 37.6 s per session, 207 h full, 63 h subset, one at a time.

Wall time notes: the proxy and Neo4j both ran on this Mac. The 8.55 s is the sum of Graphiti's LLM and embedding round trips for one session with nothing else ingesting. Ingesting several group_ids at once would divide the wall time; Vertex rate limits were not hit at this rate and were not tested at higher rates. A 150-question subset has 7,028 unique sessions when the first 150 question ids are taken in sorted order, not 6,000; the table uses the 6,000 given in the task.

## 7. Recommendation

Subset. Cost is not the blocker (the full set is 29 to 65 USD with flash-lite). The blockers are 47 hours of sequential wall time and extraction quality: flash-lite missed the answer fact in both evidence sessions of the one instance tested. Run the 6,000-session subset first, measure how often the evidence facts land in the graph, and only then decide on the full set.

## 8. Cleanup

The two smoke episodes and their entities were deleted with `smoke.py --cleanup-only` (MATCH (n {group_id: 'smoke-longmemeval-2026-09-06'}) DETACH DELETE n). The graph holds 0 nodes. If anything under that group_id reappears, exclude group_id smoke-longmemeval-2026-09-06.

## 9. Files

- /Users/muralisid/github_other/part1-tools/graphiti/smoke.py
- /Users/muralisid/github_other/part1-tools/graphiti/.venv
- /Users/muralisid/github_other/part1-tools/env/graphiti.md (this file)
- /Users/muralisid/github_other/part1-tools/env/graphiti-smoke-flashlite-json_schema.json, .log, and graphiti-smoke-llm-flashlite-json_schema.jsonl (main run)
- /Users/muralisid/github_other/part1-tools/env/graphiti-smoke-flashlite-json_object.json, .log, graphiti-smoke-llm-flashlite-json_object.jsonl
- /Users/muralisid/github_other/part1-tools/env/graphiti-smoke-flash-json_schema.json, .log, graphiti-smoke-llm-flash-json_schema.jsonl
- /Users/muralisid/github_other/part1-tools/env/graphiti-smoke.json and graphiti-smoke.log (same content as the main run)
- /Users/muralisid/github_other/part1-tools/env/neo4j.log, neo4j.pid, neo4j-install.log, graphiti-install.log
- /opt/homebrew/Cellar/neo4j/2026.07.1/libexec/conf/neo4j.conf (one line appended)

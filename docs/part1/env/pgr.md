# post-graph-rag: install, harness facts, and the smoke test

Date: 2026-09-06. Machine: the Apple Silicon Mac. Everything ran as the current user. No commits, no pushes, nothing public.

Clean-room note: this stage read only the public GitHub clone, the PyPI package, and the files under part1-tools.

## 1. What was installed

| Item | Value |
|---|---|
| Source clone | /Users/muralisid/github_other/part1-tools/post-graph-rag-src (read only reference) |
| Clone commit | 69c2e0ece5f25a41e957f2a6a803992afb5459eb, tag v1.12.0, "chore: release 1.12.0", 2026-09-05 21:44 +0100 |
| Package venv | /Users/muralisid/github_other/part1-tools/pgr/.venv (uv, CPython 3.11.16) |
| post-graph-rag from PyPI | 1.12.0 (wheel uploaded 2026-09-05). Same version string as the clone's pyproject.toml and post_graph_rag/__init__.py line 33 |
| Pulled in | post-graph 1.5.0, openai 3.8.0, asyncpg 0.31.0, pydantic 2.13.5, igraph 1.0.0, leidenalg 0.12.0, httpx2 2.12.0 |
| PyPI history | 26 releases, 0.1.0 on 2026-07-25 to 1.12.0 on 2026-09-05 |

Install command used:

    cd /Users/muralisid/github_other/part1-tools/pgr
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python "post-graph-rag==1.12.0"

## 2. The LongMemEval harness, from the clone

All paths below are relative to /Users/muralisid/github_other/part1-tools/post-graph-rag-src.

### 2.1 Which run is the published number, and its config

The README (README.md lines 50-57) and the write-up (docs/index.md lines 63-76, 400-412) report 94.0% overall on the 499 usable oracle questions. docs/index.md line 76 says the configuration was "gemini-3.6-flash for extraction and synthesis, gemini-embedding-001 at 1536 dimensions, RRF across the three retrieval channels, answers graded by a three-model majority panel, two repeats per question".

The result file that carries that number is evaluation/longmemeval/reader_sweep_nolimit.json (committed 2026-09-05 in c8bee64 "repo: ship the results behind the published claims"). Its own header says:

- index_model: gemini-3.7-flash (the graph was built by 3.7, not 3.6)
- arms (answering models): gemini-3.7-flash, gemini-3.6-flash, google/gemma-4-26b-a4b-it-maas, gpt-oss-120b
- judge_pool: gemini-3.7-flash, google/gemma-4-26b-a4b-it-maas, Meta-Llama-3.3-70B-Instruct, gpt-oss-120b
- panel for the gemini-3.6-flash arm: gemma-4-26b-a4b-it-maas, Meta-Llama-3.3-70B-Instruct, gpt-oss-120b (the arm's own family is excluded, reader_sweep.py lines 66-87)
- repeats: 2
- means_judge: gemini-3.6-flash 0.9399, gemini-3.7-flash 0.9349, gpt-oss-120b 0.8918, gemma 0.8066
- means_layered (the deterministic numeric/period/cosine score, reader_sweep.py lines 90-105): gemini-3.6-flash 0.497
- degraded: 1 instance (eaca4986, "LLM returned no usable entities or triples"), reportable: false
- by type for gemini-3.6-flash: single-session-user 0.957, single-session-assistant 1.000, knowledge-update 0.949, multi-session 0.902, temporal-reasoning 0.962, single-session-preference 0.833. These are the README table numbers.

Two points where the prose and the file differ: the file says the extraction model was gemini-3.7-flash while the prose says gemini-3.6-flash did "extraction and synthesis"; and docs/index.md line 420 names the panel as MiniMax-M2.7, gpt-oss-120b and DeepSeek-V3.2, while the file's panel for that arm is gemma, Llama-3.3-70B and gpt-oss-120b. The MiniMax/gpt-oss/DeepSeek panel is the one in evaluation/longmemeval/README.md lines 19-27 (the 20-instance 75% run) and in ablation_baseline.json.

The earlier published figure, 85.8%, is evaluation/longmemeval/reader_sweep_validity.json (same layout, gemini-3.6-flash means_judge 0.8577). tables/results-at-a-glance.csv and tables/longmemeval-by-question-type.csv still carry 85.8%. The difference between the two files is the context token budget: docs/index.md line 414 says the 4000 token default was discarding retrieved passages; 1.11.1 made the budgets unlimited (post_graph_rag/models.py lines 124-132).

A third file, evaluation/longmemeval/official_gpt4o_g36.json, regrades stored answers with gpt-4o as judge and the benchmark's own prompts (regrade_official.py lines 39-46, temperature 0, max_tokens 10, "yes" substring). It reads gemini-3.6-flash 0.782 overall and gpt-4o (reading the same post-graph-rag retrieval) 0.713, n=499. Its source file reader_sweep_gpt4_full.json is not in the repo.

Code-level settings of the 94.0% run, from evaluation/longmemeval/reader_sweep.py:

| Setting | Value | Where |
|---|---|---|
| RAGConfig model (indexing) | --index-model, default gemini-3.7-flash | lines 111-112, 203 |
| realm | rsweep_{seed}_{i}, schema_per_realm=True. reader_sweep.py only closes the connection afterwards (line 285); run.py is the one that drops each instance's schema (line 418) | lines 203-204 |
| embedding_model / embedding_dim | gemini-embedding-001 / 1536 | lines 116, 205 |
| embed_relations | True | line 205 |
| merge_strategy | rrf | line 205 |
| max_concurrent_chunks | 8 | lines 133, 206 |
| pool_min_size / pool_max_size | 1 / 4 | lines 131-132, 207 |
| extraction_prompt | CONVERSATIONAL_PROMPT from run.py lines 58-95 | line 208 |
| max_retries / retry_deadline_secs | 40 / 1800 | lines 142, 209 |
| Everything else | library defaults: gleaning_passes 1, max_hops 2, max_relation_edges 200, lexical_search True, lexical_top_k 50, relation_seed_quota 0.5 (unused under rrf), mmr off, node_distance_rerank off, contradiction_detection off, exclusive_predicate_groups empty, extract_validity True, render_relation_validity True, expand_chunks_via_mentions True, auto_decompose False, chunk_chars 2000, chunk_overlap_chars 200 | post_graph_rag/config.py |
| Answering model | rag.config.model = arm, swapped after indexing | line 239 |
| Question sent | "Today is {question_date}. {question}" | lines 220-221 |
| QueryParam | mode="mix", top_k=8, no token budgets (None = unlimited) | line 244 |
| Judge prompt | run.py JUDGE_PROMPT lines 97-109, one word CORRECT or INCORRECT | run.py lines 216-221 |
| Majority vote | judge_panel, run.py lines 191-213, over votes actually cast | |

run.py itself (the original harness) differs: one model both extracts and answers (line 347), top_k=32 (line 393), and the question is prefixed with an answer-side rule about conflicts and date arithmetic (lines 381-392). Its judge default is a single gemini-3.6-flash (line 228), but line 290 refuses a judge equal to the model, so --judges must be given. run.py preflights one session and drops the schema lme_preflight (lines 300-322), and drops each instance's schema afterwards (line 418).

### 2.2 How sessions become documents

evaluation/longmemeval/run.py, index_instance, lines 144-188:

- One document per session. The session's turns are rendered as "role: content" lines with a first line "[Conversation on YYYY-MM-DD]" (render_session, lines 112-123). The date is put in the body on purpose: metadata never reaches the prompt (lines 116-120).
- Sessions are sorted by date before indexing (line 155). The docstring says supersession resolves by document order, so a later session indexed first would invert which fact wins (lines 147-149).
- Metadata: DocumentMetadata(document=session_id, source="session://{session_id}", category="chat_session", extra={"session_date": "YYYY-MM-DD"}) (lines 160-162). The document key becomes "session://{id}::{id}" (post_graph_rag/models.py lines 92-111).
- Indexed with rag.index_document(text, metadata) (line 164). index_document is one chunk, no chunker (post_graph_rag/engine.py lines 453-462: "Index one chunk"). So a session is never split, whatever its length. The chunker (paragraph_chunker, post_graph_rag/chunking.py lines 21-59: pack paragraphs to 2000 chars, carry 200 chars forward, drop paragraphs under 40 chars and lines starting with "==") is only used by index_text (engine.py lines 313-383), which the harness does not call.
- If the conversational prompt fails on a session, the harness retries with the library's document prompt (lines 165-182). If that fails too, or a session yields nothing, the instance is DegradedRun and excluded from the denominator (lines 132-141, 183-187, 362-367).

### 2.3 What "supersession" actually does in the frozen config

Supersession is written at index time in two ways only (post_graph_rag/engine.py lines 575-606): supersede_conflicting fires when the new predicate and an older one between the same two entities are in a declared exclusive_predicate_groups set (graph_store.py lines 672-729); mark_superseded fires from LLM contradiction detection (graph_store.py lines 774-808). Both write superseded_by and t_expired into the older relation's payload. The frozen LongMemEval config declares no groups and leaves contradiction_detection off, so neither fires there. The smoke below confirms: relations_superseded was 0 on every document.

What the run does use instead: the conversational prompt asks for the conversation date in valid_from on every triple (run.py lines 77-85); valid_from and valid_to are stored (graph_store.py lines 545-547) and rendered into the answer prompt as "[from ...]" or "[valid ... to ...]" (engine.py lines 1568-1592, render_relation_validity); and the traversal channel orders relations nearest hop first, then newest asserted first (engine.py lines 1289-1304). Every relation also carries transaction time t_created and t_expired (graph_store.py lines 521-523, 552-553).

### 2.4 fetch.sh: oracle or S

evaluation/longmemeval/fetch.sh (9 lines): each argument is a variant name, default "oracle"; it downloads https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_{variant} to {variant}.json in the same directory. So "./fetch.sh" gets oracle.json and "./fetch.sh s" would get s.json. run.py, reader_sweep.py, ablate_retrieval.py and graph_vs_flat.py all default --data to oracle.json (run.py line 226, reader_sweep.py line 110). Nothing in the repo runs S or M. The README's caveat (evaluation/longmemeval/README.md lines 76-78) says the oracle variant contains only the evidence sessions and is easier than the S and M haystacks Zep report on.

### 2.5 What a query returns, and where the provenance is

query() (engine.py lines 1525-1673) returns a dict with: question, answer, mode, keywords {high_level, low_level}, retrieved_documents, retrieved_entities (names only), retrieved_graph_triples (strings "(src) --[pred]--> (tgt)"), retrieved_communities (titles), references [{reference_id, document}].

The structured form is query_data() (engine.py lines 850-1088), also reachable as query(..., QueryParam(only_need_context=True)) (lines 1536-1537). It makes no synthesis call. It returns:

- data.entities: entity_name, entity_type, description (lines 1040-1044). No id, no provenance.
- data.relationships: src_id, tgt_id (display names), edge_id, src_key, tgt_key (vertex ids), relation_type, description, weight, negated, confidence, valid_from, valid_to, t_created, t_expired, superseded_by, asserted_at, hops (_format_triple, lines 1204-1234). The relation's source chunk ids (payload "sources") are stored (graph_store.py lines 499-511, 541) but not in this output; read them from the relations table by edge_id.
- data.chunks: chunk_id (documents.id), content, metadata (source, category, collection, document, page, paragraph, space, doc_key, content_hash and any extra keys such as session_date; the smoke shows the full text is repeated inside metadata as "text" because from_dict keeps unknown payload keys) (lines 1046-1050).
- data.references, data.communities (community_id, level, title, summary, findings, rating, size, distance, score).
- metadata: query_mode, keywords, processing_info (total_entities_found, total_relations_found, final_chunks_count, communities_found).

Channels feeding data.relationships in mix mode: entity vector search then traversal to max_hops (lines 891-941), relation embedding search (lines 948-958), lexical search over relations (lines 964-974), fused by reciprocal rank fusion with k=60 (lines 1004-1011, 1328-1358), then de-duplicated. Chunks come from document vector search (lines 910-913) plus chunks that mention the matched entities (lines 979-987). In naive mode only the document vector search runs, but the keyword extraction call and the entity vector embedding still happen (lines 869-882).

For scoring retrieval without the synthesis: use query_data() and map chunk_id or metadata.document (the session id) for chunks; map edge_id to relations.payload->'sources' for relations; map entity names to doc_mentions through the unique (realm, space, lower(name)) index for entities. See pgr-schema.md.

## 3. Fitting the package to the local proxy and Postgres

Settings used: api_base http://127.0.0.1:4000/v1, api_key from part1-tools/env/.proxy_key (never printed), model gemini-3.6-flash (the chandan_model in models.json), embedding_model gemini-embedding-001, db_uri postgresql://pgr:pgr@127.0.0.1:5433/pgr_smoke, realm "smoke", schema_per_realm=True, everything else default. This matches the harness except for the realm and the missing conversational prompt (the smoke documents are reports, not chats).

One thing had to be added. models.json records gemini-embedding-001 at 3072 dimensions. post-graph creates an HNSW index on every embedding column at table creation (post_graph/client_asyncpg.py lines 338 and 340), and pgvector 0.8.6 refuses HNSW above 2000 dimensions ("column cannot have more than 2000 dimensions for hnsw index", checked in a rolled-back transaction; 1536 works). Chandan's harness uses 1536. The OpenAI embeddings API has a "dimensions" parameter, the litellm proxy passes it to Vertex as outputDimensionality, and a probe through the proxy returned 1536 floats with it and 3072 without. post-graph-rag never sends "dimensions" (post_graph_rag/llm.py lines 57-66, 76-84, 117-125), so the smoke subclasses LLMService and adds it:

    class DimLLMService(LLMService):
        def _encoding_format_kwargs(self):
            kwargs = super()._encoding_format_kwargs()
            kwargs["dimensions"] = self.config.embedding_dim
            return kwargs

    rag = GraphRAG(config, llm=DimLLMService(config))

This is in part1-tools/pgr/smoke.py. The real runs need the same wrapper (or a proxy-side default) to run at 1536.

The proxy already accepts encoding_format=float, which the package sends by default (config.py lines 163-168), so no other change was needed.

## 4. The smoke

Throwaway database pgr_smoke (owner pgr, extension vector 0.8.6), realm "smoke", so the previous stage's database pgr was not touched. Three synthetic reports about a made-up company, 556 words and 3459 characters in total, in part1-tools/pgr/smoke_docs/, dated in the body and in the file name. The fact that changes: the chief financial officer is Priya Menon in the first two documents and Tomas Ekholm from 2024-05-01 in the third; the headquarters also moves from Bristol to Cardiff. Indexed with index_document in date order, one chunk each, like the harness.

Log of the run: part1-tools/env/pgr-smoke.log. Phase boundaries in the proxy log: part1-tools/env/pgr-smoke-phases.json.

Indexing results (index_document return values):

| Document | Chars | Entities | Triples | Relations added | Superseded | Mentions | Seconds |
|---|---|---|---|---|---|---|---|
| 2023-02-14_founding_note | 1238 | 15 | 20 | 20 | 0 | 14 | 56.4 |
| 2023-09-05_autumn_update | 1126 | 14 | 16 | 16 | 0 | 14 | 42.9 |
| 2024-04-22_leadership_change | 1095 | 11 | 14 | 14 | 0 | 11 | 41.0 |

State afterwards: 18 entity vertices, 48 relations (34 distinct predicates, hold_position the commonest at 6), 39 doc_mentions edges, all 48 relations embedded, 15 relations with a validity date (for example Tomas Ekholm hold_position Harbourline Robotics valid_from 2024-05-01; Priya Menon hold_position Harbourline Robotics valid_to 2024-04-30; headquarters in Bristol valid_to 2024-03 and in Cardiff valid_from 2024-03). Aliases were merged: "Rao", "Menon", "Northgate", "Kestrel", "Kestrel picking arm", and also "the company" as an alias of Harbourline Robotics. Two relations reached weight 2 from two chunks. One self-loop was stored (Swindon depot locate_in Swindon depot) because "Swindon" was resolved onto the entity "Swindon depot" after the extractor's self-loop check.

Question asked in both modes: "Who is the chief financial officer of Harbourline Robotics now, and who held the job before? Where is the company headquartered?"

- mix, top_k=8, 11.2 s: 8 entities, 48 relations (every relation in the graph, at 2 hops from the matched entities), 3 chunks. Answer: Tomas Ekholm from 1 May 2024, Priya Menon before that from March 2022 to 30 April 2024 and now a non-executive director; headquartered in Cardiff since March 2024, previously Bristol, Bristol workshop still open. Citations [1], [2], [3].
- naive, top_k=8, 9.3 s: 0 entities, 0 relations, 3 chunks. Same answer in substance, with the same citations.
- query_data in mix mode printed the raw structure; its keyword call returned high_level ["Chief Financial Officer", "Corporate Leadership", "Company Headquarters"] and low_level ["Harbourline Robotics", "CFO"].

Both answers are correct against the documents. The full retrieved structures and both answers are in pgr-smoke.log.

## 5. Calls, tokens and cost

From part1-tools/env/requests.jsonl (the proxy's per-request log), summed by part1-tools/pgr/smoke_cost.py at the prices in models.json (gemini-3.6-flash 0.75 in and 3.75 out per 1M tokens; gemini-embedding-001 0.15 per 1M). The log is shared with every other stage on this machine, and another process was using the proxy during the smoke (16 foreign rows landed inside the index window: gemini-2.5-flash-lite chats and 5 to 12 token embeddings). Rows were attributed to the smoke by model, and for embeddings by a 20 token floor.

| Phase | Chat calls | Chat in | Chat out | Embedding calls | Embedding tokens | USD |
|---|---|---|---|---|---|---|
| index (3 chunks) | 6 | 14027 | 23536 | 9 logged (11 expected) | 2977 logged | 0.09923 |
| query_data mix | 1 | 267 | 311 | 2 | 58 | 0.00138 |
| query mix | 2 | 3726 | 1225 | 2 | 67 | 0.00740 |
| query naive | 2 | 1392 | 1230 | 2 | 62 | 0.00567 |
| total | 11 | | | 15 | | 0.11367 |

Two probes after the smoke (a one word gemini-3.6-flash call, twice, to read the usage block) cost 0.00051 more, and two 1-token embedding probes before it were under 0.000001. Whole stage: about USD 0.114. litellm's own response_cost column agrees with the models.json arithmetic to five decimals.

Calls per chunk, from the code (engine.py lines 464-519): 2 chat calls (extraction plus one gleaning pass, extractor.py lines 439 and 475) and 3 to 4 embedding requests (the chunk text, one batch for the entity descriptions, one batch for the relation texts, and one more batch when a triple names an endpoint the extractor did not return as an entity; an empty list makes no call, llm.py lines 114-115). That is 5 to 6 HTTP calls per chunk. The log shows the 6 chat calls but only 9 embedding rows where 11 were expected; the chunk-text embeddings of documents 2 and 3 are the missing rows, although both documents have their 1536-dim embedding stored. The proxy's callback appears to have dropped two rows while the other process was hammering it. Counts should be taken from the code, and tokens from the log.

Per query in mix mode: 2 chat calls (keyword extraction, synthesis) and 2 embedding calls (the question; the question plus low-level keywords). Naive mode makes the same 4 calls (engine.py lines 869-882 run for every mode). query_data alone: 1 chat and 2 embeddings.

Why the output side is so large: gemini-3.6-flash bills its reasoning tokens as completion tokens. The probe "Reply with the single word: pong" came back with completion_tokens 73, of which reasoning_tokens 72 and text_tokens 1. The extraction calls returned 2529 to 5602 completion tokens each for JSON that would be a few hundred tokens. Output tokens are 89% of the index cost at these prices.

Measured per chunk (mean chunk 1153 chars): chat in 4676 tokens, chat out 7845 tokens, embeddings about 1158 tokens (with the two unlogged rows estimated from their character counts), USD 0.0331.

Projection to 1,000 chunks of 2,000 characters: the prompt overhead is fixed at about 4160 tokens per chunk across the two chat calls (system prompt, context block and response schema), so chat input rises to about 5055 tokens. The output does not have a measured scaling law. If output stays at 7845 tokens per chunk the cost is USD 0.0335 per chunk, USD 33.5 per 1,000 chunks. If output grows in proportion to the text (x1.73) it is USD 0.0551 per chunk, USD 55.1 per 1,000 chunks. The right figure is between these and closer to the second for dense text. Embeddings are under USD 0.35 per 1,000 chunks either way. Numbers are in part1-tools/env/pgr-smoke-cost.json.

For comparison at the same token counts, gemini-2.5-flash-lite (0.10 in, 0.40 out) would be about a tenth of the price, but it is not the model of the published run.

## 6. Reproduce the smoke

Needs the Postgres server and the proxy running (section 8).

    PSQL=/opt/homebrew/opt/postgresql@17/bin/psql
    $PSQL -h 127.0.0.1 -p 5433 -d postgres -c "CREATE DATABASE pgr_smoke OWNER pgr;"
    $PSQL -h 127.0.0.1 -p 5433 -d pgr_smoke -c "CREATE EXTENSION IF NOT EXISTS vector;"

    cd /Users/muralisid/github_other/part1-tools/pgr
    .venv/bin/python smoke.py 2>&1 | tee /Users/muralisid/github_other/part1-tools/env/pgr-smoke.log
    .venv/bin/python smoke_cost.py

    # schema dump, before dropping
    $PSQL "postgresql://pgr:pgr@127.0.0.1:5433/pgr_smoke" -P pager=off -c '\dt smoke.*' -c '\d smoke.relations'

    # clean up
    $PSQL -h 127.0.0.1 -p 5433 -d postgres -c "DROP DATABASE pgr_smoke;"

smoke.py options: --db-uri, --model (default gemini-3.6-flash), --embedding-model, --embedding-dim (default 1536), --realm, --api-base. It reads the key from part1-tools/env/.proxy_key.

## 7. Cleanup done

DROP DATABASE pgr_smoke was run after the schema dump. The database pgr from the previous stage (with its vec_smoke table) was not touched. The only things left from this stage are files under part1-tools: the clone, the pgr venv, smoke.py, smoke_cost.py, smoke_docs/, and under env: pgr.md, pgr-schema.md, pgr-schema-raw.txt, pgr-smoke.log, pgr-smoke-phases.json, pgr-smoke-cost.json.

## 8. Services

Both were already running from the earlier stages and were left running. This stage started no new server.

Postgres 17, pid 50337, port 5433, data dir /Users/muralisid/github_other/part1-tools/pgdata.

    # start
    LC_ALL=C /opt/homebrew/opt/postgresql@17/bin/pg_ctl -D /Users/muralisid/github_other/part1-tools/pgdata -l /Users/muralisid/github_other/part1-tools/env/postgres.log -w start
    head -1 /Users/muralisid/github_other/part1-tools/pgdata/postmaster.pid > /Users/muralisid/github_other/part1-tools/env/postgres.pid
    # stop
    /opt/homebrew/opt/postgresql@17/bin/pg_ctl -D /Users/muralisid/github_other/part1-tools/pgdata -m fast -w stop
    # check
    /opt/homebrew/opt/postgresql@17/bin/pg_isready -h 127.0.0.1 -p 5433

litellm proxy, pid 51086, http://127.0.0.1:4000/v1.

    # start (needs VERTEX_AI_SERVICE_ACCOUNT_JSON in the environment)
    /Users/muralisid/github_other/part1-tools/litellm/start.sh
    # stop
    /Users/muralisid/github_other/part1-tools/litellm/stop.sh
    # check
    curl -s http://127.0.0.1:4000/health/liveliness

## 9. Things the next stage should know

- The requests.jsonl log is shared across stages and was busy with another client during this smoke. Attribute rows by model and time window, or give each stage its own log file (LITELLM_REQUEST_LOG is read once at proxy start, so a per-stage file means a proxy restart).
- Two embedding rows went missing from the log under that concurrency. Do not rely on the log for call counts.
- gemini-3.6-flash's reasoning tokens make indexing cost about USD 33 to 55 per 1,000 chunks of 2,000 characters at the current price. A LongMemEval oracle run indexes one chunk per session; the number of sessions in oracle.json was not measured here (the data was not fetched).
- Embedding width must be 1536 through the "dimensions" parameter; the package will not do it alone.
- The published 94.0% was indexed by gemini-3.7-flash and judged by gemma-4-26b, Llama-3.3-70B and gpt-oss-120b, per the result file, whatever the prose says. gemini-3.7-flash is not among the models the proxy serves (models.json working_models).
- The harness never runs the S or M variants and never chunks a session.

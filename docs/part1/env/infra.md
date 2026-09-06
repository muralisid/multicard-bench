# Infra changes on 2026-09-06, after the proxy and Graphiti stages

Date: 2026-09-06, 12:37 to about 12:43 local. Everything ran as the current user, no sudo, nothing committed. This file lists what changed in the bench and in part1-tools and how each change was checked.

## 1. Bench dependencies

Command, from /Users/muralisid/github_other/multicard-bench:

    uv add pyarrow "psycopg[binary]"

Result: pyproject.toml gained "pyarrow>=25.0.1" and "psycopg[binary]>=3.3.5"; uv.lock changed. pyarrow 25.0.1 was already installed as a transitive dependency of datasets and is now a direct one. psycopg 3.3.5 and psycopg-binary 3.3.5 were installed. uv run pytest -q: 73 passed right after the add. At the end of this stage the full suite shows 87 passed; this stage added tests/test_part1_infra.py (5 tests, offline) and other agents' test files landed in between.

## 2. Proxy (litellm 1.100.0 on http://127.0.0.1:4000/v1)

Files changed: part1-tools/litellm/config.yaml, request_log.py, tokens_by_model.py. start.sh and stop.sh unchanged.

Changes:

- gemini-3.7-flash and gemini-3.8-flash added to model_list on vertex_location global (both answered in proxy-verify.md; neither was in the survey). GET /v1/models now lists ten models.
- gemini-2.5-flash has reasoning_effort none in its litellm_params. litellm maps it to thinkingConfig thinkingBudget 0 for the 2.5 family. A request that sends its own reasoning_effort overrides it (router semantics: request params win over deployment params; verified, see below).
- request_log.py records three new fields per row: job_tag, tags, user. job_tag is the first metadata.tags entry when sent, else the request body user string, else null. The OpenAI python client sends the user string with user= on chat.completions.create and embeddings.create, and the tag list with extra_body={"metadata": {"tags": ["..."]}}. litellm reads body metadata.tags into litellm_params.metadata.tags and the user string into the callback kwargs; both were read from those places. Failure rows carry the fields too.
- tokens_by_model.py gained --by-tag (group by job_tag) and --since <ts prefix>.

Restart: stop.sh (pid 51086), start.sh (pid 63553) at 12:37. Liveliness answered after 4 seconds. The log before the restart is kept as env/litellm.log.1.

Verification at 12:38, all rows tagged infra-smoke in requests.jsonl, prompt "Reply with the single word: pong":

| Call | Result |
|---|---|
| gemini-3.7-flash, max_tokens 64, user tag | "pong", finish_reason length, 7 in, 60 out (59 reasoning) |
| gemini-3.8-flash, max_tokens 64, user tag | empty choices list, 7 in, 61 out (all reasoning) |
| gemini-3.8-flash, max_tokens 256 | "pong", finish_reason stop, 97 out (96 reasoning) |
| gemini-2.5-flash, max_tokens 8, metadata tag | "pong", finish_reason stop, 1 out, no reasoning tokens (thinking default off) |
| gemini-2.5-flash, max_tokens 64, reasoning_effort low, user and metadata tag | "pong", 18 out (17 reasoning): the request overrides the default |
| gemini-embedding-001, user tag | 3072 floats, 2 tokens |
| gemini-3.7-flash and gemini-3.8-flash, reasoning_effort none | HTTP 400 from Vertex: Thinking level is unsupported: THINKING_LEVEL_MINIMAL |

So thinking is off by default for gemini-2.5-flash and cannot be turned off for the 3.x models through litellm 1.100.0 (none and minimal map to thinking level minimal, which Vertex rejects; low is the lowest level it takes). For 3.6 and 3.7 a 64-token limit leaves room for a one-word answer; for 3.8 use at least 128.

Tag check: the row with user only shows job_tag and user set and tags empty; the row with metadata.tags only shows job_tag and tags set and user null; the row with both shows job_tag equal to the metadata tag, with user kept; the embedding row and the two failure rows show the user tag.

Spend for the proxy checks: 8 rows, 0.0009 USD by litellm's count. Cap was 0.05.

## 3. Neo4j (brew, community 2026.07.1)

Config file: /opt/homebrew/Cellar/neo4j/2026.07.1/libexec/conf/neo4j.conf. A copy from before the edit is at part1-tools/env/neo4j.conf.before-2026-09-06. Three lines changed, each from its commented default:

    dbms.usage_report.enabled=false
    server.memory.heap.initial_size=2g
    server.memory.heap.max_size=2g

New scripts: part1-tools/neo4j/start.sh and stop.sh.

- start.sh runs the brew wrapper in console mode under nohup, waits for "Started." in env/neo4j.log, then writes env/neo4j.pid with the pid of the JVM that listens on 7687 (found with lsof) and env/neo4j-launcher.pid with the NeoBoot launcher pid. The launcher is a small JVM (-Xmx128m) that starts the server JVM as its child and exits when the child exits.
- stop.sh sends SIGTERM to the pid in neo4j.pid (the server JVM), waits for it to exit, waits for the launcher to exit, and waits until nothing listens on 7687. Shutdown is clean: the server log prints Stopping... then Stopped.

Both stop paths were exercised. The first stop killed the launcher pid from the old pid file (55042) and the server followed in 12.6 seconds. The second stop killed the server JVM from the new pid file (63633): Stopped. after 11 seconds and the launcher exited on its own. Start takes about 5 seconds. Neo4j server-log events for these restarts, local time:

    12:38:44  Stopped.
    12:38:51  Started.
    12:41:41  Stopping...
    12:41:52  Stopped.
    12:41:57  Started.

Checks after the final start: cypher-shell RETURN 1 gives 1; the server JVM command line carries -Xms2097152k -Xmx2097152k (2 GB); env/neo4j.log for the new start contains no "Anonymous Usage Data is being sent" line (the two earlier starts each printed one); SHOW INDEXES counts 33, so Graphiti's indexes survived. The log before the first restart is kept as env/neo4j.log.1.

Current state: launcher pid 64212, server pid 64220 on bolt://127.0.0.1:7687, user neo4j, password graphiti (local only).

Noted, not changed: neo4j.conf line "server.jvm.additional=-Djava.awt.headless=true-Dunsupported.dbms.udc.source=homebrew" is two options glued into one by the brew formula. Harmless; left as is.

## 4. Planner prompt

part1-tools/prompts/planner.txt holds the design section 5 planner text, from "Label the question with exactly one shape" to "unlikely to contain.", one line for the instruction and one line per shape definition, nine lines, ASCII only, nothing else. Copied byte for byte to multicard-bench/docs/part1/prompts/planner.txt. tests/test_part1_infra.py checks the copy against the design text with whitespace collapsed.

## 5. Environment records

- proxy.md: models served, restart, job tags, thinking defaults, the two new models in the probe and price tables, the calibration-row models, spend.
- models.json: gemini-3.7-flash and gemini-3.8-flash in prices (0.75 in, 3.75 out, global, introductory through 2026-12-31, then 1.50 and 7.50), working_models and locations; gemini-3.7-flash-lite and gemini-3.8-flash-lite in failed_models; chandan_index_model gemini-3.7-flash and chandan_answer_model gemini-3.6-flash from design section 13, with chandan_model kept for older callers; thinking_defaults, proxy_min_output_tokens_note and job_tag_note.
- graphiti.md: a note in section 2 pointing at the new start and stop scripts and the config lines.
- infra.md: this file.

All four copied to multicard-bench/docs/part1/env/.

## 6. Servers left running

- Proxy: pid 63553 on 127.0.0.1:4000. Stop and start: part1-tools/litellm/stop.sh, start.sh.
- Neo4j: see section 3. Stop and start: part1-tools/neo4j/stop.sh, start.sh.
- Postgres was not touched.

## 7. Spend

0.0009 USD, all through the proxy, all rows tagged infra-smoke. No direct Vertex calls, no Azure calls.

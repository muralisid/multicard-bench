# Part 1 tools: the competitor runners and the local services

These are the scripts that drive other people's systems and the local services
Part 1 needs. They are committed here so the work can be inspected and
reproduced. They do not run inside the bench's virtual environment, because
post-graph-rag and graphiti-core bring their own dependency trees. The live
copies, their virtual environments, their logs and the databases live outside
every repository, at /Users/muralisid/github_other/part1-tools, which holds
about 34 GB, almost all of it the Postgres cluster. Nothing there is committed.

What is here, and what it is for.

## pgr/

`run_spaces.py` drives post-graph-rag exactly as its own LongMemEval harness
does: one space per question, one document per haystack session in session-date
order with the date written into the body, his conversational extraction
prompt, his query call at his shipped result limit and again at raised limits,
his synthesis call for the uncut row, then an export of his entities,
relations (with validity dates, negation, confidence and provenance), chunks
with their turn spans, aliases and relation embeddings to parquet. It meters
every call from the response usage, sends a job tag with every request, shards
the build, projects the total after the first shard and stops against a cap.
`smoke.py` and `smoke_cost.py` are the three-document smoke it was validated
with. `lists/` holds the question id lists the builds were run with.

Configuration decisions that were taken in writing, with their sources, are in
docs/part1/env/pgr.md. His schema is in docs/part1/env/pgr-schema.md and the
raw dump is `pgr-schema-raw.txt` here.

## graphiti/

`run_groups.py` drives Graphiti in pilot, run and cleanup modes: one group per
question, three ingestion variants (a text episode per session, a message
episode per session, one message episode per turn, which is Zep's own
granularity), the library's own hybrid search recipe, Zep's own context
template, and an export of edges, nodes and episodes per group. The pilot
measures cost, wall time, requests per minute and evidence presence for each
variant and applies the design's choice rule. `smoke.py` is the two-session
smoke.

The pilot ran and its table is in docs/part1/env/graphiti.md. No variant fitted
the cost cap, so by the design's rule the arm was dropped.

## litellm/

The OpenAI-compatible proxy over Vertex AI that both competitor systems talk
to. `config.yaml` lists the models and their locations; it reads the Vertex
service account and the proxy master key from the environment and holds no
secret itself. `start.sh` and `stop.sh` manage it. `request_log.py` is the
callback that appends one JSON line per request with the model, token counts,
cost, latency and the job tag. `tokens_by_model.py` sums the log by model or by
tag, which is how competitor spend is cross-checked against the runners' own
meters.

## neo4j/

`start.sh` and `stop.sh` for the local Neo4j that Graphiti uses. The
configuration changes made to it (anonymous usage reporting off, a 2 GB heap)
are recorded in docs/part1/env/infra.md.

## prompts/

`planner.txt`, the LLM planner prompt, byte-identical to the copy at
docs/part1/prompts/planner.txt that the bench reads.

## Bringing the services up

    /Users/muralisid/github_other/part1-tools/litellm/start.sh
    LC_ALL=C /opt/homebrew/opt/postgresql@17/bin/pg_ctl \
        -D /Users/muralisid/github_other/part1-tools/pgdata \
        -l /Users/muralisid/github_other/part1-tools/env/postgres.log -w start
    /Users/muralisid/github_other/part1-tools/neo4j/start.sh

The exact runner commands, with their caps and resume behaviour, are in
docs/part1/RUNBOOK.md.

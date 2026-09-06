# Gemini proxy and model survey

Date: 2026-09-06. Machine: Apple Silicon Mac. Everything runs as the current user.
Updated 2026-09-06 (infra stage, see infra.md): two models added, a thinking default for gemini-2.5-flash, job tags in the request log, proxy restarted at 12:37 (pid 63553).

## What was built

An OpenAI-compatible HTTP proxy (litellm 1.100.0) in front of Vertex AI Gemini.

- Base URL: http://127.0.0.1:4000/v1
- Master key: /Users/muralisid/github_other/part1-tools/env/.proxy_key (mode 600, one line, starts with sk-). Send it as "Authorization: Bearer <key>".
- Config: /Users/muralisid/github_other/part1-tools/litellm/config.yaml
- Venv: /Users/muralisid/github_other/part1-tools/litellm/.venv (Python 3.11, uv, litellm[proxy] 1.100.0, google-genai 2.22.0)
- Log: /Users/muralisid/github_other/part1-tools/env/litellm.log
- Pid file: /Users/muralisid/github_other/part1-tools/env/litellm.pid
- Per-request log: /Users/muralisid/github_other/part1-tools/env/requests.jsonl
- Models served: gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, gemini-3.5-flash, gemini-3.5-flash-lite (global); gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-embedding-001, text-embedding-005 (us-central1). Ten in all.

## Authentication (copied from the bench)

The bench file src/multicard/llm/vertex.py reads VERTEX_AI_SERVICE_ACCOUNT_JSON (the full service account JSON as a string), writes it to a private temp file, sets GOOGLE_APPLICATION_CREDENTIALS to that path, and builds genai.Client(vertexai=True, project=<project_id from the JSON, or VERTEX_AI_PROJECT>, location="us-central1"). VERTEX_AI_PROJECT is not set on this machine. The project id inside the JSON is scout7ai.

The proxy uses the same variable. config.yaml points vertex_credentials at os.environ/VERTEX_AI_SERVICE_ACCOUNT_JSON, so litellm reads the JSON content from the environment. No credential is written to any file under part1-tools.

## Start and stop

Start (needs VERTEX_AI_SERVICE_ACCOUNT_JSON in the environment; the login shell has it):

    /Users/muralisid/github_other/part1-tools/litellm/start.sh

Which runs:

    cd /Users/muralisid/github_other/part1-tools/litellm
    export LITELLM_MASTER_KEY="$(cat /Users/muralisid/github_other/part1-tools/env/.proxy_key)"
    export LITELLM_REQUEST_LOG=/Users/muralisid/github_other/part1-tools/env/requests.jsonl
    nohup .venv/bin/litellm --config config.yaml --host 127.0.0.1 --port 4000 > /Users/muralisid/github_other/part1-tools/env/litellm.log 2>&1 &
    echo $! > /Users/muralisid/github_other/part1-tools/env/litellm.pid

Stop:

    /Users/muralisid/github_other/part1-tools/litellm/stop.sh

Which runs:

    kill "$(cat /Users/muralisid/github_other/part1-tools/env/litellm.pid)"
    rm -f /Users/muralisid/github_other/part1-tools/env/litellm.pid

Health check:

    curl -s http://127.0.0.1:4000/health/liveliness

The proxy was started at 11:52 on 2026-09-06 and restarted with stop.sh then start.sh at 12:37 (pid 63553) to load the config changes below. It comes up in about 4 seconds.

## Request logging

config.yaml registers a custom callback, request_log.py (next to config.yaml). It appends one JSON line per request to requests.jsonl with: ts, status, model (litellm's name, for example vertex_ai/gemini-2.5-flash-lite), call_type, prompt_tokens, completion_tokens, total_tokens, response_cost (litellm's own USD estimate), latency_s, job_tag, tags, user, and error on failures. Rows written before 12:37 on 2026-09-06 have no job_tag, tags or user fields.

Job tags. The log is shared by every caller on this machine, so a client should label its requests. Two request body fields are read:

- "user": the OpenAI user string. The OpenAI python client sends it with the user= keyword on chat.completions.create and embeddings.create. Example: client.chat.completions.create(model=..., messages=..., user="pgr-build-shard-01").
- "metadata": {"tags": [...]}: litellm's tag list. The OpenAI python client sends it with extra_body={"metadata": {"tags": ["pgr-build-shard-01"]}}. Plain HTTP: put "metadata": {"tags": ["..."]} in the JSON body.

job_tag is the first metadata tag when one is sent, else the user string, else null. The row also keeps tags (the full list) and user (the raw string). Failure rows carry the tag too. Verified 12:38 on 2026-09-06 with five calls: user only, metadata.tags only, both, and an embedding with user; every row showed the expected job_tag. litellm does not forward the user field to Vertex (drop_params is on and Gemini has no such field).

To sum tokens and cost per model, or per job tag, or from a time onward:

    /Users/muralisid/github_other/part1-tools/litellm/.venv/bin/python /Users/muralisid/github_other/part1-tools/litellm/tokens_by_model.py
    /Users/muralisid/github_other/part1-tools/litellm/.venv/bin/python /Users/muralisid/github_other/part1-tools/litellm/tokens_by_model.py --by-tag
    /Users/muralisid/github_other/part1-tools/litellm/.venv/bin/python /Users/muralisid/github_other/part1-tools/litellm/tokens_by_model.py --by-tag --since 2026-09-06T12:37:00

## Thinking defaults

gemini-2.5-flash has reasoning_effort none in its litellm_params in config.yaml. litellm 1.100.0 maps that to thinkingConfig thinkingBudget 0, includeThoughts false for the 2.5 family. Verified 12:38 on 2026-09-06: "Reply with the single word: pong" with max_tokens 8 returned "pong", finish_reason stop, 7 in, 1 out, no reasoning tokens. Before the change the same call returned content null (proxy-verify.md issue 4). A request that sends its own reasoning_effort overrides the default: the same prompt with reasoning_effort low and max_tokens 64 returned "pong" with 17 reasoning tokens.

gemini-2.5-pro keeps no default and thinks unless the request sends reasoning_effort (none, low, medium, high, or minimal which litellm maps to 128 tokens for pro).

The 3.x models cannot have thinking turned off. reasoning_effort none on gemini-3.7-flash and gemini-3.8-flash came back HTTP 400 from Vertex: "Thinking level is unsupported: THINKING_LEVEL_MINIMAL" (litellm maps none and minimal to thinking level minimal for Gemini 3 and later, and low is the lowest level Vertex accepts). Callers must leave room for reasoning tokens. Measured for a one-word answer: gemini-3.7-flash used 59 reasoning tokens and returned "pong" at max_tokens 64 (finish_reason length); gemini-3.8-flash used all 64 and returned an empty choices list, then returned "pong" at max_tokens 256 with 96 reasoning tokens. Design section 13 sets a floor of 64 output tokens for every Part 1 call; for gemini-3.8-flash use at least 128.

The built-in litellm spend log needs a Postgres database and was not set up.

## Model probes (direct google-genai calls, max 5 output tokens each)

Each model was tried in us-central1 first, then global. Result is the first location that answered.

| Model | Result | Location |
|---|---|---|
| gemini-3.6-flash | works | global (404 in us-central1) |
| gemini-3.7-flash | works (added 2026-09-06 from proxy-verify.md; through the proxy: pong at 64 tokens) | global (404 in us-central1) |
| gemini-3.8-flash | works (added 2026-09-06 from proxy-verify.md; through the proxy: pong at 256 tokens, no text at 64) | global (404 in us-central1) |
| gemini-3.7-flash-lite | 404 in both (proxy-verify.md) | none |
| gemini-3.8-flash-lite | 404 in both (proxy-verify.md) | none |
| gemini-3.6-flash-lite | 404 in both | none |
| gemini-3.5-flash | works | global (404 in us-central1) |
| gemini-3.5-flash-lite | works | global (404 in us-central1) |
| gemini-3-flash | 404 in both | none |
| gemini-3-flash-lite | 404 in both | none |
| gemini-2.5-flash | works | us-central1 |
| gemini-2.5-flash-lite | works | us-central1 |
| gemini-2.5-pro | works | us-central1 |
| gemini-embedding-001 | works, 3072 dims | us-central1 |
| text-embedding-005 | works, 768 dims | us-central1 |

The 404 message for the failed models was: publisher model was not found or the caller does not have access.

JSON schema structured output (response_json_schema, a two-field object) returned valid JSON on all five working flash-class models: gemini-3.6-flash, gemini-3.5-flash, gemini-3.5-flash-lite, gemini-2.5-flash, gemini-2.5-flash-lite.

## Prices

Source: https://cloud.google.com/vertex-ai/generative-ai/pricing, fetched 2026-09-06 with curl. The page title now reads "Agent Platform Pricing". The page shows no last-updated date. HTTP Last-Modified header: not sent by server. The page says introductory pricing for Gemini 3.6 Flash runs through December 31, 2026, and standard pricing applies from January 1, 2027.

USD per 1 million tokens, standard tier, text in and text out, as printed:

| Model | Input | Output | Note |
|---|---|---|---|
| gemini-3.6-flash | 0.75 | 3.75 | Global endpoint. Introductory through 2026-12-31. From 2027-01-01: 1.50 and 7.50. Non-global: 0.825 and 4.125. |
| gemini-3.7-flash | 0.75 | 3.75 | Global endpoint. Introductory through 2026-12-31. From 2027-01-01: 1.50 and 7.50. Read during the proxy verification. |
| gemini-3.8-flash | 0.75 | 3.75 | Global endpoint. Introductory through 2026-12-31. From 2027-01-01: 1.50 and 7.50. Read during the proxy verification. |
| gemini-3.5-flash | 1.50 | 9.00 | Global endpoint. Non-global: 1.65 and 9.90. |
| gemini-3.5-flash-lite | 0.30 | 2.50 | Global endpoint. Non-global: 0.33 and 2.75. |
| gemini-2.5-flash | 0.30 | 2.50 | Audio input 1.00. |
| gemini-2.5-flash-lite | 0.10 | 0.40 | Audio input 0.30. |
| gemini-2.5-pro | 1.25 | 10.00 | Prompts up to 200K tokens. Above 200K: 2.50 and 15.00. |

Embeddings, as printed ("Price / 1,000 count (USD)", online requests, output no charge):

| Page row | Model id | Price |
|---|---|---|
| Gemini Embedding | gemini-embedding-001 | 0.00015 per 1,000 count (0.15 per 1M) |
| Embeddings for Text (Excluding Gemini Embedding) | text-embedding-005 | 0.000025 per 1,000 count (0.025 per 1M) |

The page prints "count" as the unit for embeddings. It does not say tokens or characters. litellm charged the 2-token gemini-embedding-001 test call 0.0000003 USD, which matches 0.15 per 1M tokens.

## Choices

- CHAT_MODEL = gemini-2.5-flash-lite. Cheapest working flash-class model (0.10 in, 0.40 out). JSON schema output works directly and through the proxy.
- CHANDAN_MODEL = gemini-3.6-flash. Works on the global endpoint only. 0.75 in, 3.75 out until 2026-12-31.
- Design section 13 fixes the calibration row as indexed with gemini-3.7-flash and answered with gemini-3.6-flash (chandan_index_model and chandan_answer_model in models.json). The main chandan rows use CHAT_MODEL.
- EMBED_MODEL = gemini-embedding-001. 3072 dimensions.
- The bench's own default in vertex.py is gemini-2.5-flash (0.30 in, 2.50 out). It also works and is in the proxy.

## Proxy verification (plain HTTP, curl)

All with "Authorization: Bearer <key from .proxy_key>".

- GET /v1/models lists all eight models under their Vertex ids. After the 12:37 restart it lists ten (gemini-3.7-flash and gemini-3.8-flash added).
- (a) POST /v1/chat/completions, model gemini-2.5-flash-lite, "Reply with the single word: pong": returned "pong", finish_reason stop, 7 prompt tokens, 1 completion token. Same on gemini-3.5-flash-lite.
- (b) Same endpoint with response_format type json_schema (object with city string and population integer, strict): returned {"city": "Tokyo", "population": 13960000}, 10 prompt tokens, 25 completion tokens. Same shape on gemini-3.5-flash-lite.
- (c) POST /v1/embeddings, model gemini-embedding-001, input "hello world": vector of 3072 floats, 2 prompt tokens.

Example:

    KEY="$(cat /Users/muralisid/github_other/part1-tools/env/.proxy_key)"
    curl -s http://127.0.0.1:4000/v1/chat/completions \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d '{"model":"gemini-2.5-flash-lite","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16}'

## Azure gpt-5.4 through the bench client

From the bench directory, "uv run python -c ..." built multicard.llm.azure.AzureClient(cache=False) and called generate("Reply with the single word: pong", max_output_tokens=16). It returned "pong", 13 input tokens, 5 output tokens. The client reads AZURE_GPT54_ENDPOINT (a Responses API URL on services.ai.azure.com) and AZURE_GPT54_API_KEY, not the AZURE_OPENAI_* variables. The deployment name is gpt-5.4. MCB_LLM_CACHE was pointed at a scratch file so the bench's cache was not touched.

## Spend

Direct probes: 9 chat calls at 2 input and up to 5 output tokens, 5 JSON schema calls at 12 to 77 input and 25 output tokens, 2 embedding calls. Proxy checks: 4 chat calls, 1 embedding call (litellm's own cost total for those: 0.0001 USD by the five rows in requests.jsonl; an earlier draft of this file said 0.0002). One Azure call of 13 in and 5 out. Total estimated under 0.002 USD.

Infra stage 2026-09-06 (tag infra-smoke in requests.jsonl): 7 chat calls and 1 embedding call, two of them 400 failures, 0.0009 USD by litellm's count.

## Files

- /Users/muralisid/github_other/part1-tools/litellm/config.yaml
- /Users/muralisid/github_other/part1-tools/litellm/request_log.py
- /Users/muralisid/github_other/part1-tools/litellm/tokens_by_model.py
- /Users/muralisid/github_other/part1-tools/litellm/start.sh
- /Users/muralisid/github_other/part1-tools/litellm/stop.sh
- /Users/muralisid/github_other/part1-tools/env/.proxy_key
- /Users/muralisid/github_other/part1-tools/env/litellm.log
- /Users/muralisid/github_other/part1-tools/env/litellm.pid
- /Users/muralisid/github_other/part1-tools/env/litellm-install.log
- /Users/muralisid/github_other/part1-tools/env/requests.jsonl
- /Users/muralisid/github_other/part1-tools/env/models.json
- /Users/muralisid/github_other/part1-tools/env/proxy.md
- /Users/muralisid/github_other/part1-tools/env/proxy-verify.md (second-agent check of the survey)
- /Users/muralisid/github_other/part1-tools/env/infra.md (what changed on 2026-09-06 after the survey)
- /Users/muralisid/github_other/part1-tools/env/litellm.log.1 (the log before the 12:37 restart)

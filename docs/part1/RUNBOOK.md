# Part 1 runbook

The exact commands, in order, for the full runs. Every rule comes from
docs/PART1-DESIGN.md (version 4). Every command runs from the bench root
unless it says otherwise:

    cd /Users/muralisid/github_other/multicard-bench

The Part 1 stages are registry entries of `uv run mcb run`. They take
`--corpus`, `--limit`, `--arms`, `--budget`, `--tag`, `--max-usd`,
`--sample`, `--readers`, `--workers`, `--population`. Question order is
ORDER of docs/part1/subsets.json (sha256 in docs/part1/subsets.sha256).
Outputs go under results/part1/ (results/part1_<tag>/ with a tag); the
relation vectors go under data/cache/part1/.

Wall times below were measured on the 5-question smoke (tag smoke, 243
sessions, 55 articles) on the 8-core Apple Silicon machine and scaled; the
scaled numbers are estimates, not measurements. Caps are the design's
section 11 figures split by stage.

## 0. Services

The proxy (litellm on 127.0.0.1:4000) and Neo4j must be up before the
competitor runners start. The bench stages need neither: Reader A, the
judges, the planner, the S2 tests and the overlay go to Vertex through the
bench GenerativeClient, and Reader B goes to Azure.

    # proxy (needs VERTEX_AI_SERVICE_ACCOUNT_JSON in the environment)
    /Users/muralisid/github_other/part1-tools/litellm/start.sh
    curl -s http://127.0.0.1:4000/health/liveliness
    # Neo4j, for the Graphiti runner only
    /Users/muralisid/github_other/part1-tools/neo4j/start.sh
    # Postgres, for the post-graph-rag runner (see docs/part1/env/postgres.md)

Stop them with the matching stop.sh scripts when the runners are done. No
other index build runs while Graphiti ingests (design section 5).

## 1. post-graph-rag: study-model build, both corpora

Runs from part1-tools/pgr with its own venv. One space per question on
LongMemEval in ORDER, shards of about 2,000 sessions cut on question
boundaries, projection after shard 1 against the cap. Resume: a question
whose `<out-root>/<qid>/meta.json` exists is skipped, so a stopped run is
relaunched with the same command.

    cd /Users/muralisid/github_other/part1-tools/pgr
    .venv/bin/python run_spaces.py --corpus lme \
        --ids /Users/muralisid/github_other/multicard-bench/docs/part1/subsets.json --ids-key ORDER \
        --shard-sessions 2000 --cap-usd 130 --verify \
        --job-tag pgr-lme-full
    # cap USD 130 (section 11); about 26 hours at 3.8 s per session over 25,112 sessions,
    # about USD 64 projected from the smoke (USD 0.00255 per session).
    # If shard 1 projects over the cap the build continues on GRAPHITI_150 only:
    #   --subset GRAPHITI_150   (then part1_report reads T1 on that subset, labelled)

    .venv/bin/python run_spaces.py --corpus mhrag \
        --ids /Users/muralisid/github_other/multicard-bench/docs/part1/subsets.json --ids-key ORDER \
        --cap-usd 10 --job-tag pgr-mhrag-full
    # cap USD 10; one space for the 609 articles, every non-null query in ORDER (2,255).
    # Resume: indexing is skipped when corpus/index.json matches the row count; a query with
    # its four variant files is skipped.

The calibration row (section 13), his models on CHANDAN_CAL_18, into a
separate root so the study spaces are not overwritten:

    .venv/bin/python run_spaces.py --corpus lme --calibration \
        --ids /Users/muralisid/github_other/multicard-bench/docs/part1/subsets.json --ids-key ORDER \
        --subset CHANDAN_CAL_18 --cap-usd 70 \
        --out-root /Users/muralisid/github_other/multicard-bench/data/part1/pgr_cal/lme \
        --job-tag pgr-lme-cal
    # cap USD 70; 18 questions; part1_retrieve reads data/part1/pgr_cal/lme/<qid>/ as the arm
    # chandan_live_cal (reported, never in a test).

Cross-check the metered spend against the proxy log by tag when a run ends:

    /Users/muralisid/github_other/part1-tools/litellm/.venv/bin/python \
        /Users/muralisid/github_other/part1-tools/litellm/tokens_by_model.py --by-tag \
        /Users/muralisid/github_other/part1-tools/env/requests.jsonl

## 2. Graphiti: pilot, then the run on GRAPHITI_150

Runs from part1-tools/graphiti with its own venv, Neo4j up. The pilot
ingests the first 20 sessions of the pilot question three ways and writes
the choice into docs/part1/env/graphiti.md; the run uses the chosen variant.
Resume: a question whose meta.json exists is skipped; a leftover group with
that id is deleted before ingestion.

    cd /Users/muralisid/github_other/part1-tools/graphiti
    .venv/bin/python run_groups.py pilot --qid gpt4_2f584639 --n 20 --max-usd 60 \
        --out /Users/muralisid/github_other/multicard-bench/data/part1/graphiti_pilot \
        --job-tag graphiti-pilot
    # gpt4_2f584639 is derived.GRAPHITI_PILOT in subsets.json. About 10 minutes, under USD 1.
    # Read the table and the chosen variant in docs/part1/env/graphiti.md before the run.

    .venv/bin/python run_groups.py run --variant <chosen variant> \
        --subsets /Users/muralisid/github_other/multicard-bench/docs/part1/subsets.json \
        --max-usd 60 \
        --out /Users/muralisid/github_other/multicard-bench/data/part1/graphiti \
        --job-tag graphiti-run
    # cap USD 60; 150 questions, one group at a time in ORDER (design section 2); the pilot
    # table in docs/part1/env/graphiti.md gives the cost and the seconds per session.
    # A run stopped by the cap leaves run_summary.json with partial true; part1_report then
    # records Graphiti as partial and T2 as not run.

If no variant fits the cap the pilot says Graphiti is dropped; skip this
step and part1_report records T2 as not run (graphiti_status dropped).

## 3. part1_index, per corpus

Section 4. Units and vectors through the Encoder cache (the full
LongMemEval encoding is already cached from e5: the sorted session list is
the same), the topic fit, the graph build, the relation vectors from every
post-graph-rag space present, the bridge and the overlay proposals. Every
step is skipped when its output exists, so a stopped run is relaunched with
the same command and picks up at the first missing step. Run it after the
post-graph-rag build so the relation vectors are written in one pass; a
space that appears later is encoded by part1_retrieve through the Encoder
cache instead.

    uv run mcb run part1_index --corpus lme --max-usd 15 --n-process 4
    # topics: BERTopic over 200,000 turns, tens of minutes single threaded (random_state);
    # graph: spaCy over 998,042 sub-units, about 70 minutes at one process, less at four;
    # overlay: at most 5,000 calls, about USD 0.5 (the smoke: 65 calls, USD 0.007);
    # cap USD 15 (section 11, overlay generation).

    uv run mcb run part1_index --corpus mhrag --max-usd 15 --n-process 4
    # 609 articles: a few minutes; overlay at most 500 calls.

Outputs: results/part1/<corpus>/index/{topics,graph,overlay}/ and
index.json; data/cache/part1/<corpus>/relvecs/<space_id>.npz.

## 4. part1_retrieve, per corpus

Section 5. Every arm of ours plus chandan_live (both runs, shipped and
raised), chandan_full, chandan_live_cal and graphiti where their exports
exist, rendered at 4,000 and 8,000 tokens and scored. Arms whose inputs are
absent are listed in metrics.json arms_skipped and the report marks them
absent. Not resumable inside a corpus: the stage rewrites its files, but the
planner and S2 replies are cached by prompt, so a rerun costs nothing. The
two corpora are independent and can run at the same time in two shells.

    uv run mcb run part1_retrieve --corpus lme --max-usd 35
    # 500 questions at about 15 s each (the smoke: 75 s for 5): about 2 hours;
    # model calls: one planner call per question (cap USD 10) and at most 20 S2 tests
    # per question (cap USD 25); the smoke spent USD 0.0009 on 5 questions.

    uv run mcb run part1_retrieve --corpus mhrag --max-usd 35
    # 2,556 queries at about 10 s each (the smoke: 51 s for 5): about 7 hours, most of it
    # the sequential S2 tests; about 41,000 S2 calls, about USD 4.

Outputs: results/part1/<corpus>/retrieve/{contexts_4000,contexts_8000,
scores_4000,scores_8000}.jsonl, rankings.json, candidates.json,
per_query.csv, planner.json, metrics.json.

## 5. part1_qa, both corpora in one call

Section 6. Reader A on every arm at 4,000 and 8,000 (the default), Reader B at 4,000 on its seven arms on all 500
LongMemEval questions and READER_B_MHRAG, his own reader's answers as
chandan_full_uncut, then the judge audit on the pooled sample, the primary
judge on everything, the second judge on the wrong answers of the
head-to-head arms. Run with `--corpus all` so the audit is pooled across
both corpora as section 6 says. Resume: answers.jsonl is read back and a
(question, arm, reader, budget) already answered is not asked again; the
verdicts on it are kept. A cap stop withdraws Reader B in the section 5
order (closed_book, oracle_full, chandan_full, ours_cheap, graphiti; the
head-to-head pair last) and records the withdrawn arms as partial.

    uv run mcb run part1_qa --corpus all --readers reader_a,reader_b --workers 4 --max-usd 100
    # cap USD 100 (section 11, answering and judging across both corpora and both readers);
    # Reader A: about 17 arms x 500 LongMemEval questions plus about 14 arms x 600 MHRAG_ANSWER
    # queries, about 17,000 calls; Reader B: 7 arms x 500 plus 7 x 200; judges on every record.
    # About 3 to 4 hours at 4 workers.

Outputs: results/part1/<corpus>/qa/answers.jsonl and metrics.json,
results/part1/qa_audit.json.

## 6. part1_report

Sections 7 to 9. Reads every file the earlier stages wrote, recomputes the
pooled audit from the stored verdicts, builds the failure buckets, runs the
tests and the pass rule, writes metrics.json, buckets.jsonl and REPORT.md.
No model call; a few minutes. Rerun any time.

    uv run mcb run part1_report
    # add --graphiti-status dropped|partial|run to override what the exports say

Outputs: results/part1/metrics.json, results/part1/buckets.jsonl,
results/part1/REPORT.md.

## The smoke (what was run before this runbook)

The same stages with `--sample --tag smoke` on the first 5 answerable
LongMemEval questions and the first 5 non-null MultiHop-RAG queries in
ORDER, the index restricted to their sessions (243) and to their evidence
documents plus the first 50 articles (55), Reader A only, one cap of USD
1.00 split across the stages:

    uv run mcb run part1_index --corpus lme --sample --tag smoke --max-usd 0.15
    uv run mcb run part1_index --corpus mhrag --sample --tag smoke --max-usd 0.15
    uv run mcb run part1_retrieve --corpus lme --sample --tag smoke --max-usd 0.15
    uv run mcb run part1_retrieve --corpus mhrag --sample --tag smoke --max-usd 0.15
    uv run mcb run part1_qa --corpus all --sample --tag smoke --readers reader_a --max-usd 0.70
    uv run mcb run part1_report --tag smoke

Outputs under results/part1_smoke/. The report of a limited run says so in
its head and restricts every population to the processed questions.

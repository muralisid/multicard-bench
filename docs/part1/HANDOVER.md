# TG-VGRAG Part 1: everything done so far

Written 2026-09-07, updated the same day when the gate tests, the mechanism
analysis, the published comparison and the cost result landed. Every number
here is read from a file the experiment wrote. Nothing here is published
anywhere.

The four companion documents, each with the detail behind a section below:
docs/PART1-DESIGN.md (every rule, and section 14 for every change made after a
number was read), docs/part1/COST-AND-RESULT.md, docs/part1/MECHANISM.md and
docs/part1/PUBLISHED-COMPARISON.md.

## 0. Where everything is

Two private git repositories, plus a tools directory that is deliberately
outside both. Nothing is pushed anywhere and nothing is public.

### The bench: /Users/muralisid/github_other/multicard-bench

The clean-room experiment repository. Python 3.11 with uv; run anything with
`cd /Users/muralisid/github_other/multicard-bench && uv run ...`. Its CLAUDE.md
holds the binding rules (clean room, git identity, no em dashes, boundary grep
before any push, cost caps). Part 1 commits, most recent first:

| commit | what |
|---|---|
| b69041b | this handover |
| b79da45 | design section 14 item 4, MultiHop-RAG cap amendment |
| 9e454cc | report hygiene from the independent recomputation |
| a8e922b | design amendment 3, the pooled audit figure |
| d8239df | the cross-family judge column |
| 17ceed9 | design section 14 item 3 |
| 1b5614a | retry on transient network and service errors |
| 066d194 | fused arms fill the budget; design section 14 items 1 and 2 |
| 09925ad | the Part 1 build, integration and review fixes |
| b8c1e20 | design v4, environment records, MultiHop-RAG loader, graph deps |
| d430230 | the earlier e5 LongMemEval experiment this builds on |

**Documents.**

- `docs/PART1-DESIGN.md`, version 4 plus section 14. Every rule, every arm,
  every test, every prediction, and every change made after a number was read.
  Read this first; the code implements it and cites its section numbers.
- `docs/part1/RUNBOOK.md`, the exact commands in order, with wall times, caps
  and resume behaviour for every stage.
- `docs/part1/REVIEW-FIXES.md`, what the five post-build reviewers found and
  what was done about each finding.
- `docs/part1/HANDOVER.md`, this file.
- `docs/part1/env/`, the environment records: `postgres.md`, `proxy.md`,
  `models.json` (the model survey and prices), `pgr.md` (post-graph-rag's
  configuration, the written decisions, and every observation about his
  package), `pgr-schema.md` (his Postgres schema), `graphiti.md` (including
  the pilot table), `bench.md`, `infra.md`.
- `docs/part1/prompts/planner.txt`, the planner prompt, byte-identical to the
  copy the runner reads.
- `docs/part1/subsets.json` and `subsets.sha256`, every fixed subset and the
  seeded question order, drawn before any model call.

**Code.**

- `src/multicard/part1/`: `units.py` (unit tables, chunking, fact location),
  `render.py` (the rendering rule), `score.py` (coverage and joint recall),
  `topics.py` (BERTopic), `graph.py` (noun-phrase graph, communities),
  `overlay.py` (the topic-community bridge and its placebo), `retrieve.py`
  (channels, fusion, expansion, the arms), `planner.py` (the weight table, the
  rules, the oracle, the LLM planner), `competitors.py` (reading his and Zep's
  exports as units), `evaluate.py` (readers, judges, buckets, tests, the pass
  rule), `report.py` (REPORT.md from the metrics files only), `subsets.py`.
- `src/multicard/experiments/part1.py`, the four stages, and
  `src/multicard/run.py`, which registers them as `part1_index`,
  `part1_retrieve`, `part1_qa`, `part1_report`.
- `scripts/part1_subsets.py`, which produced the subsets file.
- `tests/test_part1_*.py`, eleven files; the whole suite is 322 tests.
- Reused from the earlier work: `src/multicard/experiments/e5_longmemeval.py`
  (the speaker rule, the reader rules, the official judge prompts, the unit
  split), `src/multicard/metrics/` (ranking, paired statistics),
  `src/multicard/index/` (encoder with its on-disk cache, BM25),
  `src/multicard/llm/` (Vertex and Azure clients, the cost meter),
  `src/multicard/data/` (the LongMemEval and MultiHop-RAG loaders).

**Results.** `results/part1/`:

- `REPORT.md` and `metrics.json`, the report and the file it is generated from.
- `qa_audit.json`, the judge audit and its history.
- `<corpus>/index/{topics,graph,overlay}/`, the frozen index artifacts as
  parquet, with a diagnostics file each.
- `<corpus>/retrieve/`: `scores_4000.jsonl` and `scores_8000.jsonl` (per
  question per arm, every metric), `rankings.json`, `candidates.json` (the
  fused top 100 with per-channel provenance), `contexts_4000.jsonl` and
  `contexts_8000.jsonl` (the exact text each reader saw), `planner.json`,
  `per_query.csv`, `metrics.json`, `cost_ledger.jsonl`.
- `<corpus>/qa/answers.jsonl` and `metrics.json`, every answer with both
  judges' verdicts.
- `buckets.jsonl`, the failure bucket of every wrong answer.
- `logs/`, the stdout of every stage run, and the chain scripts that ran them.
- `results/part1_smoke/`, the 5 plus 5 question smoke, labelled and not a
  study result.

All of that is committed except three regenerable bulk files per corpus, which
are gitignored because they run to hundreds of megabytes: `contexts_*.jsonl`
(the exact text each reader saw), `candidates.json` (the fused top 100 with
per-channel provenance) and `rankings.json`. `part1_retrieve` rebuilds them
from the committed index artifacts, and every model reply it needs is cached,
so the rebuild costs CPU time and no money. The committed 177 MB includes the
frozen index artifacts (topics, the noun-phrase graph, both community variants,
the overlay proposals and link sets), so the index does not have to be refitted
either.

`results/e5_longmemeval*/` holds the earlier experiment this work builds on.

**Data**, all gitignored: `data/raw/longmemeval_s.json` (265 MB),
`data/raw/multihoprag/` (20 MB), `data/part1/pgr/` (4.2 GB, his exported
tables, one directory per question for LongMemEval), `data/part1/pgr_cal/`
(85 MB, the calibration row), `data/part1/graphiti_pilot/`,
`data/cache/part1/` (255 MB, relation vectors), `data/cache/embeddings/`
(3.1 GB, the encoder cache shared with the earlier work).

### The programme record: /Users/muralisid/github_other/multicard-retrieval-research

Private, never pushed. `program/knowledge/pov-04-topic-guided-vector-graph.md`
holds Murali's position in his own words, the adopted proposal verbatim, the
four review comments verbatim, and the execution plan. `program/worklog.md` is
the append-only diary of every session, including every number as it landed.
`program/hypothesis-register.md` carries rows H-TG-1 to H-TG-5.
`program/review-queue.md` holds the decisions he was asked for and gave.

### The competitor runners: committed at multicard-bench/tools/part1/

The scripts that drive post-graph-rag, Graphiti, the Vertex proxy and Neo4j are
committed in the bench at `tools/part1/`, with a README explaining each one:
`pgr/run_spaces.py` and its smokes and question lists, `graphiti/run_groups.py`
and its smoke, `litellm/` (the proxy config, its request-logging callback, the
spend-by-tag script, start and stop), `neo4j/` (start and stop),
`prompts/planner.txt`, and the raw dump of his Postgres schema. They hold no
secret; the configs read credentials from the environment.

They do not run inside the bench's virtual environment, because post-graph-rag
and graphiti-core bring their own dependency trees.

### The live tools directory: /Users/muralisid/github_other/part1-tools

Outside every repository and not committed. About 34 GB, of which 33 GB is the
Postgres cluster holding his extracted graph and 1 GB is the three virtual
environments. It holds the working copies of the scripts above plus:

- `pgr/.venv`, `graphiti/.venv`, `litellm/.venv`, the three environments.
- `pgdata/`, the Postgres cluster (33 GB).
- `post-graph-rag-src/`, his repository cloned at tag v1.12.0, read-only
  reference for his harness, prompts and configuration.
- `env/`, the live environment records (copies are committed at
  `docs/part1/env/`), the request log `requests.jsonl` with every model call by
  job tag, the pid files, and `.proxy_key`, which is never committed.
- `pgr/out/logs/` and `graphiti/out/logs/`, the runner logs and pid files.

Nothing here is needed to read the work. It is needed to rerun the competitor
arms, and the scripts that do that are committed.

### Services and how to bring them up

    /Users/muralisid/github_other/part1-tools/litellm/start.sh
    LC_ALL=C /opt/homebrew/opt/postgresql@17/bin/pg_ctl \
        -D /Users/muralisid/github_other/part1-tools/pgdata \
        -l /Users/muralisid/github_other/part1-tools/env/postgres.log -w start
    /Users/muralisid/github_other/part1-tools/neo4j/start.sh   # Graphiti only

Postgres is on port 5433, the proxy on 127.0.0.1:4000, Neo4j on bolt 7687.
Stop scripts sit beside each start script. Everything runs as the user; no
sudo anywhere.

### How to reproduce a stage

    cd /Users/muralisid/github_other/multicard-bench
    uv run mcb run part1_index    --corpus lme   --max-usd 15 --n-process 4
    uv run mcb run part1_retrieve --corpus lme   --max-usd 35 --workers 4
    uv run mcb run part1_qa       --corpus all   --readers reader_a,reader_b --max-usd 40
    uv run mcb run part1_report   --graphiti-status dropped

Every stage skips work whose output exists, and every model reply is cached, so
a rerun is cheap. The full commands for the competitor runners, with their caps
and resume behaviour, are in `docs/part1/RUNBOOK.md`.

## 1. What was being tested and why

Murali's position, in his words: he is not keen on graph traversal the way
traditional graph systems work, but he is strongly in favour of using a graph
to organise information; pushing everything into a graph up front is too
expensive, so the embedding space should carry that load, which is what
LazyGraphRAG did and was his intuition from the start; topic modelling is
valuable bottom up but needs a better top down, which LazyGraphRAG supplied.

The research proposal he adopted (drafted with another agent) is Topic-Guided
Vector-Graph RAG, TG-VGRAG: a dual semantic and relational index in which
BERTopic discovers topics, a LazyGraphRAG-style noun-phrase graph discovers
communities, an LLM reconciles the two, post-graph-rag-style controls protect
relationship quality, and an LLM query planner searches across chunks, topics,
communities, explicit relationships and relationship vectors.

He then split the work: "First part is to prove that we can get better results
than chandan and zep on the same dataset. Then we can move further. Otherwise
there is no point." Later he added: finish TG-VGRAG first, let running jobs
finish, then pause testing others.

His standing principle for the work: "use the latest most inexpensive model to
prove that there is not much to invent in the places where industry has already
solved, or simply take the latest success from the industry and inherit. No
point reinventing the wheel and be open about it."

## 2. The design and how it was hardened

The design is docs/PART1-DESIGN.md in the bench (multicard-bench), version 4
plus a section 14 that records every change made after a number was read.

It went through three rounds of adversarial review before any code:

- v1, agent draft. Reviewed by a three-lens panel (statistics, fairness to the
  compared systems, implementability): 33 findings, 20 kept, 9 of them
  blockers. The blockers: post-graph-rag must be indexed per question the way
  his own harness does it (his paper indexes every instance from scratch), the
  gate must not score our re-rendering of his output, the Holm families must be
  split so the gate test is not buried under 13 tests, the non-inferiority rule
  as written could not be met at true equality, the Graphiti recipe and subset
  were unspecified, the overlay had no query-time rule and its R2 and R3 arms
  were identical by construction, the planner table was not in the file, the
  index build did not fit 16 GB, and three cost caps were wrong.
- v2 applied those. A four-checker pass then found the planner prompt text was
  still missing, S(t) and S(c) were undefined, community membership was
  undefined, and several outcomes were unlabelled (a negative T1, a T5 fourth
  branch, a dropped Graphiti arm).
- v3 fixed those. v4 folded in what the environment smokes measured.

Two decisions the review left to the owner were taken by the agent under stated
assumptions and flagged in the file: D1 (a positive but non-significant T1 is
"not shown", not "failed"; zero or negative is "failed") and D2 (T2 passes on a
positive point estimate, with the CI printed beside it).

## 3. The environment, all pinned

Everything runs locally in user space on one Apple Silicon Mac, 8 cores, 16 GB.
Records are in docs/part1/env/ (postgres.md, proxy.md, models.json, pgr.md,
pgr-schema.md, graphiti.md, bench.md, infra.md).

- PostgreSQL 17.11 with pgvector 0.8.6, port 5433, cluster in user space.
- LiteLLM 1.100.0 proxy on 127.0.0.1:4000 over Vertex AI, project scout7ai,
  with per-request logging by job tag.
- post-graph-rag 1.12.0 from PyPI (clone at tag v1.12.0, commit 69c2e0e).
- Neo4j 2026.07.1 with graphiti-core 0.30.1.
- The bench itself: Python 3.11, uv, BERTopic 0.17.4, spaCy 3.8.16 with
  en_core_web_sm, python-igraph 1.0.0, leidenalg 0.12.0, pyarrow, psycopg.

Model survey on the project, with prices read from the Vertex pricing page on
2026-09-06 (USD per million tokens, in/out): gemini-2.5-flash-lite 0.10/0.40
(cheapest working flash-class, and the study model), gemini-2.5-flash
0.30/2.50, gemini-3.6-flash, gemini-3.7-flash and gemini-3.8-flash all
0.75/3.75 on the global endpoint, gemini-2.5-pro 1.25/10.00,
gemini-embedding-001 0.15. Azure gpt-5.4 is the strong reader and judge.

Model policy, fixed in design section 13 after measurement: one study model,
gemini-2.5-flash-lite, for every model call including post-graph-rag's own
extraction, because his own models under his per-question protocol project to
well over USD 1,000 on this corpus. A calibration row on 18 questions runs his
exact result-file configuration (gemini-3.7-flash index, gemini-3.6-flash
answer) so the reader can see how much his retrieval changes with his own
extractor.

## 4. Corpora

- LongMemEval_S. 500 questions, 470 with marked evidence turns, 30 abstention.
  19,829 unique sessions filling 25,112 haystack slots. Each question searches
  only its own haystack. This is the split Zep report on and the one where
  retrieval does the work; post-graph-rag's published number is on the oracle
  split, where the evidence sessions are handed to the system.
- MultiHop-RAG. 609 news articles from late 2023, 2,556 queries of which 2,255
  are non-null, each with an evidence list of 2 to 4 documents and the exact
  supporting fact excerpt. All 6,084 facts locate exactly in the corpus text,
  so fact-level scoring is exact.
- Graphiti was to run on a 150-question subset of LongMemEval. It was dropped
  (section 7 below).

## 5. What was built

Code under src/multicard/part1/ in the bench, plus two runners outside it.
Committed; 322 tests pass.

- units.py, render.py, score.py: the unit tables (turns, sub-units, chunks,
  sentences), the rendering rule (rank order until the budget is full, then
  oldest first, dates and speakers prefixed, turns cut at 2,000 characters),
  and the scoring (a symmetric coverage rule so our cut turns and his chunks
  are credited alike, JointRecall at a budget, candidate-level and
  session-level joint recall, fact-level joint recall on news).
- topics.py: BERTopic over MiniLM vectors, UMAP seeded at 13, per-unit topic
  distribution as a softmax over topic embeddings, top 5 kept sparse.
- graph.py: spaCy noun chunks per sub-unit, at most 50 per unit; pronouns and
  the top 0.5 percent of phrases by frequency dropped; co-occurrence edges
  inside a sub-unit; weights and pruning copied from Microsoft graphrag's NLP
  indexing path (PMI weights, min node frequency 2, 40th-percentile edge cut,
  ego-node removal), with the file and line numbers cited in the code; Leiden
  at seed 13; a topic-weighted variant where each edge weight is multiplied by
  one plus the cosine of the two phrases' topic vectors.
- overlay.py: the topic-community bridge. Candidate pairs share at least 5
  sub-units; disagreement is the shared count times one minus Jaccard; the top
  20 percent are flagged; one cheap model call per flagged pair asks whether
  the topic and the community describe the same subject, with a confidence.
  Proposals are stored separately and never rewrite the topic model or the
  communities. Arms: R0 none, R2 all, R3 confidence at or above 0.7 (the
  primary), P0 a placebo with the same link count and degree distribution over
  shuffled targets.
- retrieve.py and planner.py: six channels (sub-unit dense, BM25, topic
  prototypes, community prototypes, relation vectors in two renderings, entity
  seeds), weighted RRF fusion, the overlay link rule applied before fusion,
  lazy expansion over the top depth containers, the e5 speaker rule, the S2
  LazyGraphRAG pattern with cheap relevance tests, and the planner in three
  forms (LLM with a fixed prompt, deterministic rules, and an oracle from the
  benchmark labels as an upper bound).
- evaluate.py and report.py: the readers and judges with the official
  LongMemEval prompts, the judge audit that decides the primary judge before
  any test, the five failure buckets, the pre-declared tests with Holm inside
  three families, and a report generated only from the metrics files.
- Two runners outside the bench: one that drives post-graph-rag exactly as his
  LongMemEval harness does (one space per question, one document per session in
  date order, his conversational extraction prompt, then his query call at his
  shipped top_k and at raised limits, then an export of his entities,
  relations, chunks with turn spans, aliases and relation embeddings to
  parquet), and one that drives Graphiti (three ingestion variants, its own
  hybrid search recipe, Zep's own context template).

After the build, five adversarial reviewers checked every module against the
design line by line: 41 findings, 34 applied, 4 kept as disclosed gaps, 2 left
as written decisions recorded in pgr.md.

## 6. Results: LongMemEval_S

Retrieval, JointRecall at a fixed rendered-token budget, 470 answerable
questions. "cand" is candidate-level joint recall, before the budget cut.

| arm | JR@4k | cand | session JR@4k | turn R@10 | JR@8k |
|---|---|---|---|---|---|
| S4_static (all channels, equal weights, no model call) | 0.957 | 0.970 | 0.983 | 0.884 | 0.970 |
| S2_lazy (LazyGraphRAG pattern alone) | 0.955 | 0.983 | 0.985 | 0.888 | 0.970 |
| S5_planner_rules | 0.955 | 0.974 | 0.983 | 0.871 | 0.966 |
| S5_planner_oracle (upper bound) | 0.955 | 0.968 | 0.983 | 0.865 | 0.968 |
| S5_overlay_R0 (no overlay) | 0.949 | 0.970 | 0.979 | 0.863 | 0.966 |
| S5_overlay_R2 (every proposal) | 0.949 | 0.966 | 0.974 | 0.867 | 0.966 |
| S5_primary (LLM planner, overlay R3) | 0.947 | 0.962 | 0.977 | 0.869 | 0.964 |
| S5_noPGR (our layers only, none of his tables) | 0.945 | 0.953 | 0.983 | 0.858 | 0.964 |
| ours_cheap (the floor: RRF of BM25 and dense turns, speaker rule) | 0.938 | 0.949 | 0.987 | 0.863 | 0.957 |
| S5_overlay_P0 (placebo links) | 0.938 | 0.957 | 0.972 | 0.846 | 0.962 |
| ours_sentence_norule | 0.823 | 0.966 | 0.945 | 0.848 | 0.879 |
| ours_cheap_norule | 0.800 | 0.962 | 0.934 | 0.825 | 0.868 |
| S5_primary_norule | 0.766 | 0.974 | 0.904 | 0.760 | 0.870 |
| chandan_live (post-graph-rag, study model) | 0.574 | 0.998 | 0.655 | 0.556 | 0.804 |
| chandan_live_cal (his own models, 18 questions) | 0.611 | 1.000 | 0.833 | 0.611 | 0.889 |

Points to carry over:

1. The best arm is the all-channel fusion with no model call at query time.
   Adding the LLM planner and the overlay costs a little (0.957 to 0.947).
2. The overlay is indistinguishable from its own placebo at the candidate level
   and worth about 0.011 at the rendered level (R3 0.947, P0 0.938, R0 0.949),
   which is inside the noise of this test. On its own evidence the overlay is
   not reconciling anything.
3. The speaker rule (quarantine assistant turns unless the question is about
   the assistant) is worth far more than every structural layer: 0.938 against
   0.800 without it. That is a two-line declared rule.
4. His candidate-level recall is 0.998, so his index does find the evidence.
   His rendered recall at 4,000 tokens is 0.574 because his retrieval unit is
   the whole session document, about 10,000 characters, so two of them fill the
   budget. At 8,000 tokens he reaches 0.804. This is a unit-size effect, and it
   is the single most important thing to understand about the comparison.
5. His own models help him: 0.611 against 0.574 at 4k on the calibration
   subset, and 0.889 against 0.804 at 8k. The calibration row is 18 questions,
   so it is a direction, not a measurement.

Answering, all 500 questions, primary judge gpt-5.4 with the official prompts.
Reader A is the cheap study model, Reader B is gpt-5.4.

| arm | Reader A @4k | Reader B @4k | cheap judge on the Reader B rows | judge agreement |
|---|---|---|---|---|
| oracle_full (evidence sessions in full) | 0.620 | 0.858 | 0.858 | 0.964 |
| ours_cheap | 0.558 | 0.844 | 0.834 | 0.954 |
| S5_noPGR | 0.550 | pending | | |
| closed_book (no context) | 0.078 | 0.074 | 0.486 | 0.572 |

The full-arm answering pass is still running; the numbers above are from the
first pass. Two readings hold already. First, the cheap reader is the ceiling,
not retrieval: every arm sits within three points of the others under it, and
the oracle context is only six points better. Second, under a strong reader the
cheap retrieval reaches 0.844 on the S split against an evidence ceiling of
0.858, so retrieval is 1.4 points from perfect on this corpus. For context,
Zep's published number on the same split is 0.712 with gpt-4o, and our earlier
e5 experiment reached 0.690 with a flash reader.

By type under the strong reader, ours_cheap against the oracle ceiling:
knowledge-update 0.923 against 0.962, single-session-user 0.943 against 0.957,
temporal 0.797 against 0.842, preference 0.800 against 0.933, multi-session
0.759 against 0.692. Multi-session is the one type where retrieval beats the
full-evidence context, because the oracle context is long and the reader loses
facts inside it.

The judge audit decided gpt-5.4 as primary: pooled agreement with the cheap
judge was 0.848 on the first 250 records and 0.800 on 450, then 0.883 on 1,287
across both corpora, all under the 0.90 threshold the design fixed. Because the
primary judge is then the same model as Reader B, the cheap judge scored every
Reader B record as a cross-family check; it moves those rows by at most one
point, so the strong-reader result is not a same-model artefact.

## 7. Results: MultiHop-RAG

Retrieval, fact-level JointRecall at 4,000 tokens, 2,255 non-null queries, all
facts located. This is the negative result of Part 1.

| arm | fact JR@4k | doc JR@4k | cand | fact JR@8k |
|---|---|---|---|---|
| ours_cheap (RRF of BM25 and dense chunks) | 0.255 | 0.259 | 0.874 | 0.420 |
| S5_planner_rules | 0.205 | 0.209 | 0.841 | 0.345 |
| S2_lazy | 0.205 | 0.210 | 0.966 | 0.369 |
| S4_static | 0.185 | 0.188 | 0.897 | 0.361 |
| S5_overlay_R0 | 0.153 | 0.156 | 0.872 | 0.298 |
| S5_planner_oracle | 0.151 | 0.154 | 0.852 | 0.306 |
| S5_overlay_R2 | 0.145 | 0.148 | 0.784 | 0.290 |
| S5_primary | 0.136 | 0.140 | 0.844 | 0.283 |
| S5_overlay_P0 | 0.135 | 0.138 | 0.745 | 0.271 |

The one test computable so far, T8b: S5_primary against ours_cheap on
fact-level JointRecall@4k, 2,255 queries, delta minus 0.119, 95 percent CI
[-0.137, -0.101], p 0.0001, 111 wins to 380 losses, significant after Holm.
Labelled "run without post-graph-rag tables", because his MultiHop-RAG index
was not built at the time.

Two things are visible in the numbers already. Candidate-level recall stays
high (0.84 to 0.97), so the evidence is being found; it is the rendered budget
that loses it. And the ordering tracks expansion depth: the rules planner,
which labels most questions local and expands one document, is the best of the
added arms, while the oracle planner, which expands three documents for
multi-hop questions, is worse. A multi-hop question needs one chunk from each
of two to four different documents, and the lazy expansion spends the budget on
more chunks of documents already found.

Answering on MultiHop-RAG, 600 queries, gpt-5.4 judge, 4,000 tokens:

| arm | Reader A | Reader B (200 queries) |
|---|---|---|
| oracle_full | 0.787 | 0.895 |
| ours_cheap | 0.787 | 0.845 |
| S5_planner_rules | 0.797 | |
| S2_lazy | 0.780 | |
| S4_static | 0.772 | |
| S5_primary | 0.772 | 0.830 |
| closed_book | 0.700 | 0.455 |

The closed-book floor is 0.700 under the cheap reader, so most of this corpus
is answerable from the model's own memory of late-2023 news, and the whole
retrieval effect is worth about nine points there. Under the strong reader the
floor drops to 0.455 and retrieval is worth 39 points. This is why the design
required a closed-book arm.

## 8. Graphiti, dropped

The 20-session pilot on the design's pilot question measured all three
ingestion variants: one text episode per session, one message episode per
session, and one episode per turn (Zep's own granularity). Projected cost over
the 150-question subset: USD 71, 69 and 174; projected wall time one group at a
time: 74, 63 and 336 hours. The cap was USD 60, so by the design's own rule the
arm is dropped and T2 is recorded as not run. The cost driver is the library
default of putting the last ten episodes of the group into every extraction
prompt: 1.3 million input tokens for 18 sessions. Evidence presence could not
be measured because neither evidence session of the pilot question was among
the first 20 by date, which is a weakness of the pilot design and is recorded.

## 9. Cost and time

Proxy-metered, whole programme to date: USD 101. The breakdown: his LongMemEval
index USD 72.9 across four shards (13.3 hours, 500 spaces, 15 questions with a
refused document), his calibration row USD 13.3, his MultiHop-RAG index USD
12.8 for the first attempt plus USD 1.1 for the restart so far, the Graphiti
pilot USD 0.8, smokes USD 0.3. On top of that the bench-side spend (readers,
judges, planner, overlay, relevance tests) is about USD 25. Against a USD 500
cap.

Our own index costs almost nothing by comparison: the LongMemEval index took
112 minutes of CPU (topics 10 minutes, the spaCy graph 65 minutes, relation
vectors 23, overlay 11) and USD 0.28 of model calls. The MultiHop-RAG index
took 21 minutes and USD 0.03.

Index shapes, for reference: LongMemEval 199,641 turns, 998,042 sub-units,
3,421 topics with a 38 percent outlier share, 1,227,422 noun phrases, 1,670
plain communities and 1,685 topic-weighted, largest community share 11 percent;
12,830 candidate topic-community pairs, 2,566 flagged, 570 links at confidence
0.7 or above. MultiHop-RAG: 3,376 chunks, 57,073 sub-units, 95 topics, 105,540
phrases, 251 and 256 communities, 291 flagged pairs, 84 links.

## 10. What is finished and what is not

Finished: the design and its three review rounds, the environment, the code
with 322 passing tests, the frozen index on both corpora, every arm of ours
through retrieval and answering on both corpora, post-graph-rag on
LongMemEval through retrieval and answering, the calibration row on his own
models, the report with the gate tests, the mechanism analysis, the published
comparison and the cost result.

Not run, both by decision, and both recorded in design section 14: T2, the
Graphiti comparison, because no ingestion variant fitted the cost cap; and
T8a, post-graph-rag on MultiHop-RAG, which the owner stopped at 72 of 609
articles. Part 1 therefore does not pass its own rule, which requires T8a.
The report states that rather than working around it.

Not yet done: an independent recomputation of the final report. The interim
report was recomputed in full by three verifiers with their own code and every
number matched; the report has been regenerated since and that check should be
repeated before the work is quoted anywhere.

## 11. The gate result

T1, our system against post-graph-rag on LongMemEval, JointRecall at a
4,000-token budget, 469 questions after one refusal: 0.947 against 0.576,
delta plus 0.371, CI [+0.324, +0.420], 188 wins to 14 losses, significant
after Holm. Shown under D1.

On answers, same context budget, same reader, same judge, which is a
controlled comparison: under gpt-5.4, 0.832 against 0.624, plus 0.208, 121
wins to 17 losses, p below 0.0001. Under the cheap reader, 0.543 against
0.467, plus 0.076, p 0.0005.

T7, the non-inferiority control on the 120 local questions: two losses and no
wins, inside the limit of three, so the system does not regress on easy
lookups.

T8b, our full system against our own cheap floor on MultiHop-RAG: minus 0.119
fact-level joint recall, CI [-0.137, -0.101], 111 wins to 380 losses,
significant.

Family B, the mechanism tests, all flat: S5 against the cheap floor plus 0.009
(p 0.52), S5 against the static fusion minus 0.011, the overlay against no
overlay minus 0.002, the overlay against its placebo plus 0.009. T5 reads "no
measurable effect". Only the static fusion against the floor reaches plus
0.019, and it does not survive Holm.

## 12. The cost result, which is the headline

Index cost on LongMemEval_S, 500 questions, 61.2 million tokens of corpus:
post-graph-rag 144,281 model calls and USD 66.87 on the cheapest model
available, or USD 361 projected on his own published models; ours USD 0.28 for
every layer, and USD 0.00 once the overlay is dropped, which raises the score.
That is 238 times cheaper than his cheapest build and 1,283 times cheaper than
his published one, at USD 1.09 against USD 0.006 per million corpus tokens.
Our index is 112 minutes of laptop CPU; his was 13.3 hours across four
parallel shards against a hosted API.

The tightest single measurement of the thesis: his extracted tables are two of
the six channels inside our own system, and switching them off moves joint
recall by plus 0.002 on LongMemEval and by nothing at all on MultiHop-RAG,
where they return zero hits. Full detail in docs/part1/COST-AND-RESULT.md.

## 13. The measured mechanism

Four lenses over the per-query files, 57 claims each rechecked by a second
agent with its own code, 5 refuted and 47 corrected. The explanation offered
first, that lazy expansion spends the budget within documents already found,
is refuted: expansion renders zero units on MultiHop-RAG in every arm at both
budgets.

What survives: a 4,000-token budget renders about 50 units of a 55 to 62 unit
candidate list on chat memory, and about 9 of 100 on news. The machinery does
the same thing on both corpora, it reorders the top of the pool, and that
reorder is nearly free when almost everything is rendered and decisive when a
ninth of it is. The added channels move the last needed document from a median
rank of 12 to 17 against a window of nine, because under reciprocal rank
fusion a unit carrying a topic vote and a community vote outranks an evidence
chunk carrying only keyword and dense votes.

The MultiHop-RAG loss decomposes as topic and community minus 0.050, expansion
depth minus 0.020, planner weights minus 0.032, overlay minus 0.017. The one
planner row that sets the group channels to zero is the only row that beats
the cheap floor there. Full detail in docs/part1/MECHANISM.md.

## 14. Against published results

On the MultiHop-RAG paper's own retrieval metrics, computed before any budget
cut, our plain fusion of BM25 and a small local encoder reaches MRR@10 0.757,
MAP@10 0.574, Hits@10 0.992 against the paper's best published baseline
(voyage-02 with a bge-reranker-large cross-encoder) at 0.586, 0.480 and 0.747.
Our chunk is twice theirs, which flatters Hits, so the claim is narrow: a free
local fusion lands in the band of a commercial embedder with a reranker.

Our answering ceiling matches theirs independently: their GPT-4 on gold
evidence 0.89, our gpt-5.4 on the evidence documents 0.895.

Accuracy is not compared across papers, because the reader and judge move
these benchmarks more than any retrieval method does; post-graph-rag's own
repository reports 0.940, 0.858 and 0.782 for one run depending on the judge.
Full detail in docs/part1/PUBLISHED-COMPARISON.md.

## 15. The honest reading

1. On chat memory the layers help a little and the cheap parts do the work.
   Capacity and fusion earn their place. The speaker rule, two lines of
   declared logic, is worth seven times more than every structural layer put
   together.
2. On document multi-hop the layers hurt, significantly and in the same
   direction at both budgets. The mechanism is measured in section 13, and it
   is a ranking effect inside a small rendered window. The explanation this
   session first offered, that the expansion spends the budget on containers
   already found, was refuted by the verifiers: the expansion renders nothing
   at all on that corpus.
3. The overlay does not beat its placebo on either corpus. The LLM planner is
   behind the deterministic rules planner on both. The oracle planner, which is
   an upper bound on routing, is not better than the rules either. That is
   three separate pieces of evidence that the LLM-guided parts of the design
   are not paying for themselves as built.
4. Against post-graph-rag on the S split at a fixed budget, our arms are far
   ahead (0.957 against 0.574 at 4k, 0.970 against 0.804 at 8k), but the honest
   explanation is his retrieval unit, not his index quality: his candidate-level
   recall is 0.998. A system which finds the evidence can still fail to show it
   to the reader. The claim that does not depend on the budget at all is the
   cost one in section 12: his index costs USD 66.87 and 144,281 model calls,
   ours costs nothing, and inside our own system his tables are worth 0.002.
5. Under a strong reader the cheap retriever on LongMemEval is 1.4 points from
   the evidence ceiling. On that corpus there is almost nothing left for any
   index to win. The room that remains is in temporal and preference questions
   and in the reader itself.

## 16. What this suggests for the next design step

Stated as measurable changes, not a redesign:

- Make expansion breadth-first across containers rather than depth-first within
  them, and measure fact-level joint recall on MultiHop-RAG at the same budget.
  The current rule is the clearest single cause of the loss.
- Drop the overlay unless a version of it can beat its placebo on some corpus.
  The placebo control is what makes this decidable, and it should stay.
- Replace the LLM planner with the rules planner as the default and keep the
  LLM as a fallback only where the rules abstain, then re-measure. The oracle
  bound says routing itself is not where the win is.
- Treat the retrieval unit as the first-class variable. His system loses on the
  rendered budget with an excellent index. Ours wins on the rendered budget
  with a cheaper index. That is the finding worth building on.

## 17. Caveats and disclosures

- One study model does every job. His extraction ran on the cheap model, not
  the models his published run used; the 18-question calibration row is the
  bridge, and it favours his own models.
- The judge is gpt-5.4 and so is Reader B. The cross-family column exists for
  exactly this reason and shows the effect is small.
- Every change made after a number was read is in design section 14 with the
  numbers that were read: the budget-fill fix, the Graphiti drop, the judge
  audit, and the MultiHop-RAG cap amendment.
- The planner table, its prompt and the rules patterns were written with the
  earlier e5 by-type results known. The speaker rule was written by someone who
  knew the question shapes. Both are disclosed and both have no-rule rows.
- The whole interim report was independently recomputed from the per-query
  files by three verifiers with their own code. Every retrieval, answering and
  judge number matched. The findings were reporting hygiene, since fixed.

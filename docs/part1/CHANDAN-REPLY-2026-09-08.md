# His public reply about the 4,000-token limit, checked

On 2026-09-07 Chandan Rajah replied to Murali's LinkedIn comment:

> The new v1.12.0 on PyPI has a fix allowing changing the limit (4000 tokens)
> via config. Unlimited, on LongMemEval is get to above 95% accuracy and on
> ECT-QA above 80%.

This document checks that against what Part 1 actually ran, and against his
public material. Every claim below has a file or a URL behind it.

## 1. The 4,000 tokens in our study is ours, not his

It is the rendered-context budget B of design section 5, fixed before any run
and applied by one code path to every arm including our own.

- `docs/PART1-DESIGN.md:236` "Rendering rule, the same for every arm", lines
  241-243 set B to 4,000 primary and 8,000 secondary.
- `src/multicard/part1/evaluate.py:71` BUDGET_PRIMARY = 4000.
- `src/multicard/part1/retrieve.py:1161` passes the same budget to every arm in
  one comprehension. `src/multicard/part1/render.py:161` is the single `_take()`
  that enforces it for our units and his alike.
- Measured, `results/part1/REPORT.md` lines 108-122: ours_cheap 3,996.3,
  S5_primary 3,996.3, S4_static 3,999.2, chandan_live 4,000. Nobody had a
  bigger window.

## 2. We already ran his unlimited path

- Part 1 installed post-graph-rag **1.12.0** (`results/part1/REPORT.md:57`).
- In that version `max_total_tokens`, `max_entity_tokens` and
  `max_relation_tokens` all default to `None`, with his own comment saying the
  former 4,000-token default "silently withheld evidence the retriever had
  already found" (`post_graph_rag/models.py:124-132` in the pinned clone).
- We passed no token budget at all:
  `tools/part1/pgr/run_spaces.py:1010` calls
  `QueryParam(mode="mix", top_k=top_k, space=space)` and nothing else. A grep
  for the three budget names across `src/` and `tools/` returns zero hits.
- Proof from the data: his candidate list averaged 247.7 units at
  candidate-level joint recall 0.998 (`results/part1/REPORT.md:121`). A live
  4,000-token cap could not produce that.

The cap removal was released in **1.11.1** on 2026-09-02, not 1.12.0
(https://pypi.org/project/post-graph-rag/1.11.1/, commit 850f08a "fix: send
what was retrieved; remove the context cap"). 1.12.0, released 2026-09-05, is a
belief-time audit fix (commit 69c2e0e). So we were two releases past the change
he is describing.

## 3. We also let him raise his own result limit, and it never fired

Design section 5 required chandan_live to run twice: at his shipped result
limit and with the limit raised until his rendered context reaches B, taking
whichever scored higher. The ladder is top_k 8 -> 16 -> 32 -> 64 -> 128
(`run_spaces.py:81`, `run_spaces.py:466-496`).

It never took a step. Measured over all 500 exported spaces: `raised_4k.json`
and `raised_8k.json` both record top_k 8, `reused_from "shipped"`, 0 model
calls and USD 0.0. The loop condition `while tokens[k] < target` was false on
the first check every time, because his call at top_k 8 already returned
28,675 to 54,227 tokens of units against an 8,000-token target.

The gate result, `results/part1/lme/retrieve/metrics.json` key
`chosen_variant`: shipped 0.574468085106383, raised_4k 0.574468085106383.
Identical to the last digit.

## 4. So T1 is untouched

Our headline (S5_primary 0.947 vs chandan_live 0.574 at 4,000 tokens, +0.371)
was measured against his unlimited configuration. Making a limit configurable
cannot raise a number that was produced with no limit.

## 5. What must always be said beside T1

His retriever found the evidence. Candidate-level joint recall 0.998. The loss
is delivery inside a fixed budget, because his harness never chunks and one of
his LongMemEval units is a whole session of about 10,000 characters. Failure
buckets for chandan_live at 4k (`REPORT.md` lines 696-707): index 1, retrieval
0, context assembly 103, reader 63. 232 evidence turns were rendered truncated,
against 8 for S5_primary.

And the gap shrinks as the budget grows: at 8,000 tokens chandan_live goes
0.574 -> 0.804 while S5_primary goes 0.947 -> 0.964, so +0.373 becomes +0.160.
That is the honest qualifier and it is already printed in the report.

Our own `chandan_full_uncut` arm is his unlimited configuration with his own
synthesis prompt and no budget. It scored 0.692 on all questions, 0.679 on
answerable (`REPORT.md:869`). Do not use that against his 95%: different split
(ours is S with about 50 haystack sessions per question, his harness defaults
to oracle), different answering model (ours gemini-2.5-flash-lite, his
gemini-3.6-flash), different judge (ours gpt-5.4 on the official prompts, his
own multi-model panel).

## 6. The 95% figure is not published anywhere

Highest published overall figure for his system on LongMemEval is **94.0%**
(README, PyPI 1.12.0, docs site; his own majority judge panel). It traces to
`evaluation/longmemeval/reader_sweep_nolimit.json`, means_judge
0.9398797595190381, which is exactly 938/998 (499 questions, 2 repeats). That
file records `"repeats": 2`, `"degraded_count": 1` and `"reportable": false`.

Figures at or above 95% do exist in his material, but they are per-category
under that same 94.0 run: single-session-assistant 100.0%, temporal-reasoning
96.2%, single-session-user 95.7%. The 95% may be a category figure quoted as an
overall one. Not verified either way.

The full published landscape for one system on one benchmark:

| figure | reader | judge | source |
|---|---|---|---|
| 94.0% | gemini-3.6-flash | his own panel | README, PyPI, docs site |
| 93.5% | gemini-3.7-flash | his own panel | reader_sweep_nolimit.json |
| 85.8% | gemini-3.6-flash | his own panel | arXiv 2608.24921, still live at v2 |
| 81.1% | gemini-3.6-flash | older panel | docs/index.md |
| 78.16% | gemini-3.6-flash | gpt-4o, official protocol | evaluation/longmemeval/official_gpt4o_g36.json |
| 71.34% | gpt-4o | gpt-4o, official protocol | same file |

The spread is 15.8 points for the same system on the same benchmark, and all of
it is the judge and the harness. The lowest number is the one that uses Zep's
own judge and prompts. His 94.0 is an oracle number (his `fetch.sh` downloads
longmemeval_oracle by default and `run.py --data` defaults to oracle.json) set
against Zep's 71.2 on LongMemEval_S.

## 7. ECT-QA above 80% is real, with three qualifications

Published: 0.807 element-wise Correct with gemini-3.6-flash against TG-RAG
0.599, GraphRAG 0.405, LightRAG 0.406
(`evaluation/ectqa/results_native_hdfixed.json`). It survives a cross-family
regrade with a gpt-4o judge at 0.805.

1. Scale. His ECT-QA runs cover six companies at n about 78 to 84. TG-RAG's
   0.599 is on the full base specific set over 24 companies.
2. It moved fast. On 2026-09-04 the README published 0.677. On 2026-09-05, after
   an indexing gap was found and Home Depot reindexed, it became 0.807.
3. On his own numeric-F1 accuracy arm the same 78 questions score 0.718, not
   above 0.80.

The strongest published evidence for his token-limit story is on ECT-QA, not
LongMemEval: 0.3846 at `max_total_tokens: 4000` against 0.7179 with no cap, same
78 questions (`results_ctx4000_g36.json`, `results_hdfixed_g36.json`).

## 8. Worth knowing

ECT-QA is the benchmark introduced by the TG-RAG paper, "RAG Meets Temporal
Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge", arXiv
2510.13590, Han et al. That is a temporal-graph RAG system by a different team.
It is adjacent to the TG-VGRAG direction and is not the same work.

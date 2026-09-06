# Part 1: the head-to-head on LongMemEval_S and MultiHop-RAG

Generated 2026-09-06T12:00:31+00:00 from results/part1/metrics.json. Design: docs/PART1-DESIGN.md version 4. Every number below is read from that file.

Subsets file sha256: 0e2440828ec0e2f2568bb90742b21c684b73b2cc4db85738879703a5311f5082.

Commits recorded in the report:

- head: b8c1e20
- subsets_sha256_file: 0e2440828ec0e2f2568bb90742b21c684b73b2cc4db85738879703a5311f5082

Graphiti status: dropped. T2 is recorded as not run.

Limited run: every population is restricted to the processed questions, longmemeval 5, multihoprag 5. Not a study result.

## The pass rule

Part 1 passed: **no**.

The rule: T1 shown under D1, T8a positive and significant after Holm within Family A, T2 passes under D2 or is recorded as not run, and T7 holds. T8b is reported.

| quantity | value |
|---|---|
| T1 label (D1) | not shown |
| T1 delta | +0.400 |
| T1 95 percent CI | [+0.000, +0.800] |
| T1 significant after Holm | no |
| T1 shown | no |
| T8a delta | n/a |
| T8a 95 percent CI | n/a |
| T8a significant after Holm | n/a |
| T8a positive and significant | no |
| T2 not run | yes |
| T2 passes (D2, point estimate positive) | no |
| T2 strict reading (positive and significant after Holm) | no |
| T2 satisfied (passes or not run) | yes |
| T7 losses minus wins | n/a |
| T7 holds | no |

D2 reading of T2: not run (Graphiti status dropped).

Second post-graph-rag build (section 9): first build cost USD 0.64; second build ran: no.

## Setup

- tag: smoke
- encoder: sentence-transformers/all-MiniLM-L6-v2
- stages:
  - lme: index topics fitted, graph built, relation_vectors 0 spaces encoded, 5 present, 5 wanted, overlay generated, sample True, n_questions 5, containers 243, arms_skipped , chosen_variant chandan_live shipped, chandan_live_means shipped 0.4, raised_4k 0.4, metric joint_recall, n 5, exports_present chandan 5, graphiti 0, cal 0, second 0, topics n_units 2416, n_topics 87, outlier_share 0.13783112582781457, largest_topic_share 0.041390728476821195, graph plain n_communities 53, largest_community_share_nodes 0.08822681403171552, largest_community_share_units 0.10390631827318011, graph_density 0.0012437274614927427, topic_entropy mean_bits 4.153412240800286, member_weighted_bits 5.030880945865952, n_communities_with_topics 53, topic n_communities 54, largest_community_share_nodes 0.08880345987506007, largest_community_share_units 0.10311981123831163, graph_density 0.0012437274614927427, topic_entropy mean_bits 4.088406651285597, member_weighted_bits 4.957903721264548, n_communities_with_topics 54, overlay n_candidate_pairs 319, n_flagged 64, n_parse_ok 64, links R0 0, R2 64, R3 31, P0 31
  - mhrag: index topics fitted, graph built, relation_vectors 0 spaces encoded, 0 present, 1 wanted, overlay generated, sample True, n_questions 5, containers 55, arms_skipped , chosen_variant , exports_present chandan 0, graphiti 0, cal 0, second 0, topics n_units 325, n_topics 13, outlier_share 0.11076923076923077, largest_topic_share 0.15384615384615385, graph plain n_communities 47, largest_community_share_nodes 0.05174488567990373, largest_community_share_units 0.07299787384833452, graph_density 0.0021617511375350176, topic_entropy mean_bits 2.9357164956234167, member_weighted_bits 3.1886011317847593, n_communities_with_topics 47, topic n_communities 46, largest_community_share_nodes 0.07781789009225833, largest_community_share_units 0.123789274746043, graph_density 0.0021617511375350176, topic_entropy mean_bits 2.9012311572060177, member_weighted_bits 3.1416010868580084, n_communities_with_topics 46, overlay n_candidate_pairs 151, n_flagged 31, n_parse_ok 31, links R0 0, R2 31, R3 17, P0 17
- long_evidence_turns:
  - n_over_2000: 11
  - over_4000: qid 5809eb10, turn answer_sharegpt_4aJsGCH_0#0, chars 5425
- chandan_configuration:
  - package: name post-graph-rag, version 1.12.0, file /Users/muralisid/github_other/part1-tools/pgr/.venv/lib/python3.11/site-packages/post_graph_rag/__init__.py, python 3.11.16
  - index_model: gemini-2.5-flash-lite
  - answer_model: gemini-2.5-flash-lite
  - embedding_model: gemini-embedding-001
  - embedding_dim: 1536
  - extraction_prompt: CONVERSATIONAL_PROMPT from his evaluation/longmemeval/run.py
  - prompt_fallback: library document prompt on refusal, as in his run.py
- design: docs/PART1-DESIGN.md version 4
- budgets: 4000, 8000
- reader_a: gemini-2.5-flash-lite
- reader_b: gpt-5.4
- judges:
  - candidate_primary: gemini-2.5-flash-lite
  - second: gpt-5.4
  - agreement_threshold: 0.9
- price_page_date: fetched 2026-09-06; the page shows no last-updated date; HTTP Last-Modified header: not sent by server; dated statements on the page: introductory pricing for Gemini 3.6 Flash runs through December 31, 2026 and standard pricing of 1.50 in and 7.50 out applies from January 1, 2027; non-global endpoint pricing for Gemini 3 and later families applies from July 1, 2026
- corpora:
  - longmemeval: n 500, n_abstention 30, per_type knowledge-update 78, multi-session 133, single-session-assistant 56, single-session-preference 30, single-session-user 70, temporal-reasoning 133
  - multihoprag: n 2556, per_type comparison_query 856, inference_query 816, null_query 301, temporal_query 583
- subsets:
  - GRAPHITI_150: 150
  - CHANDAN_CAL_18: 18
  - MHRAG_ANSWER: 600
  - READER_B_MHRAG: 200
- seed: 13
- reader_max_output_tokens: 300
- judge_max_output_tokens: 200

Prices, USD per million tokens, read on fetched 2026-09-06; the page shows no last-updated date; HTTP Last-Modified header: not sent by server; dated statements on the page: introductory pricing for Gemini 3.6 Flash runs through December 31, 2026 and standard pricing of 1.50 in and 7.50 out applies from January 1, 2027; non-global endpoint pricing for Gemini 3 and later families applies from July 1, 2026:

| model | input | output |
|---|---|---|
| gemini-3.6-flash | 0.750 | 3.750 |
| gemini-3.7-flash | 0.750 | 3.750 |
| gemini-3.8-flash | 0.750 | 3.750 |
| gemini-3.5-flash | 1.500 | 9.000 |
| gemini-3.5-flash-lite | 0.300 | 2.500 |
| gemini-2.5-flash | 0.300 | 2.500 |
| gemini-2.5-flash-lite | 0.100 | 0.400 |
| gemini-2.5-pro | 1.250 | 10.000 |
| gemini-embedding-001 | 0.150 | 0.000 |
| text-embedding-005 | 0.025 | 0.000 |

## Retrieval

Means over the arm's population. A question with no output scores 0 and is counted under missing. Graphiti is scored at session level, so its turn columns are n/a.

### LongMemEval, answerable questions, budget 4,000 tokens

| arm | JointRecall | candidate JR | session JR | turn R@10 | nDCG@10 | session R@5 | rendered tokens | duplicate share | candidate list size | n | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_cheap | 1.000 | 1.000 | 1.000 | 1.000 | 0.797 | 0.967 | 4,000 | 0.000 | 68.2 | 5 | 0 |
| ours_cheap_norule | 1.000 | 1.000 | 1.000 | 0.867 | 0.630 | 0.967 | 4,000 | 0.000 | 100 | 5 | 0 |
| ours_sentence_norule | 0.800 | 1.000 | 1.000 | 0.833 | 0.644 | 0.967 | 4,000 | 0.000 | 100 | 5 | 0 |
| S4_static | 1.000 | 1.000 | 1.000 | 0.767 | 0.474 | 0.967 | 3,630 | 0.000 | 61.2 | 5 | 0 |
| S5_primary | 0.800 | 1.000 | 1.000 | 0.767 | 0.600 | 0.967 | 3,591 | 0.000 | 58.6 | 5 | 0 |
| S5_primary_norule | 0.600 | 1.000 | 0.800 | 0.733 | 0.471 | 0.967 | 4,000 | 0.000 | 100 | 5 | 0 |
| S5_noPGR | 0.800 | 1.000 | 1.000 | 0.733 | 0.529 | 0.967 | 3,559 | 0.000 | 58 | 5 | 0 |
| S5_planner_rules | 0.800 | 1.000 | 1.000 | 0.767 | 0.654 | 0.967 | 3,528.2 | 0.000 | 59.2 | 5 | 0 |
| S5_planner_oracle | 0.800 | 1.000 | 1.000 | 0.767 | 0.600 | 0.967 | 3,591 | 0.000 | 58.6 | 5 | 0 |
| S5_overlay_R0 | 0.800 | 1.000 | 1.000 | 0.767 | 0.545 | 0.967 | 3,615.6 | 0.000 | 60.2 | 5 | 0 |
| S5_overlay_R2 | 0.800 | 1.000 | 1.000 | 0.767 | 0.550 | 0.967 | 3,624 | 0.000 | 58.8 | 5 | 0 |
| S5_overlay_P0 | 0.800 | 1.000 | 1.000 | 0.733 | 0.535 | 0.967 | 3,621.6 | 0.000 | 58.8 | 5 | 0 |
| S2_lazy | 1.000 | 1.000 | 1.000 | 0.767 | 0.474 | 0.967 | 3,630 | 0.000 | 61.2 | 5 | 0 |
| chandan_live | 0.400 | 1.000 | 0.600 | 0.333 | 0.281 | 0.867 | 4,000 | 0.000 | 235.6 | 5 | 0 |
| chandan_full | 0.400 | 1.000 | 0.600 | 0.333 | 0.281 | 0.867 | 4,000 | 0.000 | 235.6 | 5 | 0 |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Evidence turns rendered truncated, summed over questions (section 3): truncated to fit the budget or cut at the 2,000-character limit, and the cut ones alone:

| arm | truncated evidence turns | of which cut at 2,000 characters |
|---|---|---|
| ours_cheap | 0 | 0 |
| ours_cheap_norule | 0 | 0 |
| ours_sentence_norule | 0 | 0 |
| S4_static | 0 | 0 |
| S5_primary | 0 | 0 |
| S5_primary_norule | 0 | 0 |
| S5_noPGR | 0 | 0 |
| S5_planner_rules | 0 | 0 |
| S5_planner_oracle | 0 | 0 |
| S5_overlay_R0 | 0 | 0 |
| S5_overlay_R2 | 0 | 0 |
| S5_overlay_P0 | 0 | 0 |
| S2_lazy | 0 | 0 |
| chandan_live | 3 | 0 |
| chandan_full | 3 | 0 |

Raised variant (section 5, the result limit raised until the rendered context reaches B): questions where the runner's top step did not reach B, of the questions with a raised run:

| arm | did not reach B | questions with a raised run |
|---|---|---|
| chandan_live | 0 | 5 |

### LongMemEval, answerable questions, budget 8,000 tokens

| arm | JointRecall | candidate JR | session JR | turn R@10 | nDCG@10 | session R@5 | rendered tokens | duplicate share | candidate list size | n | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_cheap | 1.000 | 1.000 | 1.000 | 1.000 | 0.797 | 0.967 | 6,536.8 | 0.000 | 68.2 | 5 | 0 |
| ours_cheap_norule | 1.000 | 1.000 | 1.000 | 0.867 | 0.630 | 0.967 | 7,998.2 | 0.000 | 100 | 5 | 0 |
| ours_sentence_norule | 1.000 | 1.000 | 1.000 | 0.833 | 0.644 | 0.967 | 7,999.8 | 0.000 | 100 | 5 | 0 |
| S4_static | 1.000 | 1.000 | 1.000 | 0.767 | 0.474 | 0.967 | 5,005.8 | 0.000 | 61.2 | 5 | 0 |
| S5_primary | 1.000 | 1.000 | 1.000 | 0.767 | 0.600 | 0.967 | 4,769 | 0.000 | 58.6 | 5 | 0 |
| S5_primary_norule | 0.800 | 1.000 | 1.000 | 0.733 | 0.471 | 0.967 | 8,000 | 0.000 | 100 | 5 | 0 |
| S5_noPGR | 1.000 | 1.000 | 1.000 | 0.733 | 0.529 | 0.967 | 5,018.4 | 0.000 | 58 | 5 | 0 |
| S5_planner_rules | 1.000 | 1.000 | 1.000 | 0.767 | 0.654 | 0.967 | 4,690.2 | 0.000 | 59.2 | 5 | 0 |
| S5_planner_oracle | 1.000 | 1.000 | 1.000 | 0.767 | 0.600 | 0.967 | 4,769 | 0.000 | 58.6 | 5 | 0 |
| S5_overlay_R0 | 0.800 | 1.000 | 1.000 | 0.767 | 0.545 | 0.967 | 4,850.6 | 0.000 | 60.2 | 5 | 0 |
| S5_overlay_R2 | 0.800 | 1.000 | 1.000 | 0.767 | 0.550 | 0.967 | 4,811.8 | 0.000 | 58.8 | 5 | 0 |
| S5_overlay_P0 | 1.000 | 1.000 | 1.000 | 0.733 | 0.535 | 0.967 | 4,835 | 0.000 | 58.8 | 5 | 0 |
| S2_lazy | 1.000 | 1.000 | 1.000 | 0.767 | 0.474 | 0.967 | 5,005.8 | 0.000 | 61.2 | 5 | 0 |
| chandan_live | 0.600 | 1.000 | 0.600 | 0.333 | 0.281 | 0.867 | 8,000 | 0.000 | 235.6 | 5 | 0 |
| chandan_full | 0.600 | 1.000 | 0.600 | 0.333 | 0.281 | 0.867 | 8,000 | 0.000 | 235.6 | 5 | 0 |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Evidence turns rendered truncated, summed over questions (section 3): truncated to fit the budget or cut at the 2,000-character limit, and the cut ones alone:

| arm | truncated evidence turns | of which cut at 2,000 characters |
|---|---|---|
| ours_cheap | 0 | 0 |
| ours_cheap_norule | 0 | 0 |
| ours_sentence_norule | 0 | 0 |
| S4_static | 0 | 0 |
| S5_primary | 0 | 0 |
| S5_primary_norule | 0 | 0 |
| S5_noPGR | 0 | 0 |
| S5_planner_rules | 0 | 0 |
| S5_planner_oracle | 0 | 0 |
| S5_overlay_R0 | 0 | 0 |
| S5_overlay_R2 | 0 | 0 |
| S5_overlay_P0 | 0 | 0 |
| S2_lazy | 0 | 0 |
| chandan_live | 0 | 0 |
| chandan_full | 0 | 0 |

Raised variant (section 5, the result limit raised until the rendered context reaches B): questions where the runner's top step did not reach B, of the questions with a raised run:

| arm | did not reach B | questions with a raised run |
|---|---|---|
| chandan_live | 0 | 5 |

### MultiHop-RAG, non-null queries, budget 4,000 tokens

| arm | fact JR (all located) | fact JR (all non-null) | document JR | candidate fact JR | candidate document JR | rendered tokens | duplicate share | candidate list size | n | missing | n all located |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_cheap | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S4_static | 0.200 | 0.200 | 0.200 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_primary | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_noPGR | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_planner_rules | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_planner_oracle | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_overlay_R0 | 0.200 | 0.200 | 0.200 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_overlay_R2 | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_overlay_P0 | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| S2_lazy | 0.200 | 0.200 | 0.200 | 1.000 | 1.000 | 4,000 | 0.000 | 100 | 5 | 0 | 5 |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

### MultiHop-RAG, non-null queries, budget 8,000 tokens

| arm | fact JR (all located) | fact JR (all non-null) | document JR | candidate fact JR | candidate document JR | rendered tokens | duplicate share | candidate list size | n | missing | n all located |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_cheap | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S4_static | 0.600 | 0.600 | 0.600 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_primary | 0.600 | 0.600 | 0.600 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_noPGR | 0.600 | 0.600 | 0.600 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_planner_rules | 0.800 | 0.800 | 0.800 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_planner_oracle | 0.800 | 0.800 | 0.800 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_overlay_R0 | 0.400 | 0.400 | 0.400 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_overlay_R2 | 0.600 | 0.600 | 0.600 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| S5_overlay_P0 | 0.800 | 0.800 | 0.800 | 1.000 | 1.000 | 7,997.4 | 0.000 | 100 | 5 | 0 | 5 |
| S2_lazy | 0.600 | 0.600 | 0.600 | 1.000 | 1.000 | 8,000 | 0.000 | 100 | 5 | 0 | 5 |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

## By question type

### longmemeval, JointRecall by type, budget 4,000 tokens

| type | n | ours_cheap | ours_cheap_norule | ours_sentence_norule | S4_static | S5_primary | S5_primary_norule | S5_noPGR | S5_planner_rules | S5_planner_oracle | S5_overlay_R0 | S5_overlay_R2 | S5_overlay_P0 | S2_lazy | chandan_live | chandan_full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| temporal-reasoning | 4 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 |

### longmemeval, JointRecall by type, budget 8,000 tokens

| type | n | ours_cheap | ours_cheap_norule | ours_sentence_norule | S4_static | S5_primary | S5_primary_norule | S5_noPGR | S5_planner_rules | S5_planner_oracle | S5_overlay_R0 | S5_overlay_R2 | S5_overlay_P0 | S2_lazy | chandan_live | chandan_full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal-reasoning | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 |

### multihoprag, fact JR (all located) by type, budget 4,000 tokens

| type | n | ours_cheap | S4_static | S5_primary | S5_noPGR | S5_planner_rules | S5_planner_oracle | S5_overlay_R0 | S5_overlay_R2 | S5_overlay_P0 | S2_lazy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison_query | 1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| temporal_query | 4 | 0.250 | 0.250 | 0.500 | 0.500 | 0.500 | 0.500 | 0.250 | 0.500 | 0.500 | 0.250 |

### multihoprag, fact JR (all located) by type, budget 8,000 tokens

| type | n | ours_cheap | S4_static | S5_primary | S5_noPGR | S5_planner_rules | S5_planner_oracle | S5_overlay_R0 | S5_overlay_R2 | S5_overlay_P0 | S2_lazy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison_query | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal_query | 4 | 1.000 | 0.500 | 0.500 | 0.500 | 0.750 | 0.750 | 0.250 | 0.500 | 0.750 | 0.500 |

## Pre-declared tests

Paired at the question level, the bench compare (10,000 permutations, percentile bootstrap CI), alpha 0.05, budget 4,000 tokens. Holm within each family. Families B and C never feed the pass rule. A test on a partial run is labelled and left out of the pass rule.

Populations: LongMemEval answerable 5, GRAPHITI_150 2, MultiHop-RAG all located 5, LOCAL_120 0, LongMemEval all 5.

### Family A, the gate

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | S5_primary vs chandan_live | joint_recall | 5 (refused 0, missing 0/0) | 0.800 vs 0.400 | +0.400 | [+0.000, +0.800] | 0.5004 | 2/3/0 | 1.0000, not significant | D1: not shown; restricted to the 5 processed longmemeval questions |
| T2 | S5_primary vs graphiti | session_joint_recall | not run: no output from graphiti on longmemeval |  |  |  |  |  |  |  |
| T8a | S5_primary vs chandan_live | fact_joint_recall | not run: no output from chandan_live on multihoprag |  |  |  |  |  |  |  |
| T8b | S5_primary vs ours_cheap | fact_joint_recall | 5 (refused 0, missing 0/0) | 0.400 vs 0.400 | +0.000 | [-0.600, +0.600] | 1.0000 | 1/3/1 | 1.0000, not significant | restricted to the 5 processed multihoprag questions |

D1 reading of T1: not shown.

### Family B, the mechanism

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T3 | S5_primary vs ours_cheap | joint_recall | 5 (refused 0, missing 0/0) | 0.800 vs 1.000 | -0.200 | [-0.600, +0.000] | 1.0000 | 0/4/1 | 1.0000, not significant | restricted to the 5 processed longmemeval questions |
| T4 | S5_primary vs S4_static | joint_recall | 5 (refused 0, missing 0/0) | 0.800 vs 1.000 | -0.200 | [-0.600, +0.000] | 1.0000 | 0/4/1 | 1.0000, not significant | restricted to the 5 processed longmemeval questions |
| T5a | S5_primary vs S5_overlay_R0 | joint_recall | 5 (refused 0, missing 0/0) | 0.800 vs 0.800 | +0.000 | [+0.000, +0.000] | 1.0000 | 0/5/0 | 1.0000, not significant | restricted to the 5 processed longmemeval questions |
| T5b | S5_primary vs S5_overlay_P0 | joint_recall | 5 (refused 0, missing 0/0) | 0.800 vs 0.800 | +0.000 | [+0.000, +0.000] | 1.0000 | 0/5/0 | 1.0000, not significant | restricted to the 5 processed longmemeval questions |
| T6 | S4_static vs ours_cheap | joint_recall | 5 (refused 0, missing 0/0) | 1.000 vs 1.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 0/5/0 | 1.0000, not significant | restricted to the 5 processed longmemeval questions |

Reported for the T5 reading, not under Holm:

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T5_P0_minus_R0 | S5_overlay_P0 vs S5_overlay_R0 | joint_recall | 5 (refused 0, missing 0/0) | 0.800 vs 0.800 | +0.000 | [+0.000, +0.000] | 1.0000 | 0/5/0 | not in Holm | restricted to the 5 processed longmemeval questions |

T5 branch: **no measurable effect**.

| pair | delta | 95 percent CI |
|---|---|---|
| R3 minus R0 | +0.000 | [+0.000, +0.000] |
| R3 minus P0 | +0.000 | [+0.000, +0.000] |
| P0 minus R0 | +0.000 | [+0.000, +0.000] |

Questions whose rendered context differs at all between R0, R3 and P0: 5. Tolerance 0.01.

### Family C, answers (exact McNemar on the discordant pairs, primary judge)

| test | arms (a vs b) | reader | population | n | accuracy a vs b | delta | wins/losses | p | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | S5_primary vs chandan_live | reader_a | LongMemEval all | 5 (refused 0, missing 0/0) | 0.600 vs 0.400 | +0.200 | 1/0 of 1 discordant | 1.0000 | 1.0000, not significant | restricted to the 5 processed longmemeval questions |
| C2 | S5_primary vs chandan_live | reader_b | LongMemEval all | not run: no answers for one of the arms under this reader |  |  |  |  |  |  |
| C3 | S5_primary vs graphiti | reader_a | GRAPHITI_150 | not run: graphiti dropped: recorded as not run |  |  |  |  |  |  |
| C4 | S5_primary vs graphiti | reader_b | GRAPHITI_150 | not run: graphiti dropped: recorded as not run |  |  |  |  |  |  |
| C5 | S5_primary vs chandan_live | reader_a | MHRAG_ANSWER | not run: no output from chandan_live on multihoprag |  |  |  |  |  |  |
| C6 | S5_primary vs chandan_live | reader_b | READER_B_MHRAG | not run: no output from chandan_live on multihoprag |  |  |  |  |  |  |

Robustness table: the wrong answers the second judge called right are flipped and the test is rerun. This never changes the primary judge.

| test | accuracy a vs b (re-judged) | delta | wins/losses | p | outcome differs |
|---|---|---|---|---|---|
| C1 | 0.600 vs 0.400 | +0.200 | 1/0 | 1.0000 | no |

### T7, non-inferiority on the local set

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T7 | S5_primary vs ours_cheap | joint_recall | not run: fewer than two paired questions |  |  |  |  |  |  |  |

T7 rule: losses minus wins is n/a; the limit is 3; not run. The CI is beside it and is not the rule.

## Predictions (section 10), written before any build

Each prediction with its fixed number and band, the measured value or paired delta with its 95 percent CI, and one label: consistent with, not confirmed (the point estimate is inside the band); not confirmed (outside the band, on the predicted side); contradicted (on the wrong side of zero, or outside a within band); untested (the quantity was not measured).

| prediction | statement | predicted | band | measured | 95 percent CI | n | label | note |
|---|---|---|---|---|---|---|---|---|
| T1_overall | T1 overall positive, about 0.02 (band 0.03 either side) | +0.02, band [-0.01, +0.05] | [-0.010, +0.050] | +0.400 | [+0.000, +0.800] | 5 | **not confirmed** | S5_primary minus chandan_live, LongMemEval answerable |
| T1_multi_session | S5_primary beats chandan_live on JointRecall@4k by at least 0.05 on multi-session (half-width about 0.07) | at least +0.05, half-width 0.07 (band from -0.02) | [-0.020, open] | n/a | n/a | n/a | **untested** | fewer than two paired questions |
| T1_temporal_reasoning | S5_primary beats chandan_live by at least 0.03 on temporal-reasoning (half-width about 0.08) | at least +0.03, half-width 0.08 (band from -0.05) | [-0.050, open] | +0.500 | [+0.000, +1.000] | 4 | **consistent with, not confirmed** | S5_primary minus chandan_live, LongMemEval answerable, temporal-reasoning |
| T1_local_set | S5_primary within 0.02 of chandan_live on the local set | within 0.02 (band [-0.02, +0.02]) | [-0.020, +0.020] | n/a | n/a | n/a | **untested** | fewer than two paired questions |
| S4_over_ours_cheap | S4_static beats ours_cheap by 0.02 to 0.04 | +0.02 to +0.04 | [+0.020, +0.040] | +0.000 | [+0.000, +0.000] | 5 | **contradicted** | S4_static minus ours_cheap, LongMemEval answerable |
| S5_gain_from_planner | Most of the S5 over S4 gain comes from the planner on temporal and multi-session, not from the overlay | planner share above the overlay share | n/a | S5_minus_S4 -0.200, overlay_R3_minus_R0 +0.000, planner -0.200, by_type temporal-reasoning S5_minus_S4 +0.000, overlay +0.000, planner +0.000, n +4.000, multi-session S5_minus_S4 n/a, overlay n/a, planner n/a, n n/a | n/a | n/a | **contradicted** | planner share = (S5_primary minus S4_static) minus (R3 minus R0); consistent when the planner share exceeds the overlay share overall and is positive on both types |
| T5_branch | T5 lands in the adds material branch | adds material | n/a | no measurable effect | n/a | n/a | **contradicted** | the T5 four-branch reading of section 9 |
| largest_community_share | The largest community share is above 50 percent without topic weighting and drops under it | plain above 0.50, topic-weighted below plain | n/a | plain_nodes +0.088, topic_nodes +0.089, plain_units +0.104, topic_units +0.103 | n/a | n/a | **contradicted** | share over the phrases of the pruned graph (nodes); the share over sub-units is beside it (design gap: the design does not say which share) |
| graphiti_gap | graphiti session-level JointRecall@4k is below S5_primary on GRAPHITI_150 by 0.02 to 0.06 (half-width about 0.04) | +0.02 to +0.06, half-width 0.04 (band [-0.02, +0.10]) | [-0.020, +0.100] | n/a | n/a | n/a | **untested** | Graphiti dropped or partial: recorded as untested |
| mhrag_comparison | MultiHop-RAG: S5_primary beats chandan_live on fact-level joint recall for comparison queries by at least 0.05 | at least +0.05 | [+0.050, open] | n/a | n/a | n/a | **untested** | no output from chandan_live on multihoprag |
| mhrag_inference | MultiHop-RAG: S5_primary beats chandan_live on fact-level joint recall for inference queries by at least 0.05 | at least +0.05 | [+0.050, open] | n/a | n/a | n/a | **untested** | no output from chandan_live on multihoprag |
| closed_book_floor_comparison_query | closed_book accuracy on answerable MultiHop-RAG queries exceeds the per-type majority-class rate by at least 0.10 (comparison_query) | at least +0.10 | [+0.100, open] | +0.000 | n/a | 1 | **contradicted** | accuracy 1.000, majority-class rate 1.000 (commonest gold answer 'no') |
| closed_book_floor_temporal_query | closed_book accuracy on answerable MultiHop-RAG queries exceeds the per-type majority-class rate by at least 0.10 (temporal_query) | at least +0.10 | [+0.100, open] | +0.000 | n/a | 4 | **contradicted** | accuracy 0.250, majority-class rate 0.250 (commonest gold answer 'larger') |
| family_C_same_sign_LongMemEval | The S5_primary minus chandan_live accuracy difference (Family C) has the same sign under Reader A and Reader B on LongMemEval | same sign under both readers | n/a | n/a | n/a | n/a | **untested** | C1 and C2; a zero delta under either reader counts as contradicted |
| family_C_same_sign_MultiHop-RAG | The S5_primary minus chandan_live accuracy difference (Family C) has the same sign under Reader A and Reader B on MultiHop-RAG | same sign under both readers | n/a | n/a | n/a | n/a | **untested** | C5 and C6; a zero delta under either reader counts as contradicted |

## Failure buckets

Bucket 5 is tested first: a wrong answer on an answerable question that the two judges disagree on. Every other wrong answer is tested against buckets 1 to 4 in order, first match wins. Bucket 5 is decidable for the head-to-head arms; for the other arms it is decidable on the audited sample only, and the undecidable count is shown.

### S2_lazy, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 1 |
| all | 2 | 0 | 0 | 0 | 0 | 2 | 1 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S2_lazy, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 1 | 1 | 2 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S4_static, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 1 |
| all | 2 | 0 | 0 | 0 | 0 | 2 | 1 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S4_static, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 1 | 1 | 2 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_noPGR, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| all | 3 | 0 | 0 | 0 | 1 | 2 | 3 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_noPGR, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 1 | 1 | 2 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_overlay_P0, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 1 |
| all | 3 | 0 | 0 | 0 | 1 | 2 | 1 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_overlay_P0, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| all | 1 | 0 | 0 | 0 | 0 | 1 | 1 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_overlay_R0, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| temporal-reasoning | 4 | 0 | 0 | 0 | 0 | 4 | 4 |
| all | 5 | 0 | 0 | 0 | 1 | 4 | 5 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_overlay_R0, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 1 | 1 | 2 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_overlay_R2, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| all | 3 | 0 | 0 | 0 | 1 | 2 | 3 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_overlay_R2, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 1 | 1 | 2 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_planner_oracle, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 1 |
| all | 2 | 0 | 0 | 0 | 0 | 2 | 1 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_planner_oracle, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 1 | 1 | 2 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_planner_rules, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| all | 2 | 0 | 0 | 0 | 0 | 2 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_planner_rules, multihoprag (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| all | 1 | 0 | 0 | 0 | 0 | 1 | 1 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_primary, longmemeval

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| all | 2 | 0 | 0 | 0 | 0 | 2 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_primary, multihoprag

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 1 | 1 | 0 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### S5_primary_norule, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 3 | 0 | 0 | 0 | 1 | 2 | 3 |
| all | 3 | 0 | 0 | 0 | 1 | 2 | 3 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### chandan_full, longmemeval

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| temporal-reasoning | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| all | 3 | 0 | 0 | 0 | 3 | 0 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### chandan_live, longmemeval

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| single-session-preference | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| temporal-reasoning | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| all | 3 | 0 | 0 | 0 | 3 | 0 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### ours_cheap, longmemeval

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| all | 2 | 0 | 0 | 0 | 0 | 2 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### ours_cheap, multihoprag

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal_query | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| all | 2 | 0 | 0 | 0 | 2 | 0 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### ours_cheap_norule, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 0 | 2 | 1 |
| all | 2 | 0 | 0 | 0 | 0 | 2 | 1 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

### ours_sentence_norule, longmemeval (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 2 | 0 | 0 | 0 | 1 | 1 | 2 |
| all | 2 | 0 | 0 | 0 | 1 | 1 | 2 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0.

## Judges

Candidate primary judge: gemini-2.5-flash-lite. Second judge: gpt-5.4. Pooled agreement on the audit sample: 12 of 13 verdicts, 0.923. Threshold 0.900. Primary judge decided before any test: **gemini-2.5-flash-lite**. Second judge column: gpt-5.4.

| arm | corpus | reader | n | agree | agreement |
|---|---|---|---|---|---|
| S2_lazy | longmemeval | reader_a | 1 | 1 | 1.000 |
| S4_static | longmemeval | reader_a | 1 | 1 | 1.000 |
| S5_overlay_P0 | longmemeval | reader_a | 2 | 2 | 1.000 |
| S5_overlay_R2 | longmemeval | reader_a | 1 | 1 | 1.000 |
| S5_planner_oracle | longmemeval | reader_a | 2 | 2 | 1.000 |
| S5_planner_rules | longmemeval | reader_a | 2 | 1 | 0.500 |
| chandan_full_uncut | longmemeval | chandan_own | 1 | 1 | 1.000 |
| chandan_live | longmemeval | reader_a | 1 | 1 | 1.000 |
| ours_cheap | longmemeval | reader_a | 1 | 1 | 1.000 |
| ours_cheap_norule | longmemeval | reader_a | 1 | 1 | 1.000 |

## Answering accuracy, primary judge

The second judge is a separate column and is never merged. A question with no output counts as wrong.

### chandan_own, longmemeval

| arm | budget | n | missing | all | answerable | abstention or null | single-session-preference | temporal-reasoning | second judge |
|---|---|---|---|---|---|---|---|---|---|
| chandan_full_uncut | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 1.000 | 0.250 | 1.000 (n 1, disagree 0) |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent |

### reader_a, longmemeval

| arm | budget | n | missing | all | answerable | abstention or null | single-session-preference | temporal-reasoning | second judge |
|---|---|---|---|---|---|---|---|---|---|
| S2_lazy | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 1, disagree 0) |
| S2_lazy | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S4_static | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 1, disagree 0) |
| S4_static | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_noPGR | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_noPGR | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_overlay_P0 | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | 0.000 (n 2, disagree 0) |
| S5_overlay_P0 | 8,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_overlay_R0 | 4,000 | 5 | 0 | 0.000 | 0.000 | n/a | 0.000 | 0.000 | n/a (n 0, disagree 0) |
| S5_overlay_R0 | 8,000 | 5 | 0 | 0.400 | 0.400 | n/a | 1.000 | 0.250 | n/a (n 0, disagree 0) |
| S5_overlay_R2 | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | 1.000 (n 1, disagree 0) |
| S5_overlay_R2 | 8,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_planner_oracle | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.500 (n 2, disagree 0) |
| S5_planner_oracle | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_planner_rules | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.500 (n 2, disagree 1) |
| S5_planner_rules | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_primary | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 2, disagree 0) |
| S5_primary | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 2, disagree 0) |
| S5_primary_norule | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 1.000 | 0.250 | n/a (n 0, disagree 0) |
| S5_primary_norule | 8,000 | 5 | 0 | 0.400 | 0.400 | n/a | 1.000 | 0.250 | n/a (n 0, disagree 0) |
| chandan_full | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | 0.000 (n 3, disagree 0) |
| chandan_full | 8,000 | 5 | 0 | 0.200 | 0.200 | n/a | 0.000 | 0.250 | 0.000 (n 4, disagree 0) |
| chandan_live | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | 0.000 (n 3, disagree 0) |
| chandan_live | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 2, disagree 0) |
| closed_book | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | n/a (n 0, disagree 0) |
| closed_book | 8,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | n/a (n 0, disagree 0) |
| oracle_full | 4,000 | 5 | 0 | 0.200 | 0.200 | n/a | 0.000 | 0.250 | n/a (n 0, disagree 0) |
| oracle_full | 8,000 | 5 | 0 | 0.200 | 0.200 | n/a | 0.000 | 0.250 | n/a (n 0, disagree 0) |
| ours_cheap | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.333 (n 3, disagree 0) |
| ours_cheap | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 2, disagree 0) |
| ours_cheap_norule | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 1, disagree 0) |
| ours_cheap_norule | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| ours_sentence_norule | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| ours_sentence_norule | 8,000 | 5 | 0 | 0.400 | 0.400 | n/a | 0.000 | 0.500 | n/a (n 0, disagree 0) |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent |

### reader_a, multihoprag

| arm | budget | n | missing | all | answerable | abstention or null | comparison_query | temporal_query | second judge |
|---|---|---|---|---|---|---|---|---|---|
| S2_lazy | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S2_lazy | 8,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| S4_static | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S4_static | 8,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_noPGR | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_noPGR | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 0.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_overlay_P0 | 4,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_overlay_P0 | 8,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_overlay_R0 | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_overlay_R0 | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 0.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_overlay_R2 | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_overlay_R2 | 8,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_planner_oracle | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_planner_oracle | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 0.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_planner_rules | 4,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| S5_planner_rules | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | n/a (n 0, disagree 0) |
| S5_primary | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 2, disagree 0) |
| S5_primary | 8,000 | 5 | 0 | 0.600 | 0.600 | n/a | 0.000 | 0.750 | 0.000 (n 2, disagree 0) |
| closed_book | 4,000 | 5 | 0 | 0.400 | 0.400 | n/a | 1.000 | 0.250 | n/a (n 0, disagree 0) |
| closed_book | 8,000 | 5 | 0 | 0.400 | 0.400 | n/a | 1.000 | 0.250 | n/a (n 0, disagree 0) |
| oracle_full | 4,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| oracle_full | 8,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | n/a (n 0, disagree 0) |
| ours_cheap | 4,000 | 5 | 0 | 0.600 | 0.600 | n/a | 1.000 | 0.500 | 0.000 (n 2, disagree 0) |
| ours_cheap | 8,000 | 5 | 0 | 0.800 | 0.800 | n/a | 1.000 | 0.750 | 0.000 (n 1, disagree 0) |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Absent on longmemeval (no output, no export found): graphiti.
Absent on multihoprag (no output, no export found): chandan_live, chandan_full.

## Cost and time

Index-time spend is charged to every arm that reads the tables it built: post-graph-rag's build to every arm that reads his entities, relations and aliases, Graphiti's to the graphiti arm. post-graph-rag's and Graphiti's spend is metered from the usage field of each response inside the runner. The cross-check against the proxy request log by time window and job tag is pending: the metrics file holds the job tags and no proxy window total.

| arm | corpus | index USD charged | components | query calls per question | query tokens in per question | query tokens out per question | query USD | seconds per question | questions |
|---|---|---|---|---|---|---|---|---|---|
| S2_lazy | longmemeval | 0.64 | topics, graph, pgr_build | 16.000 | 1,582 | 31.4 | 0.00 | 3.089 | 5 |
| S2_lazy | multihoprag | 0.00 | topics, graph | 15.000 | 1,912 | 30 | 0.00 | 2.065 | 5 |
| S4_static | longmemeval | 0.64 | topics, graph, pgr_build | 0.000 | 0 | 0 | 0.00 | 0.010 | 5 |
| S4_static | multihoprag | 0.00 | topics, graph | 0.000 | 0 | 0 | 0.00 | 0.017 | 5 |
| S5_noPGR | longmemeval | 0.01 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.009 | 5 |
| S5_noPGR | multihoprag | 0.00 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.018 | 5 |
| S5_overlay_P0 | longmemeval | 0.64 | topics, graph, overlay, pgr_build | 0.000 | 0 | 0 | 0.00 | 0.010 | 5 |
| S5_overlay_P0 | multihoprag | 0.00 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.019 | 5 |
| S5_overlay_R0 | longmemeval | 0.64 | topics, graph, pgr_build | 0.000 | 0 | 0 | 0.00 | 0.009 | 5 |
| S5_overlay_R0 | multihoprag | 0.00 | topics, graph | 0.000 | 0 | 0 | 0.00 | 0.017 | 5 |
| S5_overlay_R2 | longmemeval | 0.64 | topics, graph, overlay, pgr_build | 0.000 | 0 | 0 | 0.00 | 0.010 | 5 |
| S5_overlay_R2 | multihoprag | 0.00 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.019 | 5 |
| S5_planner_oracle | longmemeval | 0.64 | topics, graph, overlay, pgr_build | 0.000 | 0 | 0 | 0.00 | 0.009 | 5 |
| S5_planner_oracle | multihoprag | 0.00 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.018 | 5 |
| S5_planner_rules | longmemeval | 0.64 | topics, graph, overlay, pgr_build | 0.000 | 0 | 0 | 0.00 | 0.009 | 5 |
| S5_planner_rules | multihoprag | 0.00 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.019 | 5 |
| S5_primary | longmemeval | 0.64 | topics, graph, overlay, pgr_build | 1.000 | 204 | 1.2 | 0.00 | 0.010 | 5 |
| S5_primary | multihoprag | 0.00 | topics, graph, overlay | 1.000 | 247 | 3 | 0.00 | 0.019 | 5 |
| S5_primary_norule | longmemeval | 0.64 | topics, graph, overlay, pgr_build | 0.000 | 0 | 0 | 0.00 | 0.016 | 5 |
| S5_primary_norule | multihoprag | 0.00 | topics, graph, overlay | n/a | n/a | n/a | 0.00 | n/a | 0 |
| chandan_full | longmemeval | 0.64 | pgr_build | 3.000 | 223.2 | 66.6 | 0.00 | 2.498 | 5 |
| chandan_full_uncut | longmemeval | 0.64 | pgr_build | n/a | n/a | n/a | 0.00 | n/a | 0 |
| chandan_live | longmemeval | 0.64 | pgr_build | 3.000 | 223.2 | 66.6 | 0.00 | 2.498 | 5 |
| closed_book | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 5 |
| closed_book | multihoprag | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 5 |
| oracle_full | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 5 |
| oracle_full | multihoprag | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 5 |
| ours_cheap | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.031 | 5 |
| ours_cheap | multihoprag | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.020 | 5 |
| ours_cheap_norule | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.016 | 5 |
| ours_sentence_norule | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.015 | 5 |
| graphiti | longmemeval | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_live | multihoprag | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | multihoprag | absent | absent | absent | absent | absent | absent | absent | absent |

Index-time builds as metered:

| corpus | component | USD | calls | tokens in | tokens out | seconds | charged to |
|---|---|---|---|---|---|---|---|
| longmemeval | topics | 0.00 | 0 | 0 | 0 | 17.9 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| longmemeval | graph | 0.00 | 0 | 0 | 0 | 48.2 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| longmemeval | overlay | 0.01 | 64 | 42,901 | 6,495 | 19.5 | S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R2, S5_overlay_P0 |
| longmemeval | pgr_build | 0.64 | 1,405 | 2,119,450 | 971,314 | 1,305.7 | (default charge list) |
| multihoprag | topics | 0.00 | 0 | 0 | 0 | 9.0 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| multihoprag | graph | 0.00 | 0 | 0 | 0 | 12.5 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| multihoprag | overlay | 0.00 | 31 | 18,632 | 3,006 | 10.9 | S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R2, S5_overlay_P0 |

Answering:

```
{
 "longmemeval": {
  "answering_cap_usd": 0.34,
  "judging_stopped": null,
  "n_records": 175,
  "partial": {},
  "populations": {
   "chandan_own": 5,
   "chandan_subset": null,
   "per_arm": {
    "S2_lazy": 5,
    "S4_static": 5,
    "S5_noPGR": 5,
    "S5_overlay_P0": 5,
    "S5_overlay_R0": 5,
    "S5_overlay_R2": 5,
    "S5_planner_oracle": 5,
    "S5_planner_rules": 5,
    "S5_primary": 5,
    "S5_primary_norule": 5,
    "chandan_full": 5,
    "chandan_live": 5,
    "closed_book": 5,
    "oracle_full": 5,
    "ours_cheap": 5,
    "ours_cheap_norule": 5,
    "ours_sentence_norule": 5
   },
   "reader_a": 5,
   "reader_b": 5,
   "second_build_subset": null
  },
  "readers": [
   "reader_a"
  ]
 },
 "multihoprag": {
  "answering_cap_usd": 0.34,
  "judging_stopped": null,
  "n_records": 120,
  "partial": {},
  "populations": {
   "chandan_own": 5,
   "chandan_subset": null,
   "per_arm": {
    "S2_lazy": 5,
    "S4_static": 5,
    "S5_noPGR": 5,
    "S5_overlay_P0": 5,
    "S5_overlay_R0": 5,
    "S5_overlay_R2": 5,
    "S5_planner_oracle": 5,
    "S5_planner_rules": 5,
    "S5_primary": 5,
    "closed_book": 5,
    "oracle_full": 5,
    "ours_cheap": 5
   },
   "reader_a": 5,
   "reader_b": 5,
   "second_build_subset": null
  },
  "readers": [
   "reader_a"
  ]
 }
}
```

Proxy log cross-check:

```
{
 "pgr_build_lme_job_tags": [
  "pgr-lme-full-s0",
  "pgr-lme-full-s1",
  "pgr-lme-full-s2",
  "pgr-lme-full-s3"
 ]
}
```

Caps:

```
{
 "index_lme": 0.15,
 "index_mhrag": 0.15,
 "qa_lme": 0.4,
 "qa_mhrag": 0.4,
 "retrieve_lme": 0.15,
 "retrieve_mhrag": 0.15
}
```

Meter totals:

```
{
 "index_lme": {
  "by_tier": {
   "vertex-flash": {
    "calls": 64,
    "tokens_in": 42901,
    "tokens_out": 6495,
    "usd": 0.006888
   }
  },
  "prices_as_of": "2026-08-19",
  "prices_usd_per_mtok": {
   "azure-gpt54": {
    "in": 1.25,
    "out": 10.0
   },
   "encoder-local": {
    "in": 0.0,
    "out": 0.0
   },
   "generative-cheap": {
    "in": 0.1,
    "out": 0.4
   },
   "generative-frontier": {
    "in": 3.0,
    "out": 15.0
   },
   "generative-mid": {
    "in": 0.4,
    "out": 1.6
   },
   "vertex-flash": {
    "in": 0.1,
    "out": 0.4
   },
   "vertex-partner-claude": {
    "in": 3.0,
    "out": 15.0
   },
   "vertex-partner-llama": {
    "in": 0.25,
    "out": 0.75
   },
   "vertex-pro": {
    "in": 1.25,
    "out": 5.0
   }
  },
  "total_calls": 64,
  "total_usd": 0.006888
 },
 "index_mhrag": {
  "by_tier": {
   "vertex-flash": {
    "calls": 31,
    "tokens_in": 18632,
    "tokens_out": 3006,
    "usd": 0.003066
   }
  },
  "prices_as_of": "2026-08-19",
  "prices_usd_per_mtok": {
   "azure-gpt54": {
    "in": 1.25,
    "out": 10.0
   },
   "encoder-local": {
    "in": 0.0,
    "out": 0.0
   },
   "generative-cheap": {
    "in": 0.1,
    "out": 0.4
   },
   "generative-frontier": {
    "in": 3.0,
    "out": 15.0
   },
   "generative-mid": {
    "in": 0.4,
    "out": 1.6
   },
   "vertex-flash": {
    "in": 0.1,
    "out": 0.4
   },
   "vertex-partner-claude": {
    "in": 3.0,
    "out": 15.0
   },
   "vertex-partner-llama": {
    "in": 0.25,
    "out": 0.75
   },
   "vertex-pro": {
    "in": 1.25,
    "out": 5.0
   }
  },
  "total_calls": 31,
  "total_usd": 0.003066
 },
 "qa_lme": {
  "by_tier": {
   "azure-gpt54": {
    "calls": 9,
    "tokens_in": 2629,
    "tokens_out": 45,
    "usd": 0.003736
   },
   "gemini-2.5-flash-lite": {
    "calls": 196,
    "tokens_in": 843255,
    "tokens_out": 8256,
    "usd": 0.087628
   }
  },
  "prices_as_of": "2026-08-19",
  "prices_usd_per_mtok": {
   "azure-gpt54": {
    "in": 1.25,
    "out": 10.0
   },
   "encoder-local": {
    "in": 0.0,
    "out": 0.0
   },
   "gemini-2.5-flash": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-2.5-flash-lite": {
    "in": 0.1,
    "out": 0.4
   },
   "gemini-2.5-pro": {
    "in": 1.25,
    "out": 10.0
   },
   "gemini-3.5-flash": {
    "in": 1.5,
    "out": 9.0
   },
   "gemini-3.5-flash-lite": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-3.6-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.7-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.8-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-embedding-001": {
    "in": 0.15,
    "out": 0.0
   },
   "generative-cheap": {
    "in": 0.1,
    "out": 0.4
   },
   "generative-frontier": {
    "in": 3.0,
    "out": 15.0
   },
   "generative-mid": {
    "in": 0.4,
    "out": 1.6
   },
   "text-embedding-005": {
    "in": 0.025,
    "out": 0.0
   },
   "vertex-flash": {
    "in": 0.1,
    "out": 0.4
   },
   "vertex-partner-claude": {
    "in": 3.0,
    "out": 15.0
   },
   "vertex-partner-llama": {
    "in": 0.25,
    "out": 0.75
   },
   "vertex-pro": {
    "in": 1.25,
    "out": 5.0
   }
  },
  "total_calls": 205,
  "total_usd": 0.091364
 },
 "qa_mhrag": {
  "by_tier": {
   "azure-gpt54": {
    "calls": 9,
    "tokens_in": 2629,
    "tokens_out": 45,
    "usd": 0.003736
   },
   "gemini-2.5-flash-lite": {
    "calls": 196,
    "tokens_in": 843255,
    "tokens_out": 8256,
    "usd": 0.087628
   }
  },
  "prices_as_of": "2026-08-19",
  "prices_usd_per_mtok": {
   "azure-gpt54": {
    "in": 1.25,
    "out": 10.0
   },
   "encoder-local": {
    "in": 0.0,
    "out": 0.0
   },
   "gemini-2.5-flash": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-2.5-flash-lite": {
    "in": 0.1,
    "out": 0.4
   },
   "gemini-2.5-pro": {
    "in": 1.25,
    "out": 10.0
   },
   "gemini-3.5-flash": {
    "in": 1.5,
    "out": 9.0
   },
   "gemini-3.5-flash-lite": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-3.6-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.7-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.8-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-embedding-001": {
    "in": 0.15,
    "out": 0.0
   },
   "generative-cheap": {
    "in": 0.1,
    "out": 0.4
   },
   "generative-frontier": {
    "in": 3.0,
    "out": 15.0
   },
   "generative-mid": {
    "in": 0.4,
    "out": 1.6
   },
   "text-embedding-005": {
    "in": 0.025,
    "out": 0.0
   },
   "vertex-flash": {
    "in": 0.1,
    "out": 0.4
   },
   "vertex-partner-claude": {
    "in": 3.0,
    "out": 15.0
   },
   "vertex-partner-llama": {
    "in": 0.25,
    "out": 0.75
   },
   "vertex-pro": {
    "in": 1.25,
    "out": 5.0
   }
  },
  "total_calls": 205,
  "total_usd": 0.091364
 },
 "retrieve_lme": {
  "by_tier": {
   "vertex-flash": {
    "calls": 44,
    "tokens_in": 4484,
    "tokens_out": 86,
    "usd": 0.000483
   }
  },
  "prices_as_of": "2026-08-19",
  "prices_usd_per_mtok": {
   "azure-gpt54": {
    "in": 1.25,
    "out": 10.0
   },
   "encoder-local": {
    "in": 0.0,
    "out": 0.0
   },
   "gemini-2.5-flash": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-2.5-flash-lite": {
    "in": 0.1,
    "out": 0.4
   },
   "gemini-2.5-pro": {
    "in": 1.25,
    "out": 10.0
   },
   "gemini-3.5-flash": {
    "in": 1.5,
    "out": 9.0
   },
   "gemini-3.5-flash-lite": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-3.6-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.7-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.8-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-embedding-001": {
    "in": 0.15,
    "out": 0.0
   },
   "generative-cheap": {
    "in": 0.1,
    "out": 0.4
   },
   "generative-frontier": {
    "in": 3.0,
    "out": 15.0
   },
   "generative-mid": {
    "in": 0.4,
    "out": 1.6
   },
   "text-embedding-005": {
    "in": 0.025,
    "out": 0.0
   },
   "vertex-flash": {
    "in": 0.1,
    "out": 0.4
   },
   "vertex-partner-claude": {
    "in": 3.0,
    "out": 15.0
   },
   "vertex-partner-llama": {
    "in": 0.25,
    "out": 0.75
   },
   "vertex-pro": {
    "in": 1.25,
    "out": 5.0
   }
  },
  "total_calls": 44,
  "total_usd": 0.000483
 },
 "retrieve_mhrag": {
  "by_tier": {
   "vertex-flash": {
    "calls": 26,
    "tokens_in": 3380,
    "tokens_out": 52,
    "usd": 0.000359
   }
  },
  "prices_as_of": "2026-08-19",
  "prices_usd_per_mtok": {
   "azure-gpt54": {
    "in": 1.25,
    "out": 10.0
   },
   "encoder-local": {
    "in": 0.0,
    "out": 0.0
   },
   "gemini-2.5-flash": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-2.5-flash-lite": {
    "in": 0.1,
    "out": 0.4
   },
   "gemini-2.5-pro": {
    "in": 1.25,
    "out": 10.0
   },
   "gemini-3.5-flash": {
    "in": 1.5,
    "out": 9.0
   },
   "gemini-3.5-flash-lite": {
    "in": 0.3,
    "out": 2.5
   },
   "gemini-3.6-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.7-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-3.8-flash": {
    "in": 0.75,
    "out": 3.75
   },
   "gemini-embedding-001": {
    "in": 0.15,
    "out": 0.0
   },
   "generative-cheap": {
    "in": 0.1,
    "out": 0.4
   },
   "generative-frontier": {
    "in": 3.0,
    "out": 15.0
   },
   "generative-mid": {
    "in": 0.4,
    "out": 1.6
   },
   "text-embedding-005": {
    "in": 0.025,
    "out": 0.0
   },
   "vertex-flash": {
    "in": 0.1,
    "out": 0.4
   },
   "vertex-partner-claude": {
    "in": 3.0,
    "out": 15.0
   },
   "vertex-partner-llama": {
    "in": 0.25,
    "out": 0.75
   },
   "vertex-pro": {
    "in": 1.25,
    "out": 5.0
   }
  },
  "total_calls": 26,
  "total_usd": 0.000359
 }
}
```

## MultiHop-RAG fact location

Non-null queries 5; all facts located for 5 (1.000). Facts 11, located 11 (1.000), fallback 0, straddling a chunk boundary 0, unresolved 0.

## Missing outputs, refused ids and partial runs

| arm | corpus | budget | questions with no output (scored 0, wrong) |
|---|---|---|---|
| ours_cheap | longmemeval | 4,000 | 0 |
| ours_cheap | longmemeval | 8,000 | 0 |
| ours_cheap | multihoprag | 4,000 | 0 |
| ours_cheap | multihoprag | 8,000 | 0 |
| ours_cheap_norule | longmemeval | 4,000 | 0 |
| ours_cheap_norule | longmemeval | 8,000 | 0 |
| ours_sentence_norule | longmemeval | 4,000 | 0 |
| ours_sentence_norule | longmemeval | 8,000 | 0 |
| S4_static | longmemeval | 4,000 | 0 |
| S4_static | longmemeval | 8,000 | 0 |
| S4_static | multihoprag | 4,000 | 0 |
| S4_static | multihoprag | 8,000 | 0 |
| S5_primary | longmemeval | 4,000 | 0 |
| S5_primary | longmemeval | 8,000 | 0 |
| S5_primary | multihoprag | 4,000 | 0 |
| S5_primary | multihoprag | 8,000 | 0 |
| S5_primary_norule | longmemeval | 4,000 | 0 |
| S5_primary_norule | longmemeval | 8,000 | 0 |
| S5_noPGR | longmemeval | 4,000 | 0 |
| S5_noPGR | longmemeval | 8,000 | 0 |
| S5_noPGR | multihoprag | 4,000 | 0 |
| S5_noPGR | multihoprag | 8,000 | 0 |
| S5_planner_rules | longmemeval | 4,000 | 0 |
| S5_planner_rules | longmemeval | 8,000 | 0 |
| S5_planner_rules | multihoprag | 4,000 | 0 |
| S5_planner_rules | multihoprag | 8,000 | 0 |
| S5_planner_oracle | longmemeval | 4,000 | 0 |
| S5_planner_oracle | longmemeval | 8,000 | 0 |
| S5_planner_oracle | multihoprag | 4,000 | 0 |
| S5_planner_oracle | multihoprag | 8,000 | 0 |
| S5_overlay_R0 | longmemeval | 4,000 | 0 |
| S5_overlay_R0 | longmemeval | 8,000 | 0 |
| S5_overlay_R0 | multihoprag | 4,000 | 0 |
| S5_overlay_R0 | multihoprag | 8,000 | 0 |
| S5_overlay_R2 | longmemeval | 4,000 | 0 |
| S5_overlay_R2 | longmemeval | 8,000 | 0 |
| S5_overlay_R2 | multihoprag | 4,000 | 0 |
| S5_overlay_R2 | multihoprag | 8,000 | 0 |
| S5_overlay_P0 | longmemeval | 4,000 | 0 |
| S5_overlay_P0 | longmemeval | 8,000 | 0 |
| S5_overlay_P0 | multihoprag | 4,000 | 0 |
| S5_overlay_P0 | multihoprag | 8,000 | 0 |
| S2_lazy | longmemeval | 4,000 | 0 |
| S2_lazy | longmemeval | 8,000 | 0 |
| S2_lazy | multihoprag | 4,000 | 0 |
| S2_lazy | multihoprag | 8,000 | 0 |
| chandan_live | longmemeval | 4,000 | 0 |
| chandan_live | longmemeval | 8,000 | 0 |
| chandan_full | longmemeval | 4,000 | 0 |
| chandan_full | longmemeval | 8,000 | 0 |

No refused ids.

Partial retrieval runs (a cap stopped the stage): none.

Answering withdrawn under the answering cap (section 5 order; only the Family C tests under that reader are labelled partial): none.

## Disclosures (sections 12 and 13)

- No component is tuned. No index is rebuilt except the conditional second post-graph-rag build.
- One encoder (all-MiniLM-L6-v2) and one seed (13) for everything of ours.
- No question, answer or evidence flag reaches the overlay prompt, the topic model, the graph, or the planner table. The oracle planner is a bound, never a system.
- The planner table, the planner prompt definitions and the rules patterns were written with the e5 by-type results on all 500 LongMemEval questions known to the author.
- The e5 speaker rule was written by someone who knew the question shapes. It is kept in our arms as a declared component. It is not applied to the competitors' units: his chunks span both roles and cutting them is not as shipped, and Graphiti facts carry no role. Its share is shown by the no-rule rows.
- The post-graph-rag run reproduces the configuration of the shipped package version recorded in pgr.md, with the models set by section 13. In his frozen configuration supersession never fires (no exclusive predicate groups, contradiction detection off), so the later document closes earlier fact mechanism is not part of his LongMemEval result.
- Reader A wording: the READER_RULES say the excerpts start with who spoke. His chunks and Graphiti facts do not name a speaker in that form. This mismatch is noted, not fixed.
- Model policy (section 13): one study model, gemini-2.5-flash-lite, for every model call in Part 1 except Reader B and the second judge (Azure gpt-5.4, applied to every arm alike). Embeddings are each system's own default: gemini-embedding-001 at 1,536 dimensions for post-graph-rag, 3,072 for Graphiti, MiniLM for ours.
- This differs from the models post-graph-rag's README run used. The calibration row on CHANDAN_CAL_18 (indexed with gemini-3.7-flash, answered with gemini-3.6-flash) is reported beside the study-model row and is never in a test.
- Every call in Part 1 sets an output limit of at least 64 tokens.
- Zep's paper does not name its reranker. The search here is the library's COMBINED_HYBRID_SEARCH_RRF recipe with no cross-encoder.
- Graphiti is scored at session level only: a fact covers every session whose episode it cites, and an ENTITY summary line covers no session.
- Added after reading the design: section 5 fixes the truncation rule (a unit larger than the remaining budget is truncated to fit, flagged and counted) under chandan_live only. The rendering here applies that rule to every arm alike, ours included, so a truncated tail unit can cover an evidence turn under the half rule in any arm.
- Design-text error found at implementation: section 5 describes Graphiti's COMBINED_HYBRID_SEARCH_RRF recipe as BM25, cosine and BFS. The recipe as shipped in graphiti-core 0.30.1 has no BFS method in any scope (search_config_recipes.py); it is used as shipped, BM25 and cosine under RRF.
- Graph hub rule, pronouns (design gap): a phrase is a pronoun when its chunk root is PRON in more than half of its occurrences (majority vote over occurrences).
- Graph hub rule order (design gap): the hub rule runs before graphrag's counting and pruning: co-occurrence, PMI and the node and edge rules see the surviving phrases only, and graphrag's one ego node is the highest-degree content phrase left after the hub rule.
- Graph phrase normalisation: every article token (a, an, the) stripped from the lemma.
- graphrag's ego-node rule removed one content phrase after the hub rule: 'ability'.
- The calibration row (chandan_live_cal, CHANDAN_CAL_18 with his own models) was not run.
- This run is tagged smoke and is a sample smoke, not a study result.
- Arms absent on longmemeval (no output, no export found): graphiti.
- Arms absent on multihoprag (no output, no export found): chandan_live, chandan_full.

## Published numbers beside chandan_live (section 12)

| number | value | source | judge | models | note |
|---|---|---|---|---|---|
| chandan_readme_94.0 | 0.940 | post-graph-rag README, oracle variant, 499 questions | his own three-model majority panel, two repeats per question | indexed with gemini-3.7-flash, answered with gemini-3.6-flash, gemini-embedding-001 at 1536 dimensions | marked not reportable in his result file |
| chandan_official_78.2 | 0.782 | post-graph-rag evaluation/longmemeval/official_gpt4o_g36.json, n 499 | gpt-4o with the official LongMemEval prompts | answered with gemini-3.6-flash | the official protocol; gpt-4o reading the same retrieval scored 0.713 |
| chandan_paper_v2_85.8 | 0.858 | post-graph-rag paper v2 (reader_sweep_validity.json) | his own judge panel | gemini-3.6-flash, with the 4,000 token context budget of that version | the budget was made unlimited in 1.11.1, which gives the 94.0 row |
| zep_71.2 | 0.712 | Zep paper, LongMemEval_S | the official LongMemEval prompts | gpt-4o reader | the S variant, the one Part 1 runs on; his 94.0 and 85.8 are on the oracle variant |

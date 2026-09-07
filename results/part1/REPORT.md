# Part 1: the head-to-head on LongMemEval_S and MultiHop-RAG

Generated 2026-09-06T22:25:06+00:00 from results/part1/metrics.json. Design: docs/PART1-DESIGN.md version 4. Every number below is read from that file.

Subsets file sha256: 0e2440828ec0e2f2568bb90742b21c684b73b2cc4db85738879703a5311f5082.

Commits recorded in the report:

- head: 9e454cc
- subsets_sha256_file: 0e2440828ec0e2f2568bb90742b21c684b73b2cc4db85738879703a5311f5082

Graphiti status: dropped. T2 is recorded as not run.

## The pass rule

Part 1 passed: **no**.

The rule: T1 shown under D1, T8a positive and significant after Holm within Family A, T2 passes under D2 or is recorded as not run, and T7 holds. T8b is reported.

| quantity | value |
|---|---|
| T1 label (D1) | not run |
| T1 delta | n/a |
| T1 95 percent CI | n/a |
| T1 significant after Holm | n/a |
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
| T8a note | S5_primary run without post-graph-rag tables |
| T8b note | S5_primary run without post-graph-rag tables |

D2 reading of T2: not run (Graphiti status dropped).

Second post-graph-rag build (section 9): first build cost so far USD 60.63 (longmemeval: partial build snapshot, 467 of 500 spaces, build in progress; multihoprag: partial build snapshot, 0 of 1 spaces, build stopped at its cap); second-build rule: not evaluated, build incomplete; second build ran: no.

## Setup

- tag: 
- encoder: sentence-transformers/all-MiniLM-L6-v2
- stages:
  - lme: index topics fitted, graph built, relation_vectors 116 spaces encoded, 0 present, 500 wanted, overlay generated, sample False, n_questions 500, containers 19829, arms_skipped , chosen_variant , exports_present chandan 0, graphiti 0, cal 0, second 0, topics n_units 199641, n_topics 3421, outlier_share 0.3795613125560381, largest_topic_share 0.005029027103651054, graph plain n_communities 1670, largest_community_share_nodes 0.0913023234494455, largest_community_share_units 0.10969859146646278, graph_density 3.1974720116958864e-05, topic_entropy mean_bits 2.771227610677477, member_weighted_bits 9.271119151349733, n_communities_with_topics 1670, topic n_communities 1685, largest_community_share_nodes 0.0931261504464275, largest_community_share_units 0.11148998735554444, graph_density 3.1974720116958864e-05, topic_entropy mean_bits 2.7812383730601504, member_weighted_bits 9.140311580761013, n_communities_with_topics 1685, overlay n_candidate_pairs 12830, n_flagged 2566, n_parse_ok 2566, links R0 0, R2 2566, R3 570, P0 570
  - mhrag: index topics fitted, graph built, relation_vectors 0 spaces encoded, 0 present, 1 wanted, overlay generated, sample False, n_questions 2556, containers 609, arms_skipped , chosen_variant , exports_present chandan 0, graphiti 0, cal 0, second 0, topics n_units 3376, n_topics 95, outlier_share 0.15136255924170616, largest_topic_share 0.07316350710900474, graph plain n_communities 251, largest_community_share_nodes 0.08065564635958396, largest_community_share_units 0.11705352698972897, graph_density 0.0002940287606719218, topic_entropy mean_bits 3.4369701333111, member_weighted_bits 5.551516258641589, n_communities_with_topics 251, topic n_communities 256, largest_community_share_nodes 0.09040676077265973, largest_community_share_units 0.1301977286095652, graph_density 0.0002940287606719218, topic_entropy mean_bits 3.4140820354158494, member_weighted_bits 5.454004704366372, n_communities_with_topics 256, overlay n_candidate_pairs 1454, n_flagged 291, n_parse_ok 289, links R0 0, R2 289, R3 84, P0 84
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
| ours_cheap | 0.938 | 0.949 | 0.987 | 0.863 | 0.708 | 0.938 | 3,996.3 | 0.000 | 62.2 | 470 | 0 |
| ours_cheap_norule | 0.800 | 0.962 | 0.934 | 0.825 | 0.632 | 0.945 | 3,999.8 | 0.000 | 100 | 470 | 0 |
| ours_sentence_norule | 0.823 | 0.966 | 0.945 | 0.848 | 0.689 | 0.959 | 3,999.8 | 0.000 | 100 | 470 | 0 |
| S5_noPGR | 0.945 | 0.953 | 0.983 | 0.858 | 0.645 | 0.944 | 3,996.3 | 0.000 | 55.0 | 470 | 0 |
| S4_static | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary_norule | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_rules | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_oracle | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R2 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_P0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S2_lazy | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Absent rows: S4_static (not run in the retrieve pass); S5_primary (not run in the retrieve pass); S5_primary_norule (not run in the retrieve pass); S5_planner_rules (not run in the retrieve pass); S5_planner_oracle (not run in the retrieve pass); S5_overlay_R0 (not run in the retrieve pass); S5_overlay_R2 (not run in the retrieve pass); S5_overlay_P0 (not run in the retrieve pass); S2_lazy (not run in the retrieve pass); chandan_live (export present, not run in the retrieve pass); chandan_full (export present, not run in the retrieve pass); graphiti (no export).

Evidence turns rendered truncated, summed over questions (section 3): truncated to fit the budget or cut at the 2,000-character limit, and the cut ones alone:

| arm | truncated evidence turns | of which cut at 2,000 characters |
|---|---|---|
| ours_cheap | 8 | 8 |
| ours_cheap_norule | 11 | 11 |
| ours_sentence_norule | 11 | 11 |
| S5_noPGR | 8 | 8 |

### LongMemEval, answerable questions, budget 8,000 tokens

| arm | JointRecall | candidate JR | session JR | turn R@10 | nDCG@10 | session R@5 | rendered tokens | duplicate share | candidate list size | n | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_cheap | 0.957 | 0.949 | 0.994 | 0.863 | 0.708 | 0.938 | 6,199.9 | 0.000 | 62.2 | 470 | 0 |
| ours_cheap_norule | 0.868 | 0.962 | 0.970 | 0.825 | 0.632 | 0.945 | 7,999.7 | 0.000 | 100 | 470 | 0 |
| ours_sentence_norule | 0.879 | 0.966 | 0.977 | 0.848 | 0.689 | 0.959 | 7,999.9 | 0.000 | 100 | 470 | 0 |
| S5_noPGR | 0.964 | 0.953 | 0.996 | 0.858 | 0.645 | 0.944 | 6,800.9 | 0.000 | 55.0 | 470 | 0 |
| S4_static | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary_norule | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_rules | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_oracle | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R2 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_P0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S2_lazy | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Absent rows: S4_static (not run in the retrieve pass); S5_primary (not run in the retrieve pass); S5_primary_norule (not run in the retrieve pass); S5_planner_rules (not run in the retrieve pass); S5_planner_oracle (not run in the retrieve pass); S5_overlay_R0 (not run in the retrieve pass); S5_overlay_R2 (not run in the retrieve pass); S5_overlay_P0 (not run in the retrieve pass); S2_lazy (not run in the retrieve pass); chandan_live (export present, not run in the retrieve pass); chandan_full (export present, not run in the retrieve pass); graphiti (no export).

Evidence turns rendered truncated, summed over questions (section 3): truncated to fit the budget or cut at the 2,000-character limit, and the cut ones alone:

| arm | truncated evidence turns | of which cut at 2,000 characters |
|---|---|---|
| ours_cheap | 8 | 8 |
| ours_cheap_norule | 11 | 11 |
| ours_sentence_norule | 11 | 11 |
| S5_noPGR | 8 | 8 |

### MultiHop-RAG, non-null queries, budget 4,000 tokens

| arm | fact JR (all located) | fact JR (all non-null) | document JR | candidate fact JR | candidate document JR | rendered tokens | duplicate share | candidate list size | n | missing | n all located | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_cheap | 0.255 | 0.255 | 0.259 | 0.874 | 0.874 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 |  |
| S4_static | 0.185 | 0.185 | 0.188 | 0.897 | 0.898 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_primary | 0.136 | 0.136 | 0.140 | 0.844 | 0.845 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_noPGR | 0.136 | 0.136 | 0.140 | 0.844 | 0.845 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 |  |
| S5_planner_rules | 0.205 | 0.205 | 0.209 | 0.841 | 0.842 | 3,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_planner_oracle | 0.151 | 0.151 | 0.154 | 0.852 | 0.852 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_overlay_R0 | 0.153 | 0.153 | 0.156 | 0.872 | 0.873 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_overlay_R2 | 0.145 | 0.145 | 0.148 | 0.784 | 0.785 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_overlay_P0 | 0.135 | 0.135 | 0.138 | 0.745 | 0.746 | 3,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S2_lazy | 0.205 | 0.205 | 0.210 | 0.966 | 0.967 | 3,999.9 | 0.000 | 218.2 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Absent rows: chandan_live (no export); chandan_full (no export).

### MultiHop-RAG, non-null queries, budget 8,000 tokens

| arm | fact JR (all located) | fact JR (all non-null) | document JR | candidate fact JR | candidate document JR | rendered tokens | duplicate share | candidate list size | n | missing | n all located | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_cheap | 0.420 | 0.420 | 0.422 | 0.874 | 0.874 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 |  |
| S4_static | 0.361 | 0.361 | 0.362 | 0.897 | 0.898 | 7,999.9 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_primary | 0.283 | 0.283 | 0.288 | 0.844 | 0.845 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_noPGR | 0.283 | 0.283 | 0.288 | 0.844 | 0.845 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 |  |
| S5_planner_rules | 0.345 | 0.345 | 0.348 | 0.841 | 0.842 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_planner_oracle | 0.306 | 0.306 | 0.310 | 0.852 | 0.852 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_overlay_R0 | 0.298 | 0.298 | 0.301 | 0.872 | 0.873 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_overlay_R2 | 0.290 | 0.290 | 0.295 | 0.784 | 0.785 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S5_overlay_P0 | 0.271 | 0.271 | 0.274 | 0.745 | 0.746 | 7,999.8 | 0.000 | 100 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| S2_lazy | 0.369 | 0.369 | 0.371 | 0.966 | 0.967 | 7,999.8 | 0.000 | 218.2 | 2,255 | 0 | 2,255 | run without post-graph-rag tables |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Absent rows: chandan_live (no export); chandan_full (no export).

## By question type

### longmemeval, JointRecall by type, budget 4,000 tokens

| type | n | ours_cheap | ours_cheap_norule | ours_sentence_norule | S5_noPGR |
|---|---|---|---|---|---|
| knowledge-update | 72 | 1.000 | 0.958 | 0.958 | 1.000 |
| multi-session | 121 | 0.926 | 0.579 | 0.645 | 0.934 |
| single-session-assistant | 56 | 0.929 | 0.982 | 0.982 | 0.911 |
| single-session-preference | 30 | 0.800 | 0.700 | 0.733 | 0.800 |
| single-session-user | 64 | 0.969 | 0.984 | 0.984 | 0.969 |
| temporal-reasoning | 127 | 0.937 | 0.772 | 0.787 | 0.961 |

### longmemeval, JointRecall by type, budget 8,000 tokens

| type | n | ours_cheap | ours_cheap_norule | ours_sentence_norule | S5_noPGR |
|---|---|---|---|---|---|
| knowledge-update | 72 | 1.000 | 0.986 | 0.972 | 1.000 |
| multi-session | 121 | 0.959 | 0.702 | 0.719 | 0.975 |
| single-session-assistant | 56 | 0.946 | 1.000 | 1.000 | 0.929 |
| single-session-preference | 30 | 0.867 | 0.833 | 0.867 | 0.867 |
| single-session-user | 64 | 0.969 | 1.000 | 1.000 | 0.969 |
| temporal-reasoning | 127 | 0.953 | 0.843 | 0.866 | 0.969 |

### multihoprag, fact JR (all located) by type, budget 4,000 tokens

| type | n | ours_cheap | S4_static | S5_primary | S5_noPGR | S5_planner_rules | S5_planner_oracle | S5_overlay_R0 | S5_overlay_R2 | S5_overlay_P0 | S2_lazy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison_query | 856 | 0.386 | 0.276 | 0.171 | 0.171 | 0.257 | 0.167 | 0.216 | 0.193 | 0.164 | 0.306 |
| inference_query | 816 | 0.099 | 0.055 | 0.075 | 0.075 | 0.088 | 0.076 | 0.054 | 0.074 | 0.072 | 0.069 |
| temporal_query | 583 | 0.283 | 0.235 | 0.172 | 0.172 | 0.293 | 0.232 | 0.199 | 0.177 | 0.180 | 0.249 |

### multihoprag, fact JR (all located) by type, budget 8,000 tokens

| type | n | ours_cheap | S4_static | S5_primary | S5_noPGR | S5_planner_rules | S5_planner_oracle | S5_overlay_R0 | S5_overlay_R2 | S5_overlay_P0 | S2_lazy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison_query | 856 | 0.574 | 0.499 | 0.343 | 0.343 | 0.405 | 0.343 | 0.383 | 0.361 | 0.325 | 0.519 |
| inference_query | 816 | 0.232 | 0.167 | 0.176 | 0.176 | 0.196 | 0.180 | 0.162 | 0.179 | 0.162 | 0.174 |
| temporal_query | 583 | 0.458 | 0.429 | 0.345 | 0.345 | 0.463 | 0.427 | 0.365 | 0.343 | 0.345 | 0.424 |

## Pre-declared tests

Paired at the question level, the bench compare (10,000 permutations, percentile bootstrap CI), alpha 0.05, budget 4,000 tokens. Holm within each family. Families B and C never feed the pass rule. A test on a partial run is labelled and left out of the pass rule.

Populations: LongMemEval answerable 470, GRAPHITI_150 150, MultiHop-RAG all located 2,255, LOCAL_120 120, LongMemEval all 500.

### Family A, the gate

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | S5_primary vs chandan_live | joint_recall | not run: no output from S5_primary, chandan_live on longmemeval |  |  |  |  |  |  |  |
| T2 | S5_primary vs graphiti | session_joint_recall | not run: no output from S5_primary, graphiti on longmemeval |  |  |  |  |  |  |  |
| T8a | S5_primary vs chandan_live | fact_joint_recall | not run: no output from chandan_live on multihoprag |  |  |  |  |  |  |  |
| T8b | S5_primary vs ours_cheap | fact_joint_recall | 2,255 (refused 0, missing 0/0) | 0.136 vs 0.255 | -0.119 | [-0.137, -0.101] | 0.0001 | 111/1,764/380 | 0.0001, significant | S5_primary run without post-graph-rag tables |

D1 reading of T1: not run.

### Family B, the mechanism

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T3 | S5_primary vs ours_cheap | joint_recall | not run: no output from S5_primary on longmemeval |  |  |  |  |  |  |  |
| T4 | S5_primary vs S4_static | joint_recall | not run: no output from S5_primary, S4_static on longmemeval |  |  |  |  |  |  |  |
| T5a | S5_primary vs S5_overlay_R0 | joint_recall | not run: no output from S5_primary, S5_overlay_R0 on longmemeval |  |  |  |  |  |  |  |
| T5b | S5_primary vs S5_overlay_P0 | joint_recall | not run: no output from S5_primary, S5_overlay_P0 on longmemeval |  |  |  |  |  |  |  |
| T6 | S4_static vs ours_cheap | joint_recall | not run: no output from S4_static on longmemeval |  |  |  |  |  |  |  |

Reported for the T5 reading, not under Holm:

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T5_P0_minus_R0 | S5_overlay_P0 vs S5_overlay_R0 | joint_recall | not run: no output from S5_overlay_P0, S5_overlay_R0 on longmemeval |  |  |  |  |  |  |  |

T5 branch: **not run**.

| pair | delta | 95 percent CI |
|---|---|---|
| R3 minus R0 | n/a | n/a |
| R3 minus P0 | n/a | n/a |
| P0 minus R0 | n/a | n/a |

Questions whose rendered context differs at all between R0, R3 and P0: 0. Tolerance 0.01.

### Family C, answers (exact McNemar on the discordant pairs, primary judge)

| test | arms (a vs b) | reader | population | n | accuracy a vs b | delta | wins/losses | p | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | S5_primary vs chandan_live | reader_a | LongMemEval all | not run: no output from S5_primary, chandan_live on longmemeval |  |  |  |  |  |  |
| C2 | S5_primary vs chandan_live | reader_b | LongMemEval all | not run: no output from S5_primary, chandan_live on longmemeval |  |  |  |  |  |  |
| C3 | S5_primary vs graphiti | reader_a | GRAPHITI_150 | not run: graphiti dropped: recorded as not run |  |  |  |  |  |  |
| C4 | S5_primary vs graphiti | reader_b | GRAPHITI_150 | not run: graphiti dropped: recorded as not run |  |  |  |  |  |  |
| C5 | S5_primary vs chandan_live | reader_a | MHRAG_ANSWER | not run: no output from chandan_live on multihoprag |  |  |  |  |  |  |
| C6 | S5_primary vs chandan_live | reader_b | READER_B_MHRAG | not run: no output from chandan_live on multihoprag |  |  |  |  |  |  |

### T7, non-inferiority on the local set

| test | arms (a vs b) | metric | n | means a vs b | delta | 95 percent CI | p | wins/ties/losses | Holm adjusted p, verdict | label |
|---|---|---|---|---|---|---|---|---|---|---|
| T7 | S5_primary vs ours_cheap | joint_recall | not run: no output from S5_primary on longmemeval |  |  |  |  |  |  |  |

T7 rule: losses minus wins is n/a; the limit is 3; not run. The CI is beside it and is not the rule.

## Predictions (section 10), written before any build

Each prediction with its fixed number and band, the measured value or paired delta with its 95 percent CI, and one label: consistent with, not confirmed (the point estimate is inside the band); not confirmed (outside the band, on the predicted side); contradicted (on the wrong side of zero, or outside a within band); untested (the quantity was not measured).

| prediction | statement | predicted | band | measured | 95 percent CI | n | label | note |
|---|---|---|---|---|---|---|---|---|
| T1_overall | T1 overall positive, about 0.02 (band 0.03 either side) | +0.02, band [-0.01, +0.05] | [-0.010, +0.050] | n/a | n/a | n/a | **untested** | no output from S5_primary, chandan_live on longmemeval |
| T1_multi_session | S5_primary beats chandan_live on JointRecall@4k by at least 0.05 on multi-session (half-width about 0.07) | at least +0.05, half-width 0.07 (band from -0.02) | [-0.020, open] | n/a | n/a | n/a | **untested** | no output from S5_primary, chandan_live on longmemeval |
| T1_temporal_reasoning | S5_primary beats chandan_live by at least 0.03 on temporal-reasoning (half-width about 0.08) | at least +0.03, half-width 0.08 (band from -0.05) | [-0.050, open] | n/a | n/a | n/a | **untested** | no output from S5_primary, chandan_live on longmemeval |
| T1_local_set | S5_primary within 0.02 of chandan_live on the local set | within 0.02 (band [-0.02, +0.02]) | [-0.020, +0.020] | n/a | n/a | n/a | **untested** | no output from S5_primary, chandan_live on longmemeval |
| S4_over_ours_cheap | S4_static beats ours_cheap by 0.02 to 0.04 | +0.02 to +0.04 | [+0.020, +0.040] | n/a | n/a | n/a | **untested** | no output from S4_static on longmemeval |
| S5_gain_from_planner | Most of the S5 over S4 gain comes from the planner on temporal and multi-session, not from the overlay | planner share above the overlay share | n/a | n/a | n/a | n/a | **untested** | no output from S5_primary, S4_static on longmemeval |
| T5_branch | T5 lands in the adds material branch | adds material | n/a | not run | n/a | n/a | **untested** | the T5 four-branch reading of section 9 |
| largest_community_share | The largest community share is above 50 percent without topic weighting and drops under it | plain above 0.50, topic-weighted below plain | n/a | plain_nodes +0.091, topic_nodes +0.093, plain_units +0.110, topic_units +0.111 | n/a | n/a | **contradicted** | share over the phrases of the pruned graph (nodes); the share over sub-units is beside it (design gap: the design does not say which share) |
| graphiti_gap | graphiti session-level JointRecall@4k is below S5_primary on GRAPHITI_150 by 0.02 to 0.06 (half-width about 0.04) | +0.02 to +0.06, half-width 0.04 (band [-0.02, +0.10]) | [-0.020, +0.100] | n/a | n/a | n/a | **untested** | Graphiti dropped or partial: recorded as untested |
| mhrag_comparison | MultiHop-RAG: S5_primary beats chandan_live on fact-level joint recall for comparison queries by at least 0.05 | at least +0.05 | [+0.050, open] | n/a | n/a | n/a | **untested** | no output from chandan_live on multihoprag |
| mhrag_inference | MultiHop-RAG: S5_primary beats chandan_live on fact-level joint recall for inference queries by at least 0.05 | at least +0.05 | [+0.050, open] | n/a | n/a | n/a | **untested** | no output from chandan_live on multihoprag |
| closed_book_floor_comparison_query | closed_book accuracy on answerable MultiHop-RAG queries exceeds the per-type majority-class rate by at least 0.10 (comparison_query) | at least +0.10 | [+0.100, open] | +0.087 | n/a | 150 | **not confirmed** | accuracy 0.620, majority-class rate 0.533 (commonest gold answer 'yes') |
| closed_book_floor_inference_query | closed_book accuracy on answerable MultiHop-RAG queries exceeds the per-type majority-class rate by at least 0.10 (inference_query) | at least +0.10 | [+0.100, open] | +0.527 | n/a | 150 | **consistent with, not confirmed** | accuracy 0.887, majority-class rate 0.360 (commonest gold answer 'sam bankman-fried') |
| closed_book_floor_temporal_query | closed_book accuracy on answerable MultiHop-RAG queries exceeds the per-type majority-class rate by at least 0.10 (temporal_query) | at least +0.10 | [+0.100, open] | +0.093 | n/a | 150 | **not confirmed** | accuracy 0.547, majority-class rate 0.453 (commonest gold answer 'no') |
| family_C_same_sign_LongMemEval | The S5_primary minus chandan_live accuracy difference (Family C) has the same sign under Reader A and Reader B on LongMemEval | same sign under both readers | n/a | n/a | n/a | n/a | **untested** | C1 and C2; a zero delta under either reader counts as contradicted |
| family_C_same_sign_MultiHop-RAG | The S5_primary minus chandan_live accuracy difference (Family C) has the same sign under Reader A and Reader B on MultiHop-RAG | same sign under both readers | n/a | n/a | n/a | n/a | **untested** | C5 and C6; a zero delta under either reader counts as contradicted |

## Failure buckets

Bucket 5 is tested first: a wrong answer on an answerable question that the two judges disagree on. Every other wrong answer is tested against buckets 1 to 4 in order, first match wins. Bucket 5 is decidable for the head-to-head arms; for the other arms it is decidable on the audited sample only, and the undecidable count is shown.

### S2_lazy, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 50 | 2 | 0 | 0 | 41 | 7 | 42 |
| inference_query | 9 | 0 | 0 | 2 | 7 | 0 | 8 |
| temporal_query | 66 | 2 | 0 | 3 | 53 | 8 | 58 |
| all | 125 | 4 | 0 | 5 | 101 | 15 | 108 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S4_static, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 50 | 0 | 0 | 4 | 40 | 6 | 46 |
| inference_query | 12 | 0 | 0 | 6 | 6 | 0 | 11 |
| temporal_query | 68 | 5 | 0 | 9 | 46 | 8 | 58 |
| all | 130 | 5 | 0 | 19 | 92 | 14 | 115 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_noPGR, longmemeval, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| knowledge-update | 19 | 0 | 0 | 0 | 1 | 18 | 19 |
| multi-session | 81 | 6 | 0 | 7 | 0 | 68 | 71 |
| single-session-assistant | 6 | 0 | 0 | 2 | 2 | 2 | 6 |
| single-session-preference | 23 | 0 | 0 | 0 | 5 | 18 | 20 |
| single-session-user | 8 | 1 | 0 | 0 | 0 | 7 | 7 |
| temporal-reasoning | 85 | 2 | 0 | 3 | 3 | 77 | 75 |
| all | 222 | 9 | 0 | 12 | 11 | 190 | 198 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 5. Knowledge-update bucket 3 cases where the clause fired: 1, of which the gold turn is the earlier of the two by timestamp: 1.

### S5_noPGR, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 56 | 1 | 0 | 9 | 43 | 3 | 49 |
| inference_query | 8 | 0 | 0 | 5 | 3 | 0 | 8 |
| temporal_query | 71 | 0 | 0 | 7 | 60 | 4 | 63 |
| all | 135 | 1 | 0 | 21 | 106 | 7 | 120 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_overlay_P0, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 71 | 1 | 0 | 19 | 48 | 3 | 63 |
| inference_query | 12 | 0 | 0 | 6 | 6 | 0 | 12 |
| temporal_query | 70 | 2 | 0 | 24 | 41 | 3 | 64 |
| all | 153 | 3 | 0 | 49 | 95 | 6 | 139 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_overlay_R0, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 66 | 0 | 0 | 8 | 52 | 6 | 60 |
| inference_query | 12 | 0 | 0 | 7 | 5 | 0 | 10 |
| temporal_query | 62 | 1 | 0 | 8 | 48 | 5 | 57 |
| all | 140 | 1 | 0 | 23 | 105 | 11 | 127 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_overlay_R2, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 61 | 2 | 0 | 16 | 38 | 5 | 56 |
| inference_query | 3 | 0 | 0 | 1 | 2 | 0 | 3 |
| temporal_query | 73 | 0 | 0 | 23 | 48 | 2 | 68 |
| all | 137 | 2 | 0 | 40 | 88 | 7 | 127 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_planner_oracle, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 56 | 2 | 0 | 9 | 42 | 3 | 48 |
| inference_query | 8 | 0 | 0 | 5 | 3 | 0 | 8 |
| temporal_query | 63 | 2 | 0 | 7 | 49 | 5 | 54 |
| all | 127 | 4 | 0 | 21 | 94 | 8 | 110 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 1. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_planner_rules, multihoprag, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 51 | 0 | 0 | 4 | 42 | 5 | 48 |
| inference_query | 8 | 0 | 0 | 6 | 2 | 0 | 8 |
| temporal_query | 59 | 1 | 0 | 12 | 37 | 9 | 50 |
| all | 118 | 1 | 0 | 22 | 81 | 14 | 106 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_primary, multihoprag, reader_a, budget 4,000 tokens

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 56 | 14 | 0 | 7 | 32 | 3 | 0 |
| inference_query | 8 | 0 | 0 | 5 | 3 | 0 | 0 |
| temporal_query | 71 | 16 | 0 | 7 | 44 | 4 | 0 |
| all | 135 | 30 | 0 | 19 | 79 | 7 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### S5_primary, multihoprag, reader_b, budget 4,000 tokens

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 11 | 3 | 0 | 4 | 2 | 2 | 0 |
| temporal_query | 13 | 3 | 0 | 0 | 10 | 0 | 0 |
| all | 24 | 6 | 0 | 4 | 12 | 2 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### ours_cheap, longmemeval, reader_a, budget 4,000 tokens

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| knowledge-update | 19 | 0 | 0 | 0 | 1 | 18 | 0 |
| multi-session | 79 | 44 | 0 | 4 | 0 | 31 | 0 |
| single-session-assistant | 6 | 0 | 0 | 2 | 2 | 2 | 0 |
| single-session-preference | 19 | 0 | 0 | 1 | 5 | 13 | 0 |
| single-session-user | 9 | 6 | 0 | 0 | 0 | 3 | 0 |
| temporal-reasoning | 86 | 28 | 0 | 4 | 1 | 53 | 0 |
| all | 218 | 78 | 0 | 11 | 9 | 120 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 6. Knowledge-update bucket 3 cases where the clause fired: 1, of which the gold turn is the earlier of the two by timestamp: 1.

### ours_cheap, longmemeval, reader_b, budget 4,000 tokens

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| knowledge-update | 5 | 1 | 0 | 0 | 1 | 3 | 0 |
| multi-session | 31 | 2 | 0 | 5 | 0 | 24 | 0 |
| single-session-assistant | 3 | 0 | 0 | 1 | 2 | 0 | 0 |
| single-session-preference | 6 | 1 | 0 | 0 | 2 | 3 | 0 |
| single-session-user | 4 | 1 | 0 | 0 | 0 | 3 | 0 |
| temporal-reasoning | 27 | 4 | 0 | 6 | 1 | 16 | 0 |
| all | 76 | 9 | 0 | 12 | 6 | 49 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 1. Knowledge-update bucket 3 cases where the clause fired: 1, of which the gold turn is the earlier of the two by timestamp: 1.

### ours_cheap, multihoprag, reader_a, budget 4,000 tokens

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 59 | 11 | 0 | 3 | 35 | 10 | 0 |
| inference_query | 7 | 1 | 0 | 2 | 4 | 0 | 0 |
| temporal_query | 57 | 17 | 0 | 7 | 27 | 6 | 0 |
| all | 123 | 29 | 0 | 12 | 66 | 16 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### ours_cheap, multihoprag, reader_b, budget 4,000 tokens

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| comparison_query | 8 | 2 | 0 | 1 | 4 | 1 | 0 |
| inference_query | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| temporal_query | 9 | 2 | 0 | 0 | 5 | 2 | 0 |
| all | 18 | 4 | 0 | 1 | 10 | 3 | 0 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 0. Knowledge-update bucket 3 cases where the clause fired: 0, of which the gold turn is the earlier of the two by timestamp: 0.

### ours_cheap_norule, longmemeval, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| knowledge-update | 15 | 0 | 0 | 0 | 2 | 13 | 13 |
| multi-session | 87 | 5 | 0 | 8 | 38 | 36 | 80 |
| single-session-assistant | 4 | 0 | 0 | 0 | 2 | 2 | 4 |
| single-session-preference | 22 | 0 | 0 | 1 | 7 | 14 | 19 |
| single-session-user | 12 | 0 | 0 | 0 | 0 | 12 | 12 |
| temporal-reasoning | 93 | 3 | 0 | 7 | 19 | 64 | 86 |
| all | 233 | 8 | 0 | 16 | 68 | 141 | 214 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 6. Knowledge-update bucket 3 cases where the clause fired: 1, of which the gold turn is the earlier of the two by timestamp: 1.

### ours_sentence_norule, longmemeval, reader_a, budget 4,000 tokens (bucket 5 on the audited sample)

| type | wrong | 5 judge disagreement | 1 index failure | 2 retrieval failure | 3 context assembly failure | 4 reader failure | bucket 5 undecidable |
|---|---|---|---|---|---|---|---|
| knowledge-update | 13 | 0 | 0 | 0 | 2 | 11 | 11 |
| multi-session | 81 | 5 | 0 | 6 | 30 | 40 | 71 |
| single-session-assistant | 3 | 0 | 0 | 0 | 2 | 1 | 3 |
| single-session-preference | 20 | 0 | 0 | 1 | 5 | 14 | 19 |
| single-session-user | 12 | 1 | 0 | 0 | 1 | 10 | 10 |
| temporal-reasoning | 89 | 1 | 0 | 7 | 15 | 66 | 79 |
| all | 218 | 7 | 0 | 14 | 55 | 142 | 193 |

Evidence units truncated or half covered whose cut-off part does not contain the gold answer, counted as inside: 0. Knowledge-update cases where the superseding clause could not fire: 5. Knowledge-update bucket 3 cases where the clause fired: 1, of which the gold turn is the earlier of the two by timestamp: 1.

Knowledge-update bucket 3 cases where the gold turn is the earlier of the two by timestamp, all arms and readers: 5 of 5 cases where the clause fired. The section 8 rule names the turn holding the gold answer as the superseding turn; when the question asks about the earlier fact that turn is the superseded one and the clause fires backwards, so the reader can discount these.

## Judges

Candidate primary judge: gemini-2.5-flash-lite. Second judge: gpt-5.4. Pooled agreement on the audit sample: 1,136 of 1,287 verdicts, 0.883. Threshold 0.900. Primary judge decided before any test: **gpt-5.4**. Second judge column: gemini-2.5-flash-lite.

| arm | corpus | reader | n | agree | agreement |
|---|---|---|---|---|---|
| S2_lazy | multihoprag | reader_a | 60 | 56 | 0.933 |
| S4_static | multihoprag | reader_a | 60 | 53 | 0.883 |
| S5_noPGR | longmemeval | reader_a | 50 | 41 | 0.820 |
| S5_noPGR | multihoprag | reader_a | 60 | 56 | 0.933 |
| S5_overlay_P0 | multihoprag | reader_a | 60 | 55 | 0.917 |
| S5_overlay_R0 | multihoprag | reader_a | 60 | 53 | 0.883 |
| S5_overlay_R2 | multihoprag | reader_a | 60 | 58 | 0.967 |
| S5_planner_oracle | multihoprag | reader_a | 60 | 54 | 0.900 |
| S5_planner_rules | multihoprag | reader_a | 60 | 58 | 0.967 |
| S5_primary | multihoprag | reader_a | 60 | 53 | 0.883 |
| S5_primary | multihoprag | reader_b | 20 | 20 | 1.000 |
| chandan_full_uncut | longmemeval | chandan_own | 37 | 35 | 0.946 |
| closed_book | longmemeval | reader_a | 50 | 30 | 0.600 |
| closed_book | longmemeval | reader_b | 50 | 28 | 0.560 |
| closed_book | multihoprag | reader_a | 60 | 53 | 0.883 |
| closed_book | multihoprag | reader_b | 20 | 18 | 0.900 |
| oracle_full | longmemeval | reader_a | 50 | 44 | 0.880 |
| oracle_full | longmemeval | reader_b | 50 | 46 | 0.920 |
| oracle_full | multihoprag | reader_a | 60 | 58 | 0.967 |
| oracle_full | multihoprag | reader_b | 20 | 20 | 1.000 |
| ours_cheap | longmemeval | reader_a | 50 | 40 | 0.800 |
| ours_cheap | longmemeval | reader_b | 50 | 48 | 0.960 |
| ours_cheap | multihoprag | reader_a | 60 | 57 | 0.950 |
| ours_cheap | multihoprag | reader_b | 20 | 19 | 0.950 |
| ours_cheap_norule | longmemeval | reader_a | 50 | 41 | 0.820 |
| ours_sentence_norule | longmemeval | reader_a | 50 | 42 | 0.840 |

Decision as recorded by the qa stage: made on 1,287 pooled verdicts, 1,136 agreeing, agreement 0.883, primary gpt-5.4.

No qa audit history is kept yet: the history starts with the next qa invocation, and the first-pass figure is the one recorded above.

## Answering accuracy, primary judge

The second judge is a separate column and is never merged. A question with no output counts as wrong.

### chandan_own, longmemeval

| arm | budget | n | missing | all | answerable | abstention or null | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning | second judge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chandan_full_uncut | 4,000 | 379 | 0 | 0.691 | 0.671 | 0.962 | 0.754 | 0.594 | 0.976 | 0.440 | 0.865 | 0.612 | 0.757 (n 37, disagree 2) |
| S4_static | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary_norule | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_rules | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_oracle | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R2 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_P0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S2_lazy | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

### reader_a, longmemeval

| arm | budget | n | missing | all | answerable | abstention or null | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning | second judge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S5_noPGR | 4,000 | 500 | 0 | 0.550 | 0.528 | 0.900 | 0.718 | 0.391 | 0.893 | 0.233 | 0.886 | 0.361 | 0.700 (n 50, disagree 9) |
| S5_noPGR | 8,000 | 500 | 0 | 0.544 | 0.519 | 0.933 | 0.731 | 0.383 | 0.893 | 0.300 | 0.843 | 0.346 | n/a (n 0, disagree 0) |
| closed_book | 4,000 | 500 | 0 | 0.078 | 0.019 | 1.000 | 0.090 | 0.090 | 0.054 | 0.000 | 0.086 | 0.083 | 0.420 (n 50, disagree 20) |
| closed_book | 8,000 | 500 | 0 | 0.078 | 0.019 | 1.000 | 0.090 | 0.090 | 0.054 | 0.000 | 0.086 | 0.083 | n/a (n 0, disagree 0) |
| oracle_full | 4,000 | 500 | 0 | 0.620 | 0.600 | 0.933 | 0.846 | 0.429 | 0.982 | 0.433 | 0.871 | 0.436 | 0.840 (n 50, disagree 6) |
| oracle_full | 8,000 | 500 | 0 | 0.620 | 0.600 | 0.933 | 0.846 | 0.429 | 0.982 | 0.433 | 0.871 | 0.436 | n/a (n 0, disagree 0) |
| ours_cheap | 4,000 | 500 | 0 | 0.558 | 0.536 | 0.900 | 0.718 | 0.406 | 0.893 | 0.367 | 0.871 | 0.353 | 0.420 (n 245, disagree 80) |
| ours_cheap | 8,000 | 500 | 0 | 0.550 | 0.528 | 0.900 | 0.744 | 0.414 | 0.911 | 0.300 | 0.843 | 0.323 | 0.338 (n 222, disagree 75) |
| ours_cheap_norule | 4,000 | 500 | 0 | 0.532 | 0.504 | 0.967 | 0.795 | 0.346 | 0.929 | 0.267 | 0.829 | 0.301 | 0.760 (n 50, disagree 9) |
| ours_cheap_norule | 8,000 | 500 | 0 | 0.542 | 0.521 | 0.867 | 0.769 | 0.368 | 0.946 | 0.333 | 0.843 | 0.301 | n/a (n 0, disagree 0) |
| ours_sentence_norule | 4,000 | 500 | 0 | 0.560 | 0.536 | 0.933 | 0.821 | 0.391 | 0.946 | 0.333 | 0.829 | 0.323 | 0.620 (n 50, disagree 8) |
| ours_sentence_norule | 8,000 | 500 | 0 | 0.546 | 0.523 | 0.900 | 0.821 | 0.338 | 0.964 | 0.267 | 0.843 | 0.323 | n/a (n 0, disagree 0) |
| S4_static | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary_norule | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_rules | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_oracle | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R2 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_P0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S2_lazy | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

### reader_a, multihoprag

| arm | budget | n | missing | all | answerable | abstention or null | comparison_query | inference_query | null_query | temporal_query | second judge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2_lazy | 4,000 | 600 | 0 | 0.780 | 0.722 | 0.953 | 0.667 | 0.940 | 0.953 | 0.560 | 0.783 (n 60, disagree 4) |
| S2_lazy | 8,000 | 600 | 0 | 0.847 | 0.804 | 0.973 | 0.767 | 0.967 | 0.973 | 0.680 | n/a (n 0, disagree 0) |
| S4_static | 4,000 | 600 | 0 | 0.772 | 0.711 | 0.953 | 0.667 | 0.920 | 0.953 | 0.547 | 0.783 (n 60, disagree 7) |
| S4_static | 8,000 | 600 | 0 | 0.850 | 0.809 | 0.973 | 0.767 | 0.973 | 0.973 | 0.687 | n/a (n 0, disagree 0) |
| S5_noPGR | 4,000 | 600 | 0 | 0.772 | 0.700 | 0.987 | 0.627 | 0.947 | 0.987 | 0.527 | 0.717 (n 60, disagree 4) |
| S5_noPGR | 8,000 | 600 | 0 | 0.807 | 0.756 | 0.960 | 0.673 | 0.973 | 0.960 | 0.620 | n/a (n 0, disagree 0) |
| S5_overlay_P0 | 4,000 | 600 | 0 | 0.738 | 0.660 | 0.973 | 0.527 | 0.920 | 0.973 | 0.533 | 0.783 (n 60, disagree 5) |
| S5_overlay_P0 | 8,000 | 600 | 0 | 0.802 | 0.747 | 0.967 | 0.667 | 0.940 | 0.967 | 0.633 | n/a (n 0, disagree 0) |
| S5_overlay_R0 | 4,000 | 600 | 0 | 0.757 | 0.689 | 0.960 | 0.560 | 0.920 | 0.960 | 0.587 | 0.700 (n 60, disagree 7) |
| S5_overlay_R0 | 8,000 | 600 | 0 | 0.817 | 0.769 | 0.960 | 0.693 | 0.973 | 0.960 | 0.640 | n/a (n 0, disagree 0) |
| S5_overlay_R2 | 4,000 | 600 | 0 | 0.768 | 0.696 | 0.987 | 0.593 | 0.980 | 0.987 | 0.513 | 0.850 (n 60, disagree 2) |
| S5_overlay_R2 | 8,000 | 600 | 0 | 0.827 | 0.778 | 0.973 | 0.693 | 0.973 | 0.973 | 0.667 | n/a (n 0, disagree 0) |
| S5_planner_oracle | 4,000 | 600 | 0 | 0.778 | 0.718 | 0.960 | 0.627 | 0.947 | 0.960 | 0.580 | 0.750 (n 60, disagree 6) |
| S5_planner_oracle | 8,000 | 600 | 0 | 0.822 | 0.776 | 0.960 | 0.673 | 0.973 | 0.960 | 0.680 | n/a (n 0, disagree 0) |
| S5_planner_rules | 4,000 | 600 | 0 | 0.797 | 0.738 | 0.973 | 0.660 | 0.947 | 0.973 | 0.607 | 0.800 (n 60, disagree 2) |
| S5_planner_rules | 8,000 | 600 | 0 | 0.837 | 0.791 | 0.973 | 0.707 | 0.967 | 0.973 | 0.700 | n/a (n 0, disagree 0) |
| S5_primary | 4,000 | 600 | 0 | 0.772 | 0.700 | 0.987 | 0.627 | 0.947 | 0.987 | 0.527 | 0.418 (n 184, disagree 32) |
| S5_primary | 8,000 | 600 | 0 | 0.807 | 0.756 | 0.960 | 0.673 | 0.973 | 0.960 | 0.620 | 0.200 (n 110, disagree 22) |
| closed_book | 4,000 | 600 | 0 | 0.700 | 0.684 | 0.747 | 0.620 | 0.887 | 0.747 | 0.547 | 0.800 (n 60, disagree 7) |
| closed_book | 8,000 | 600 | 0 | 0.700 | 0.684 | 0.747 | 0.620 | 0.887 | 0.747 | 0.547 | n/a (n 0, disagree 0) |
| oracle_full | 4,000 | 600 | 0 | 0.787 | 0.800 | 0.747 | 0.727 | 0.987 | 0.747 | 0.687 | 0.833 (n 60, disagree 2) |
| oracle_full | 8,000 | 600 | 0 | 0.787 | 0.800 | 0.747 | 0.727 | 0.987 | 0.747 | 0.687 | n/a (n 0, disagree 0) |
| ours_cheap | 4,000 | 600 | 0 | 0.787 | 0.727 | 0.967 | 0.607 | 0.953 | 0.967 | 0.620 | 0.442 (n 172, disagree 30) |
| ours_cheap | 8,000 | 600 | 0 | 0.837 | 0.802 | 0.940 | 0.733 | 0.980 | 0.940 | 0.693 | 0.247 (n 89, disagree 22) |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

### reader_b, longmemeval

| arm | budget | n | missing | all | answerable | abstention or null | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning | second judge | cheap judge | agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| closed_book | 4,000 | 500 | 0 | 0.074 | 0.017 | 0.967 | 0.077 | 0.083 | 0.000 | 0.100 | 0.086 | 0.083 | 0.486 (n 500, disagree 214) | 0.486 (n 500) | 0.572 |
| oracle_full | 4,000 | 500 | 0 | 0.858 | 0.857 | 0.867 | 0.962 | 0.692 | 0.982 | 0.933 | 0.957 | 0.842 | 0.858 (n 500, disagree 18) | 0.858 (n 500) | 0.964 |
| ours_cheap | 4,000 | 500 | 0 | 0.844 | 0.838 | 0.933 | 0.923 | 0.759 | 0.946 | 0.800 | 0.943 | 0.797 | 0.834 (n 500, disagree 23) | 0.834 (n 500) | 0.954 |
| S4_static | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_primary_norule | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_rules | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_planner_oracle | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_R2 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S5_overlay_P0 | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| S2_lazy | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| graphiti | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Cheap judge column: gemini-2.5-flash-lite scored every Reader B record, not only the wrong ones, because the primary judge gpt-5.4 is the same model as Reader B (design section 14 item 3).

### reader_b, multihoprag

| arm | budget | n | missing | all | answerable | abstention or null | comparison_query | inference_query | null_query | temporal_query | second judge | cheap judge | agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S5_primary | 4,000 | 200 | 0 | 0.830 | 0.840 | 0.800 | 0.780 | 1.000 | 0.800 | 0.740 | 0.850 (n 200, disagree 8) | 0.850 (n 200) | 0.960 |
| closed_book | 4,000 | 200 | 0 | 0.455 | 0.320 | 0.860 | 0.080 | 0.560 | 0.860 | 0.320 | 0.590 (n 200, disagree 29) | 0.590 (n 200) | 0.855 |
| oracle_full | 4,000 | 200 | 0 | 0.895 | 0.907 | 0.860 | 0.900 | 1.000 | 0.860 | 0.820 | 0.875 (n 200, disagree 6) | 0.875 (n 200) | 0.970 |
| ours_cheap | 4,000 | 200 | 0 | 0.845 | 0.880 | 0.740 | 0.840 | 0.980 | 0.740 | 0.820 | 0.855 (n 200, disagree 6) | 0.855 (n 200) | 0.970 |
| chandan_live | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| chandan_full | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |

Cheap judge column: gemini-2.5-flash-lite scored every Reader B record, not only the wrong ones, because the primary judge gpt-5.4 is the same model as Reader B (design section 14 item 3).

Absent on longmemeval: S4_static (not run in the retrieve pass); S5_primary (not run in the retrieve pass); S5_primary_norule (not run in the retrieve pass); S5_planner_rules (not run in the retrieve pass); S5_planner_oracle (not run in the retrieve pass); S5_overlay_R0 (not run in the retrieve pass); S5_overlay_R2 (not run in the retrieve pass); S5_overlay_P0 (not run in the retrieve pass); S2_lazy (not run in the retrieve pass); chandan_live (export present, not run in the retrieve pass); chandan_full (export present, not run in the retrieve pass); graphiti (no export).
Absent on multihoprag: chandan_live (no export); chandan_full (no export).

## Cost and time

Index-time spend is charged to every arm that reads the tables it built: post-graph-rag's build to every arm that reads his entities, relations and aliases, chandan_live and chandan_full included, Graphiti's to the graphiti arm. post-graph-rag's and Graphiti's spend is metered from the usage field of each response inside the runner. The cross-check against the proxy request log by time window and job tag is pending: the metrics file holds the job tags and no proxy window total.

| arm | corpus | index USD charged | components | query calls per question | query tokens in per question | query tokens out per question | query USD | seconds per question | questions | state |
|---|---|---|---|---|---|---|---|---|---|---|
| S2_lazy | longmemeval | 60.63 | topics, graph, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S2_lazy | multihoprag | 0.00 | topics, graph | 17.201 | 2,147.1 | 37.7 | 0.59 | 3.579 | 2,556 | run |
| S4_static | longmemeval | 60.63 | topics, graph, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S4_static | multihoprag | 0.00 | topics, graph | 0.000 | 0 | 0 | 0.00 | 0.041 | 2,556 | run |
| S5_noPGR | longmemeval | 0.28 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.035 | 500 | run |
| S5_noPGR | multihoprag | 0.03 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.034 | 2,556 | run |
| S5_overlay_P0 | longmemeval | 60.91 | topics, graph, overlay, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S5_overlay_P0 | multihoprag | 0.03 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.043 | 2,556 | run |
| S5_overlay_R0 | longmemeval | 60.63 | topics, graph, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S5_overlay_R0 | multihoprag | 0.00 | topics, graph | 0.000 | 0 | 0 | 0.00 | 0.033 | 2,556 | run |
| S5_overlay_R2 | longmemeval | 60.91 | topics, graph, overlay, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S5_overlay_R2 | multihoprag | 0.03 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.042 | 2,556 | run |
| S5_planner_oracle | longmemeval | 60.91 | topics, graph, overlay, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S5_planner_oracle | multihoprag | 0.03 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.034 | 2,556 | run |
| S5_planner_rules | longmemeval | 60.91 | topics, graph, overlay, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S5_planner_rules | multihoprag | 0.03 | topics, graph, overlay | 0.000 | 0 | 0 | 0.00 | 0.033 | 2,556 | run |
| S5_primary | longmemeval | 60.91 | topics, graph, overlay, pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | not run in the retrieve pass |
| S5_primary | multihoprag | 0.03 | topics, graph, overlay | 1.000 | 238.9 | 2.3 | 0.06 | 0.256 | 2,556 | run |
| chandan_full | longmemeval | 60.63 | pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | 2.996 | 201.4 | 61.0 | 0.02 | 2.635 | 467 | export present, not run in the retrieve pass |
| chandan_full_uncut | longmemeval | 60.63 | pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | n/a | n/a | n/a | 0.00 | n/a | 0 | run |
| chandan_live | longmemeval | 60.63 | pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | 2.996 | 201.4 | 61.0 | 0.02 | 2.635 | 467 | export present, not run in the retrieve pass |
| closed_book | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 500 | run |
| closed_book | multihoprag | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 2,556 | run |
| oracle_full | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 500 | run |
| oracle_full | multihoprag | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.000 | 2,556 | run |
| ours_cheap | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.069 | 500 | run |
| ours_cheap | multihoprag | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.034 | 2,556 | run |
| ours_cheap_norule | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.037 | 500 | run |
| ours_sentence_norule | longmemeval | 0.00 | none | 0.000 | 0 | 0 | 0.00 | 0.038 | 500 | run |
| planner | longmemeval | 0.00 | none | 1.000 | 198.0 | 1.7 | 0.01 | 0.000 | 500 | pseudo-arm |
| S5_primary_norule | longmemeval | absent | absent | absent | absent | absent | absent | absent | absent | not run in the retrieve pass |
| graphiti | longmemeval | absent | absent | absent | absent | absent | absent | absent | absent | no export |
| chandan_live | multihoprag | absent | absent | absent | absent | absent | absent | absent | absent | no export |
| chandan_full | multihoprag | absent | absent | absent | absent | absent | absent | absent | absent | no export |

The planner row on longmemeval is a pseudo-arm, not a system: its calls are the LLM planner decisions shared by the S5 arms (S5_primary, S5_primary_norule, S5_noPGR, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0), made once per question and cached. At the study price its 500 calls cost USD 0.01. In the last pass 500 of 500 were served from the cache and 0 were paid. The pass that paid for them is not in the ledger: the ledger began after it, so its spend is known only from the log of that pass.

Index-time builds as metered:

| corpus | component | USD | calls | tokens in | tokens out | seconds | charged to |
|---|---|---|---|---|---|---|---|
| longmemeval | topics | 0.00 | 0 | 0 | 0 | 620.6 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| longmemeval | graph | 0.00 | 0 | 0 | 0 | 3,915.6 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| longmemeval | overlay | 0.28 | 2,566 | 1,782,503 | 257,991 | 659.0 | S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R2, S5_overlay_P0 |
| longmemeval | pgr_build (partial build snapshot, 467 of 500 spaces, build in progress) | 60.63 | 131,521 | 198,024,569 | 93,581,726 | 164,099.8 | (default charge list) |
| multihoprag | topics | 0.00 | 0 | 0 | 0 | 62.6 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| multihoprag | graph | 0.00 | 0 | 0 | 0 | 214.9 | S4_static, S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy |
| multihoprag | overlay | 0.03 | 291 | 178,695 | 34,947 | 72.8 | S5_primary, S5_primary_norule, S5_noPGR, S5_planner_rules, S5_planner_oracle, S5_overlay_R2, S5_overlay_P0 |

Post-graph-rag builds (the runner's run logs under data/part1/pgr; a log with no end time is a runner still writing spaces; the runner meter covers index and query calls together):

| corpus | state | in progress | running job tags | stopped | spaces exported | spaces wanted | population | runner meter USD so far |
|---|---|---|---|---|---|---|---|---|
| longmemeval | partial build snapshot, 467 of 500 spaces, build in progress | yes | pgr-lme-full-s1, pgr-lme-full-s2, pgr-lme-full-s3 | none | 467 | 500 | ORDER | 62.40 |
| multihoprag | partial build snapshot, 0 of 1 spaces, build stopped at its cap | no | none | pgr-mhrag-full: cap: spend 10.0071 USD exceeds the cap of 10.00 | 0 | 1 | ORDER | 10.01 |

Spend ledger (section 11): every invocation of each stage, appended at the end of the invocation, a capped one included; the sum per stage is read against the cap of that stage.

No ledger rows yet: the ledger starts with the first invocation of a stage after it was added, so the passes before it are known only from their last-pass meters below and the run logs.

Answering:

```
{
 "longmemeval": {
  "answering_cap_usd": 34.0,
  "judging_stopped": null,
  "n_records": 7879,
  "partial": {},
  "populations": {
   "chandan_own": 500,
   "chandan_subset": null,
   "per_arm": {
    "S5_noPGR": 500,
    "closed_book": 500,
    "oracle_full": 500,
    "ours_cheap": 500,
    "ours_cheap_norule": 500,
    "ours_sentence_norule": 500
   },
   "reader_a": 500,
   "reader_b": 500,
   "second_build_subset": null
  },
  "readers": [
   "reader_a",
   "reader_b"
  ]
 },
 "multihoprag": {
  "answering_cap_usd": 34.0,
  "judging_stopped": null,
  "n_records": 15200,
  "partial": {},
  "populations": {
   "chandan_own": 600,
   "chandan_subset": null,
   "per_arm": {
    "S2_lazy": 600,
    "S4_static": 600,
    "S5_noPGR": 600,
    "S5_overlay_P0": 600,
    "S5_overlay_R0": 600,
    "S5_overlay_R2": 600,
    "S5_planner_oracle": 600,
    "S5_planner_rules": 600,
    "S5_primary": 600,
    "closed_book": 600,
    "oracle_full": 600,
    "ours_cheap": 600
   },
   "reader_a": 600,
   "reader_b": 200,
   "second_build_subset": null
  },
  "readers": [
   "reader_a",
   "reader_b"
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

Caps of the last pass of each stage:

```
{
 "index_lme": 15.0,
 "index_mhrag": 15.0,
 "qa_lme": 40.0,
 "qa_mhrag": 40.0,
 "retrieve_lme": 35.0,
 "retrieve_mhrag": 35.0
}
```

Last-pass meter totals (the meter of the most recent invocation of each stage on each corpus; earlier passes are in the ledger above):

```
{
 "index_lme": {
  "by_tier": {
   "vertex-flash": {
    "calls": 2566,
    "tokens_in": 1782503,
    "tokens_out": 257991,
    "usd": 0.281447
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
  "total_calls": 2566,
  "total_usd": 0.281447
 },
 "index_mhrag": {
  "by_tier": {
   "vertex-flash": {
    "calls": 291,
    "tokens_in": 178695,
    "tokens_out": 34947,
    "usd": 0.031848
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
  "total_calls": 291,
  "total_usd": 0.031848
 },
 "qa_lme": {
  "by_tier": {
   "azure-gpt54": {
    "calls": 7022,
    "tokens_in": 3938730,
    "tokens_out": 55025,
    "usd": 5.473663
   },
   "gemini-2.5-flash-lite": {
    "calls": 10511,
    "tokens_in": 53624563,
    "tokens_out": 380454,
    "usd": 5.514638
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
  "total_calls": 17533,
  "total_usd": 10.9883
 },
 "qa_mhrag": {
  "by_tier": {
   "azure-gpt54": {
    "calls": 7022,
    "tokens_in": 3938730,
    "tokens_out": 55025,
    "usd": 5.473663
   },
   "gemini-2.5-flash-lite": {
    "calls": 10511,
    "tokens_in": 53624563,
    "tokens_out": 380454,
    "usd": 5.514638
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
  "total_calls": 17533,
  "total_usd": 10.9883
 },
 "retrieve_lme": {
  "by_tier": {},
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
  "total_calls": 0,
  "total_usd": 0
 },
 "retrieve_mhrag": {
  "by_tier": {
   "vertex-flash": {
    "calls": 34869,
    "tokens_in": 4584336,
    "tokens_out": 76857,
    "usd": 0.489176
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
  "total_calls": 34869,
  "total_usd": 0.489176
 }
}
```

## MultiHop-RAG fact location

Non-null queries 2,255; all facts located for 2,255 (1.000). Facts 6,084, located 6,084 (1.000), fallback 0, straddling a chunk boundary 0, unresolved 0.

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
| S5_noPGR | longmemeval | 4,000 | 0 |
| S5_noPGR | longmemeval | 8,000 | 0 |
| S5_noPGR | multihoprag | 4,000 | 0 |
| S5_noPGR | multihoprag | 8,000 | 0 |
| S4_static | multihoprag | 4,000 | 0 |
| S4_static | multihoprag | 8,000 | 0 |
| S5_primary | multihoprag | 4,000 | 0 |
| S5_primary | multihoprag | 8,000 | 0 |
| S5_planner_rules | multihoprag | 4,000 | 0 |
| S5_planner_rules | multihoprag | 8,000 | 0 |
| S5_planner_oracle | multihoprag | 4,000 | 0 |
| S5_planner_oracle | multihoprag | 8,000 | 0 |
| S5_overlay_R0 | multihoprag | 4,000 | 0 |
| S5_overlay_R0 | multihoprag | 8,000 | 0 |
| S5_overlay_R2 | multihoprag | 4,000 | 0 |
| S5_overlay_R2 | multihoprag | 8,000 | 0 |
| S5_overlay_P0 | multihoprag | 4,000 | 0 |
| S5_overlay_P0 | multihoprag | 8,000 | 0 |
| S2_lazy | multihoprag | 4,000 | 0 |
| S2_lazy | multihoprag | 8,000 | 0 |

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
- graphrag's ego-node rule removed one content phrase after the hub rule: 'another option'.
- The calibration row (chandan_live_cal, CHANDAN_CAL_18 with his own models) was not run.
- On multihoprag the arms that read post-graph-rag's tables (S4_static, S5_primary, S5_planner_rules, S5_planner_oracle, S5_overlay_R0, S5_overlay_R2, S5_overlay_P0, S2_lazy) were run without post-graph-rag tables (source: index steps relation_vectors (the spaces present when the index was built)); their rows and every test that reads them carry that label.
- Arms absent on longmemeval: S4_static (not run in the retrieve pass); S5_primary (not run in the retrieve pass); S5_primary_norule (not run in the retrieve pass); S5_planner_rules (not run in the retrieve pass); S5_planner_oracle (not run in the retrieve pass); S5_overlay_R0 (not run in the retrieve pass); S5_overlay_R2 (not run in the retrieve pass); S5_overlay_P0 (not run in the retrieve pass); S2_lazy (not run in the retrieve pass); chandan_live (export present, not run in the retrieve pass); chandan_full (export present, not run in the retrieve pass); graphiti (no export).
- Arms absent on multihoprag: chandan_live (no export); chandan_full (no export).

## Published numbers beside chandan_live (section 12)

| number | value | source | judge | models | note |
|---|---|---|---|---|---|
| chandan_readme_94.0 | 0.940 | post-graph-rag README, oracle variant, 499 questions | his own three-model majority panel, two repeats per question | indexed with gemini-3.7-flash, answered with gemini-3.6-flash, gemini-embedding-001 at 1536 dimensions | marked not reportable in his result file |
| chandan_official_78.2 | 0.782 | post-graph-rag evaluation/longmemeval/official_gpt4o_g36.json, n 499 | gpt-4o with the official LongMemEval prompts | answered with gemini-3.6-flash | the official protocol; gpt-4o reading the same retrieval scored 0.713 |
| chandan_paper_v2_85.8 | 0.858 | post-graph-rag paper v2 (reader_sweep_validity.json) | his own judge panel | gemini-3.6-flash, with the 4,000 token context budget of that version | the budget was made unlimited in 1.11.1, which gives the 94.0 row |
| zep_71.2 | 0.712 | Zep paper, LongMemEval_S | the official LongMemEval prompts | gpt-4o reader | the S variant, the one Part 1 runs on; his 94.0 and 85.8 are on the oracle variant |

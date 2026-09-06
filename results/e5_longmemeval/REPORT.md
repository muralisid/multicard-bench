# LongMemEval_S with a flat multi-vector retriever

Written 2026-09-05. Experiment e5_longmemeval in multicard-bench. Every number here is read from the metrics files that the experiment writes. Per-query outputs are in the same folder.

## In one paragraph

On the S split of LongMemEval, 500 questions over histories of about 50 chat sessions each, a retriever with no graph and no model call at index time found the answer turn in its top 10 for 86.3% of the 470 answerable questions. The single largest step was cutting each turn into sentences and embedding every sentence: 72.7% to 83.7% with the same 22-million-parameter encoder. Grouping sentences by purpose (cards) did not beat grouping them by position, for any of four taxonomies. Fusing keyword search with dense search helped, and a declared speaker rule helped again. End to end, with gemini-2.5-flash reading the top 10 turns and the benchmark's official judge prompts, the best arm scored 69.0%. The same reader given the evidence sessions in full scored 74.8%. So under this reader, cheap retrieval closes most of the distance to perfect retrieval. The remaining wrong answers split about half and half between the reader missing a fact that was in front of it and the retriever bringing only part of the evidence, mostly on multi-session questions.

## Setup

- Data: LongMemEval_S, 500 questions, 19829 unique sessions, 199,641 turns. 30 abstention questions have no evidence and are scored only in the answering stage.
- Units: whole turns; sentences of user turns and paragraphs or list items of assistant turns (998,042 units); purpose cards over user sentences with a position control of identical sentences and identical group counts.
- Encoder: sentence-transformers/all-MiniLM-L6-v2, on CPU. Keyword search: BM25. Fusion: reciprocal rank fusion, k=60.
- Taxonomies: one hand-written (about me, things I did and when, counts, preferences, the request) and three designed by gemini-2.5-flash from topic-stratified samples of user turns. The designer never saw a question, an answer or an evidence flag.
- Topic map: two-pass discovery (UMAP, HDBSCAN) over 6,000 session openings, 79 topics, 27% unclustered. Used for the designer samples.
- Speaker rule (declared, not tuned): assistant turns are quarantined unless the question refers to the assistant.
- Reader: gemini-2.5-flash, top 10 turns sorted by date, each prefixed with its date and speaker, with a latest-statement-wins rule and a date-arithmetic instruction. Judges: azure:gpt-5.4 (primary) and gemini-2.5-flash, both with the benchmark's official prompts. Agreement between the two judges was 0.94 to 0.96 on every arm.
- Fixed before any number was read: the primary metric (turn-level Recall@10), the four tests below, the predictions, and the reader prompt. One reader-prompt change was made after a six-question smoke test and is recorded in the code. Two answering arms (rrf_sentence, rrf_turn_route) were added after the retrieval results were read and are labelled as such.

## Retrieval ladder, 470 answerable questions

| arm | what it is | turn R@5 | turn R@10 | turn nDCG@10 | session R@5 |
|---|---|---|---|---|---|
| rrf_turn_route | rrf_turn with the speaker rule | 0.780 | 0.863 | 0.708 | 0.938 |
| rrf_sentence | keyword + dense sentences, fused | 0.756 | 0.848 | 0.689 | 0.959 |
| rrf_cards_m11 | keyword + dense cards m11 | 0.729 | 0.838 | 0.670 | 0.956 |
| sentence | one vector per sentence or list item | 0.713 | 0.837 | 0.666 | 0.964 |
| rrf_cards_m17 | keyword + dense cards m17 | 0.735 | 0.836 | 0.672 | 0.956 |
| rrf_cards_m13 | keyword + dense cards m13 | 0.717 | 0.825 | 0.652 | 0.956 |
| rrf_turn | keyword + dense turns, fused | 0.681 | 0.825 | 0.632 | 0.945 |
| rrf_cards_hand | keyword over cards + dense cards, fused | 0.707 | 0.822 | 0.650 | 0.951 |
| cards_m17 | purpose cards, model taxonomy seed 17 | 0.673 | 0.811 | 0.623 | 0.954 |
| cards_m11 | purpose cards, model taxonomy seed 11 | 0.676 | 0.809 | 0.620 | 0.959 |
| pos_m11 | position control | 0.672 | 0.806 | 0.627 | 0.954 |
| pos_m17 | position control | 0.668 | 0.805 | 0.626 | 0.952 |
| cards_hand | purpose cards, hand taxonomy | 0.668 | 0.804 | 0.618 | 0.955 |
| pos_hand | position control for cards_hand | 0.667 | 0.797 | 0.612 | 0.952 |
| cards_m13 | purpose cards, model taxonomy seed 13 | 0.634 | 0.784 | 0.572 | 0.947 |
| pos_m13 | position control | 0.634 | 0.780 | 0.583 | 0.950 |
| bm25_turn | keyword search over turns | 0.677 | 0.779 | 0.644 | 0.885 |
| bm25_cards_hand | keyword search over cards | 0.622 | 0.731 | 0.592 | 0.854 |
| turn | one vector per turn | 0.571 | 0.727 | 0.519 | 0.933 |
| dense_session_mean | one vector per session (mean of turns) | - | - | - | 0.903 |
| bm25_session | keyword search over whole sessions | - | - | - | 0.909 |

Turn-level Recall@10 by question type:

| type | n | bm25_turn | turn | sentence | cards_hand | pos_hand | rrf_turn | rrf_sentence | rrf_turn_route |
|---|---|---|---|---|---|---|---|---|---|
| knowledge-update | 72 | 0.882 | 0.843 | 0.965 | 0.903 | 0.917 | 0.942 | 0.926 | 0.988 |
| multi-session | 121 | 0.662 | 0.606 | 0.754 | 0.736 | 0.726 | 0.717 | 0.765 | 0.782 |
| single-session-assistant | 56 | 0.929 | 0.982 | 0.964 | 0.964 | 0.964 | 0.964 | 0.964 | 0.911 |
| single-session-preference | 30 | 0.528 | 0.667 | 0.772 | 0.783 | 0.783 | 0.678 | 0.700 | 0.694 |
| single-session-user | 64 | 0.930 | 0.898 | 0.938 | 0.938 | 0.922 | 0.930 | 0.969 | 0.953 |
| temporal-reasoning | 127 | 0.751 | 0.591 | 0.750 | 0.681 | 0.662 | 0.782 | 0.805 | 0.841 |

## The four tests, decided in advance

- **H1, dilution at session level. HELD.** (turn vs dense_session_mean, sess_r5): 0.933 vs 0.903, delta +0.029, 95% CI [+0.014, +0.045], p 0.0001, wins/ties/losses 42/413/15.
- **H2, purpose cards beat the position control, hand taxonomy. NOT SUPPORTED.** (cards_hand vs pos_hand, turn_r10): 0.804 vs 0.797, delta +0.008, 95% CI [-0.004, +0.020], p 0.1966, wins/ties/losses 17/444/9.
- **H2, same test, model taxonomy m11. NOT SUPPORTED.** (cards_m11 vs pos_m11, turn_r10): 0.809 vs 0.806, delta +0.003, 95% CI [-0.009, +0.015], p 0.6263, wins/ties/losses 19/437/14.
- **H2, same test, model taxonomy m13. NOT SUPPORTED.** (cards_m13 vs pos_m13, turn_r10): 0.784 vs 0.780, delta +0.003, 95% CI [-0.010, +0.016], p 0.6108, wins/ties/losses 20/432/18.
- **H2, same test, model taxonomy m17. NOT SUPPORTED.** (cards_m17 vs pos_m17, turn_r10): 0.811 vs 0.805, delta +0.006, 95% CI [-0.005, +0.016], p 0.3079, wins/ties/losses 20/438/12.
- **H3, fusion beats keyword search alone. HELD.** (rrf_turn vs bm25_turn, turn_r10): 0.825 vs 0.779, delta +0.046, 95% CI [+0.024, +0.068], p 0.0001, wins/ties/losses 58/390/22.
- **H3, fusion beats dense turns alone. HELD.** (rrf_turn vs turn, turn_r10): 0.825 vs 0.727, delta +0.098, 95% CI [+0.069, +0.127], p 0.0001, wins/ties/losses 101/347/22.
- **H4, fusion over cards beats fusion over turns. NOT SUPPORTED.** (rrf_cards_hand vs rrf_turn, turn_r10): 0.822 vs 0.825, delta -0.003, 95% CI [-0.021, +0.015], p 0.7354, wins/ties/losses 31/399/40.

Not pre-declared, reported because they explain the ladder:

- **Sentences beat turns (capacity).** (sentence vs turn, turn_r10): 0.837 vs 0.727, delta +0.110, 95% CI [+0.080, +0.140], p 0.0001, wins/ties/losses 113/332/25.
- **Cards lose to plain sentences (grouping at all costs recall).** (cards_hand vs sentence, turn_r10): 0.804 vs 0.837, delta -0.032, 95% CI [-0.050, -0.015], p 0.0001, wins/ties/losses 15/416/39.
- **Keyword search beats dense whole turns.** (turn vs bm25_turn, turn_r10): 0.727 vs 0.779, delta -0.053, 95% CI [-0.086, -0.018], p 0.0023, wins/ties/losses 54/317/99.
- **The speaker rule on top of fusion.** (rrf_turn_route vs rrf_turn, turn_r10): 0.863 vs 0.825, delta +0.038, 95% CI [+0.022, +0.054], p 0.0001, wins/ties/losses 48/417/5.

The prediction written in advance was that cards would beat the position control on single-session-user and multi-session questions. They did not, on any type. The preference type was predicted to stay weakest and did.

## Answering, 500 questions

| arm | context | accuracy, all 500 | answerable (470) | abstention (30) | accuracy under the Gemini judge |
|---|---|---|---|---|---|
| bm25_turn | 8,583 chars | 66.0% | 63.8% | 100.0% | 66.8% |
| rrf_turn | 8,994 chars | 67.0% | 64.9% | 100.0% | 66.8% |
| rrf_cards_hand | 10,177 chars | 65.8% | 63.8% | 96.7% | 68.2% |
| rrf_cards_m13 | 10,233 chars | 66.4% | 64.3% | 100.0% | 67.2% |
| rrf_sentence (added after) | 9,931 chars | 68.6% | 66.6% | 100.0% | 68.4% |
| rrf_turn_route (added after) | 3,734 chars | 69.0% | 67.0% | 100.0% | 68.4% |
| oracle_full: evidence sessions in full | 21,568 chars | 74.8% | 73.4% | 96.7% | 75.0% |

By question type, primary judge:

| type | bm25_turn | rrf_turn | rrf_sentence | rrf_turn_route | oracle_full |
|---|---|---|---|---|---|
| knowledge-update | 83.3% | 89.7% | 89.7% | 87.2% | 91.0% |
| multi-session | 49.6% | 48.1% | 51.1% | 53.4% | 63.2% |
| single-session-assistant | 87.5% | 91.1% | 89.3% | 85.7% | 92.9% |
| single-session-preference | 30.0% | 46.7% | 40.0% | 40.0% | 30.0% |
| single-session-user | 84.3% | 84.3% | 87.1% | 88.6% | 92.9% |
| temporal-reasoning | 61.7% | 57.9% | 61.7% | 63.2% | 69.9% |

Published numbers for context. They were produced with different readers and different judges, so read them as neighbourhoods, not head-to-heads.

| system | split | reader | judge | accuracy |
|---|---|---|---|---|
| Zep / Graphiti (arXiv 2501.13956) | S | gpt-4o | gpt-4o, official prompts | 71.2% |
| Full-context baseline (same paper) | S | gpt-4o | gpt-4o | 60.2% |
| post-graph-rag (arXiv 2608.24921) | oracle | gemini-3.6-flash | panel of three, own prompt | 85.8% |
| GPT-4o reading the evidence sessions (LongMemEval paper, Fig. 3b) | oracle | gpt-4o | gpt-4o | 87.0% |
| this work, best arm | S | gemini-2.5-flash | gpt-5.4, official prompts | 69.0% |
| this work, evidence sessions in full | oracle | gemini-2.5-flash | gpt-5.4, official prompts | 74.8% |

## Where the wrong answers come from

For each wrong answer on an answerable question, was the evidence inside the ten turns the reader saw?

| arm | wrong | all evidence present (reading failure) | part of the evidence present | no evidence present |
|---|---|---|---|---|
| bm25_turn | 170 | 58 | 59 | 53 |
| rrf_turn | 165 | 69 | 66 | 30 |
| rrf_sentence | 157 | 73 | 57 | 27 |
| rrf_turn_route | 155 | 76 | 50 | 29 |

Best arm, by type:

| type | wrong | reading | partial evidence | no evidence |
|---|---|---|---|---|
| knowledge-update | 10 | 10 | 0 | 0 |
| multi-session | 62 | 21 | 37 | 4 |
| single-session-assistant | 8 | 4 | 0 | 4 |
| single-session-preference | 18 | 8 | 3 | 7 |
| single-session-user | 8 | 6 | 0 | 2 |
| temporal-reasoning | 49 | 27 | 10 | 12 |

Multi-session questions need two or three evidence turns. Most of their failures are partial retrieval: some of the turns made the top 10 and some did not. That is the place where links across sessions should pay, and where a flat retriever is weakest.

## Cost and time

- Retrieval stage: 166 minutes on one Mac CPU, of which about 150 minutes were encoding 1.2 million units once. Model calls: three taxonomy designs, USD 0.0018.
- Answering stage: 21 + 12 minutes, 3,500 reader calls and 7,000 judge calls, USD 1.33 at the prices in the bench's dated table.
- Index-time model cost of this retriever on the S corpus: under one US cent. A pipeline that sends every 2,000-character chunk to a model twice would spend roughly 200 million input tokens on the same corpus.

## Caveats

- One run, one seed for the encoder side, which is deterministic. The three taxonomy seeds agree with each other and with the hand taxonomy, so the card result is not one draw.
- The reader is gemini-2.5-flash and the primary judge is gpt-5.4. Neither matches the readers or judges in the published numbers. The comparison rows above are context, not a controlled test. The two judges here agreed on 94 to 96 percent of answers, and the accuracy under either is within two points.
- The encoder is small (22M parameters, 256-token window). A larger encoder would lift every dense arm. Whether it changes the gaps between arms is untested.
- The speaker rule was written from a review of the data, before any result, but by someone who knew the question shapes. It costs recall on questions about what the assistant said.
- The preference type is weak under every arm, including the full-evidence control (30.0%). The reader prompt asks for brief answers, and the official rubric wants the answer to use the user's personal information. That is a prompt problem, not a retrieval problem, and it is not fixed here.
- Abstention is near perfect because the reader was told to say when the facts are not there. That instruction may cost a few answerable questions; it is the same trade every memory system makes.

## What this suggests for a graph memory system

- A sentence-level dense channel plus keyword search, fused by rank, is cheap and finds the answer turn for about 85 percent of questions on the S split. It can sit beside graph traversal as one more channel in the same fusion.
- Purpose cards are not needed. Plain sentences did better than any grouping of them.
- Under this reader, perfect retrieval is worth about six points over the best cheap retrieval. The bigger lever is the reader and its prompt, which is consistent with the temporal-grounding ablation in the post-graph-rag paper.
- The failures a graph could target are concrete: multi-session questions whose evidence is spread across sessions, and temporal questions that need two dated endpoints.
- To measure the channel fairly inside a graph system: index once, hold the graph fixed, and add or remove the channel at query time. Re-indexing per arm measures the extractor, not the channel.

# Our system against published results

Written 2026-09-07. Every number of ours is computed from the per-query files
by a script in the scratchpad; every published number is quoted from the paper
named beside it. Where a comparison is not like for like, the difference is
stated in the same sentence as the number.

## The rule this document follows

Accuracy numbers on these benchmarks are not comparable across papers, because
the reader and the judge move them by more than any retrieval method does. Our
own earlier experiment measured that directly: the same retrieval, read by
gemini-2.5-flash, scored 69.0 on LongMemEval_S, and the same evidence read by a
stronger model scored far higher. So accuracy tables here are context, never a
claim of superiority.

Retrieval metrics with a fixed definition and no model in the loop are
comparable, provided the retrieval unit is stated. That is where the real
comparison lives.

## 1. MultiHop-RAG, retrieval, against the paper's own table

The MultiHop-RAG paper (arXiv 2401.15391, Table 5) reports MRR@10, MAP@10,
Hits@10 and Hits@4 over the same 609 articles and the same non-null queries we
ran. Its best row is voyage-02 embeddings with the bge-reranker-large reranker.
Its chunk is 256 tokens; ours is about 500 tokens with no overlap.

Our rows below are computed on the ranking before any budget cut, which is what
the paper measures. n is 2,255 for every row.

Document level: a document counts as relevant when the query's evidence list
names it, and the unit ranking is folded to the first appearance of each
document.

| arm | MRR@10 | MAP@10 | Hits@10 | Hits@4 |
|---|---|---|---|---|
| ours_cheap | 0.757 | 0.574 | 0.992 | 0.917 |
| S2_lazy | 0.736 | 0.537 | 0.987 | 0.894 |
| S4_static | 0.706 | 0.512 | 0.985 | 0.877 |
| S5_overlay_R0 | 0.681 | 0.485 | 0.979 | 0.849 |
| S5_planner_rules | 0.664 | 0.489 | 0.983 | 0.848 |
| S5_planner_oracle | 0.635 | 0.454 | 0.972 | 0.837 |
| S5_overlay_R2 | 0.634 | 0.449 | 0.966 | 0.840 |
| S5_primary | 0.618 | 0.438 | 0.970 | 0.826 |
| S5_overlay_P0 | 0.582 | 0.411 | 0.955 | 0.804 |
| **published best: voyage-02 with bge-reranker-large** | **0.586** | **0.480** | **0.747** | **0.663** |

Unit level, closer in kind to the paper's chunk-level rows: a retrieved chunk
counts as relevant when it belongs to an evidence document.

| arm | MRR@10 | MAP@10 | Hits@10 | Hits@4 |
|---|---|---|---|---|
| ours_cheap | 0.745 | 0.362 | 0.973 | 0.883 |
| S4_static | 0.693 | 0.297 | 0.962 | 0.840 |
| S5_primary | 0.601 | 0.251 | 0.930 | 0.776 |
| **published best** | **0.586** | **0.480** | **0.747** | **0.663** |

Reading. On the metrics the paper itself publishes, our plain fused retrieval
is ahead of its best baseline on MRR and clearly ahead on both Hits figures,
and its MAP is higher at document level and lower at unit level. Our chunk is
twice the size of theirs, which flatters Hits, so the honest claim is narrow:
a reciprocal rank fusion of BM25 and a 22-million-parameter encoder, with no
reranker and no model call, lands in the same band as a commercial embedding
model with a cross-encoder reranker on this benchmark. It does not need to be
better than that to be interesting. It is two orders of magnitude cheaper.

The second reading matters more for our own design. Every arm that adds our
layers is below the plain fusion on every one of these metrics, and the
placebo overlay arm is the worst of all. The added layers do not merely spend
the budget badly. They also push evidence down the ranking.

## 2. Where our MultiHop-RAG result actually comes from

Joint hit at k: every evidence document of the query appears in the top k of
the ranking, before any budget cut. This is the ranking-level twin of our
JointRecall at a token budget, so the difference between the two columns is
exactly what the rendering budget costs.

| arm | joint@4 | joint@10 | joint@20 | joint@50 | joint@100 | JointRecall@4k | JointRecall@8k |
|---|---|---|---|---|---|---|---|
| ours_cheap | 0.335 | 0.577 | 0.870 | 0.964 | 0.969 | 0.255 | 0.420 |
| S2_lazy (see the note below) | 0.275 | 0.557 | 0.885 | 0.966 | 0.969 | 0.205 | 0.369 |
| S4_static | 0.256 | 0.543 | 0.882 | 0.966 | 0.969 | 0.185 | 0.361 |
| S5_planner_rules | 0.248 | 0.520 | 0.825 | 0.933 | 0.944 | 0.205 | 0.345 |
| S5_primary | 0.191 | 0.467 | 0.794 | 0.936 | 0.947 | 0.136 | 0.283 |
| S5_overlay_P0 | 0.178 | 0.434 | 0.692 | 0.917 | 0.938 | 0.135 | 0.271 |

Two corrections to how these rows may be used, both found by the adversarial
recheck of the mechanism analysis (docs/part1/MECHANISM.md). First, S2_lazy
logs a candidate list of 218 units where every other arm logs 100, so its
candidate-level recall is not comparable across arms and is not quoted here.
Second, at the 8,000-token budget the arms are not budget matched on
LongMemEval: they fill 6,200 to 7,586 of the 8,000 tokens because the
candidate list runs out, so cross-arm claims there are unsound. The 4,000-token
rows are matched and are the ones to read.

Three facts follow.

1. Every arm finds the evidence. At depth 100 all of them carry every evidence
   document for about 95 percent of queries. Nothing is missing from the index.
2. The ranking order is where the added layers first lose. At depth 10 the
   plain fusion carries all the evidence for 57.7 percent of queries and the
   full system for 46.7 percent. That is a ranking failure, not a budget one.
3. The budget then takes another large bite from everyone. The plain fusion
   drops from 57.7 percent at depth 10 to 25.5 percent inside a 4,000-token
   context, because a 4,000-token budget holds about nine of our chunks and
   the ranking has to have put all two to four evidence documents inside them.

The mechanism behind point 2 is measured in docs/part1/MECHANISM.md and is not
what this session first assumed. The added channels do not fetch junk from
nowhere; they reorder documents the keyword and dense channels already found,
and under reciprocal rank fusion a unit carrying a topic vote and a community
vote outranks an evidence chunk carrying only keyword and dense votes. The
median rank at which every evidence document has first appeared moves from 12
under the plain fusion to 15.5 with equal channel weights and 17 with the
planner weights, against a rendered window of nine. On chat memory the same
reorder is nearly free, because a 4,000-token budget renders about 50 of a
55 to 62 unit candidate list.

The difficulty scales exactly with how many documents a question needs, for the
plain fusion at depth 10:

| evidence documents needed | queries | joint@10 |
|---|---|---|
| 2 | 1,169 | 0.760 |
| 3 | 774 | 0.444 |
| 4 | 312 | 0.218 |

By question type at depth 10, plain fusion against the full system: comparison
0.692 against 0.514, inference 0.381 against 0.333, temporal 0.683 against
0.583. Inference queries are the hardest for everything.

## 3. MultiHop-RAG, answering, as context only

The paper (Table 6) gave its readers the top 6 retrieved chunks, or the
ground-truth evidence as a ceiling, and scored against the gold answer.

| system | reader | context | accuracy |
|---|---|---|---|
| published | GPT-4 | ground-truth evidence | 0.89 |
| ours, oracle_full | gpt-5.4 | evidence documents in full | 0.895 |
| published | GPT-4 | top 6 retrieved chunks | 0.56 |
| ours, ours_cheap | gpt-5.4 | 4,000 tokens of fused chunks | 0.845 |
| ours, S5_primary | gpt-5.4 | 4,000 tokens | 0.830 |
| ours, ours_cheap | gemini-2.5-flash-lite | 4,000 tokens | 0.787 |
| ours, closed_book | gpt-5.4 | nothing | 0.455 |
| ours, closed_book | gemini-2.5-flash-lite | nothing | 0.700 |

Two things are worth noting and one is a warning. The ceilings agree closely:
their GPT-4 on gold evidence reaches 0.89 and our gpt-5.4 on the evidence
documents reaches 0.895, which says our pipeline and judging are behaving
sanely against an independent implementation. The retrieved-setting gap, 0.845
against 0.56, is not a like-for-like comparison: the readers differ, and a
2026 model has read this 2023 news corpus, which is why the closed-book arm is
in the design at all. Under the cheap reader closed-book alone answers 70
percent of these queries.

## 4. LongMemEval, as context only

No published retrieval anchor is comparable. The LongMemEval paper's retrieval
numbers are on the M variant, about 500 sessions per question, not the S
variant we ran, which our earlier work established.

Accuracy on the S split, every row with its own reader and judge:

| system | reader | judge | accuracy |
|---|---|---|---|
| ours, oracle_full | gpt-5.4 | gpt-5.4 | 0.858 |
| ours, ours_cheap | gpt-5.4 | gpt-5.4 | 0.844 |
| Zep, arXiv 2501.13956 | gpt-4o | gpt-4o | 0.712 |
| full-context baseline, same paper | gpt-4o | gpt-4o | 0.602 |
| ours, ours_cheap | gemini-2.5-flash-lite | gpt-5.4 | 0.558 |
| our earlier e5 experiment | gemini-2.5-flash | gpt-5.4 | 0.690 |
| ours, closed_book | gpt-5.4 | gpt-5.4 | 0.074 |

The two rows that matter are the first two, and they are ours, so they compare
only to each other: cheap retrieval is 1.4 points from the evidence ceiling on
this corpus. Everything else in the table uses a different reader. The rows
below 0.712 are also ours, with a weaker reader, which is the plainest
demonstration in this document that the reader dominates.

For the oracle split, which we did not run as a study arm, the published
landscape is: GPT-4o reading the evidence sessions directly 0.870 (LongMemEval
paper), post-graph-rag 0.858 in its paper, 0.940 in its repository README under
its own judge panel, and 0.782 when its own repository regrades that run under
the official protocol with a gpt-4o judge. Those three numbers for one system
are the reason this document does not compare accuracy across papers.

## 5. What can and cannot be claimed

Can be claimed:

- On the MultiHop-RAG paper's own retrieval metrics, a cheap fusion of BM25 and
  a small local encoder is in the same band as the published best baseline,
  which uses a commercial embedding model and a cross-encoder reranker, with
  the chunk-size caveat stated.
- Our added layers are worse than that fusion on every published metric, on
  both corpora, at ranking level as well as at budget level. This is our own
  result against our own baseline and needs no cross-paper comparison.
- Our answering ceiling matches an independent implementation's ceiling on
  MultiHop-RAG (0.895 against 0.89), which validates the pipeline.

Cannot be claimed:

- That we beat Zep, or post-graph-rag, on answer accuracy. The readers and
  judges differ and we could not match theirs; the OpenAI key is dead and the
  Azure deployment carries no gpt-4o.
- That our retrieval beats the MultiHop-RAG baselines exactly, because the
  chunk size differs.
- Anything about Graphiti or about post-graph-rag on MultiHop-RAG, since both
  arms were stopped.

# The cost result: a better answer at an index cost of nothing

Written 2026-09-07. Every number is read from a file the experiment wrote.
This is the headline finding of Part 1 and it is stated here on its own,
because it is easy to lose inside the mechanism analysis.

## The claim

On LongMemEval_S, at the same rendered-token budget, with the same reader and
the same judge, our index puts more of the marked evidence in front of the
reader than post-graph-rag does, at an index cost of nothing.

One precision that the claim needs, and that an earlier draft of this document
did not carry. The arms that score 0.947 and 0.957 read post-graph-rag's
extracted entities, relations and aliases as two of their six channels, so
those two arms do rest on his upfront extraction even though they add little
because of it. The arms that send no document to a language model before the
question is asked are S5_noPGR at 0.945 and ours_cheap at 0.938, and both of
them are still far above his 0.574. So the no-upfront-extraction version of
the claim stands on those two rows, and the difference between them and the
arms that use his tables is 0.002.

## Index cost, 500 questions, 61.2 million tokens of corpus

post-graph-rag indexes each question's haystack from scratch, as its own
LongMemEval harness does, at two extraction calls per session document. Our
index embeds every unit once, fits topics, builds a noun-phrase graph and its
communities, and computes the topic-community alignment. Only the overlay
calls a model.

| system | index-time model calls | index cost | query calls per question | JointRecall at 4,000 tokens |
|---|---|---|---|---|
| post-graph-rag, study model (gemini-2.5-flash-lite, the cheapest available) | 140,784 | USD 64.97 | 2 (USD 1.90 over 500) | 0.574 |
| post-graph-rag, his own published models (gemini-3.7-flash index) | not run in full | USD 361 projected from the 17 questions that run metered | 2 | 0.611 on the 18 calibration questions |
| ours, every layer including the overlay | 2,566 | USD 0.28 | 1 (the planner) | 0.947 |
| ours, overlay dropped, which raises the score | 0 | USD 0.00 | 1 | 0.949 |
| ours, overlay dropped and none of his tables (S5_noPGR) | 0 | USD 0.00 | 1 | 0.945 |
| ours_cheap, no model call at index or query time | 0 | USD 0.00 | 0 | 0.938 |

Ratios on index cost alone: 231 times cheaper than his cheapest build, and
without limit once the overlay is dropped. On his own models the projection is
USD 361 over the 17 questions that run metered, or USD 391 taken per session
over the same run; either way it is over a thousand times our index cost.
Per million tokens of the 61.2 million his protocol indexes: his USD 1.06 for
the index and USD 1.09 including his query calls, ours USD 0.005.

Wall time for our index: 112 minutes of CPU on one laptop, of which 65 minutes
is the noun-phrase graph stage and 62 of those are spaCy over 998,042
sub-units. His build took 13.3 hours across four parallel shards against a
hosted API.

## Query cost

| arm | model calls per question |
|---|---|
| ours_cheap | 0 |
| S4_static, every channel fused | 0 |
| S5_primary | 1, the planner |
| S2_lazy | up to 20 relevance tests |
| post-graph-rag, mix mode | 2, keyword extraction and synthesis |

## The result at that cost

Retrieval, JointRecall at a fixed rendered budget, 469 to 470 answerable
questions:

| arm | 4,000 tokens | 8,000 tokens |
|---|---|---|
| S4_static | 0.957 | 0.970 |
| S5_primary | 0.947 | 0.964 |
| ours_cheap | 0.938 | 0.957 |
| post-graph-rag | 0.574 | 0.804 |

Answers, same context budget, same reader, same judge, so this is a controlled
comparison:

| arm | reader gpt-5.4 | reader gemini-2.5-flash-lite |
|---|---|---|
| S5_primary | 0.832 | 0.542 |
| post-graph-rag | 0.624 | 0.466 |

Paired, at 4,000 tokens: T1 on retrieval is plus 0.371, CI [+0.324, +0.420],
188 wins to 14 losses, significant after Holm. On answers under the strong
reader, plus 0.208, 121 wins to 17 losses, p below 0.0001; under the cheap
reader, plus 0.076, p 0.0005.

## Why he loses, stated fairly

His index is good. His candidate-level recall on LongMemEval is 0.998, meaning
the evidence is in what his system retrieves for almost every question. He
loses at the rendering step, because his retrieval unit is a whole session
document of about 10,000 characters, so two of them fill a 4,000-token
context. At 8,000 tokens he recovers to 0.804.

So the honest form of the claim is not that his extraction is worthless. It is
that his extraction is expensive and his unit is coarse, and the second problem
wastes the first. Our index costs nothing and its unit fits the budget.

## What the extraction bought inside our own system

Our S4 and S5 arms read his entities, relations and aliases as two of their six
channels. Switching those channels off changes almost nothing:

| | his tables on | off | difference |
|---|---|---|---|
| LongMemEval, 4,000 tokens | 0.947 | 0.945 | +0.002 |
| LongMemEval, 8,000 tokens | 0.964 | 0.964 | 0.000 |
| MultiHop-RAG, both budgets | identical | identical | 0.000 |

USD 64.97 and 140,784 model calls of upfront extraction were worth two
thousandths of joint recall on one corpus and nothing on the other. On
MultiHop-RAG the relation and entity channels returned zero hits in the entire
candidates file.

## What this settles about the design question

The owner's position was that pushing everything into a graph up front is too
expensive and the embedding space should carry that load. On this evidence
that is right, and the measurement is unusually clean, because the expensive
upfront extraction is present as a channel in our own system and can be
switched off without changing anything else.

Two qualifications belong with the claim. First, the corpus is chat memory
with a per-question haystack, and the same test on a corpus where entities
recur across many documents may read differently; MultiHop-RAG could not
answer that because his index there was stopped. Second, the parts of our
system that do reason about structure, the topic and community channels, the
overlay and the planner, are not what wins either: the best arm on this corpus
is the all-channel static fusion, which makes no model call, and the cheap
fusion is within two points of it. The win is the representation and the
budget discipline, not the reasoning layer.

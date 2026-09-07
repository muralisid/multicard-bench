# The cost result: a better answer with no model calls at index time

Written 2026-09-07. Every number is read from a file the experiment wrote.
This is the headline finding of Part 1 and it is stated here on its own,
because it is easy to lose inside the mechanism analysis.

## The claim

On LongMemEval_S, at the same rendered-token budget, with the same reader and
the same judge, our index puts more of the marked evidence in front of the
reader than post-graph-rag does, and it does so without sending a single
document to a language model before the question is asked.

## Index cost, 500 questions, 61.2 million tokens of corpus

post-graph-rag indexes each question's haystack from scratch, as its own
LongMemEval harness does, at two extraction calls per session document. Our
index embeds every unit once, fits topics, builds a noun-phrase graph and its
communities, and computes the topic-community alignment. Only the overlay
calls a model.

| system | model calls at index time | cost | JointRecall at 4,000 tokens |
|---|---|---|---|
| post-graph-rag, study model (gemini-2.5-flash-lite, the cheapest available) | 144,281 | USD 66.87 | 0.574 |
| post-graph-rag, his own published models (gemini-3.7-flash index) | not run in full | USD 361 projected from the 17-question calibration row | 0.611 on those 17 |
| ours, every layer including the overlay | 2,566 | USD 0.28 | 0.947 |
| ours, overlay dropped, which raises the score | 0 | USD 0.00 | 0.949 |
| ours_cheap, no model call at index or query time | 0 | USD 0.00 | 0.938 |

Ratios: 238 times cheaper than his cheapest build, 1,283 times cheaper than
his published configuration, and without limit once the overlay is dropped.
Per million tokens of corpus indexed: his USD 1.09, ours USD 0.006.

Wall time for our index: 112 minutes of CPU on one laptop, of which 65 minutes
is spaCy over 998,042 sub-units. His build took 13.3 hours across four
parallel shards against a hosted API.

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
| S5_primary | 0.832 | 0.543 |
| post-graph-rag | 0.624 | 0.467 |

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

USD 66.87 and 144,281 model calls of upfront extraction were worth two
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

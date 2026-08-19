# multicard-bench

Experiments, reference code, and the paper for multi-card retrieval. Private while the work is tested; the owner flips it public when it is ready.

## The idea, in plain words

Think of a library. Every book gets one index card. The card tries to say everything about the book in one line. That is how most search systems work today. Each document gets one number-summary, called a vector. It is like a smoothie: all the flavors blended into one. If you ask about just the banana, the smoothie tastes only a little like banana.

We do it differently. We write several cards for the same book. One card says what problem it talks about. One card says who it is about. One card says what it asks you to do. When you ask a question, we check the kind of card that fits your question. It is like a fruit salad: every fruit is still there, and you can find each one.

There is also a helper at the door. Before anyone reads with care, the helper makes one quick pass and sorts out which books belong on the topic shelf at all. So the slow, careful reading stays small and cheap, even when the library is huge.

One more trick, and it is the new one. Who gets the results matters. When a robot (an AI model) will read them and write a summary, we slip in a few surprising cards on purpose. The robot is good at using odd clues. When a person will read them, we keep the list focused, because to a busy reader the surprises just look like noise. This repo tests whether that trick really works, with fair experiments anyone can rerun.

## What the experiments test

**E1, many cards vs one vector.** We measure whether several purpose cards per document beat one blended vector when a question targets one aspect of a document. Theory says the blended vector's match fades as documents carry more aspects; the cards keep it. We test against strong baselines (BM25, plain chunks, hybrid fusion, rerankers, a small late-interaction model) so a win means something. A matched "plain chunks" control checks that the win comes from the purpose framing, not just from splitting documents up.

**E2, the door helper and the bill.** We measure how well the cheap relevance gate separates on-topic from off-topic on a public email corpus, and we meter the full cost: the two-pass way (embed everything cheaply, use the expensive model only on discovered topics) versus sending every document through a large model. The claim is a scaling law: the gap widens as the corpus grows.

**E3, who reads the results.** Same pool of candidates, two readers. For an AI writer we deliberately add low-similarity picks; for a human report we keep a focused blend. We score both with blind judges from three model families plus objective checks (question coverage, factual support, distractor counts), and we test the interaction: does the right policy really depend on the reader?

## How to run

One command per experiment (arrives with Sprint R1):

```
uv run mcb run <experiment_id>
```

Every run is seeded, cached, budget-capped, and logged; two runs with the same seed produce byte-identical outputs. Datasets are downloaded by script with checksums; nothing is redistributed.

## Paper and DOI

Placeholders until release: paper PDF (paper/), Zenodo DOI, arXiv listing.

## Provenance

This repo is a clean-room build from docs/PATTERN_SPEC.md (see docs/CLEANROOM.md). The pattern originated in the authors' production experience in the social media domain; here it is evaluated on public corpora only, and every reported number regenerates from this repo.

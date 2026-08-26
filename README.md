# multicard-bench

The experiments, the code, and the data behind the retrieval research published at
[agenticarchitectureskills.com/patterns](https://www.agenticarchitectureskills.com/patterns).

Every number quoted on that site regenerates from this repository. If you want to check a
claim, this is where you check it.

## The idea, in plain words

Think of a library. Every book gets one index card, and that card tries to say everything
about the book in one line. That is how most search works today: each document gets one
number-summary, called a vector. It is like a smoothie, with all the flavours blended into
one. If you ask about just the banana, the smoothie tastes only a little like banana.

The idea we tested was to write several cards for the same book instead. One card says what
problem it covers, one says who it is about, one says what it asks you to do. When you ask a
question, the search looks at the kind of card that fits your question. It is like a fruit
salad: every fruit is still there, and you can find each one.

Two more ideas sat alongside it. A helper at the door makes one quick cheap pass and sorts
out which books belong on the topic shelf at all, so the slow expensive reading stays small.
And a guess that who reads the results should change what you return: more variety when an
AI will summarise them, a tighter list when a person will read them.

## What we found, including what did not work

The short version: the first idea is real but for a plainer reason than we expected, the
second is useful but much smaller than we claimed, and the third did not hold up.

**Several cards beat one blended vector. Confirmed.** On documents covering many aspects, the
single blended vector collapses as you add aspects, from 0.815 down to 0.294 on our quality
measure, while any multi-card version stays above 0.63. On a public stress test built for
this, the blended vector scored 0.314 against 0.988 for per-aspect cards.

**But most of that win is having more cards, not smarter cards.** This is the finding that
cost us the most to learn. If you cut the same document into arbitrary fixed-size chunks, you
also get several vectors, and that alone recovers most of the benefit. A deliberately
mindless three-word window recovered 0.610 of the 0.674 we had been crediting to clever
purpose framing. So the honest claim is much narrower than the one we started with.

**On real prose the clever cards actually lost.** Against a fair control with the same number
of vectors per document, purpose cards scored 0.635 against plain chunking's 0.667, losing 70
comparisons to 36. Where they win is where questions target one aspect of a document; where
questions concern a whole finding, they hurt. We report both.

**Plain keyword search remains essential.** On queries that name a specific thing, old
fashioned keyword matching scored 0.997 against 0.314 for the blended vector. Combining the
two costs almost nothing and is the single easiest thing on this list to adopt.

**The cheap door helper works, and is not where the money is saved.** It separates on-topic
from off-topic well, at 0.933 by the usual measure. But if you insist on keeping 95 percent
of the genuinely relevant material, it only throws away about a third of the corpus. That is
a saving of one-and-a-bit times, not the order of magnitude the original framing implied.

**Our cost claim was wrong in an interesting way. Corrected.** We claimed a widening scaling
law: the more documents, the bigger the saving. It is not. Once you hold the summaries to a
useful level of detail, the advantage *shrinks* as the corpus grows. What survives is a large
constant saving, roughly 200 times at comparable detail, which is still worth having but is a
different and smaller claim. The apparent widening came from our own summaries quietly
getting coarser as the corpus grew.

**The reader-dependent variety idea did not hold. Not supported.** This was the most novel
claim and it failed. The effect measured positive under a judge from the same model family as
the writer, weaker under a more capable judge from that family, and vanished entirely under a
judge from a different company. Shrinking as you move away from the generator's own family is
the signature of the judge marking its own homework, not a real effect. We are not claiming
it.

**One thing we did not predict turned out to be the most solid result.** Harvesting varied
results reduced unsupported statements in generated summaries, and unlike the finding above,
this one held up under judges from every family we tried. We went looking for one thing and
found another.

## What is in here

- `src/` the experiment code, one command per experiment
- `results/` committed outputs for 20 experiments, including per-query results so you can
  recompute the statistics yourself rather than trusting our summary numbers
- `paper/` the write-up in LaTeX, with its figures
- `docs/` how the experiments are specified, preregistered, and run reproducibly

## How to run it

```
uv sync --python 3.11
uv run mcb run <experiment_id>
```

Every run is seeded, cached and budget-capped, and two runs with the same seed produce
byte-identical output. The corpora are public and download on first use with checksums.
Nothing here redistributes a dataset we do not own.

## How this was kept honest

The decision rules were written down and frozen before any result was read, so a
disappointing number could not be reinterpreted after the fact. Every arm is reported,
including the ones that beat our own method. Where a model was used to grade output, we
repeated the grading with a model from a different company, which is what exposed the failed
finding above. Five headline results in this programme reversed at least once, and the
reversals are published rather than quietly dropped.

## Licence

Code is MIT (`LICENSE`). The documentation, results and paper are CC BY-SA 4.0
(`LICENSE-CONTENT.md`), matching the guide the research supports. Attribution: Murali Sid,
https://www.agenticarchitectureskills.com

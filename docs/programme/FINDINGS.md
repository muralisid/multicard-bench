# What we did, what held, and what did not

Written 2026-08-20 as the single document to read if you want to understand this programme. Everything else is either raw material or a chronological log. **Read this instead of the sprint brief**, which is an append-only diary that still contains conclusions later withdrawn; it is kept that way deliberately, so the record shows what was believed and when, but it is a poor way to learn what is true.

Published 2026-08-24 alongside the benchmark. The patent hold this document was originally written under was lifted, and the decision was to publish the record in full rather than only the conclusions that survived.

---

## 1. The question, in one paragraph

A document that is about several things at once gets one embedding, and that embedding is an average of everything the document says. The proposal was to give each document several embeddings instead, one per purpose ("what is being asked", "what was promised", "what is the risk"), so a query about one aspect matches the card for that aspect instead of a blurred average. Around that sat three supporting claims: that a cheap filter can discard most of a corpus before expensive processing, that doing the expensive processing per discovered topic rather than per document is dramatically cheaper, and that how much diversity you put in a result set should depend on whether a model or a person is going to read it.

The last of those is the patent kernel. The others were already published by you in March, so they were never the valuable part.

## 2. How the experiments are built, and why

The whole design exists to separate one thing from another thing that looks identical from the outside.

**The confound.** Giving a document several cards means giving it several embeddings. Cutting the same document into arbitrary chunks also means several embeddings. If cards beat a single pooled vector, that proves nothing about purposes, because chunks would beat it too. So every experiment carries three arms:

| arm | what it is | what it isolates |
|---|---|---|
| **pooled** | one embedding of the whole document | the baseline everyone uses |
| **chunk** | the same text cut into fixed word windows | capacity alone, no semantics |
| **card** | the same text split by purpose | capacity plus alignment |

The gap between pooled and chunk is what more embeddings buy. The gap between chunk and card is what purposes buy. Only the second one is the hypothesis. **Matching the number of units per document between chunk and card is what makes the comparison mean anything**, and getting that wrong was the single largest error in this programme.

**The corpora form a ladder of realism**, chosen so the answer can differ along it, which it does:

| corpus | what it is | why it is in the study |
|---|---|---|
| **S0, synthetic** | documents built from k passages from k unrelated topics; queries target one | the best case: aspects genuinely exist, queries genuinely target them, and k is a dial we control |
| **LIMIT** | 46 documents, each a person and 44 things they like; queries ask who likes one thing | a published stress test designed to break single-vector retrieval; the decomposition is free and perfect |
| **SciFact** | 5,183 real scientific abstracts, 300 real claims, human relevance judgements | the honest case: real prose nobody wrote for us, real labels, and queries about a whole finding rather than one aspect |
| **Enron** | 20,000 real corporate emails | the enterprise case, used for the gate and the cost study |

The point of the ladder is that a technique which only works on the top rung is not a technique, it is an artefact of the test.

## 3. The answers

### What held

**Multi-vector beats single-vector, decisively.** On the synthetic corpus, pooled retrieval collapses from 0.815 nDCG@10 at one aspect per document to 0.294 at ten. Splitting the document up, by any method, recovers most of that. This is the least surprising and best-supported finding in the programme.

**Purpose alignment helps, but only under a condition, and the condition is sharp.** Where documents carry several distinct aspects and queries target one of them, cards beat matched chunks by 0.188 nDCG@10 at ten aspects, while using 25 percent fewer embeddings. Where queries are about a document as a whole, cards *lose* to chunks by 0.032 at matched capacity, on human-judged data. The mechanism is intuitive once seen: an aspect boundary throws away evidence that a blind window happens to keep together, and if no card is the unit the query asks about, that loss buys nothing.

This is your own validity analysis being confirmed. You listed "the card taxonomy must align with the query distribution" as a *condition* of validity. It is.

**A cheap gate does discriminate** (ROC-AUC 0.933 separating Enron business email from Usenet).

**The diversity harvest is efficient.** Within a gated pool, taking a quota from the low-similarity tail buys subtopic coverage about twice as cheaply as MMR (+0.315 coverage for -0.046 relevance, against MMR's +0.314 for -0.098) and admits fewer answerless documents than either MMR or a DPP. The mechanism: MMR and DPP diversify the whole list so the top drifts off-relevance, while the harvest keeps a relevance head and spends only the tail.

**Harvesting makes outputs more grounded.** Summaries written from a diverse selection contain less unsupported content (-0.067 for the human-facing condition, p=0.0018). This was not predicted, needs no judge to measure, and survived every check. It is the most defensible thing the consumer study produced.

### What did not hold

**"Alignment, not count" as an unconditional claim.** This was the paper's thesis and it is false. On LIMIT, a semantically blind three-word sliding window recovers 0.610 of the 0.674 improvement I had attributed to purposes. Bare attribute strings beat purpose-framed cards. Most of the benefit is capacity.

**The economics scaling law.** The cost advantage appeared to widen with corpus size (N^0.435). It does not. That was clusters getting coarser as the corpus grew: at a fixed useful granularity the advantage *shrinks* (N^-0.273). What remains is a real constant-factor saving of roughly 200x, itself flattered by pricing local compute at zero and by comparing two designs that produce different deliverables. Fourteen topics over ten thousand messages is not a summary of anything.

**The gate as the source of the saving.** At the high recall such a filter is supposed to hold, it discards only 33.7 percent of a corpus, and at 0.99 recall only 12.1 percent. It contributes a linear factor of one to two. The order of magnitude comes from processing topics instead of documents. Also: three sets of *deliberately irrelevant* anchor phrases reach 0.860 AUC against the purpose-built ones at 0.933, and plain logistic regression on the raw embeddings reaches 0.995. The elaborate weighting earns 0.002 over equal weights.

**The consumer flip, which is your patent kernel.** Three judges scored the same 240 generations: the interaction measured +0.071 (p=0.009), +0.047 (p=0.211), and +0.008 (p=0.763), shrinking monotonically as the judge moved away from the generator's own model family and vanishing at the cross-family judge. That is the signature of a judging artefact. It is not disproved, all three estimates are positive and 60 tasks is small, but there is currently no evidence for it.

## 4. The part worth your intuition: five reversals and what caused each

Every headline in this programme reversed at least once. The causes are more useful than the results.

| what I first reported | what was wrong | how it was caught |
|---|---|---|
| Cards beat chunks 9:1 on LIMIT | The chunk control had 4 units per document against the cards' 44 | Adversarial reviewer re-ran it with matched counts |
| Aspect dilution follows the theory on synthetic data | The "cards" were the exact passages the queries were generated from, and 81 percent of true positives were unlabelled because filler sentences realised real facts | Same reviewer, reading the generator |
| Multi-card is catastrophic on SciFact (-0.193) | The card builder kept only the top two spans per aspect, so cards held 44.6 percent of each document against the chunks' 100 percent | Measured the text coverage after the result looked too extreme |
| Diversity coverage is nearly free | An integer-division bug gave 2 distractors per task instead of 40, so nothing could be lost and every relevance metric sat at its ceiling | Same reviewer |
| The consumer interaction is significant | One judge, from the same family as the generator | A cross-family judge, on your suggestion |

Two patterns run through all five.

**A control that is not matched is not a control.** Three of the five come from an arm that was quietly weaker than its comparison: fewer units, less text, no distractors. The comparison then measures the handicap rather than the hypothesis.

**A measurement that cannot be wrong is usually not measuring.** The synthetic experiment could not have produced a negative result, because the cards were the ground truth. The first coverage metric fired on shared vocabulary, so outputs "covered" 4.3 subtopics from selections containing 1.8, which is arithmetically impossible and is exactly the check that caught it. **Ask what result would falsify this, and if there isn't one, the experiment is decoration.**

## 5. What you can and cannot say

**Can say:** several embeddings per document beat one on aspect-rich material; purpose alignment helps when the taxonomy matches the query distribution and hurts when it does not, with both sides measured; the two-pass design is about two orders of magnitude cheaper at comparable topic granularity; the outlier harvest is a more efficient way to buy coverage than MMR or a DPP, and produces better-grounded summaries.

**Cannot say:** that alignment rather than capacity explains multi-card's benefit; that the cost advantage widens with corpus size; that the gate is where the saving comes from; that diversity should be conditioned on the consumer.

**For the patent:** the kernel is still filable, since patents need novelty and not empirical proof, and the retrieval half of the claim genuinely stands. What you no longer have is evidence for the consumer conditioning that the claims are anchored on. That matters at the month-10 decision on whether to spend real money on a complete or PCT filing, not now.

## 6. Reading order

1. **This file.**
2. `program/hypothesis-register.md`, six rows, each with its final disposition and the numbers behind it. The one-screen version of the above.
3. `program/reviews/r1-red-team-2026-08-20.md`, the adversarial review that found four fatal defects. The most valuable single artifact here, and the best argument for the process.
4. `../multicard-bench/README.md`, the plain-language explanation of the idea itself.
5. `../multicard-bench/src/multicard/experiments/`, one file per experiment, each opening with a docstring saying what it tests and what it cannot show. Read `e0_dilution.py` and `e1_beir.py` first; they carry the main comparison.
6. `program/sprints/sprint-r1-dilution.md` only if you want the chronology, including what was believed before it was corrected.
7. `program/ip/disclosure-dates.md` before the filing, because it is the file the patent decisions rest on.

To rerun anything: `uv run mcb run <name>` in the bench repo, where the names are `e0_dilution`, `e1_limit`, `e1_scifact`, `e2_gate`, `e2_economics`, `e3_diversity`, `e3b_consumer`, `e1_anchor_sensitivity`. Everything is seeded and cached; two runs produce byte-identical output, and a test enforces it.

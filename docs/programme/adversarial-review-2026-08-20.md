# Red-team review of Sprint R1: REJECT (1.5 / 5)

Run 2026-08-20 by the adversarial reviewer against multicard-bench at commit 1cec616. The reviewer re-ran experiments rather than reading code, and falsified the central claim. This is the most valuable artifact the programme has produced so far and it must be acted on before anything is published or quoted.

## The one-line summary

**"Alignment, not count" does not survive.** The benefit attributed to purpose-aligned cards is, on the evidence in this repository, mostly the benefit of having more embeddings per document. Three of the four experiments additionally have construction defects that make their headline numbers wrong.

## Fatal findings

**F1. The chunk control was unmatched by eleven times, and matching it destroys the claim.** On LIMIT the control had 4.0 chunks per document against 44.0 cards, because the chunk width was set to 30 words on documents averaging 73. Re-running with only the chunk granularity varied, nDCG@10 is a smooth monotone function of units per document: 0.390 at 4 units, 0.611 at 10.8, 0.892 at 24.7, 0.924 at 36.8, against 0.988 for cards at 44. A semantically blind three-word sliding window, with no taxonomy and no anchors, recovers 0.610 of the 0.674 attributed to alignment. The residual card advantage over that window is +0.064, which is smaller than the +0.076 the paper labelled the pure capacity effect. Worse, dropping the template and using the bare attribute string beats the cards outright (+0.010, p=0.0001). The card text was also being reconstructed as "subject likes attribute" while LIMIT queries are literally "Who likes X?", so the card text was the query template with the answer substituted, against a raw-text control.

**F2. SciFact contradicts the thesis and was absent from the paper.** Independently found by both the reviewer and this programme in the same hours. Cards 0.452 against chunks 0.673 and pooled 0.645, a loss of 0.221 against the control, Holm-significant.

**F3. The synthetic corpus's relevance labels are 81 percent incomplete.** Passage filler sentences draw attributes and actions from the same pool in the same template, so they realise real facts that were never labelled. At k=10 there are 33.3 labelled relevant documents per query against 166.0 that actually state the queried fact. Corrected, the headline changes from 0.273 to 0.622 for pooled, and a blind 20-word chunk control reaches 0.965 against oracle cards at 0.970.

**F4. The synthetic experiment is tautological.** Its "cards" are the exact passages the queries were generated from, and the real card builder is never called. Max-over-cards similarity is flat across k by construction, not by measurement, because the card text is byte-identical at every k.

**F5. The diversity experiment's subtopic labels are wrong for 73.7 percent of documents,** by the same filler mechanism as F3, which makes both alpha-nDCG and subtopic recall uninterpretable.

## Major findings

**F6.** The diversity experiment's distractor pool is 2 documents per task, not 240: an integer-division bug made the parameter inert. So "coverage is nearly free" is a saturation artefact, not a property of gating. Roughly 97 percent of the pool has gain 1.0 and nDCG cannot detect a penalty.

**F7.** The diversity query is sampled independently of the facts defining its subtopics; in 27 percent of tasks the query names an attribute no subtopic-bearing document is about.

**F8.** The gate's "purpose-specific anchors" claim fails a null control. Three sets of deliberately irrelevant anchors reach ROC-AUC 0.860 against 0.933 for the purpose anchors, one purpose-specific set reaches 0.852, and plain logistic regression on the raw embeddings reaches 0.995. The anchor gate is 0.062 AUC worse than the trivial supervised baseline on the same vectors, and most of its discrimination comes from projecting onto three directions of an already-separable space.

**F9.** The committed economics exponent was fitted through a single data point, and the committed guard now makes that result unreproducible by the committed code, contradicting the reproducibility statement. Encoder compute is priced at exactly zero, so the cost ratio is set by zeroing the denominator, and the two designs produce different deliverables.

**F10.** Card count is itself a function of anchor quality (3.74 cards per document with the real anchors, 1.10 with junk), so in this implementation the taxonomy knob and the capacity knob are the same knob, while the chunk control has no such knob.

## What held up

The statistical core is correct: Holm step-down verified against the textbook definition over 2,000 randomised comparisons, the permutation test is properly two-sided, the bootstrap is a valid percentile interval, and nDCG, recall, MRR, RRF, alpha-DCG and subtopic recall all match their standard definitions on hand-worked cases. Chunk windows cover 100 percent of tokens, so the control was not disadvantaged that way. The candidate-depth asymmetry is unprincipled but numerically inert: identical to three decimals at every depth tested. There is no test-set leakage anywhere; anchors never see queries or judgements.

## Assessment

The reviewer is right on every fatal finding. Two of them (F2 and part of F9) this programme had already found independently in the same session, which is some evidence the process works, but F1, F3, F4 and F5 were missed and they are the ones that matter most, because F1 removes the paper's headline contribution and F3 and F4 remove the evidence base for its first result.

The honest position now is that **the programme has no established claim that purpose-aligned cards beat an equally-sized set of naive chunks.** That is not the same as the hypothesis being false: every experiment that appeared to support it was flawed, and the corrected experiments have not been run. But it does mean nothing may be published, submitted, or quoted until they are.

## Consequences

1. The ECIR 2027 abstract deadline of 21 September is now at risk and should be treated as unlikely rather than planned.
2. The paper's abstract and both results sections must be rewritten from scratch once corrected numbers exist.
3. The patent is unaffected in its kernel, since the consumer-adaptive selection claim does not depend on multi-card retrieval beating chunking, but the multi-card supporting embodiments are weaker than the specification's background implies. Worth raising with a professional adviser at the complete or PCT stage rather than now.

## Addendum, same day: the economics scaling claim is a hyperparameter artefact

Re-running the economics with topic coverage reported alongside cost, at two clustering settings, settles the question the reviewer raised at F9 and adds a finding of its own.

At coarse clustering (minimum cluster size 25), the setting used for the original result:

| N | topics K | coverage of survivors | documents per topic | cost ratio |
|---|---|---|---|---|
| 500 | 2 | 49.6% | 62 | 325x |
| 2,000 | 2 | 59.8% | 299 | 1,481x |
| 5,000 | 7 | 97.6% | 349 | 947x |
| 10,000 | 7 | 94.8% | 677 | 1,824x |
| 20,000 | 14 | 67.9% | 485 | 1,841x |

At useful granularity (minimum cluster size 10):

| N | topics K | coverage of survivors | documents per topic | cost ratio |
|---|---|---|---|---|
| 500 | 3 | 54.0% | 45 | 216x |
| 5,000 | 16 | 97.1% | 152 | 418x |
| 10,000 | 88 | 32.1% | 18 | 152x |
| 20,000 | 132 | 26.3% | 20 | 204x |

**The direction of the headline finding reverses with the hyperparameter.** At coarse clustering the cost advantage grows as N^0.435 and topics grow sublinearly as N^0.52. At finer clustering topics grow superlinearly as N^1.245 and the advantage *shrinks* as N^-0.273. The claim that the advantage widens with corpus size was therefore an artefact of the clusters getting coarser as N grew, not a property of the design.

What survives is smaller and more defensible: at comparable topic granularity the two-pass design costs roughly two hundred times less than sending every document to a generative model, which is a real and useful saving. What does not survive is the scaling law. Two further caveats stand from the reviewer: encoder compute is priced at zero, which flatters the ratio, and the two designs produce different deliverables, so the ratio applies to topic discovery rather than to per-document understanding.

Coverage also exposes how coarse the coarse setting is. Fourteen topics over ten thousand messages, at 485 documents per topic, is not a summary of a corpus in any sense a practitioner would accept.

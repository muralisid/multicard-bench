# Sprint R1 brief: aspect dilution measured

Produced 2026-08-19 by the research agent. Feeds P1, and partly P2. Private; the public artifacts are the code, results, and figure in multicard-bench at commit 01c60f9.

## Headline dispositions

- **P1 (aspect dilution is real and material): SUPPORTED on synthetic data, with a caveat on magnitude.** Pooling a document's k aspects into one vector degrades aspect-directed retrieval severely; representing the same document as k cards and scoring by the maximum does not. At k=10 the pooled representation reaches nDCG@10 of 0.273 against 0.672 for max-over-cards, a gain of +0.399 (95% CI +0.360 to +0.438, paired permutation p < 0.0002, Holm-corrected across 21 tests, 186 wins against 13 losses over 200 queries). Every k from 2 upward is significant in the same direction. This is synthetic evidence only; C1 and C2 decide whether it survives on real corpora.
- **The theory's shape holds, its magnitude does not.** Measured dilution follows k^-0.32; the near-orthogonality argument predicts k^-0.5. The paper should report the fitted exponent and explain the gap rather than claim agreement.
- **The mechanism behaves exactly as the theory says it should.** Mean cosine to relevant documents under max-over-cards is flat across k (0.498 at k=1, 0.493 at k=10, varying by under 0.02 throughout), while the pooled vector decays monotonically from 0.498 to 0.235. The aspect signal is preserved by cards and diluted by pooling, which is the claim.
- **A sanity check passes that would have caught a whole class of bugs:** at k=1 the two representations are byte-identical in every metric (delta exactly 0.000, 200 ties), as they must be, since a one-aspect document has one card.

## Evidence

Corpus S0: documents assembled from k passages drawn from k of ten unrelated topic pools (legal, logistics, finance, engineering, personnel, marketing, facilities, research, security, procurement), 500 documents per k, k in {1,2,3,4,5,7,10}, 200 queries per k, encoder all-MiniLM-L6-v2, seed 13.

| k | nDCG@10 pooled | nDCG@10 max-card | delta | 95% CI | relevant per query |
|---|---|---|---|---|---|
| 1 | 0.682 | 0.682 | +0.000 | exact tie | 3.7 |
| 2 | 0.395 | 0.571 | +0.177 | +0.145 to +0.207 | 6.6 |
| 3 | 0.308 | 0.565 | +0.257 | +0.226 to +0.288 | 9.7 |
| 4 | 0.271 | 0.556 | +0.285 | +0.253 to +0.317 | 13.2 |
| 5 | 0.300 | 0.585 | +0.285 | +0.251 to +0.318 | 16.6 |
| 7 | 0.282 | 0.656 | +0.374 | +0.339 to +0.408 | 23.4 |
| 10 | 0.273 | 0.672 | +0.399 | +0.360 to +0.438 | 33.3 |

## Two problems found and fixed, both worth recording

**The first version of the experiment was ill-posed and its retrieval numbers were meaningless.** Queries were generic templates drawn from a topic pool, and relevance was defined as one arbitrarily chosen document. Because hundreds of documents contained passages from the same pool, dozens were equally good answers, and nDCG@10 sat near 0.08 for every system. The fix was to define relevance by construction: each passage realises a specific (pool, attribute, action) fact, a query asks about one fact, and every document carrying a passage that realises it is relevant. Relevance sets then contain between four and thirty-three documents and the metrics mean something. The lesson generalises to C1 and C2: relevance has to come from the generative process or from human labels, never from an assumption that one document is the answer.

**The determinism gate failed three times before passing.** First run: 799 of 1400 rows differed between two seeded runs, from float32 similarity wobble of about 1e-8. Rounding the recorded output to six decimals did not fix it, because values sitting on a rounding boundary still flipped. Pinning BLAS threads cut the differences to 35 rows but did not eliminate them, because macOS Accelerate parallelises regardless of the thread environment variables. Accumulating the similarity matmuls in float64 fixed it: two seeded runs are now byte-identical. Worth carrying forward as a standing requirement, since every later experiment inherits the same matmul.

## Caveat to carry into the paper

The relevance-set size grows with k as a side effect of the construction (3.7 documents per query at k=1 rising to 33.3 at k=10), so absolute nDCG is not comparable across columns. The pooled-versus-cards comparison at each k is unaffected, because both systems see the same relevance sets, and that comparison is the claim. The paper must state this rather than let a reader infer that retrieval improves with k.

## What was built

In multicard-bench: the package skeleton and single entrypoint (`uv run mcb run <id>`); the synthetic corpus generator with constructed relevance; an encoder with an on-disk cache keyed by model and text; ranking metrics (nDCG, recall, MRR, judged@k, success@k, reciprocal rank fusion) with fourteen hand-worked golden tests, all passing; paired significance testing (permutation test, bootstrap confidence intervals, Holm correction); the analysis script; and Figure 1 in three panels (aspect signal, measured dilution against theory, retrieval consequence).

Environment note for reproducibility: the machine is an Intel Mac, so torch is pinned to 2.2.2, the last release with macOS x86_64 wheels, and transformers to the 4.x line that works with it. Everything runs on CPU. A full run takes about six minutes cold and is cached thereafter.

## Gates

- Determinism: PASS, byte-identical across two seeded runs.
- Golden metric tests: PASS, 14 of 14.
- EVAL-BOUNDARY: PASS, zero HARD hits against the blocklist.
- EVAL-STYLE: PASS, no em dashes in source.
- EVAL-BUDGET: no API spend; the encoder is local and free.
- LIMIT sanity gate: NOT YET RUN. That is the remaining R1 item and moves to the next session along with the Enron and KILT ingest.

---

# Sprint R1, part 2: the LIMIT capacity test

Added 2026-08-19. Feeds P2 and P6. Artifacts in multicard-bench, experiment `e1_limit`.

## Headline dispositions

- **P2 (the win is purpose framing, not just chunking): STRONGLY SUPPORTED, and this is the study's most important result so far.** On identical source text, splitting a document into fixed word windows recovers almost nothing, while splitting it into purpose-aligned cards recovers almost everything. The card effect is roughly nine times the chunking effect. Anyone who dismisses multi-card retrieval as "chunking with extra steps" has to explain this table.
- **The R1 sanity gate passes.** Pooled single-vector retrieval is weak on a 46-document corpus (nDCG@10 0.314) while BM25 is near perfect (0.997), reproducing the published LIMIT finding that motivated the dataset.
- **P6 (identifier queries need the lexical hybrid): SUPPORTED IN PART.** LIMIT queries name one attribute and behave like exact-match lookups, and BM25 dominates dense retrieval there by a wide margin. Fusing the two costs nothing and matches the better of them.

## Results

46 documents, 44.0 attributes each, 2,024 cards, 186 chunks, 1,000 queries scored. Encoder all-MiniLM-L6-v2, seed 13, document-level scoring throughout, significance against the pooled baseline by paired permutation with Holm correction.

| system | nDCG@10 | R@2 | R@10 | Success@10 | delta vs pooled | p |
|---|---|---|---|---|---|---|
| bm25 | 0.997 | 0.991 | 1.000 | 1.000 | +0.683 | 0.0001 |
| dense-pooled | 0.314 | 0.162 | 0.480 | 0.729 | baseline | |
| dense-chunk | 0.390 | 0.225 | 0.549 | 0.799 | +0.076 | 0.0001 |
| dense-multicard | 0.988 | 0.965 | 0.997 | 0.999 | +0.674 | 0.0001 |
| hybrid-rrf | 0.997 | 0.990 | 1.000 | 1.000 | +0.683 | 0.0001 |

## Reading it honestly

The chunk control is what makes this result mean something. Chunking and carding both increase the number of embeddings per document from one to many, so if the benefit came from capacity alone the two rows would be close. They are not: +0.076 against +0.674. What separates them is that a card corresponds to the unit the query asks about, an attribute, while a window of thirty words spans several attributes and dilutes each of them exactly as the pooled vector does, only less severely.

Three caveats belong in the paper next to this table.

First, LIMIT hands over a perfect decomposition. The documents are comma-delimited attribute lists, so the card boundaries are unambiguous and align exactly with the query distribution. Real corpora do not come pre-decomposed, and the card taxonomy has to be designed and may be wrong. This result is therefore an existence proof and an upper bound on what alignment can buy, not evidence about the general case. Enron and Wikipedia decide the general case.

Second, the storage multiplier here is 44 cards per document, not the four to seven the pattern normally implies. On a corpus of attribute lists the natural card count is large, and the economy argument that multi-card is far cheaper than per-token late interaction weakens accordingly. The paper should state the multiplier it measures on each corpus rather than assert a small constant.

Third, BM25 wins outright on this dataset and the hybrid only matches it. That is the honest headline for LIMIT specifically: for exact-attribute lookup, lexical retrieval is the right tool, and the interesting claim is that purpose-aligned cards let a dense retriever reach the same place, which matters on the semantic queries where BM25 does not.

## Gates

Determinism: PASS, byte-identical across two seeded runs. Tests: 27 of 27 pass, including new tests for Enron quoted-chain stripping and for card construction. EVAL-BOUNDARY: clean. No API spend.

## Still open in R1

The Enron corpus is downloading (the canonical CMU archive, roughly 423 MB). The parser, dedup, span chunker, card builders, and candidate aspect taxonomy are written and tested against fixtures, and run as soon as the archive lands. The Wikipedia and KILT ingest follows.

---

# Sprint R1, part 3: consumer-dependent diversity, retrieval side

Added 2026-08-19. Feeds P5. Experiment `e3_diversity`, 120 constructed queries, selection budget 10, subtopics known by construction so coverage is counted exactly rather than inferred.

## Headline disposition

**P5, retrieval half: the coverage mechanism works, and the proposed operator is not the best one.** Every diversity policy raises subtopic coverage significantly over pure relevance, and on this task the determinantal point process is the strongest (S-recall 0.879 against 0.628 for relevance ranking, a gain of +0.251), ahead of maximal marginal relevance at low lambda (0.818) and ahead of the ascending-similarity outlier harvest this study proposed (0.772 at a 40 percent quota, 0.716 at 20 percent).

That is a result worth stating plainly rather than burying, because it reshapes the paper's claim in a way that makes it stronger and more defensible. The contribution is not a new diversification operator. Outlier harvesting is a serviceable one and a cheap one, but a reader with a determinantal point process will do better on coverage. The contribution is the conditioning variable: how much coverage to buy, with whatever operator, should depend on who consumes the results. Framed that way the result supports the thesis instead of denting it, and it points at the strongest implementation, which is a known operator with its diversity parameter set per consumer.

## Results

| policy | S-recall | alpha-nDCG | nDCG@10 | intra-list distance | distractor rate |
|---|---|---|---|---|---|
| relevance only | 0.628 | 0.692 | 0.965 | 0.249 | 0.033 |
| MMR, lambda 0.7 | 0.666 | 0.723 | 0.964 | 0.279 | 0.032 |
| MMR, lambda 0.3 | 0.818 | 0.840 | 0.963 | 0.338 | 0.036 |
| DPP | 0.879 | 0.877 | 0.963 | 0.352 | 0.035 |
| cluster round robin | 0.666 | 0.724 | 0.967 | 0.275 | 0.028 |
| outlier harvest, 20 percent | 0.716 | 0.734 | 0.966 | 0.274 | 0.031 |
| outlier harvest, 40 percent | 0.772 | 0.775 | 0.967 | 0.274 | 0.028 |

Coverage gains against relevance ranking are significant for every policy (Holm-corrected paired permutation, p at or below 0.012 throughout, and p below 0.0002 for the four strongest).

## Two honest caveats

**Coverage came almost free here, and that is probably an artefact of the setup.** No policy lost measurable relevance: every nDCG delta is within noise. The pool was gated to its upper half before selection, so nearly everything in it was relevant and spending part of the budget away from the centre cost nothing. On a weaker gate, or with a stricter relevance criterion, the trade would bite. The paper should not claim that diversity is free; it should report that under a strong gate the trade-off nearly vanishes, which is itself an argument for gating.

**Distractor risk did not materialise, on this task.** Harvested outliers were no more likely to be answerless than relevance-ranked selections, at roughly three percent for every policy. That is the guard working as designed, since the pool was gated, but the constructed distractors here are mild compared with the answerless-adjacent material a real corpus contains.

## What remains for P5

The half that decides the claim is the consumer half: whether the coverage a policy buys actually helps a generative model synthesising an answer, and actually hurts a person reading a bounded report. That requires downstream judgement, which requires model calls, which costs money and needs the owner's approval before any spend. Everything up to that point now exists and is tested.

---

# Sprint R1, part 4: gate separability and what the gate really costs

Added 2026-08-20. Feeds P3, and qualifies P4. Experiment `e2_gate`, 4,000 Enron messages as on-purpose against 4,000 Usenet posts as off-purpose, labels exact by provenance, fitted on one half and reported on the held-out half.

## Headline dispositions

- **P3 (gate separability): SUPPORTED, on the easy version of the question.** Held-out ROC-AUC is 0.933 and PR-AUC 0.935, comfortably above the 0.85 target the register set. The gate discriminates, so the pass-one filter is a filter rather than a random sample, and the architecture's premise survives.
- **Two findings qualify that, and both are more interesting than the headline.**

## The weighted composite earns almost nothing

Fitting the anchor-set weights by logistic regression gives ROC-AUC 0.933. Giving the three anchor sets equal weight gives 0.931. The elaborate weighted composite, which reads as the sophisticated part of the design, is worth two thousandths of AUC over simply averaging.

That is worth saying out loud in the paper, because it is the kind of claim practitioners tune for weeks. On this evidence the load is carried by having several purpose-specific anchor sets at all, not by how they are mixed. The fitted weights are also not what intuition would suggest: the objective facet dominates rather than the domain facet, which is the opposite of how such a gate is usually hand-tuned.

## The gate discards far less than the framing implies

This is the finding that most needs to reach the economics section. The two-pass story is usually told as though the gate throws away most of the corpus. At a high recall target it does not.

| recall target | achieved recall | precision | fraction of corpus discarded |
|---|---|---|---|
| 0.80 | 0.781 | 0.910 | 57.7% |
| 0.90 | 0.901 | 0.830 | 46.5% |
| 0.95 | 0.954 | 0.710 | 33.7% |
| 0.98 | 0.981 | 0.605 | 20.0% |
| 0.99 | 0.993 | 0.557 | 12.1% |

At the recall a high-recall pre-filter is supposed to hold, 0.95, the gate discards about a third of the corpus, not most of it. Push to 0.99 and it discards an eighth. The saving from gating is therefore real but modest, and it is bought directly with recall.

The consequence for the economic argument is a useful correction rather than a refutation. The dominant saving in the two-pass design does not come from the gate at all: it comes from the second pass, where generative cost attaches to discovered topics instead of to documents. The gate contributes a linear factor of between one and two; the topic-versus-document substitution contributes the order of magnitude. The paper should attribute the saving accordingly instead of crediting the filter.

## Caveat on the labels

Separating corporate email from Usenet is easier than separating on-purpose from off-purpose material inside one corpus, which is what a deployed gate faces. These numbers are an upper bound on discrimination and a floor on adequacy: failure here would have been disqualifying, and success here is necessary but not sufficient. The within-corpus question needs the human-audited Enron sample and stays open at the labelling gate.

---

# Sprint R1, part 5: the first real-corpus test contradicts the hypothesis

Added 2026-08-20. Feeds P1, P2 and P4. This is the most important entry in the brief so far and it should be read before any of the earlier ones are quoted anywhere.

## What happened

SciFact is the first corpus in this programme with genuine human relevance labels and prose that nobody constructed for the experiment: 5,183 scientific abstracts, 300 real claims, judgements made by people. Multi-card retrieval lost, badly.

| system | nDCG@10 | R@10 | R@100 | delta vs pooled | p |
|---|---|---|---|---|---|
| BM25 | 0.662 | 0.774 | 0.876 | +0.017 | 0.42 |
| dense, pooled | 0.645 | 0.783 | 0.925 | baseline | |
| dense, chunked | 0.673 | 0.803 | 0.936 | +0.028 | 0.050 |
| dense, multi-card extractive | 0.452 | 0.599 | 0.792 | **-0.193** | 0.0001 |
| dense, multi-card template | 0.451 | 0.610 | 0.799 | **-0.194** | 0.0001 |
| hybrid RRF | 0.579 | 0.731 | 0.919 | -0.066 | 0.0009 |

Multi-card was 0.221 nDCG@10 below the chunking control, with 136 losses against 39 wins. On LIMIT the same comparison ran nine to one the other way. Taken at face value, P1 and P2 are contradicted on the first real corpus.

## Why, and why the result does not stand as written

The result is real but the experiment was not a fair test, and the flaw is mine rather than the hypothesis's. Measuring how much of each abstract survived into its cards gave the answer: **the cards contained 44.6 percent of the document text on average, and as little as 8.6 percent for some documents, while the chunk control contained 100 percent.**

The card builder kept only the top two spans per aspect. On LIMIT that was harmless, because every attribute became its own card and nothing was dropped. On prose it discards more than half the document. The comparison therefore measured information loss, not purpose alignment, and a system reading 45 percent of each document losing to one reading all of it is not an interesting finding.

## What was changed, and what is now being measured

Card construction now partitions: every span is assigned to its single best-matching aspect, so the cards together contain the whole document and differ from chunks only in where the boundaries fall, which is the comparison the claim is actually about. The lossy top-spans variant is retained as an explicit ablation, because the size of the gap between the two is itself the finding: it quantifies how much of multi-card retrieval's fate rests on coverage rather than on aspect structure. A test now asserts that partition mode loses no text, so this cannot regress silently.

The corrected run is in progress. Three outcomes are possible and all three are publishable:

1. Partitioned cards match or beat chunking, in which case the LIMIT result generalises and the earlier failure was purely an implementation defect.
2. Partitioned cards close most of the gap but still trail chunking, in which case aspect structure costs something on prose whose queries are not aspect-directed, and the honest claim narrows to corpora where queries target aspects.
3. Partitioned cards still lose badly, in which case P1 and P2 are contradicted on real prose and the paper reports a negative result with the LIMIT case as the boundary condition.

**No claim from the synthetic corpus or from LIMIT should be repeated in public until this is settled.** SciFact queries are claims about a whole finding rather than questions aimed at one aspect, which is precisely the misalignment the original validity analysis listed as a condition of validity, so outcome 2 would confirm the analysis rather than refute it.

## The economics result, which did complete

| N | survivors | topics K | two-pass | full-LLM | ratio |
|---|---|---|---|---|---|
| 2,000 | 1,000 | 2 | $0.0001 | $0.1836 | 1,505x |
| 5,000 | 2,500 | 7 | $0.0005 | $0.4402 | 947x |
| 10,000 | 5,000 | 7 | $0.0005 | $0.8481 | 1,824x |
| 20,000 | 10,000 | 14 | $0.0009 | $1.7064 | 1,841x |

Topics grow as N^0.52, comfortably sublinear, and the cost ratio grows as N^0.44, so the advantage widens with corpus size exactly as the scaling argument predicts. P4's shape is confirmed.

**The magnitude must not be quoted without its caveat.** A ratio near 1,800x compares two designs that do not produce the same output: the full-LLM design reads every message, while the two-pass design names fourteen topics. That is a fair comparison only when the task is to discover topic structure, and it is not a fair comparison when the task requires per-document understanding. The paper must state the task the ratio applies to, or a reviewer will correctly call it a category error. A second caveat: fourteen clusters over ten thousand survivors is a small number, and the fraction of survivors left as clustering noise needs measuring before the naming step can be described as covering the corpus.

---

# Sprint R1, part 6: the corrected SciFact run, and what it settles

Added 2026-08-20, after the red-team review and the lossiness fix.

## The diagnosis was right, and the hypothesis still does not win

Partitioning the cards so they cover the whole document recovered almost all of the loss: multi-card went from 0.452 to 0.635 nDCG@10, a gain of +0.183. The lossy variant is retained in the table as the ablation that proves the point.

| system | nDCG@10 | R@100 | vs pooled | p |
|---|---|---|---|---|
| BM25 | 0.662 | 0.876 | +0.017 | 0.42 |
| dense, pooled | 0.645 | 0.925 | baseline | |
| dense, chunked (5.5 units/doc) | 0.673 | 0.936 | +0.028 | 0.050 |
| dense, multi-card partitioned (2.8 units/doc) | 0.635 | 0.923 | -0.010 | 0.48 |
| dense, multi-card template | 0.634 | 0.922 | -0.011 | 0.42 |
| dense, multi-card lossy (the earlier defect) | 0.452 | 0.792 | -0.193 | 0.0001 |
| hybrid, cards fused with BM25 | **0.685** | **0.948** | +0.040 | 0.012 |

Read plainly: on the one corpus in this programme with human relevance judgements, **multi-card retrieval is statistically indistinguishable from a single pooled vector** (-0.010, p=0.48) and **loses to naive chunking** (-0.037, p=0.005). The best system on the table is the hybrid, and what it adds is lexical matching, not aspect structure.

## The remaining confound, and the test now running

Cards produce 2.8 units per document against the chunk control's 5.5, so the cards are working with half the embedding budget. Given the reviewer's finding that retrieval quality tracks units per document almost perfectly, that difference alone could account for the 0.037 gap. A matched run with the chunk width widened to give the same units per document is in progress, and it is the test that decides P2 on real prose. Three outcomes:

1. Matched chunks fall to card level: the gap was capacity, cards are neither better nor worse, and the honest claim becomes that multi-vector representations help while the choice of boundary does not.
2. Matched chunks stay ahead: aspect structure actively costs something on prose whose queries are not aspect-directed.
3. Cards pull ahead at matched capacity: alignment contributes, and the LIMIT finding partly survives.

Outcome 1 is the most likely given everything else seen so far, and it would be a publishable negative result rather than a failure. It would also vindicate the original validity analysis, which listed query-aligned taxonomy as a condition of validity rather than a guarantee: SciFact claims are assertions about a whole finding, not questions aimed at one aspect of a document.

## What the programme can honestly say today

Nothing about alignment. The defensible claims after the review are narrower and mostly about method: that multi-vector representation beats single-vector pooling on aspect-rich material, that the size of that benefit tracks the number of embeddings rather than their semantic framing, that lossy card construction is catastrophic and easy to introduce by accident, and that a lexical hybrid remains the strongest cheap addition on real prose.

## The matched-capacity test: outcome 2, and it settles P2 on real prose

With the chunk control widened to 2.7 units per document against the cards' 2.8, so both sides carry the same embedding budget:

| system | nDCG@10 | R@100 |
|---|---|---|
| dense, pooled | 0.645 | 0.925 |
| dense, chunked at matched capacity | 0.667 | 0.943 |
| dense, multi-card partitioned | 0.635 | 0.923 |
| hybrid, cards fused with BM25 | 0.685 | 0.948 |

Multi-card minus chunk at matched capacity: **-0.032, 95% CI [-0.057, -0.008], p=0.0115, 70 losses to 36 wins.** Equalising the budget moved the gap from -0.037 to -0.032, which is to say it moved it almost not at all.

**Capacity was not the explanation. On this corpus purpose-aligned cards are genuinely, if modestly, worse than cutting the same text into arbitrary windows.** That is outcome 2 of the three anticipated, and it is the strongest evidence the programme has produced about the hypothesis on real prose.

It also confirms the original validity analysis rather than refuting it. That analysis listed a query-aligned taxonomy as a condition of validity, not a guarantee. SciFact claims are assertions about a whole finding, so no card is the unit the query asks about, and imposing an aspect boundary then destroys evidence that a blind window happens to keep together. Where the taxonomy does match the query distribution, as on LIMIT where each query names one attribute, cards do win. The condition is doing exactly the work the analysis said it would.

## Consolidated position on P1 and P2

Across a synthetic corpus, an adversarial capacity benchmark, and a human-judged real corpus, the defensible summary is:

1. Multi-vector representation beats single-vector pooling on aspect-rich material. This holds everywhere it was tested.
2. The size of that benefit tracks the number of embeddings per document, not their semantic framing. On LIMIT a blind three-word window recovers 0.610 of the 0.674 attributed to alignment, and the residual is smaller than the chunking effect itself.
3. Where the taxonomy matches the query distribution, aspect framing adds a little. Where it does not, it subtracts a little, significantly so on SciFact at matched capacity.
4. Lossy card construction is catastrophic and easy to introduce by accident: -0.193 nDCG@10, four to five times any other effect measured here.

The paper's original thesis, that alignment rather than count explains the benefit, is contradicted. What is left is a clean negative result with a mechanism, a condition under which the positive case holds, and a cautionary implementation finding. That is publishable, and arguably more useful to a practitioner than the positive claim would have been, but it is a different paper.

## Venue consequence

ECIR's full-paper deadline of 5 October is not realistic for a paper that now needs rebuilding from a different thesis. ECIR runs a reproducibility track with a later deadline (abstract 12 October, paper 19 October per the call), and a paper whose contribution is "the reported mechanism is not the operative one, here is the controlled comparison that shows it, and here is the condition that decides it" is a natural fit for that track rather than the full track. Recommend retargeting, subject to the owner's decision.

---

# Sprint R1, part 7: the corrected synthetic experiment, and the condition that decides everything

Added 2026-08-20. The synthetic experiment now uses cards the builder infers rather than the passages the corpus was generated from, adds the chunk control it never had, and runs on a corpus whose filler no longer leaks unlabelled relevance. It reverses the direction of the SciFact finding, and the reversal is the point.

## Result

| k | pooled | chunk (units/doc) | inferred card (units/doc) | oracle card | card minus chunk | p |
|---|---|---|---|---|---|---|
| 1 | 0.815 | 0.832 (2.8) | 0.799 (1.9) | 0.815 | -0.033 | 0.057 |
| 2 | 0.485 | 0.738 (4.8) | 0.734 (3.5) | 0.765 | -0.004 | 0.78 |
| 3 | 0.394 | 0.677 (6.7) | 0.684 (4.8) | 0.723 | +0.007 | 0.63 |
| 4 | 0.355 | 0.657 (8.0) | 0.678 (5.8) | 0.691 | +0.021 | 0.12 |
| 5 | 0.377 | 0.638 (9.1) | 0.702 (6.7) | 0.762 | **+0.064** | 0.0001 |
| 7 | 0.356 | 0.676 (10.9) | 0.731 (8.2) | 0.848 | **+0.055** | 0.0005 |
| 10 | 0.294 | 0.633 (12.8) | 0.742 (9.6) | 0.859 | **+0.109** | 0.0001 |

Holm-corrected across the seven comparisons; the three significant rows survive correction.

## What it shows

**Cards beat chunks from k=5 upward, while carrying fewer embeddings per document.** At k=10 the cards use 9.6 units per document against the chunk control's 12.8 and still win by 0.109. That is the alignment effect the earlier experiments failed to demonstrate, measured this time against a control that is not handicapped and with cards the builder had to infer. Below k=4 there is no effect, and at k=1 chunking is marginally ahead, which is what should happen when a document has one aspect and there is nothing to align to.

Pooled retrieval collapses as k rises, from 0.815 to 0.294, so the multi-vector benefit is large and unambiguous. Oracle cards, which are the generating passages and therefore an upper bound rather than a method, run about 0.12 above the inferred cards at high k, showing how much a perfect decomposition would be worth over an inferred one.

## The condition, stated precisely

Putting this beside the matched-capacity SciFact result gives a clean two-sided finding, and it is the programme's real contribution:

- **Where documents carry several distinct aspects and queries target one of them, purpose-aligned cards beat naive chunking at equal or lower cost, and the advantage grows with the number of aspects** (synthetic, +0.109 at k=10 with 25 percent fewer embeddings).
- **Where queries are about a document as a whole, purpose-aligned cards lose to naive chunking even at matched capacity** (SciFact, -0.032, p=0.012), because an aspect boundary destroys evidence a blind window keeps together.
- **The number of embeddings per document explains most of the benefit in the middle of that range**, which is why the LIMIT result overstated alignment: its control was eleven times smaller, and a blind three-word window recovers most of the gap once matched.

That is not the thesis the programme started with, and it is a better one. It converts an unconditional claim that the evidence does not support into a conditional claim with both sides measured and a mechanism that explains the sign change.

## Caveat on the synthetic anchors

The aspect anchors given to the builder are rendered from the same templates the corpus generator uses, so the taxonomy is effectively supplied by an oracle even though the assignment of spans to aspects is inferred. That mirrors practice, where a human designs the taxonomy and the system assigns, but it is generous. A fairer version would hand-write paraphrased anchors, and the honest expectation is that the effect shrinks. This should be run before the number is published.

## The anchor caveat resolves favourably, and strengthens the finding

Replacing the generator-derived anchors with hand-written ones, the same concepts in a practitioner's words and none of the generator's phrasing, does not shrink the effect. It roughly doubles it.

| k | card minus chunk, generator anchors | card minus chunk, hand-written anchors | units, card/chunk |
|---|---|---|---|
| 3 | +0.007 (ns) | +0.023 (ns) | 4.4 / 6.0 |
| 5 | +0.064 (p=0.0001) | **+0.099** (p=0.0001) | 6.5 / 8.8 |
| 7 | +0.055 (p=0.0005) | **+0.098** (p=0.0001) | 8.1 / 10.9 |
| 10 | +0.109 (p=0.0001) | **+0.188** (p=0.0001) | 9.8 / 13.0 |

The reason is instructive rather than lucky. A generator-derived anchor is one rendered sentence carrying one specific attribute and action, so it acts as a prototype for a single fact rather than for the aspect as a whole, and span assignment inherits that narrowness. A hand-written anchor describes the aspect ("contracts, liability, counsel and formal obligations between parties"), which is a better prototype and assigns spans better. The oracle taxonomy was not an advantage; it was a handicap.

That removes the last open caveat on the positive half. At k=10, cards written from a human-designed taxonomy beat a matched chunk control by 0.188 nDCG@10 while using 25 percent fewer embeddings per document.

## Sprint R1 closes here

All six register rows now carry evidence-backed dispositions. The programme's position, stated as it would be to a reviewer:

1. Multi-vector representation beats single-vector pooling on aspect-rich material, decisively and everywhere tested.
2. Purpose-aligned cards beat naive chunking when documents carry several aspects and queries target one of them, by up to 0.188 nDCG@10 at fewer embeddings, and the advantage grows with aspect count.
3. Purpose-aligned cards lose to naive chunking when queries concern whole documents, by 0.032 at matched capacity on human-judged data.
4. Between those poles, embedding count explains most of the benefit, which is why an unmatched control made alignment look far stronger than it is.
5. Lossy card construction costs 0.193 nDCG@10, several times any other effect here, and is easy to introduce without noticing.
6. A cheap anchor gate separates on-purpose from off-purpose material at 0.933 AUC but discards only a third of a corpus at high recall, and three sets of nonsense anchors reach 0.860, so the mechanism is projection onto several directions rather than purpose specificity.
7. The two-pass cost advantage is a large constant factor at comparable topic granularity, not a widening scaling law; the widening was clusters coarsening as the corpus grew.
8. Every diversity policy buys coverage over pure relevance, and a determinantal point process buys more of it than the outlier harvest this programme proposed, so the contribution is the consumer conditioning rather than the operator.

Outstanding before any of this is published: the E3 defects the review found (subtopic labels, the inert distractor parameter, query independence) must be fixed and that experiment rerun, the junk-anchor control belongs in the gate table, and the paper needs rebuilding around the conditional thesis rather than the original one.

---

# Sprint R1, part 8: the corrected diversity experiment reverses part 3

Added 2026-08-20. With a distractor pool that can actually be lost (40 per task rather than the 2 an integer-division bug had produced) and a query drawn from the facts it is scored against, the diversity result changes materially, and the earlier conclusion in part 3 is withdrawn.

| policy | S-recall | alpha-nDCG | nDCG@10 | answerless rate | coverage gained | relevance given up | coverage per unit of relevance |
|---|---|---|---|---|---|---|---|
| relevance only | 0.372 | 0.542 | 0.827 | 0.168 | baseline | baseline | |
| MMR, lambda 0.7 | 0.405 | 0.566 | 0.802 | 0.197 | +0.033 | -0.026 | 1.3 |
| MMR, lambda 0.3 | 0.686 | 0.695 | 0.729 | 0.285 | +0.314 | -0.098 | 3.2 |
| DPP | **0.746** | 0.706 | 0.699 | 0.328 | +0.374 | -0.128 | 2.9 |
| cluster round robin | 0.545 | 0.626 | 0.774 | 0.229 | +0.173 | -0.053 | 3.3 |
| outlier harvest, 20% | 0.567 | 0.619 | 0.808 | 0.198 | +0.196 | -0.019 | **10.3** |
| outlier harvest, 40% | 0.686 | 0.677 | 0.781 | 0.236 | +0.315 | -0.046 | **6.8** |

All deltas Holm-corrected and significant.

## What changed and why it matters

Part 3 concluded that the outlier harvest was beaten by a determinantal point process and that the contribution therefore lay in the consumer conditioning rather than the operator. That conclusion rested on a saturated task where nothing could be lost, and it does not survive a corpus with real distractors.

**The trade is now real, and the outlier harvest sits on the efficient frontier.** Every policy pays for coverage in relevance, which is what should happen. At matched coverage the harvest is roughly twice as efficient as maximal marginal relevance: the 40 percent harvest buys +0.315 subtopic recall for -0.046 nDCG, while MMR at lambda 0.3 buys an indistinguishable +0.314 for -0.098. The DPP buys the most coverage of any policy, +0.374, but pays -0.128 for it, the worst rate on the table. The 20 percent harvest is the most efficient point measured, at ten units of coverage per unit of relevance.

**It is also the safest.** The harvest admits fewer answerless documents than the operators that match its coverage: 0.236 at the 40 percent setting against 0.285 for MMR and 0.328 for the DPP.

The mechanism explains the sign. MMR and a DPP diversify the entire selection, so the top of the list moves away from relevance. The harvest keeps a relevance head, six of ten items at the 40 percent setting, and spends only the tail quota on the low-similarity region. The head holds the answer while the tail buys coverage, which is exactly the behaviour the consumer-conditioning argument wants: a machine consumer takes the whole set and benefits from the tail, while a human reader can be given the head alone.

## Disposition

P5's retrieval half is now **supported, and the operator claim is restored**: within a gated pool, ascending-similarity harvesting is a more efficient and safer way to buy coverage than MMR or a DPP. The half that remains unsettled is the one the whole thesis rests on, whether that coverage helps a generative consumer and hurts a human one, and that needs judges and therefore the owner's approval to spend.

This is also the second time in this sprint that a headline conclusion was an artefact of a defect in the harness rather than a property of the world. Part 3 and part 5 both had to be withdrawn. The lesson for the programme is that no disposition should be written from a first run, and the red-team review should run before results are recorded rather than after.

---

# Sprint R1, part 9: the consumer study, and how far it actually gets

Added 2026-08-20. This is the experiment the patent kernel rests on, run on Google Cloud credits at a total cost of about four cents.

## Design

Same gated pool, same query, same synthesis model throughout; only the selection policy and the consumer vary. The machine consumer is asked for a comprehensive account and given a 600-word allowance; the human consumer is asked for a brief a colleague will read once and given 100 words. Consumers differ in budget as well as framing, which an earlier version did not, and that earlier version consequently tested nothing but wording.

The endpoint is a blinded extraction judge: it sees a summary and a numbered list of candidate facts and reports which the summary actually states. It never learns which policy produced the text, and the task is extraction rather than preference, so the usual position and self-preference biases apply less. The prediction under test is an interaction, not a main effect.

## Result, and the honest bound on it

| judge | machine gain | human gain | interaction | p |
|---|---|---|---|---|
| gemini-2.5-flash | +0.022 (p=0.30) | **-0.049 (p=0.010)** | **+0.071 [+0.022, +0.125]** | **0.009** |
| gemini-2.5-pro | +0.043 (p=0.078) | -0.004 (p=0.88) | +0.047 [-0.024, +0.117] | 0.211 |

**Both judges put the interaction in the direction the claim predicts, and only one of them makes it significant.** The stricter judge finds the same positive sign at half the confidence, and it does so because the effect that carried significance with the first judge, harvesting hurting the human reader, largely disappears under the second. What survives across both is a marginal benefit to the machine consumer and no benefit to the human one.

The correct statement is therefore that **the consumer interaction is suggestive and not established.** It is the first evidence in this programme that points the kernel's way, and it is not enough to claim the kernel is demonstrated.

A secondary finding is consistent across judges and metrics and deserves its own line: harvesting reduces unsupported content in the output for both consumers, significantly for the human one (-0.067 to -0.069, p below 0.002). Outputs built from a diverse selection stay closer to their evidence. That was not predicted and is worth following up.

## Three defects found on the way, each of which would have produced a wrong answer

**The first coverage metric fired without evidence.** Counting a majority of a subtopic's words as coverage meant that shared topic vocabulary triggered matches, and outputs "covered" 4.3 subtopics from selections containing 1.8. Caught by an arithmetic sanity check that an output cannot cover more subtopics than its selection contains. That check now runs with the experiment.

**The replacement metric was too strict, and reversed the sign.** Requiring both exact phrases missed ordinary paraphrase, and under it the interaction ran negative and non-significant. The sign of the headline depended on the instrument, which is a fragility that belongs in the paper rather than in a footnote.

**The response cache silently served stale text.** Its key covered model, temperature and prompt but not the token budget, so a fix to output truncation appeared to do nothing because the old truncated responses were being returned. Any generation parameter that can change the text is now in the key.

## What the claim needs before it can be published

1. **Cross-family judging.** Both judges here are Gemini judging Gemini output. Claude and Llama are available through Vertex Model Garden but return 404 until each is enabled once in the console, which is a person's action, not the agent's. Until then the result carries a self-preference risk that a reviewer will name immediately.
2. **More tasks.** Sixty is small for an interaction, and the confidence interval under the stricter judge crosses zero by a margin that more data would resolve either way.
3. **A human-facing endpoint that is not a proxy.** Judged fact coverage is a reasonable stand-in for what a machine consumer needs. It is a poor stand-in for whether a person found the brief useful, which is the other half of the claim and ultimately needs human raters.

## The cross-family judge settles it, and the answer is negative

Murali suggested Azure GPT-5.4 in place of the Model Garden partner models, which was the better choice anyway: a different family from a different vendor, judging the same 240 fixed generations, with nothing else varying.

| judge | machine gain | human gain | interaction | 95% CI | p |
|---|---|---|---|---|---|
| gemini-2.5-flash | +0.022 | -0.049 | **+0.071** | [+0.022, +0.125] | **0.009** |
| gemini-2.5-pro | +0.043 | -0.004 | +0.047 | [-0.024, +0.117] | 0.211 |
| Azure GPT-5.4 (cross-family) | +0.014 | +0.007 | **+0.008** | [-0.039, +0.056] | 0.763 |

**The effect shrinks monotonically as the judge moves away from the generator's own family, and vanishes at the cross-family judge.** That is the signature of a judging artefact rather than a property of the text. The single significant result came from the weakest judge, drawn from the same family as the model that wrote the summaries, which is exactly the configuration most prone to self-preference.

**P5's consumer half is therefore not supported.** The claim that the right diversity policy depends on the consumer has no empirical backing from this study. It is not disproved either: the point estimates are positive under all three judges, and 60 tasks cannot resolve an effect of the size the cross-family judge suggests. But nothing here may be cited as evidence for it.

## What is robust, because it never depended on a judge

Two endpoints are computed lexically and are identical across all three runs by construction:

- Harvesting reduces unsupported content in the output, for the human consumer significantly (-0.067, p=0.0018) and for the machine consumer marginally (-0.048, p=0.067). Summaries built from a diverse selection stay closer to their evidence.
- Harvesting slightly reduces strict phrase coverage for the machine consumer (-0.046, p=0.063) and does nothing for the human one.

The grounding finding was not predicted, survives every judge because it needs none, and is the most defensible result in the consumer study. It belongs in the paper on its own terms.

## Method note worth carrying forward

Had the study stopped at one judge it would have reported a significant consumer interaction, and that claim would have been the paper's headline and the patent's empirical support. Three judges cost four cents and about an hour. The rule this programme should adopt: **a claim resting on a generative judge is provisional until a judge from another family and vendor reproduces it**, and the first judge should never be from the same family as the generator.

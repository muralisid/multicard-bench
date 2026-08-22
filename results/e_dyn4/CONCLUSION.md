# e_dyn4: does the selection-from-substrate architecture (pov-03) prove the hypothesis?

Run 2026-08-22 on the Apple Silicon machine. No preregistration (owner's choice
2026-08-22); rigor held by decision rules fixed in code before results were read,
robustness across topic-model seeds, and honest reporting of every arm. Encoder
all-MiniLM-L6-v2 throughout (a weak instruction baseline; a stronger one would
widen, not narrow, the gap below). Metrics: nDCG@10 under each instruction's own
human qrels, and FollowIR p-MRR for instruction following (instruction-blind arms
are 0 by construction).

## 1. Synthetic architecture test (validated AnchorCardBuilder, 5 topic-model seeds)
Question: does the cross-over that round 3 got with an objective-SEEDED topic
model survive moving the objective LATE, to selection over a shared UNSEEDED
substrate? Answer: NO. Two-sided cross-over held at 1 of 5 seeds. The A direction
is robustly positive (+0.034 to +0.099, significant in 4/5); the B direction is
near zero or the wrong sign (-0.038 to +0.025). Same fragility seen across
machines. Moving the objective late does not create a stable cross-over.

## 2. FollowIR real human-judged data
Arms: pooled (bare query), instr_pooled (instruction prepended to the query),
chunks, cards_corpus (one objective-blind view set), cards_obj (view set selected
per instruction from the shared substrate).

Core17 (20 queries, 3 substrate seeds, mean-pool cards):
  nDCG@10 og:      pooled .393  instr_pooled .394  chunks .403  cards_corpus .400  cards_obj .375
  nDCG@10 changed: pooled .243  instr_pooled .242  chunks .247  cards_corpus .217  cards_obj .222
  p-MRR: instr_pooled -.013  cards_obj +.007   (both ~0: no instruction following)

Core17 (seed 13, VALIDATED concat card representation, caveat check):
  nDCG@10 og:      pooled .393  instr_pooled .394  chunks .403  cards_corpus .409  cards_obj .349
  nDCG@10 changed: pooled .243  instr_pooled .242  chunks .247  cards_corpus .247  cards_obj .213
  p-MRR: instr_pooled -.013  cards_obj -.002
  -> cards_obj is the worst arm and ~0.06 below cards_corpus. Objective
     conditioning HURTS even with the validated representation; the mean-pool
     shortcut was not the cause (if anything it was kinder to cards_obj).

News21 (32 queries, substrate seeds 11 and 13, mean-pool cards):
  nDCG@10 og:      pooled .386  instr_pooled .416  chunks .364  cards_corpus ~.327  cards_obj ~.331
  nDCG@10 changed: pooled .210  instr_pooled .265  chunks .205  cards_corpus ~.190  cards_obj ~.194
  p-MRR: instr_pooled +.005  cards_obj ~+.008
  -> the instruction-prepended encoder is the BEST arm; all card arms are worst.

Robust04 not run (skipped for time; Core17 + News21 + synthetic are consistent).

## 3. Conclusion
The pov-03 selection-from-substrate architecture does not prove the hypothesis.
- The late-objective cross-over is not a stable effect (1/5 seeds).
- On real human-judged instruction-following data, objective-conditioned card
  selection is the WORST arm: worse than a single corpus-only card set, and worse
  than or no better than a plain instruction-prepended encoder.
- Card re-selection does not follow the instruction (p-MRR ~0), while even the
  weak instruction-prepended encoder shows the only positive standard-metric
  advantage (News21).
This is consistent with round 4 (conditioning is monotonically harmful), the
cross-over's fragility across machines, and the prior-art warning that
instruction-conditioned embeddings subsume this idea.

The corpus-only card set (cards_corpus) remains competitive with chunks/pooled,
which is the one durable positive: multi-view indexing built from the corpus is a
reasonable representation; conditioning WHICH views on the objective is what does
not pay off.

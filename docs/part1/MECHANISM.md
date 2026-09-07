# Why the added layers gain on chat memory and lose on news

Written 2026-09-07 from the per-query files. Four independent lenses proposed
explanations, 57 of their claims were rechecked adversarially by a second
agent that recomputed each number with its own code, and 5 claims were refuted
outright and 47 corrected in part. What follows is what survived that pass.
The scripts are in the scratchpad recorded in the programme worklog.

The first explanation this session offered, that the lazy expansion spends the
budget on more chunks of documents already found, is REFUTED. Lazy expansion
renders zero units and zero tokens on MultiHop-RAG in all nine arms at both
budgets, because the render order puts the whole 100-unit fused list ahead of
the expansion list and 100 news chunks overfill a 4,000-token budget long
before the expansion units are reached. The account below replaces it.

1. Why the same machinery gains on chat and loses on news

The machinery does the same thing on both corpora. It widens the candidate pool a little and it reorders the top of that pool a lot. What changes is how much of that pool the token budget lets through. A chat turn costs about 73 to 83 tokens once the declared speaker rule has dropped the assistant turns, so 4,000 tokens renders about 48 to 55 units out of a candidate list of about 55 to 62. That is nearly the whole list. The reorder is therefore almost free, the small pool gain is banked, and the render step costs about one point of joint recall. A news chunk costs about 424 to 443 tokens, so the same 4,000 tokens renders about 9 units out of 100. There the score is not pool recall at all. It is precision in the first nine slots, and the render step costs 62 to 71 points of what retrieval already found. A news question needs 2.62 different documents on average and joint recall needs all of them, so pushing one document out of the nine loses the whole question. The extra channels do exactly that. They move the rank at which the last needed document first appears from a median of 12 under the cheap floor to 15.5 with equal channel weights and 17 with the planner weights, against a window of nine. Same reorder, two budgets, opposite sign.

2. Evidence

1. At 4,000 tokens news renders 9.22 to 9.43 units at 424 to 434 tokens each; chat renders 48.0 to 55.0 units at 73 to 83 tokens each.
2. That is 9.2 to 9.4 percent of a 100-unit news candidate list, against 48 to 55 units of a 55 to 62 unit chat list.
3. The render step costs 62 to 71 points on news (ours_cheap 0.8736 candidate to 0.2554 rendered, S4_static 0.8971 to 0.1854) and 1.1 to 1.5 points on chat (0.9489 to 0.9383, 0.9702 to 0.9574).
4. Median rank at which every evidence document has appeared on news, over the queries where every evidence document is inside the logged 100-unit candidate list: ours_cheap 12.0, S2_lazy 14.0, planner rules 15.0, S4_static 15.5, planner oracle 16.0, S5_primary 17.0, overlay P0 19.0. Over all 2,255 queries the same medians are 13, 15, 16, 16, 18, 19 and 21.
5. Share of news queries with all evidence documents inside the first nine ranks: 0.4151, 0.3614, 0.3477, 0.3401, 0.3002, 0.2887, 0.2749, against measured joint recall 0.2554, 0.2053, 0.2053, 0.1854, 0.1508, 0.1361, 0.1348.
6. Read the same rankings deeper and the equal-weight arm catches the floor: 0.8727 against 0.8767 at k=50 and 0.9685 against 0.9694 at k=100, while S5_primary is still 0.0226 behind at k=100.
7. A news query needs 2 documents on 1,169 queries, 3 on 774 and 4 on 312, mean 2.620.
8. The relation and entity channels never fire on news, 0 hits in the whole candidates file against 274,126 for relation on chat, and S5_noPGR equals S5_primary at 0.1361 and 0.2834.
9. Lazy expansion renders 0 units and 0 tokens on news, in all nine arms, at both budgets.
10. On chat, container expansion is 0.4 to 1.7 percent of rendered tokens and deep filler from the fused tail is 9.9 to 12.4 percent, over the arms with an expansion depth above zero.
11. The news loss is a ladder of four arm steps, not four isolated mechanisms: ours_cheap to S2_lazy -0.0501, S2_lazy to S4_static -0.0200, S4_static to S5_overlay_R0 -0.0324, S5_overlay_R0 to S5_primary -0.0169, total -0.1193. Each rung differs from the one above by more than one thing, so the labels are the arms, and the attribution to a mechanism is an inference from the arm definitions rather than a measured decomposition.
12. The lexical weight row, the only one that sets topic and community to zero, is the one place the system beats the floor on news: 0.3491 against 0.3047 on 699 queries; the thematic row (topic 3, community 2) scores 0.1348 against 0.4000 on 230 queries.
13. The LLM planner labels 1,312 of 2,255 news queries cross-topic, which sets community 3 and topic 2 against dense 1 and bm25 1.
14. No overlay is the best overlay on both corpora: news R0 0.1530, R2 0.1455, R3 0.1361, placebo P0 0.1348; chat R0 0.9489, R2 0.9489, R3 0.9468, P0 0.9383.
15. The chat gain is 12 wins to 3 losses for S4_static and 13 wins to 9 losses for S5_primary, out of 470; the two-sided sign test on the discordant pairs gives 0.035 and 0.52, and the 10,000-permutation paired test that the design fixes gives 0.036 and 0.520; the news loss is 118 to 276 and 111 to 380 out of 2,255, both p below 1e-14.
16. The declared speaker rule, which runs on chat and on no news arm, is worth 0.9383 against 0.8000 for the floor and 0.9468 against 0.7660 for S5_primary.
17. Doubling the news budget to 8,000 tokens doubles the slots to about 18 and lifts every arm, and the ordering holds (floor 0.4200, S2_lazy 0.3694, S4_static 0.3605, rules 0.3446, S5_primary 0.2834).

3. What this says about the design

Topic and community channels. On news these two are the whole of the added machinery, because the other two never fire there. They are also the whole of the cost. Every increase in their weight lowers the score, and the only weight row that beats the cheap floor on news is the one that sets both to zero. Their failure mode is not that they drag in junk from nowhere. They reorder documents the lexical and dense channels already found, and under reciprocal rank fusion a unit that collects a topic vote and a community vote outranks an evidence chunk that only has bm25 and dense behind it. That is harmless when 50 slots are rendered and fatal when 9 are. On chat they are not the source of the gain either: the best chat arm is S4_static, which makes no model call, and it wins by 12 questions to 3.

Lazy expansion. It earns nothing on news and it cannot, because the render order puts the whole fused candidate list before the expansion list and 100 news chunks overfill 4,000 tokens long before the expansion units are reached. Measured, it contributes 0 units at both budgets. On chat it fires, but what fires is mostly deep filler from the fused tail, not expansion of the top containers. Container expansion is under 3 percent of rendered tokens on every chat arm. So the depth column of the planner table has almost no work to do on either corpus as the system is currently ordered.

Overlay. It does not earn its place on either corpus. No overlay scores highest on both, the R3 link set is 1.7 points below no overlay on news, and the difference between the real links and the placebo on news is 0.0013. Its failure mode is that it perturbs group scores across a large part of a small index, and that perturbation is paid at the nine-slot cut.

Planner. It is the third largest of four contributors to the news loss, worth -0.0324, not the whole story. Its specific fault is the cross-topic row, which it assigns to 1,312 of 2,255 news queries and which gives the two harmful channels more weight than dense and bm25 together. The rules planner scores higher on news only because its lexical regex fires on 699 queries and switches the group channels off. Correctness of the shape label is not the axis: the oracle planner, which is right about the question type, scores 0.1508, below the rules planner's 0.2053.

Speaker rule. It is the biggest single lever on chat and there is no equivalent on news. Any cross-corpus claim about the channels has to carry that caveat, because the two corpora are not running the same arm.

4. What a corrected version would have to change, measurably

a. Make the group-channel weight a function of how many slots the budget buys, not of the question shape. The measurable target on news is the share of queries with every evidence document inside the rendered depth: the floor is at 0.4151, S4_static at 0.3401 and S5_primary at 0.2887. A corrected arm has to beat 0.4151 at nine slots.
b. Run a weight sweep on topic and community at fixed pool and fixed slot count, and report joint recall at the sweep points. The two end points are already known from arms in the file: weight 0 gives 0.2554 and weight 1 gives 0.1854.
c. Change the render order so expansion units are interleaved into the fused head rather than appended after all 100. The test is whether the news expansion count moves off 0 at 4,000 tokens.
d. Set the overlay default to R0 until an overlay beats it. The bar is 0.1530 on news and 0.9489 on chat.
e. Either build the post-graph-rag tables for news or drop the S5_noPGR row from the news table. As it stands it is byte-identical to S5_primary on all 2,255 queries and is not an ablation.
f. Report the chat result with its paired counts. The shipped arm is 13 wins to 9 losses out of 470, and the two-sided sign test on the discordant pairs gives p = 0.52.
g. Do not use S2_lazy's candidate-level recall in a cross-arm table. Its logged candidate list is 218 units against 100 for every other arm.
h. Do not make cross-arm claims from the chat 8,000-token row. Arms fill 6,200 to 7,586 of the 8,000 tokens because the candidate list runs out, so the arms are not budget matched there.

5. What the data cannot settle

- Whether the topic and community channels would help news at a budget that renders 50 or more chunks. My k-curves say S4_static catches the floor at k=50 (0.8727 against 0.8767), but no run exists at that budget, so this is an extrapolation over ranking prefixes and not a rendered measurement.
- Whether the relation and entity channels would help or hurt news. They were never built for that corpus and produce zero rows. Nothing in these files can say.
- Whether news chunk size or news content is the binding cause. The 4,000 and 8,000 rows are consistent with size, but no re-chunking run exists.
- Whether the chat gain is real for the shipped system. The evidence is thin: 13 wins and 9 losses on 470 questions, p = 0.52. Only S4_static, which makes no model call, clears p below 0.05, and that is uncorrected across the nine arms tested.
- Whether the extra channels cause the deeper ranks rather than merely coinciding with them. The ordering is monotone across nine arms, but the only interventions available here are the arms themselves. No weight sweep was run.
- Why the 4-document news questions are inert. Every arm scores between 0.003 and 0.013 there on 312 questions, so nothing can be separated at that end.
# Sprint R2 brief: the dynamic pipeline, three rounds

Produced 2026-08-21. Feeds H-DYN-1, 3, 4, 6. Artifacts in multicard-bench under results/e_dyn and results/e_dyn2.

## Round 1: the thesis fails

The preregistered battery ran and returned: H-DYN-4 (cross-over) no effect in either direction; H-DYN-3 (ladder) reversed, with corpus-only design beating corpus-plus-objective by 0.068 (p=0.0020); H-DYN-6 (sampler) mechanism working (minority coverage 60 to 100 percent) but no retrieval benefit and the negative control firing on the balanced corpus; H-DYN-1 nominally supported against a baseline that turned out to be unsound, since e5-small-v2 is not instruction-following and the objective text degraded its queries below the plain pooled baseline.

## The diagnosis, prompted by the owner

Murali asked whether the topic modelling was actually LLM-guided and whether noise was LLM-cleaned. Checking the code rather than reasoning about it: the round-1 pipeline contained **zero gate calls, zero seed-topic guidance, and zero outlier cleaning**. BERTopic supports all three (seed_topic_list, zeroshot_topic_list, reduce_outliers, merge_topics). Round 1 tested a stripped-down reconstruction, not the method.

The gate omission is the mechanistically important one. Under partition-mode carding every span must land in some view, so a taxonomy aimed at three departments had to absorb the seven it was not aimed at, and the narrower objective-conditioned taxonomy was polluted more than the broader corpus-only one. That is the sign the ladder showed.

## Round 2: all three stages restored

Objective-derived anchor gate (applied identically to every arm), LLM-derived seed topics guiding BERTopic, reduce_outliers plus an LLM pass naming topics and flagging incoherent ones, then stratified sampling and view design.

**Ladder:** the objective penalty disappears. L0 pooled 0.261, L1 chunks 0.483, L4 corpus-only 0.540, L5 objective 0.530; objective increment -0.010 (p=0.4587), indistinguishable from zero where round 1 had a significant penalty. On a single workload, objective conditioning neither helps nor hurts against a well-formed corpus-only taxonomy.

**Cross-over:** two confounds were found and removed before the result was believed. First, each taxonomy was initially scored over its own gated pool, which made the gate carry the comparison, since a taxonomy for the wrong objective was judged on a pool that had already discarded its answers; scoring both over a common per-workload pool halved the effect from +0.137 to +0.062. Second, a per-query writer schema bug crashed the run after the numbers were computed.

## The cross-over, with its gate-strength sweep

| gate keep | relevant retained | A direction (T_A vs T_B on A) | B direction (T_B vs T_A on B) | preregistered verdict |
|---|---|---|---|---|
| 0.35 | 45.6% | **+0.067 (p=0.0001)** | +0.023 (p=0.1492) | NOT SUPPORTED |
| 0.50 | 64.1% | **+0.062 (p=0.0001)** | **+0.062 (p=0.0001)** | SUPPORTED |
| 0.75 | 83.3% | **+0.156 (p=0.0001)** | +0.006 (p=0.5909) | NOT SUPPORTED |

**Read honestly: the cross-over is half-robust.** The A direction holds at every operating point, strongly and consistently. The B direction reaches significance at exactly one of three, and the preregistered rule requires both directions. Reporting only the 0.50 column would be choosing the hyperparameter that flatters the claim, which is the error that produced the phantom economics scaling law in Sprint R1, so the whole sweep is reported.

The asymmetry has a visible cause rather than being noise: the B taxonomy was built on a degenerate topic model, 3 topics against 12 for A, despite receiving 8 seed topics. A design step that collapses to three topics cannot produce a well-separated view set, so the B direction is testing a weak taxonomy rather than the principle. That is a fixable quality failure in the pipeline, diagnosed and recorded before any attempt to fix it.

## Standing position

Objective conditioning of the view set produces genuinely objective-specific taxonomies, demonstrated robustly in one direction and fragilely in the other, and only inside an objective-gated pool. It does not beat a well-formed corpus-only taxonomy on a single workload. Both findings are conditional on the gate, without which the effect vanishes entirely. Next: repair the degenerate topic model, then rerun the full sweep and report old and new together.

## Round 3: after the declared repair, both tables side by side

The repair (extended min_cluster_size sweep, degenerate models now raise rather than proceed silently) was declared in preregistration amendment 2 before it was attempted, with the diagnostic being the topic count recorded in round 2's own metrics. It worked as diagnosed: the taxonomies are now balanced (T_A five views, T_B four) instead of 12 topics against 3.

**Cross-over, pre-repair and post-repair:**

| gate keep | A direction, before | A direction, after | B direction, before | B direction, after | verdict before | verdict after |
|---|---|---|---|---|---|---|
| 0.35 | +0.067 (p=0.0001) | +0.070 (p=0.0001) | +0.023 (p=0.1492) | **+0.066 (p=0.0029)** | NOT SUPPORTED | **SUPPORTED** |
| 0.50 | +0.062 (p=0.0001) | +0.114 (p=0.0001) | +0.062 (p=0.0001) | **+0.035 (p=0.0153)** | SUPPORTED | **SUPPORTED** |
| 0.75 | +0.156 (p=0.0001) | +0.156 (p=0.0001) | +0.006 (p=0.5909) | +0.006 (p=0.5909) | NOT SUPPORTED | NOT SUPPORTED |

Two of three operating points now satisfy the preregistered both-directions rule, against one of three before. The A direction is robust everywhere and grows monotonically as the gate loosens (+0.070, +0.114, +0.156). The B direction is significant at the two tighter gates and vanishes at the loosest. The asymmetry is real and unexplained: T_B's views came back oddly action-centric (logistics_actions, incident_actions, facilities_actions and so on), which may make them less discriminative, but that is a hypothesis and not a finding.

**Ladder, across the same sweep: unstable, and that is itself the result.** The objective increment measures +0.019 (p=0.2515) at keep 0.35, **-0.061 (p=0.0018)** at 0.50, and -0.012 (p=0.5626) at 0.75. It changes sign across operating points and reaches significance in the harmful direction at one of them. No consistent effect exists in either direction, so the single-workload comparison should not be quoted at all, in either direction, without its sweep.

## The distinction this sprint establishes, and it matters

Two claims that sound alike are supported differently by this evidence.

**Supported: view sets are genuinely objective-specific.** A taxonomy designed under objective A serves A's workload better than one designed under objective B does, and the reverse, on the same documents, the same gate, the same encoder and builder, at two of three gate settings and in the A direction at all three. Objective conditioning changes the index in a way that matters for the matching workload.

**Not supported: objective conditioning beats corpus-only conditioning.** The ladder gives no stable evidence that an objective-designed taxonomy outperforms a well-formed corpus-only taxonomy on a given workload, and at one setting it is significantly worse.

These are compatible and the combination is the honest story: the objective determines WHICH specialisation the index acquires, and each specialisation serves its own workload best, but specialisation is not free and a broad corpus-only taxonomy remains a strong competitor on any single workload. For the patent that is enough to describe a technical effect tied to the objective input. For a paper it is a nuanced result that must be framed as such, not as a win.

Everything above holds only inside an objective-gated pool. Without the gate (round 1) the effect is absent entirely.

## Round 4: conditioning by level on a varied workload, and the stopping rule

Preregistered before implementation at commit 55d1632, on the owner's argument that the conditioning signal should be standing business context plus schema metadata rather than a narrow query objective, and that guided topic modelling must not suppress discovered structure. Workload deliberately varied: 200 queries across six departments, representing several agents in one line of business.

**H-DYN-7: every level of added context made retrieval worse, monotonically.**

| conditioning level | what the designer saw | nDCG@10 |
|---|---|---|
| L0 pooled | (single vector baseline) | 0.303 |
| L1 chunks | (blind windows) | 0.489 |
| **C0 sample only** | 30 stratified documents | **0.562** |
| C1 business context | + what the line of business does | 0.515 |
| C2 context plus metadata | + schema-style description of the data sources | 0.494 |
| C3 narrow objective | + the immediate query objective | 0.457 |

C2 against C0: **-0.068, 95% CI [-0.090, -0.046], p=0.0001, Holm-significant.** C1 against C0: -0.047 (p=0.0001). C3 against C2: -0.037 (p=0.0179). Each addition of context cost quality, and the effect is monotone in how much context was supplied.

The convergence condition did not fire: mean pairwise cosine between the four taxonomies is 0.833, so context genuinely produced different view sets. They are different and they are worse. The taxonomies show why: C0 derived six views from the corpus itself (fault_log, system_maint, financial_risk, legal_compliance, logistics_supply, it_incidents), while C1 to C3 produced four views organised around the business narrative (commercial_agreements, operations_logistics, finance_reporting, risk_compliance). The business framing is more elegant and covers the corpus less well, and on a workload spanning six departments coverage is what matters.

**H-DYN-8 is inconclusive because its diagnostic was wrong.** All three topic modes returned identical counts (10 topics, 16.0 percent noise), and the "topics beyond supplied" measure was computed as topic count minus supplied count, which reads zero whenever the two are equal regardless of whether the topics are actually the supplied ones. That measures arithmetic, not preservation of discovered structure. No conclusion may be drawn about zero-shot versus seeded from this run; testing it properly requires matching discovered topics to supplied ones by content rather than by counting.

## The preregistered stopping rule, honoured

Round 4's preregistration stated: "If C2 does not beat C0, the conditioning claim does not survive at any level and the programme stops pursuing it." C2 did not beat C0; it was significantly worse. **The programme therefore stops pursuing the conditioning claim, and no further architectural revision will be made in pursuit of it.**

## What the four rounds establish, taken together

1. **Conditioning the view design on purpose, at any level tested, does not improve retrieval and generally harms it.** Narrow objective (rounds 1 and 3), business context, and schema metadata (round 4) all failed, and round 4 shows the harm growing with the amount of context supplied.
2. **Conditioning does produce genuinely different taxonomies, and each serves its own matching workload better than a rival taxonomy does** (round 3 cross-over, supported at two of three gate settings). Specialisation is real; it is just not better than a good general taxonomy, and it is worse when the workload is broader than the conditioning.
3. **The gate is load-bearing for everything.** Without it (round 1) no effect exists at all.
4. **A corpus-derived taxonomy is a strong system.** C0 at 0.562 beat blind chunks at 0.489 and pooled at 0.303, which is the one robust positive across the whole programme and is also the least novel part of it.

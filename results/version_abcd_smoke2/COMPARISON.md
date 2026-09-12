# B/C/D paired hypothesis screen

BEAM 10M excluded. Same sampled questions, reader, judge and 4,000-token evidence budget. Larger-dataset samples are not full benchmark results.

| Dataset | Cohort | Comparison | Scope | Paired/requested | Metric | Reference | New | Delta | Fact setup USD | Answer USD | Judge USD |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| beam_100k | headline | B-A | screen/complete | 10/10 | beam_rubric_mean | 0.2250 | 0.2250 | 0.0000 | 0.0725 | 0.0554 | 0.0030 |
| beam_100k | headline | C-B | screen/complete | 10/10 | beam_rubric_mean | 0.2250 | 0.3250 | 0.1000 | 0.0725 | 0.0559 | 0.0029 |
| beam_100k | headline | D-C | screen/complete | 10/10 | beam_rubric_mean | 0.3250 | 0.3250 | 0.0000 | 0.0725 | 0.0546 | 0.0029 |
| locomo | headline | B-A | screen/complete | 8/8 | judge | 0.5000 | 0.3750 | -0.1250 | 0.0067 | 0.0451 | 0.0001 |
| locomo | headline | C-B | screen/complete | 8/8 | judge | 0.3750 | 0.3750 | 0.0000 | 0.0067 | 0.0452 | 0.0001 |
| locomo | headline | D-C | screen/complete | 8/8 | judge | 0.3750 | 0.2500 | -0.1250 | 0.0067 | 0.0454 | 0.0001 |
| locomo | adversarial | B-A | screen/complete | 2/2 | judge | 1.0000 | 1.0000 | 0.0000 | 0.0034 | 0.0116 | 0.0000 |
| locomo | adversarial | C-B | screen/complete | 2/2 | judge | 1.0000 | 1.0000 | 0.0000 | 0.0034 | 0.0122 | 0.0000 |
| locomo | adversarial | D-C | screen/complete | 2/2 | judge | 1.0000 | 1.0000 | 0.0000 | 0.0034 | 0.0117 | 0.0000 |

Measured new API spend: answer=$0.2653, extract=$0.0827, judge=$0.0049.

Fact setup is the shared extraction cost allocated to this cohort by history/question count. It is shown for each arm as a standalone deployment cost; the experiment pays it once across B/C/D, so do not sum it across arms. CSV includes answering plus setup cost per correct answer. Judge cost is evaluation overhead. Logical serving cost includes cache reuse; actual incremental spend is in COSTS.csv. Local compute is recorded in preparation artifacts, not priced as free.

CSV includes exploratory 95% bootstrap intervals clustered by history, intervention counts and cost per correct. A single shared corpus has no cluster interval. F1 is separate from exact match; BEAM has no binary cost-per-correct. D is a graph of discovered evidence, not the entire corpus. No cross-dataset average or superiority claim is made.

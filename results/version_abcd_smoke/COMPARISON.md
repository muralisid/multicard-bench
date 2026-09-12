# B/C/D paired hypothesis screen

BEAM 10M excluded. Same sampled questions, reader, judge and 4,000-token evidence budget. Larger-dataset samples are not full benchmark results.

| Dataset | Cohort | Comparison | Scope | Paired/requested | Metric | Reference | New | Delta | Answer USD | Judge USD |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| factconsolidation_sh_6k | headline | B-A | screen/complete | 2/2 | substring_exact_match | 1.0000 | 1.0000 | 0.0000 | 0.0088 | 0.0000 |
| factconsolidation_sh_6k | headline | C-B | screen/complete | 2/2 | substring_exact_match | 1.0000 | 1.0000 | 0.0000 | 0.0089 | 0.0000 |
| factconsolidation_sh_6k | headline | D-C | screen/complete | 2/2 | substring_exact_match | 1.0000 | 1.0000 | 0.0000 | 0.0085 | 0.0000 |

Measured new API spend: answer=$0.0263, extract=$0.0086.

Extraction is paid once and shared across B/C/D; it is additional to the serving dollars in the table. Judge cost is evaluation overhead. Logical serving cost includes cache reuse; actual incremental spend is in COSTS.csv. Local compute is recorded in preparation artifacts, not priced as free.

CSV includes exploratory 95% bootstrap intervals clustered by history, intervention counts and cost per correct. A single shared corpus has no cluster interval. F1 is separate from exact match; BEAM has no binary cost-per-correct. D is a graph of discovered evidence, not the entire corpus. No cross-dataset average or superiority claim is made.

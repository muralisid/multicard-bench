# Version A measured results

Generated from saved records. Partial scores apply only to scored questions. No cross-dataset average.

| Dataset | Cohort | Scope/status | Scored / requested / full | Metric | Score | Correct | Answer USD | Judge USD | USD/correct |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| factconsolidation_mh_6k | all | smoke/incomplete | 2 / 2 / 100 | substring exact match | 0.5000 | 1 | 0.0108 | 0.0000 | 0.0108 |
| locomo | headline | smoke/incomplete | 2 / 2 / 1540 | judge accuracy | 1.0000 | 2 | 0.0093 | 0.0000 | 0.0046 |
| locomo | adversarial | smoke/incomplete | 0 / 0 / 446 | judge accuracy | pending | N/A | 0.0000 | 0.0000 | pending |
| personamem_v2_32k | all | smoke/incomplete | 2 / 2 / 5000 | MCQ accuracy | 0.0000 | 0 | 0.0123 | 0.0000 | pending |

Answer USD is logical serving cost, including reuse from cache. Judge USD is evaluation overhead. For F1, USD/correct uses exact matches. BEAM graded scores have no binary cost/correct.

| Dataset | Indexed groups | Source tokens indexed | Mean context tokens | Retrieval p50/p95 seconds | Serving p50/p95 seconds | Config SHA256 |
|---|---:|---:|---:|---:|---:|---|
| factconsolidation_mh_6k | 1/1 | 7,082 | 4000.0 | 0.0410/0.0521 | 3.3613/4.1369 | 5c1f424f7a7c |
| locomo | 1/10 | 15,790 | 3994.5 | 0.0714/0.1166 | 3.3126/3.6085 | f43587afc953 |
| personamem_v2_32k | 1/200 | 33,728 | 4000.0 | 0.0713/0.1142 | 6.7554/6.9920 | 6f6c869def79 |

| Dataset | Metered API USD | Unresolved reservation USD | Committed USD | Done / pending / error calls | Recorded QA errors |
|---|---:|---:|---:|---:|---:|
| factconsolidation_mh_6k | 0.0108 | 0.0000 | 0.0108 | 2 / 0 / 0 | 0 |
| locomo | 0.0093 | 0.0000 | 0.0093 | 4 / 0 / 0 | 0 |
| personamem_v2_32k | 0.0123 | 0.0000 | 0.0123 | 2 / 0 / 0 | 0 |

The persistent meter includes both answering and judging, plus orphan calls. Reservations are holds, not measured spending. Full hashes, source variants, protocol differences, per-type scores, token usage and latency are retained in JSON/CSV files.

# Version A measured results

Generated from saved records. Partial scores apply only to scored questions. No cross-dataset average.

| Dataset | Cohort | Scope/status | Scored / requested / full | Metric | Score | Correct | Answer USD | Judge USD | USD/correct |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| beam_100k | all | smoke/incomplete | 1 / 1 / 400 | graded rubric mean | 1.0000 | N/A | 0.0055 | 0.0001 | pending |

Answer USD is logical serving cost, including reuse from cache. Judge USD is evaluation overhead. For F1, USD/correct uses exact matches. BEAM graded scores have no binary cost/correct.

| Dataset | Indexed groups | Source tokens indexed | Mean context tokens | Retrieval p50/p95 seconds | Serving p50/p95 seconds | Config SHA256 |
|---|---:|---:|---:|---:|---:|---|
| beam_100k | 1/20 | 145,805 | 4000.0 | 0.0274/0.0274 | 6.1389/6.1389 | c2523626aa92 |

| Dataset | Metered API USD | Unresolved reservation USD | Committed USD | Done / pending / error calls | Recorded QA errors |
|---|---:|---:|---:|---:|---:|
| beam_100k | 0.0057 | 0.0000 | 0.0057 | 2 / 0 / 0 | 0 |

The persistent meter includes both answering and judging, plus orphan calls. Reservations are holds, not measured spending. Full hashes, source variants, protocol differences, per-type scores, token usage and latency are retained in JSON/CSV files.

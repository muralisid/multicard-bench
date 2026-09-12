# Version A measured results

Generated from saved records. Partial scores apply only to scored questions. No cross-dataset average.

Deferred at the user's request: beam_10m. Cached work is preserved and excluded from the active results below. See SCOPE.json.

| Dataset | Cohort | Scope/status | Answered / scored / requested / full | Metric | Score | Correct | Answer USD | Judge USD | USD/correct |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 2wikimultihopqa | all | full/complete | 1000 / 1000 / 1000 / 1000 | F1 / exact match | 0.6572 / 0.6050 | 605 | 5.4627 | 0.0000 | 0.0090 |
| beam_100k | all | full/complete | 400 / 400 / 400 / 400 | graded rubric mean | 0.3640 | N/A | 2.1661 | 0.1331 | pending |
| beam_1m | all | full/complete | 700 / 700 / 700 / 700 | graded rubric mean | 0.3169 | N/A | 3.7657 | 0.3055 | pending |
| beam_500k | all | full/complete | 700 / 700 / 700 / 700 | graded rubric mean | 0.3118 | N/A | 3.8016 | 0.2462 | pending |
| factconsolidation_mh_262k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 0.0900 | 9 | 0.6328 | 0.0000 | 0.0703 |
| factconsolidation_mh_32k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 0.1700 | 17 | 0.6196 | 0.0000 | 0.0364 |
| factconsolidation_mh_64k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 0.1500 | 15 | 0.6266 | 0.0000 | 0.0418 |
| factconsolidation_mh_6k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 0.6000 | 60 | 0.5845 | 0.0000 | 0.0097 |
| factconsolidation_sh_262k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 0.7400 | 74 | 0.4987 | 0.0000 | 0.0067 |
| factconsolidation_sh_32k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 0.9200 | 92 | 0.4541 | 0.0000 | 0.0049 |
| factconsolidation_sh_64k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 0.8600 | 86 | 0.4640 | 0.0000 | 0.0054 |
| factconsolidation_sh_6k | all | full/complete | 100 / 100 / 100 / 100 | substring exact match | 1.0000 | 100 | 0.4503 | 0.0000 | 0.0045 |
| locomo | headline | full/complete | 1540 / 1540 / 1540 / 1540 | judge accuracy | 0.6130 | 944 | 8.3686 | 0.0152 | 0.0089 |
| locomo | adversarial | full/complete | 446 / 446 / 446 / 446 | judge accuracy | 0.8789 | 392 | 2.6217 | 0.0035 | 0.0067 |
| longmemeval_oracle | all | full/complete | 500 / 500 / 500 / 500 | judge accuracy | 0.8000 | 400 | 2.3372 | 0.0088 | 0.0058 |
| longmemeval_s | all | full/complete | 500 / 500 / 500 / 500 | judge accuracy | 0.7820 | 391 | 2.5403 | 0.0088 | 0.0065 |
| multihoprag | all | full/complete | 2556 / 2556 / 2556 / 2556 | judge accuracy | 0.4953 | 1266 | 14.0299 | 0.0344 | 0.0111 |
| musique | all | full/complete | 1000 / 1000 / 1000 / 1000 | F1 / exact match | 0.4867 / 0.3840 | 384 | 5.8547 | 0.0000 | 0.0152 |
| personamem_v2_128k | all | full/complete | 5000 / 5000 / 5000 / 5000 | MCQ accuracy | 0.4204 | 2102 | 29.5937 | 0.0000 | 0.0141 |
| personamem_v2_32k | all | full/complete | 5000 / 5000 / 5000 / 5000 | MCQ accuracy | 0.4490 | 2245 | 29.3788 | 0.0000 | 0.0131 |

Answer USD is logical serving cost, including reuse from cache. Judge USD is evaluation overhead. For F1, USD/correct uses exact matches. BEAM graded scores have no binary cost/correct.

| Dataset | Indexed groups | Source tokens indexed | Mean context tokens | Retrieval p50/p95 seconds | Serving p50/p95 seconds | Config SHA256 |
|---|---:|---:|---:|---:|---:|---|
| 2wikimultihopqa | 1/1 | 624,234 | 3998.9 | 0.0439/0.0519 | 4.1033/5.5233 | 412c63d7698d |
| beam_100k | 20/20 | 2,845,367 | 3999.7 | 0.0677/0.1081 | 4.6042/6.0991 | 476236457261 |
| beam_1m | 35/35 | 39,717,906 | 3999.7 | 0.0643/0.1122 | 4.6167/6.1764 | 1595ff9332ea |
| beam_500k | 35/35 | 19,317,471 | 3999.8 | 0.0614/0.0879 | 4.5187/6.1474 | ef8d6107d7b9 |
| factconsolidation_mh_262k | 1/1 | 321,873 | 4000.0 | 0.0538/0.0804 | 5.1434/5.7749 | 7cc46f7bfff4 |
| factconsolidation_mh_32k | 1/1 | 38,279 | 3998.2 | 0.0929/0.1399 | 5.2056/5.9824 | ee70506aab6c |
| factconsolidation_mh_64k | 1/1 | 77,984 | 3999.7 | 0.0665/0.1483 | 5.0025/5.8688 | 793b9ee12611 |
| factconsolidation_mh_6k | 1/1 | 7,082 | 4000.0 | 0.0363/0.0544 | 4.2804/5.6834 | aaac2861a268 |
| factconsolidation_sh_262k | 1/1 | 321,873 | 4000.0 | 0.0526/0.0726 | 2.6038/5.1654 | 7a0a9ba948a7 |
| factconsolidation_sh_32k | 1/1 | 38,279 | 3998.3 | 0.0414/0.0534 | 2.3810/3.7234 | 423ea7903708 |
| factconsolidation_sh_64k | 1/1 | 77,984 | 3999.8 | 0.0722/0.1316 | 2.5004/4.6320 | d806a7e405ef |
| factconsolidation_sh_6k | 1/1 | 7,082 | 4000.0 | 0.0355/0.0469 | 2.4725/3.4333 | 0eb8af723fc8 |
| locomo | 10/10 | 196,490 | 3997.0 | 0.0266/0.0332 | 3.6014/5.7360 | a44e0e42c720 |
| longmemeval_oracle | 500/500 | 3,029,881 | 3623.8 | 0.0427/0.0777 | 3.6022/5.8701 | e754b731c347 |
| longmemeval_s | 500/500 | 54,701,150 | 3999.6 | 0.0512/0.0668 | 3.6813/5.7239 | 72b234dcc94c |
| multihoprag | 1/1 | 1,423,563 | 3999.5 | 0.0434/0.0806 | 4.4588/5.9664 | 454f4011c0a5 |
| musique | 1/1 | 1,267,600 | 3998.9 | 0.0374/0.0489 | 4.6020/5.7473 | ea17ffc3550b |
| personamem_v2_128k | 200/200 | 23,001,920 | 3999.0 | 0.0674/0.1074 | 4.7814/6.0645 | 11df218c2a1f |
| personamem_v2_32k | 200/200 | 6,718,637 | 3999.1 | 0.0692/0.1043 | 4.8221/6.1262 | cef4bc2db955 |

| Dataset | Metered API USD | Unresolved reservation USD | Committed USD | Done / pending / error calls | Recorded QA errors |
|---|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 5.4627 | 0.0000 | 5.4627 | 1000 / 0 / 0 | 1 |
| beam_100k | 2.2986 | 0.0000 | 2.2986 | 1935 / 0 / 0 | 1 |
| beam_1m | 4.0677 | 0.0000 | 4.0677 | 4383 / 0 / 0 | 1 |
| beam_500k | 4.0468 | 0.0000 | 4.0468 | 3608 / 0 / 0 | 1 |
| factconsolidation_mh_262k | 0.6328 | 0.0000 | 0.6328 | 100 / 0 / 0 | 0 |
| factconsolidation_mh_32k | 0.6196 | 0.0000 | 0.6196 | 100 / 0 / 0 | 0 |
| factconsolidation_mh_64k | 0.6266 | 0.0000 | 0.6266 | 100 / 0 / 0 | 0 |
| factconsolidation_mh_6k | 0.5845 | 0.0000 | 0.5845 | 100 / 0 / 0 | 0 |
| factconsolidation_sh_262k | 0.4987 | 0.0000 | 0.4987 | 100 / 0 / 0 | 0 |
| factconsolidation_sh_32k | 0.4541 | 0.0000 | 0.4541 | 100 / 0 / 0 | 0 |
| factconsolidation_sh_64k | 0.4640 | 0.0000 | 0.4640 | 100 / 0 / 0 | 0 |
| factconsolidation_sh_6k | 0.4503 | 0.0000 | 0.4503 | 100 / 0 / 0 | 0 |
| locomo | 10.9475 | 0.0000 | 10.9475 | 3949 / 0 / 0 | 0 |
| longmemeval_oracle | 2.3460 | 0.0000 | 2.3460 | 1000 / 0 / 0 | 0 |
| longmemeval_s | 2.5475 | 0.0000 | 2.5475 | 897 / 0 / 0 | 0 |
| multihoprag | 14.0643 | 0.0000 | 14.0643 | 5112 / 0 / 0 | 0 |
| musique | 5.8485 | 0.0000 | 5.8485 | 999 / 0 / 0 | 0 |
| personamem_v2_128k | 29.5937 | 0.0000 | 29.5937 | 5000 / 0 / 0 | 0 |
| personamem_v2_32k | 29.3788 | 0.0000 | 29.3788 | 5000 / 0 / 0 | 0 |

The persistent meter includes both answering and judging, plus orphan calls. Reservations are holds, not measured spending. Full hashes, source variants, protocol differences, per-type scores, token usage and latency are retained in JSON/CSV files.

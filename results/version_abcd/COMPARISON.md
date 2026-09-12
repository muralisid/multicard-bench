# B/C/D paired hypothesis screen

BEAM 10M excluded. Same sampled questions, reader, judge and 4,000-token evidence budget. Larger-dataset samples are not full benchmark results.

| Dataset | Cohort | Comparison | Scope | Paired/requested | Metric | Reference | New | Delta | Delta vs A | Fact setup USD | Answer USD | Judge USD |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | headline | B-A | screen/complete | 100/100 | f1 | 0.5718 | 0.5806 | 0.0087 | 0.0087 | 0.3432 | 0.5590 | 0.0000 |
| 2wikimultihopqa | headline | C-B | screen/complete | 100/100 | f1 | 0.5806 | 0.6089 | 0.0283 | 0.0371 | 0.3432 | 0.5587 | 0.0000 |
| 2wikimultihopqa | headline | D-C | screen/complete | 100/100 | f1 | 0.6089 | 0.5541 | -0.0548 | -0.0178 | 0.3432 | 0.5676 | 0.0000 |
| beam_100k | headline | B-A | screen/complete | 100/100 | beam_rubric_mean | 0.3902 | 0.3797 | -0.0105 | -0.0105 | 0.5894 | 0.5371 | 0.0326 |
| beam_100k | headline | C-B | screen/complete | 100/100 | beam_rubric_mean | 0.3797 | 0.3710 | -0.0087 | -0.0192 | 0.5894 | 0.5361 | 0.0326 |
| beam_100k | headline | D-C | screen/complete | 100/100 | beam_rubric_mean | 0.3710 | 0.3810 | 0.0100 | -0.0092 | 0.5894 | 0.5462 | 0.0326 |
| beam_1m | headline | B-A | screen/complete | 100/100 | beam_rubric_mean | 0.2667 | 0.2757 | 0.0090 | 0.0090 | 0.6658 | 0.5403 | 0.0437 |
| beam_1m | headline | C-B | screen/complete | 100/100 | beam_rubric_mean | 0.2757 | 0.2808 | 0.0051 | 0.0141 | 0.6658 | 0.5429 | 0.0438 |
| beam_1m | headline | D-C | screen/complete | 100/100 | beam_rubric_mean | 0.2808 | 0.2898 | 0.0090 | 0.0231 | 0.6658 | 0.5433 | 0.0439 |
| beam_500k | headline | B-A | screen/complete | 100/100 | beam_rubric_mean | 0.2677 | 0.2471 | -0.0205 | -0.0205 | 0.6474 | 0.5443 | 0.0332 |
| beam_500k | headline | C-B | screen/complete | 100/100 | beam_rubric_mean | 0.2471 | 0.2412 | -0.0059 | -0.0264 | 0.6474 | 0.5468 | 0.0329 |
| beam_500k | headline | D-C | screen/complete | 100/100 | beam_rubric_mean | 0.2412 | 0.2739 | 0.0327 | 0.0062 | 0.6474 | 0.5495 | 0.0328 |
| factconsolidation_mh_262k | headline | B-A | full/complete | 100/100 | substring_exact_match | 0.0900 | 0.0800 | -0.0100 | -0.0100 | 0.3498 | 0.6176 | 0.0000 |
| factconsolidation_mh_262k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.0800 | 0.0800 | 0.0000 | -0.0100 | 0.3498 | 0.6396 | 0.0000 |
| factconsolidation_mh_262k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.0800 | 0.0800 | 0.0000 | -0.0100 | 0.3498 | 0.6390 | 0.0000 |
| factconsolidation_mh_32k | headline | B-A | full/complete | 100/100 | substring_exact_match | 0.1700 | 0.1800 | 0.0100 | 0.0100 | 0.0951 | 0.5995 | 0.0000 |
| factconsolidation_mh_32k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.1800 | 0.0800 | -0.1000 | -0.0900 | 0.0951 | 0.6388 | 0.0000 |
| factconsolidation_mh_32k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.0800 | 0.2500 | 0.1700 | 0.0800 | 0.0951 | 0.6208 | 0.0000 |
| factconsolidation_mh_64k | headline | B-A | full/complete | 100/100 | substring_exact_match | 0.1500 | 0.1600 | 0.0100 | 0.0100 | 0.1749 | 0.6038 | 0.0000 |
| factconsolidation_mh_64k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.1600 | 0.1500 | -0.0100 | 0.0000 | 0.1749 | 0.6323 | 0.0000 |
| factconsolidation_mh_64k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.1500 | 0.1600 | 0.0100 | 0.0100 | 0.1749 | 0.6217 | 0.0000 |
| factconsolidation_mh_6k | headline | B-A | full/complete | 100/100 | substring_exact_match | 0.6000 | 0.5700 | -0.0300 | -0.0300 | 0.0173 | 0.5678 | 0.0000 |
| factconsolidation_mh_6k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.5700 | 0.5500 | -0.0200 | -0.0500 | 0.0173 | 0.5750 | 0.0000 |
| factconsolidation_mh_6k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.5500 | 0.5900 | 0.0400 | -0.0100 | 0.0173 | 0.5658 | 0.0000 |
| factconsolidation_sh_262k | headline | B-A | full/complete | 100/100 | substring_exact_match | 0.7400 | 0.7900 | 0.0500 | 0.0500 | 0.3546 | 0.4885 | 0.0000 |
| factconsolidation_sh_262k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.7900 | 0.6000 | -0.1900 | -0.1400 | 0.3546 | 0.5126 | 0.0000 |
| factconsolidation_sh_262k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.6000 | 0.6900 | 0.0900 | -0.0500 | 0.3546 | 0.4909 | 0.0000 |
| factconsolidation_sh_32k | headline | B-A | full/complete | 100/100 | substring_exact_match | 0.9200 | 0.9200 | 0.0000 | 0.0000 | 0.0959 | 0.4504 | 0.0000 |
| factconsolidation_sh_32k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.9200 | 0.7900 | -0.1300 | -0.1300 | 0.0959 | 0.4853 | 0.0000 |
| factconsolidation_sh_32k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.7900 | 0.9100 | 0.1200 | -0.0100 | 0.0959 | 0.4588 | 0.0000 |
| factconsolidation_sh_64k | headline | B-A | full/complete | 100/100 | substring_exact_match | 0.8600 | 0.9000 | 0.0400 | 0.0400 | 0.1699 | 0.4599 | 0.0000 |
| factconsolidation_sh_64k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.9000 | 0.7400 | -0.1600 | -0.1200 | 0.1699 | 0.4812 | 0.0000 |
| factconsolidation_sh_64k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.7400 | 0.9000 | 0.1600 | 0.0400 | 0.1699 | 0.4539 | 0.0000 |
| factconsolidation_sh_6k | headline | B-A | full/complete | 100/100 | substring_exact_match | 1.0000 | 0.8700 | -0.1300 | -0.1300 | 0.0177 | 0.4427 | 0.0000 |
| factconsolidation_sh_6k | headline | C-B | full/complete | 100/100 | substring_exact_match | 0.8700 | 0.8400 | -0.0300 | -0.1600 | 0.0177 | 0.4605 | 0.0000 |
| factconsolidation_sh_6k | headline | D-C | full/complete | 100/100 | substring_exact_match | 0.8400 | 0.8600 | 0.0200 | -0.1400 | 0.0177 | 0.4517 | 0.0000 |
| locomo | headline | B-A | screen/complete | 77/77 | judge | 0.6104 | 0.5455 | -0.0649 | -0.0649 | 0.0714 | 0.4267 | 0.0007 |
| locomo | headline | C-B | screen/complete | 77/77 | judge | 0.5455 | 0.5455 | 0.0000 | -0.0649 | 0.0714 | 0.4332 | 0.0007 |
| locomo | headline | D-C | screen/complete | 77/77 | judge | 0.5455 | 0.5584 | 0.0130 | -0.0519 | 0.0714 | 0.4386 | 0.0007 |
| locomo | adversarial | B-A | screen/complete | 23/23 | judge | 0.9130 | 1.0000 | 0.0870 | 0.0870 | 0.0208 | 0.1373 | 0.0002 |
| locomo | adversarial | C-B | screen/complete | 23/23 | judge | 1.0000 | 1.0000 | 0.0000 | 0.0870 | 0.0208 | 0.1396 | 0.0002 |
| locomo | adversarial | D-C | screen/complete | 23/23 | judge | 1.0000 | 0.9130 | -0.0870 | 0.0000 | 0.0208 | 0.1370 | 0.0002 |
| longmemeval_oracle | headline | B-A | screen/complete | 100/100 | judge | 0.7600 | 0.7400 | -0.0200 | -0.0200 | 0.4704 | 0.4696 | 0.0017 |
| longmemeval_oracle | headline | C-B | screen/complete | 100/100 | judge | 0.7400 | 0.8000 | 0.0600 | 0.0400 | 0.4704 | 0.4832 | 0.0017 |
| longmemeval_oracle | headline | D-C | screen/complete | 100/100 | judge | 0.8000 | 0.7600 | -0.0400 | 0.0000 | 0.4704 | 0.4860 | 0.0017 |
| longmemeval_s | headline | B-A | screen/complete | 100/100 | judge | 0.7200 | 0.7500 | 0.0300 | 0.0300 | 0.4606 | 0.5153 | 0.0017 |
| longmemeval_s | headline | C-B | screen/complete | 100/100 | judge | 0.7500 | 0.7400 | -0.0100 | 0.0200 | 0.4606 | 0.5200 | 0.0017 |
| longmemeval_s | headline | D-C | screen/complete | 100/100 | judge | 0.7400 | 0.7500 | 0.0100 | 0.0300 | 0.4606 | 0.5193 | 0.0017 |
| multihoprag | headline | B-A | screen/complete | 100/100 | judge | 0.5300 | 0.5400 | 0.0100 | 0.0100 | 0.3983 | 0.5492 | 0.0013 |
| multihoprag | headline | C-B | screen/complete | 100/100 | judge | 0.5400 | 0.5600 | 0.0200 | 0.0300 | 0.3983 | 0.5485 | 0.0013 |
| multihoprag | headline | D-C | screen/complete | 100/100 | judge | 0.5600 | 0.5700 | 0.0100 | 0.0400 | 0.3983 | 0.5432 | 0.0013 |
| musique | headline | B-A | screen/complete | 100/100 | f1 | 0.5301 | 0.4976 | -0.0325 | -0.0325 | 0.4004 | 0.5859 | 0.0000 |
| musique | headline | C-B | screen/complete | 100/100 | f1 | 0.4976 | 0.4986 | 0.0010 | -0.0315 | 0.4004 | 0.5858 | 0.0000 |
| musique | headline | D-C | screen/complete | 100/100 | f1 | 0.4986 | 0.5382 | 0.0395 | 0.0081 | 0.4004 | 0.5855 | 0.0000 |
| personamem_v2_128k | headline | B-A | screen/complete | 100/100 | mcq | 0.4100 | 0.4000 | -0.0100 | -0.0100 | 0.2592 | 0.5930 | 0.0000 |
| personamem_v2_128k | headline | C-B | screen/complete | 100/100 | mcq | 0.4000 | 0.3900 | -0.0100 | -0.0200 | 0.2592 | 0.5937 | 0.0000 |
| personamem_v2_128k | headline | D-C | screen/complete | 100/100 | mcq | 0.3900 | 0.4200 | 0.0300 | 0.0100 | 0.2592 | 0.5871 | 0.0000 |
| personamem_v2_32k | headline | B-A | screen/complete | 100/100 | mcq | 0.4500 | 0.5000 | 0.0500 | 0.0500 | 0.3159 | 0.5906 | 0.0000 |
| personamem_v2_32k | headline | C-B | screen/complete | 100/100 | mcq | 0.5000 | 0.5000 | 0.0000 | 0.0500 | 0.3159 | 0.5918 | 0.0000 |
| personamem_v2_32k | headline | D-C | screen/complete | 100/100 | mcq | 0.5000 | 0.4400 | -0.0600 | -0.0100 | 0.3159 | 0.5935 | 0.0000 |

Measured new API spend: answer=$26.5129, extract=$5.6084, judge=$0.1638.

Fact setup is the shared extraction cost allocated to this cohort by history/question count. It is shown for each arm as a standalone deployment cost; the experiment pays it once across B/C/D, so do not sum it across arms. CSV includes answering plus setup cost per correct answer. Judge cost is evaluation overhead. Logical serving cost includes cache reuse; actual incremental spend is in COSTS.csv. Local compute is recorded in preparation artifacts, not priced as free.

CSV includes exploratory 95% bootstrap intervals clustered by history, intervention counts and cost per correct. A single shared corpus has no cluster interval. F1 is separate from exact match; BEAM has no binary cost-per-correct. D is a graph of discovered evidence, not the entire corpus. No cross-dataset average or superiority claim is made.

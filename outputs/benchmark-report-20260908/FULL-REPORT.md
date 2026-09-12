# TG-VGRAG experiment results

Generated from saved run outputs at 2026-09-08T03:26:12.761929+00:00.

Version A is complete: **20,142 questions across 19 active variants**. B/C/D are complete on **19/19 variants**. The B/C/D design is a 100-question paired screen per variant, except the eight 100-question FactConsolidation variants, which are full tests. BEAM 10M remains deferred.

The results are mixed. The baseline remains competitive. Fact compression, temporal validity and graph ranking each help some cases and hurt others. These measurements do not establish an overall winner or industry superiority.

The audit found **21 stored fact intervals with an end before the start**. C and D include this known defect. Their scores below are the measured results before any correction, and the defect's effect on accuracy has not been isolated.

## One summary table

All scores use a 0-100 display scale. F1 and rubric mean are graded scores, not percent correct. Compare B/C/D with **A matched**, not A full. LoCoMo adversarial is a separate cohort, not another dataset.

| Test | Metric | A full N | A full | Matched N | A matched | B | C | D | Best matched | B/C/D scope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LoCoMo | Judge accuracy | 1,540 | 61.30 | 77 | 61.04 | 54.55 | 54.55 | 55.84 | A | screen |
| LoCoMo adversarial | Judge accuracy | 446 | 87.89 | 23 | 91.30 | 100.00 | 100.00 | 91.30 | B/C | screen |
| MuSiQue | F1 | 1,000 | 48.67 | 100 | 53.01 | 49.76 | 49.86 | 53.82 | D | screen |
| 2WikiMultiHopQA | F1 | 1,000 | 65.72 | 100 | 57.18 | 58.06 | 60.89 | 55.41 | C | screen |
| LongMemEval S | Judge accuracy | 500 | 78.20 | 100 | 72.00 | 75.00 | 74.00 | 75.00 | B/D | screen |
| LongMemEval oracle | Judge accuracy | 500 | 80.00 | 100 | 76.00 | 74.00 | 80.00 | 76.00 | C | screen |
| MultiHop-RAG | Judge accuracy | 2,556 | 49.53 | 100 | 53.00 | 54.00 | 56.00 | 57.00 | D | screen |
| FactConsolidation SH 6K | Substring EM | 100 | 100.00 | 100 | 100.00 | 87.00 | 84.00 | 86.00 | A | full |
| FactConsolidation MH 6K | Substring EM | 100 | 60.00 | 100 | 60.00 | 57.00 | 55.00 | 59.00 | A | full |
| FactConsolidation SH 32K | Substring EM | 100 | 92.00 | 100 | 92.00 | 92.00 | 79.00 | 91.00 | A/B | full |
| FactConsolidation MH 32K | Substring EM | 100 | 17.00 | 100 | 17.00 | 18.00 | 8.00 | 25.00 | D | full |
| FactConsolidation SH 64K | Substring EM | 100 | 86.00 | 100 | 86.00 | 90.00 | 74.00 | 90.00 | B/D | full |
| FactConsolidation MH 64K | Substring EM | 100 | 15.00 | 100 | 15.00 | 16.00 | 15.00 | 16.00 | B/D | full |
| FactConsolidation SH 262K | Substring EM | 100 | 74.00 | 100 | 74.00 | 79.00 | 60.00 | 69.00 | B | full |
| FactConsolidation MH 262K | Substring EM | 100 | 9.00 | 100 | 9.00 | 8.00 | 8.00 | 8.00 | A | full |
| PersonaMem v2 32K | MCQ accuracy | 5,000 | 44.90 | 100 | 45.00 | 50.00 | 50.00 | 44.00 | B/C | screen |
| PersonaMem v2 128K | MCQ accuracy | 5,000 | 42.04 | 100 | 41.00 | 40.00 | 39.00 | 42.00 | D | screen |
| BEAM 100K | Rubric mean | 400 | 36.40 | 100 | 39.02 | 37.97 | 37.10 | 38.10 | A | screen |
| BEAM 500K | Rubric mean | 700 | 31.18 | 100 | 26.77 | 24.71 | 24.13 | 27.39 | D | screen |
| BEAM 1M | Rubric mean | 700 | 31.69 | 100 | 26.67 | 27.57 | 28.08 | 28.98 | D | screen |

## Test variables

| Variable | Setting | Purpose |
| --- | --- | --- |
| Arm A | BM25 + local MiniLM dense retrieval; raw source evidence | Zero ingestion LLM calls; baseline |
| Arm B | A + cached facts from detected source text | Test whether fact compression improves evidence per token |
| Arm C | B + validity intervals, supersession and source replacement | Test whether handling changed facts reduces stale answers |
| Arm D | C + topic index, community index and relation traversal | Test whether graph links improve evidence ranking |
| Reader / provider | Gemini 3.6 Flash / Google Vertex AI | Same reader across every arm |
| Judge / extractor | Gemini 2.5 Flash Lite / Google Vertex AI | Judging only where required; B/C/D fact extraction |
| Evidence budget | 4,000 MiniLM WordPiece tokens including source labels | Same reader evidence allowance; nominal corpus size is separate |
| Answer allowance | 768 provider output tokens; includes hidden reasoning | Empty answers score zero; no quality retries |
| Judge allowance | 64 output tokens; BEAM rubric 512 | Preserve dataset-specific metric |
| A retrieval | BM25 + all-MiniLM-L6-v2; equal RRF weights; k=60; top 100/channel | Local lexical and embedding search |
| Dense windows | 200 tokens, overlap 30; max window score folded to parent source | Long source units remain retrievable |
| Sample | 100 questions/dataset; proportional type strata; seed 13 | Same IDs in A/B/C/D; all 100-question MAB variants are full |
| B detection | Up to 8 units from first 30 A ranks; at most 12 fixed 500-token pieces/query | Bound extraction cost; not exhaustive fact coverage |
| B extraction | At most 16 facts/piece; 2,048 output tokens; units over 1,000 tokens stay raw | Quote/object grounding required; invalid output falls back to raw |
| Fact cache | Key includes namespace, source piece, metadata, prompt and model parameters | Reuse extraction across arms and questions; new protocol means new key |
| Memory state | Discover all sampled queries' evidence per history before answering | Warm shared memory; setup charged separately; not online cold-start replay |
| C supersession | Matching subject/predicate/scope; exclusive state; strict date or benchmark serial | Preserve history/events/multi-value facts; no guessed temporal order |
| D topics | TF-IDF + MiniBatchKMeans, seed 13, up to 32 topics | Local topic membership and separate inverted index |
| D communities | Leiden on source projection with shared entities; filter entity hubs | Local community membership and separate inverted index |
| D relationships | Typed source/entity/fact nodes and edges; two-hop traversal | Graph covers discovered evidence, not all corpus facts |
| D fusion weights | Base 1.0; topic 0.2; community 0.2; relation 0.4; RRF k=60 | Fixed before screening; no evaluation-answer tuning |
| API price inputs | Reader input/output $0.75/$3.75 per million; Lite $0.10/$0.40 | Recorded-token cost estimates; not a reconciled cloud invoice |
| Concurrency / caps | 16 QA workers; A cap $150; B/C/D shared cap $60; program cap $500 | Persistent reservation ledger before each request |

## Hypothesis findings

| Hypothesis | Observed result | What it means |
| --- | --- | --- |
| A: inexpensive indexing | All 19 active variants indexed with zero ingestion LLM calls. Recorded local index/build-load duration totals 647.1 minutes across groups. | The zero-ingestion-LLM design is implemented and measured. Indexing still needs local compute and storage. This run does not establish a vendor total-cost comparison. |
| B: lazy fact compression | PersonaMem v2 32K: A 45.00 to B 50.00; LoCoMo: A 61.04 to B 54.55; FactConsolidation SH 6K: A 100.00 to B 87.00 | Mixed quality. B is useful on some tests, but bounded fact extraction can lose useful source information. The screen does not isolate which losses came from extraction versus evidence packing. |
| C: temporal validity | FactConsolidation SH 262K: B 79.00 to C 60.00; FactConsolidation SH 64K: B 90.00 to C 74.00; LongMemEval oracle: B 74.00 to C 80.00 | The current temporal implementation does not consistently improve answers. It hurts all four single-hop FactConsolidation variants compared with B. Improvement in oracle cannot establish full-history retrieval gains. |
| C: actual invalidation coverage | Only 20 stored facts are superseded outside FactConsolidation in prepared datasets. Both LongMemEval variants and both PersonaMem variants have zero. | C also changes validity labels and filters by supplied dates. Score movement without supersession is not evidence that old-fact invalidation helped. |
| C: interval correctness | 21 stored facts have an end date before their start date. LongMemEval S: 4; LongMemEval oracle: 3; MultiHop-RAG: 4; BEAM 100K: 3; BEAM 500K: 4; BEAM 1M: 3 | This is an implementation defect. Correct start-date fallback and interval validation before relying on C or D for temporal memory. The current measurements preserve the defect; its accuracy effect is not isolated. |
| D: combined graph ranking | FactConsolidation MH 32K: C 8.00 to D 25.00; MultiHop-RAG: A 53.00 to D 57.00; 2WikiMultiHopQA: A 57.18 to D 55.41 | The combined graph helps some tests and hurts others. Recovery from C does not always beat A. Separate topic, community and relationship contributions have not been measured. |
| Extraction robustness | MuSiQue: 409/739 pieces fell back to raw; LongMemEval S: 594/958 pieces fell back to raw; MultiHop-RAG: 395/501 pieces fell back to raw | Grounding/format checks frequently reject extraction on natural text. These are retained raw-source fallbacks, not missing answers. Improve extraction coverage before assuming the fact layer is fully exercised. |
| Cost plus quality | Per-arm tables include questions correct and setup-plus-answer dollars per correct answer. BEAM uses graded rubric scores and has no binary cost-per-correct figure. | Lower cost is useful only alongside sufficient quality. Shared-cache experiment savings are not the same as standalone deployment economics. |

## Full Version A quality and cost

USD below are logical model costs. Answer generation is serving cost; judging is evaluation overhead. A has zero ingestion LLM calls, with local indexing time and storage recorded separately.

| Test | Arm | N | Score /100 | Correct | EM /100 | Fact setup | Answers | Judging | Serving/Q | Serving/correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | A full | 1,540 | 61.30 | 944 | N/A | $0.0000 | $8.3686 | $0.0152 | $0.0054 | $0.0089 |
| LoCoMo adversarial | A full | 446 | 87.89 | 392 | N/A | $0.0000 | $2.6217 | $0.0035 | $0.0059 | $0.0067 |
| MuSiQue | A full | 1,000 | 48.67 | 384 | 38.40 | $0.0000 | $5.8547 | $0.0000 | $0.0059 | $0.0152 |
| 2WikiMultiHopQA | A full | 1,000 | 65.72 | 605 | 60.50 | $0.0000 | $5.4627 | $0.0000 | $0.0055 | $0.0090 |
| LongMemEval S | A full | 500 | 78.20 | 391 | N/A | $0.0000 | $2.5403 | $0.0088 | $0.0051 | $0.0065 |
| LongMemEval oracle | A full | 500 | 80.00 | 400 | N/A | $0.0000 | $2.3372 | $0.0088 | $0.0047 | $0.0058 |
| MultiHop-RAG | A full | 2,556 | 49.53 | 1,266 | N/A | $0.0000 | $14.0299 | $0.0344 | $0.0055 | $0.0111 |
| FactConsolidation SH 6K | A full | 100 | 100.00 | 100 | N/A | $0.0000 | $0.4503 | $0.0000 | $0.0045 | $0.0045 |
| FactConsolidation MH 6K | A full | 100 | 60.00 | 60 | N/A | $0.0000 | $0.5845 | $0.0000 | $0.0058 | $0.0097 |
| FactConsolidation SH 32K | A full | 100 | 92.00 | 92 | N/A | $0.0000 | $0.4541 | $0.0000 | $0.0045 | $0.0049 |
| FactConsolidation MH 32K | A full | 100 | 17.00 | 17 | N/A | $0.0000 | $0.6196 | $0.0000 | $0.0062 | $0.0364 |
| FactConsolidation SH 64K | A full | 100 | 86.00 | 86 | N/A | $0.0000 | $0.4640 | $0.0000 | $0.0046 | $0.0054 |
| FactConsolidation MH 64K | A full | 100 | 15.00 | 15 | N/A | $0.0000 | $0.6266 | $0.0000 | $0.0063 | $0.0418 |
| FactConsolidation SH 262K | A full | 100 | 74.00 | 74 | N/A | $0.0000 | $0.4987 | $0.0000 | $0.0050 | $0.0067 |
| FactConsolidation MH 262K | A full | 100 | 9.00 | 9 | N/A | $0.0000 | $0.6328 | $0.0000 | $0.0063 | $0.0703 |
| PersonaMem v2 32K | A full | 5,000 | 44.90 | 2,245 | N/A | $0.0000 | $29.3788 | $0.0000 | $0.0059 | $0.0131 |
| PersonaMem v2 128K | A full | 5,000 | 42.04 | 2,102 | N/A | $0.0000 | $29.5937 | $0.0000 | $0.0059 | $0.0141 |
| BEAM 100K | A full | 400 | 36.40 | N/A | N/A | $0.0000 | $2.1661 | $0.1331 | $0.0054 | N/A |
| BEAM 500K | A full | 700 | 31.18 | N/A | N/A | $0.0000 | $3.8016 | $0.2462 | $0.0054 | N/A |
| BEAM 1M | A full | 700 | 31.69 | N/A | N/A | $0.0000 | $3.7657 | $0.3055 | $0.0054 | N/A |

## Matched A/B/C/D quality and cost

Fact setup is shown for each arm as its standalone allocation. The experiment pays shared extraction once across B/C/D. Serving/Q and serving/correct include that setup, exclude judging and unpriced local compute. Correct counts on F1 datasets use exact match; BEAM has no binary correct count.

| Test | Arm | N | Score /100 | Correct | EM /100 | Fact setup | Answers | Judging | Serving/Q | Serving/correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | A | 77 | 61.04 | 47 | N/A | $0.0000 | $0.4300 | $0.0007 | $0.0056 | $0.0091 |
| LoCoMo | B | 77 | 54.55 | 42 | N/A | $0.0714 | $0.4267 | $0.0007 | $0.0065 | $0.0119 |
| LoCoMo | C | 77 | 54.55 | 42 | N/A | $0.0714 | $0.4332 | $0.0007 | $0.0066 | $0.0120 |
| LoCoMo | D | 77 | 55.84 | 43 | N/A | $0.0714 | $0.4386 | $0.0007 | $0.0066 | $0.0119 |
| LoCoMo adversarial | A | 23 | 91.30 | 21 | N/A | $0.0000 | $0.1357 | $0.0002 | $0.0059 | $0.0065 |
| LoCoMo adversarial | B | 23 | 100.00 | 23 | N/A | $0.0208 | $0.1373 | $0.0002 | $0.0069 | $0.0069 |
| LoCoMo adversarial | C | 23 | 100.00 | 23 | N/A | $0.0208 | $0.1396 | $0.0002 | $0.0070 | $0.0070 |
| LoCoMo adversarial | D | 23 | 91.30 | 21 | N/A | $0.0208 | $0.1370 | $0.0002 | $0.0069 | $0.0075 |
| MuSiQue | A | 100 | 53.01 | 46 | 46.00 | $0.0000 | $0.5798 | $0.0000 | $0.0058 | $0.0126 |
| MuSiQue | B | 100 | 49.76 | 46 | 46.00 | $0.4004 | $0.5859 | $0.0000 | $0.0099 | $0.0214 |
| MuSiQue | C | 100 | 49.86 | 44 | 44.00 | $0.4004 | $0.5858 | $0.0000 | $0.0099 | $0.0224 |
| MuSiQue | D | 100 | 53.82 | 47 | 47.00 | $0.4004 | $0.5855 | $0.0000 | $0.0099 | $0.0210 |
| 2WikiMultiHopQA | A | 100 | 57.18 | 51 | 51.00 | $0.0000 | $0.5516 | $0.0000 | $0.0055 | $0.0108 |
| 2WikiMultiHopQA | B | 100 | 58.06 | 53 | 53.00 | $0.3432 | $0.5590 | $0.0000 | $0.0090 | $0.0170 |
| 2WikiMultiHopQA | C | 100 | 60.89 | 56 | 56.00 | $0.3432 | $0.5587 | $0.0000 | $0.0090 | $0.0161 |
| 2WikiMultiHopQA | D | 100 | 55.41 | 51 | 51.00 | $0.3432 | $0.5676 | $0.0000 | $0.0091 | $0.0179 |
| LongMemEval S | A | 100 | 72.00 | 72 | N/A | $0.0000 | $0.5184 | $0.0017 | $0.0052 | $0.0072 |
| LongMemEval S | B | 100 | 75.00 | 75 | N/A | $0.4606 | $0.5153 | $0.0017 | $0.0098 | $0.0130 |
| LongMemEval S | C | 100 | 74.00 | 74 | N/A | $0.4606 | $0.5200 | $0.0017 | $0.0098 | $0.0133 |
| LongMemEval S | D | 100 | 75.00 | 75 | N/A | $0.4606 | $0.5193 | $0.0017 | $0.0098 | $0.0131 |
| LongMemEval oracle | A | 100 | 76.00 | 76 | N/A | $0.0000 | $0.4752 | $0.0017 | $0.0048 | $0.0063 |
| LongMemEval oracle | B | 100 | 74.00 | 74 | N/A | $0.4704 | $0.4696 | $0.0017 | $0.0094 | $0.0127 |
| LongMemEval oracle | C | 100 | 80.00 | 80 | N/A | $0.4704 | $0.4832 | $0.0017 | $0.0095 | $0.0119 |
| LongMemEval oracle | D | 100 | 76.00 | 76 | N/A | $0.4704 | $0.4860 | $0.0017 | $0.0096 | $0.0126 |
| MultiHop-RAG | A | 100 | 53.00 | 53 | N/A | $0.0000 | $0.5399 | $0.0013 | $0.0054 | $0.0102 |
| MultiHop-RAG | B | 100 | 54.00 | 54 | N/A | $0.3983 | $0.5492 | $0.0013 | $0.0095 | $0.0175 |
| MultiHop-RAG | C | 100 | 56.00 | 56 | N/A | $0.3983 | $0.5485 | $0.0013 | $0.0095 | $0.0169 |
| MultiHop-RAG | D | 100 | 57.00 | 57 | N/A | $0.3983 | $0.5432 | $0.0013 | $0.0094 | $0.0165 |
| FactConsolidation SH 6K | A | 100 | 100.00 | 100 | N/A | $0.0000 | $0.4503 | $0.0000 | $0.0045 | $0.0045 |
| FactConsolidation SH 6K | B | 100 | 87.00 | 87 | N/A | $0.0177 | $0.4427 | $0.0000 | $0.0046 | $0.0053 |
| FactConsolidation SH 6K | C | 100 | 84.00 | 84 | N/A | $0.0177 | $0.4605 | $0.0000 | $0.0048 | $0.0057 |
| FactConsolidation SH 6K | D | 100 | 86.00 | 86 | N/A | $0.0177 | $0.4517 | $0.0000 | $0.0047 | $0.0055 |
| FactConsolidation MH 6K | A | 100 | 60.00 | 60 | N/A | $0.0000 | $0.5845 | $0.0000 | $0.0058 | $0.0097 |
| FactConsolidation MH 6K | B | 100 | 57.00 | 57 | N/A | $0.0173 | $0.5678 | $0.0000 | $0.0059 | $0.0103 |
| FactConsolidation MH 6K | C | 100 | 55.00 | 55 | N/A | $0.0173 | $0.5750 | $0.0000 | $0.0059 | $0.0108 |
| FactConsolidation MH 6K | D | 100 | 59.00 | 59 | N/A | $0.0173 | $0.5658 | $0.0000 | $0.0058 | $0.0099 |
| FactConsolidation SH 32K | A | 100 | 92.00 | 92 | N/A | $0.0000 | $0.4541 | $0.0000 | $0.0045 | $0.0049 |
| FactConsolidation SH 32K | B | 100 | 92.00 | 92 | N/A | $0.0959 | $0.4504 | $0.0000 | $0.0055 | $0.0059 |
| FactConsolidation SH 32K | C | 100 | 79.00 | 79 | N/A | $0.0959 | $0.4853 | $0.0000 | $0.0058 | $0.0074 |
| FactConsolidation SH 32K | D | 100 | 91.00 | 91 | N/A | $0.0959 | $0.4588 | $0.0000 | $0.0055 | $0.0061 |
| FactConsolidation MH 32K | A | 100 | 17.00 | 17 | N/A | $0.0000 | $0.6196 | $0.0000 | $0.0062 | $0.0364 |
| FactConsolidation MH 32K | B | 100 | 18.00 | 18 | N/A | $0.0951 | $0.5995 | $0.0000 | $0.0069 | $0.0386 |
| FactConsolidation MH 32K | C | 100 | 8.00 | 8 | N/A | $0.0951 | $0.6388 | $0.0000 | $0.0073 | $0.0917 |
| FactConsolidation MH 32K | D | 100 | 25.00 | 25 | N/A | $0.0951 | $0.6208 | $0.0000 | $0.0072 | $0.0286 |
| FactConsolidation SH 64K | A | 100 | 86.00 | 86 | N/A | $0.0000 | $0.4640 | $0.0000 | $0.0046 | $0.0054 |
| FactConsolidation SH 64K | B | 100 | 90.00 | 90 | N/A | $0.1699 | $0.4599 | $0.0000 | $0.0063 | $0.0070 |
| FactConsolidation SH 64K | C | 100 | 74.00 | 74 | N/A | $0.1699 | $0.4812 | $0.0000 | $0.0065 | $0.0088 |
| FactConsolidation SH 64K | D | 100 | 90.00 | 90 | N/A | $0.1699 | $0.4539 | $0.0000 | $0.0062 | $0.0069 |
| FactConsolidation MH 64K | A | 100 | 15.00 | 15 | N/A | $0.0000 | $0.6266 | $0.0000 | $0.0063 | $0.0418 |
| FactConsolidation MH 64K | B | 100 | 16.00 | 16 | N/A | $0.1749 | $0.6038 | $0.0000 | $0.0078 | $0.0487 |
| FactConsolidation MH 64K | C | 100 | 15.00 | 15 | N/A | $0.1749 | $0.6323 | $0.0000 | $0.0081 | $0.0538 |
| FactConsolidation MH 64K | D | 100 | 16.00 | 16 | N/A | $0.1749 | $0.6217 | $0.0000 | $0.0080 | $0.0498 |
| FactConsolidation SH 262K | A | 100 | 74.00 | 74 | N/A | $0.0000 | $0.4987 | $0.0000 | $0.0050 | $0.0067 |
| FactConsolidation SH 262K | B | 100 | 79.00 | 79 | N/A | $0.3546 | $0.4885 | $0.0000 | $0.0084 | $0.0107 |
| FactConsolidation SH 262K | C | 100 | 60.00 | 60 | N/A | $0.3546 | $0.5126 | $0.0000 | $0.0087 | $0.0145 |
| FactConsolidation SH 262K | D | 100 | 69.00 | 69 | N/A | $0.3546 | $0.4909 | $0.0000 | $0.0085 | $0.0123 |
| FactConsolidation MH 262K | A | 100 | 9.00 | 9 | N/A | $0.0000 | $0.6328 | $0.0000 | $0.0063 | $0.0703 |
| FactConsolidation MH 262K | B | 100 | 8.00 | 8 | N/A | $0.3498 | $0.6176 | $0.0000 | $0.0097 | $0.1209 |
| FactConsolidation MH 262K | C | 100 | 8.00 | 8 | N/A | $0.3498 | $0.6396 | $0.0000 | $0.0099 | $0.1237 |
| FactConsolidation MH 262K | D | 100 | 8.00 | 8 | N/A | $0.3498 | $0.6390 | $0.0000 | $0.0099 | $0.1236 |
| PersonaMem v2 32K | A | 100 | 45.00 | 45 | N/A | $0.0000 | $0.5899 | $0.0000 | $0.0059 | $0.0131 |
| PersonaMem v2 32K | B | 100 | 50.00 | 50 | N/A | $0.3159 | $0.5906 | $0.0000 | $0.0091 | $0.0181 |
| PersonaMem v2 32K | C | 100 | 50.00 | 50 | N/A | $0.3159 | $0.5918 | $0.0000 | $0.0091 | $0.0182 |
| PersonaMem v2 32K | D | 100 | 44.00 | 44 | N/A | $0.3159 | $0.5935 | $0.0000 | $0.0091 | $0.0207 |
| PersonaMem v2 128K | A | 100 | 41.00 | 41 | N/A | $0.0000 | $0.5906 | $0.0000 | $0.0059 | $0.0144 |
| PersonaMem v2 128K | B | 100 | 40.00 | 40 | N/A | $0.2592 | $0.5930 | $0.0000 | $0.0085 | $0.0213 |
| PersonaMem v2 128K | C | 100 | 39.00 | 39 | N/A | $0.2592 | $0.5937 | $0.0000 | $0.0085 | $0.0219 |
| PersonaMem v2 128K | D | 100 | 42.00 | 42 | N/A | $0.2592 | $0.5871 | $0.0000 | $0.0085 | $0.0201 |
| BEAM 100K | A | 100 | 39.02 | N/A | N/A | $0.0000 | $0.5334 | $0.0325 | $0.0053 | N/A |
| BEAM 100K | B | 100 | 37.97 | N/A | N/A | $0.5894 | $0.5372 | $0.0326 | $0.0113 | N/A |
| BEAM 100K | C | 100 | 37.10 | N/A | N/A | $0.5894 | $0.5361 | $0.0326 | $0.0113 | N/A |
| BEAM 100K | D | 100 | 38.10 | N/A | N/A | $0.5894 | $0.5462 | $0.0326 | $0.0114 | N/A |
| BEAM 500K | A | 100 | 26.77 | N/A | N/A | $0.0000 | $0.5462 | $0.0331 | $0.0055 | N/A |
| BEAM 500K | B | 100 | 24.71 | N/A | N/A | $0.6474 | $0.5443 | $0.0332 | $0.0119 | N/A |
| BEAM 500K | C | 100 | 24.13 | N/A | N/A | $0.6474 | $0.5468 | $0.0329 | $0.0119 | N/A |
| BEAM 500K | D | 100 | 27.39 | N/A | N/A | $0.6474 | $0.5495 | $0.0328 | $0.0120 | N/A |
| BEAM 1M | A | 100 | 26.67 | N/A | N/A | $0.0000 | $0.5403 | $0.0436 | $0.0054 | N/A |
| BEAM 1M | B | 100 | 27.57 | N/A | N/A | $0.6658 | $0.5403 | $0.0437 | $0.0121 | N/A |
| BEAM 1M | C | 100 | 28.08 | N/A | N/A | $0.6658 | $0.5429 | $0.0438 | $0.0121 | N/A |
| BEAM 1M | D | 100 | 28.98 | N/A | N/A | $0.6658 | $0.5433 | $0.0439 | $0.0121 | N/A |

## Paired changes

Score differences are points on a 0-100 scale. Wins/losses/ties count per-question score changes. Intervals are exploratory history-cluster bootstrap intervals; N/A means only one shared corpus, not zero uncertainty.

| Test | Change | N | Histories | Delta | 95% low | 95% high | Wins | Losses | Ties |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | B-A | 77 | 10 | -6.49 | -12.50 | +0.00 | 2 | 7 | 68 |
| LoCoMo | C-B | 77 | 10 | +0.00 | -9.38 | +7.89 | 4 | 4 | 69 |
| LoCoMo | D-C | 77 | 10 | +1.30 | -5.88 | +9.38 | 4 | 3 | 70 |
| LoCoMo | C-A | 77 | 10 | -6.49 | -15.38 | +1.18 | 2 | 7 | 68 |
| LoCoMo | D-A | 77 | 10 | -5.19 | -10.81 | +1.39 | 1 | 5 | 71 |
| LoCoMo adversarial | B-A | 23 | 9 | +8.70 | +0.00 | +20.00 | 2 | 0 | 21 |
| LoCoMo adversarial | C-B | 23 | 9 | +0.00 | +0.00 | +0.00 | 0 | 0 | 23 |
| LoCoMo adversarial | D-C | 23 | 9 | -8.70 | -15.38 | +0.00 | 0 | 2 | 21 |
| LoCoMo adversarial | C-A | 23 | 9 | +8.70 | +0.00 | +20.00 | 2 | 0 | 21 |
| LoCoMo adversarial | D-A | 23 | 9 | +0.00 | -12.00 | +9.68 | 1 | 1 | 21 |
| MuSiQue | B-A | 100 | 1 | -3.25 | N/A | N/A | 10 | 16 | 74 |
| MuSiQue | C-B | 100 | 1 | +0.10 | N/A | N/A | 6 | 6 | 88 |
| MuSiQue | D-C | 100 | 1 | +3.95 | N/A | N/A | 16 | 13 | 71 |
| MuSiQue | C-A | 100 | 1 | -3.15 | N/A | N/A | 11 | 18 | 71 |
| MuSiQue | D-A | 100 | 1 | +0.81 | N/A | N/A | 12 | 14 | 74 |
| 2WikiMultiHopQA | B-A | 100 | 1 | +0.87 | N/A | N/A | 6 | 8 | 86 |
| 2WikiMultiHopQA | C-B | 100 | 1 | +2.83 | N/A | N/A | 5 | 2 | 93 |
| 2WikiMultiHopQA | D-C | 100 | 1 | -5.48 | N/A | N/A | 5 | 10 | 85 |
| 2WikiMultiHopQA | C-A | 100 | 1 | +3.71 | N/A | N/A | 9 | 7 | 84 |
| 2WikiMultiHopQA | D-A | 100 | 1 | -1.78 | N/A | N/A | 5 | 11 | 84 |
| LongMemEval S | B-A | 100 | 100 | +3.00 | -4.00 | +10.00 | 8 | 5 | 87 |
| LongMemEval S | C-B | 100 | 100 | -1.00 | -8.00 | +5.00 | 5 | 6 | 89 |
| LongMemEval S | D-C | 100 | 100 | +1.00 | -4.00 | +6.00 | 4 | 3 | 93 |
| LongMemEval S | C-A | 100 | 100 | +2.00 | -5.00 | +9.00 | 7 | 5 | 88 |
| LongMemEval S | D-A | 100 | 100 | +3.00 | -4.00 | +9.00 | 7 | 4 | 89 |
| LongMemEval oracle | B-A | 100 | 100 | -2.00 | -7.00 | +3.00 | 3 | 5 | 92 |
| LongMemEval oracle | C-B | 100 | 100 | +6.00 | +2.00 | +11.00 | 6 | 0 | 94 |
| LongMemEval oracle | D-C | 100 | 100 | -4.00 | -11.00 | +2.00 | 3 | 7 | 90 |
| LongMemEval oracle | C-A | 100 | 100 | +4.00 | -2.00 | +10.00 | 7 | 3 | 90 |
| LongMemEval oracle | D-A | 100 | 100 | +0.00 | -8.00 | +8.00 | 8 | 8 | 84 |
| MultiHop-RAG | B-A | 100 | 1 | +1.00 | N/A | N/A | 7 | 6 | 87 |
| MultiHop-RAG | C-B | 100 | 1 | +2.00 | N/A | N/A | 8 | 6 | 86 |
| MultiHop-RAG | D-C | 100 | 1 | +1.00 | N/A | N/A | 5 | 4 | 91 |
| MultiHop-RAG | C-A | 100 | 1 | +3.00 | N/A | N/A | 5 | 2 | 93 |
| MultiHop-RAG | D-A | 100 | 1 | +4.00 | N/A | N/A | 7 | 3 | 90 |
| FactConsolidation SH 6K | B-A | 100 | 1 | -13.00 | N/A | N/A | 0 | 13 | 87 |
| FactConsolidation SH 6K | C-B | 100 | 1 | -3.00 | N/A | N/A | 0 | 3 | 97 |
| FactConsolidation SH 6K | D-C | 100 | 1 | +2.00 | N/A | N/A | 3 | 1 | 96 |
| FactConsolidation SH 6K | C-A | 100 | 1 | -16.00 | N/A | N/A | 0 | 16 | 84 |
| FactConsolidation SH 6K | D-A | 100 | 1 | -14.00 | N/A | N/A | 0 | 14 | 86 |
| FactConsolidation MH 6K | B-A | 100 | 1 | -3.00 | N/A | N/A | 20 | 23 | 57 |
| FactConsolidation MH 6K | C-B | 100 | 1 | -2.00 | N/A | N/A | 11 | 13 | 76 |
| FactConsolidation MH 6K | D-C | 100 | 1 | +4.00 | N/A | N/A | 13 | 9 | 78 |
| FactConsolidation MH 6K | C-A | 100 | 1 | -5.00 | N/A | N/A | 20 | 25 | 55 |
| FactConsolidation MH 6K | D-A | 100 | 1 | -1.00 | N/A | N/A | 18 | 19 | 63 |
| FactConsolidation SH 32K | B-A | 100 | 1 | +0.00 | N/A | N/A | 5 | 5 | 90 |
| FactConsolidation SH 32K | C-B | 100 | 1 | -13.00 | N/A | N/A | 2 | 15 | 83 |
| FactConsolidation SH 32K | D-C | 100 | 1 | +12.00 | N/A | N/A | 14 | 2 | 84 |
| FactConsolidation SH 32K | C-A | 100 | 1 | -13.00 | N/A | N/A | 3 | 16 | 81 |
| FactConsolidation SH 32K | D-A | 100 | 1 | -1.00 | N/A | N/A | 4 | 5 | 91 |
| FactConsolidation MH 32K | B-A | 100 | 1 | +1.00 | N/A | N/A | 8 | 7 | 85 |
| FactConsolidation MH 32K | C-B | 100 | 1 | -10.00 | N/A | N/A | 4 | 14 | 82 |
| FactConsolidation MH 32K | D-C | 100 | 1 | +17.00 | N/A | N/A | 18 | 1 | 81 |
| FactConsolidation MH 32K | C-A | 100 | 1 | -9.00 | N/A | N/A | 3 | 12 | 85 |
| FactConsolidation MH 32K | D-A | 100 | 1 | +8.00 | N/A | N/A | 17 | 9 | 74 |
| FactConsolidation SH 64K | B-A | 100 | 1 | +4.00 | N/A | N/A | 7 | 3 | 90 |
| FactConsolidation SH 64K | C-B | 100 | 1 | -16.00 | N/A | N/A | 2 | 18 | 80 |
| FactConsolidation SH 64K | D-C | 100 | 1 | +16.00 | N/A | N/A | 18 | 2 | 80 |
| FactConsolidation SH 64K | C-A | 100 | 1 | -12.00 | N/A | N/A | 5 | 17 | 78 |
| FactConsolidation SH 64K | D-A | 100 | 1 | +4.00 | N/A | N/A | 8 | 4 | 88 |
| FactConsolidation MH 64K | B-A | 100 | 1 | +1.00 | N/A | N/A | 6 | 5 | 89 |
| FactConsolidation MH 64K | C-B | 100 | 1 | -1.00 | N/A | N/A | 8 | 9 | 83 |
| FactConsolidation MH 64K | D-C | 100 | 1 | +1.00 | N/A | N/A | 9 | 8 | 83 |
| FactConsolidation MH 64K | C-A | 100 | 1 | +0.00 | N/A | N/A | 8 | 8 | 84 |
| FactConsolidation MH 64K | D-A | 100 | 1 | +1.00 | N/A | N/A | 9 | 8 | 83 |
| FactConsolidation SH 262K | B-A | 100 | 1 | +5.00 | N/A | N/A | 10 | 5 | 85 |
| FactConsolidation SH 262K | C-B | 100 | 1 | -19.00 | N/A | N/A | 6 | 25 | 69 |
| FactConsolidation SH 262K | D-C | 100 | 1 | +9.00 | N/A | N/A | 13 | 4 | 83 |
| FactConsolidation SH 262K | C-A | 100 | 1 | -14.00 | N/A | N/A | 5 | 19 | 76 |
| FactConsolidation SH 262K | D-A | 100 | 1 | -5.00 | N/A | N/A | 10 | 15 | 75 |
| FactConsolidation MH 262K | B-A | 100 | 1 | -1.00 | N/A | N/A | 2 | 3 | 95 |
| FactConsolidation MH 262K | C-B | 100 | 1 | +0.00 | N/A | N/A | 3 | 3 | 94 |
| FactConsolidation MH 262K | D-C | 100 | 1 | +0.00 | N/A | N/A | 3 | 3 | 94 |
| FactConsolidation MH 262K | C-A | 100 | 1 | -1.00 | N/A | N/A | 2 | 3 | 95 |
| FactConsolidation MH 262K | D-A | 100 | 1 | -1.00 | N/A | N/A | 3 | 4 | 93 |
| PersonaMem v2 32K | B-A | 100 | 77 | +5.00 | +1.02 | +9.38 | 5 | 0 | 95 |
| PersonaMem v2 32K | C-B | 100 | 77 | +0.00 | +0.00 | +0.00 | 0 | 0 | 100 |
| PersonaMem v2 32K | D-C | 100 | 77 | -6.00 | -14.55 | +2.04 | 4 | 10 | 86 |
| PersonaMem v2 32K | C-A | 100 | 77 | +5.00 | +1.02 | +9.38 | 5 | 0 | 95 |
| PersonaMem v2 32K | D-A | 100 | 77 | -1.00 | -10.53 | +7.55 | 8 | 9 | 83 |
| PersonaMem v2 128K | B-A | 100 | 78 | -1.00 | -5.72 | +3.16 | 2 | 3 | 95 |
| PersonaMem v2 128K | C-B | 100 | 78 | -1.00 | -3.26 | +0.00 | 0 | 1 | 99 |
| PersonaMem v2 128K | D-C | 100 | 78 | +3.00 | -4.67 | +11.22 | 8 | 5 | 87 |
| PersonaMem v2 128K | C-A | 100 | 78 | -2.00 | -7.07 | +2.11 | 2 | 4 | 94 |
| PersonaMem v2 128K | D-A | 100 | 78 | +1.00 | -5.16 | +8.17 | 6 | 5 | 89 |
| BEAM 100K | B-A | 100 | 20 | -1.05 | -6.13 | +3.48 | 3 | 6 | 91 |
| BEAM 100K | C-B | 100 | 20 | -0.88 | -6.25 | +4.83 | 5 | 5 | 90 |
| BEAM 100K | D-C | 100 | 20 | +1.00 | -4.36 | +6.03 | 16 | 12 | 72 |
| BEAM 100K | C-A | 100 | 20 | -1.93 | -5.96 | +1.74 | 2 | 6 | 92 |
| BEAM 100K | D-A | 100 | 20 | -0.92 | -5.95 | +3.38 | 13 | 12 | 75 |
| BEAM 500K | B-A | 100 | 32 | -2.05 | -5.61 | +1.63 | 6 | 8 | 86 |
| BEAM 500K | C-B | 100 | 32 | -0.59 | -4.68 | +3.38 | 4 | 9 | 87 |
| BEAM 500K | D-C | 100 | 32 | +3.27 | -1.47 | +8.18 | 17 | 3 | 80 |
| BEAM 500K | C-A | 100 | 32 | -2.64 | -6.60 | +1.07 | 3 | 10 | 87 |
| BEAM 500K | D-A | 100 | 32 | +0.62 | -4.75 | +6.02 | 13 | 9 | 78 |
| BEAM 1M | B-A | 100 | 34 | +0.90 | -2.01 | +4.12 | 4 | 5 | 91 |
| BEAM 1M | C-B | 100 | 34 | +0.51 | -2.33 | +2.99 | 7 | 3 | 90 |
| BEAM 1M | D-C | 100 | 34 | +0.90 | -2.60 | +4.91 | 11 | 8 | 81 |
| BEAM 1M | C-A | 100 | 34 | +1.41 | -1.45 | +4.51 | 8 | 4 | 88 |
| BEAM 1M | D-A | 100 | 34 | +2.31 | -1.30 | +6.16 | 12 | 6 | 82 |

## Results by question type

| Test | Question type | Metric | Full N | A full | Matched N | A | B | C | D |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | adversarial | Judge accuracy | 446 | 87.89 | 23 | 91.30 | 100.00 | 100.00 | 91.30 |
| LoCoMo | multi_hop | Judge accuracy | 282 | 34.04 | 14 | 42.86 | 28.57 | 35.71 | 42.86 |
| LoCoMo | open_domain | Judge accuracy | 96 | 20.83 | 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| LoCoMo | single_hop | Judge accuracy | 841 | 73.25 | 42 | 78.57 | 76.19 | 71.43 | 73.81 |
| LoCoMo | temporal | Judge accuracy | 321 | 66.04 | 16 | 50.00 | 37.50 | 43.75 | 37.50 |
| MuSiQue | 2hop | F1 | 518 | 54.20 | 52 | 58.22 | 58.32 | 57.49 | 63.18 |
| MuSiQue | 3hop1 | F1 | 243 | 53.35 | 24 | 58.43 | 54.73 | 59.88 | 54.61 |
| MuSiQue | 3hop2 | F1 | 73 | 29.25 | 7 | 30.71 | 27.14 | 17.14 | 32.73 |
| MuSiQue | 4hop1 | F1 | 108 | 39.65 | 11 | 49.09 | 30.91 | 21.82 | 35.45 |
| MuSiQue | 4hop2 | F1 | 27 | 14.98 | 3 | 16.67 | 33.33 | 66.67 | 33.33 |
| MuSiQue | 4hop3 | F1 | 31 | 26.02 | 3 | 22.22 | 0.00 | 0.00 | 22.22 |
| 2WikiMultiHopQA | bridge_comparison | F1 | 235 | 82.41 | 24 | 63.97 | 70.83 | 79.17 | 65.28 |
| 2WikiMultiHopQA | comparison | F1 | 244 | 96.77 | 24 | 100.00 | 100.00 | 100.00 | 100.00 |
| 2WikiMultiHopQA | compositional | F1 | 413 | 35.19 | 41 | 23.98 | 22.52 | 24.55 | 21.54 |
| 2WikiMultiHopQA | inference | F1 | 108 | 76.03 | 11 | 72.70 | 71.12 | 71.12 | 62.79 |
| LongMemEval S | knowledge-update | Judge accuracy | 78 | 89.74 | 15 | 86.67 | 93.33 | 86.67 | 100.00 |
| LongMemEval S | multi-session | Judge accuracy | 133 | 72.18 | 27 | 74.07 | 70.37 | 74.07 | 70.37 |
| LongMemEval S | single-session-assistant | Judge accuracy | 56 | 94.64 | 11 | 81.82 | 81.82 | 81.82 | 81.82 |
| LongMemEval S | single-session-preference | Judge accuracy | 30 | 23.33 | 6 | 0.00 | 33.33 | 33.33 | 0.00 |
| LongMemEval S | single-session-user | Judge accuracy | 70 | 92.86 | 14 | 85.71 | 92.86 | 92.86 | 92.86 |
| LongMemEval S | temporal-reasoning | Judge accuracy | 133 | 75.19 | 27 | 66.67 | 66.67 | 62.96 | 70.37 |
| LongMemEval oracle | knowledge-update | Judge accuracy | 78 | 92.31 | 15 | 86.67 | 100.00 | 100.00 | 93.33 |
| LongMemEval oracle | multi-session | Judge accuracy | 133 | 75.19 | 27 | 70.37 | 62.96 | 77.78 | 74.07 |
| LongMemEval oracle | single-session-assistant | Judge accuracy | 56 | 96.43 | 11 | 81.82 | 81.82 | 81.82 | 81.82 |
| LongMemEval oracle | single-session-preference | Judge accuracy | 30 | 23.33 | 6 | 0.00 | 0.00 | 0.00 | 16.67 |
| LongMemEval oracle | single-session-user | Judge accuracy | 70 | 91.43 | 14 | 92.86 | 92.86 | 100.00 | 100.00 |
| LongMemEval oracle | temporal-reasoning | Judge accuracy | 133 | 77.44 | 27 | 81.48 | 74.07 | 77.78 | 66.67 |
| MultiHop-RAG | comparison_query | Judge accuracy | 856 | 26.29 | 33 | 30.30 | 33.33 | 27.27 | 30.30 |
| MultiHop-RAG | inference_query | Judge accuracy | 816 | 75.37 | 32 | 78.13 | 84.38 | 87.50 | 90.63 |
| MultiHop-RAG | null_query | Judge accuracy | 301 | 93.36 | 12 | 100.00 | 100.00 | 100.00 | 100.00 |
| MultiHop-RAG | temporal_query | Judge accuracy | 583 | 24.87 | 23 | 26.09 | 17.39 | 30.43 | 26.09 |
| FactConsolidation SH 6K | single_hop | Substring EM | 100 | 100.00 | 100 | 100.00 | 87.00 | 84.00 | 86.00 |
| FactConsolidation MH 6K | multi_hop | Substring EM | 100 | 60.00 | 100 | 60.00 | 57.00 | 55.00 | 59.00 |
| FactConsolidation SH 32K | single_hop | Substring EM | 100 | 92.00 | 100 | 92.00 | 92.00 | 79.00 | 91.00 |
| FactConsolidation MH 32K | multi_hop | Substring EM | 100 | 17.00 | 100 | 17.00 | 18.00 | 8.00 | 25.00 |
| FactConsolidation SH 64K | single_hop | Substring EM | 100 | 86.00 | 100 | 86.00 | 90.00 | 74.00 | 90.00 |
| FactConsolidation MH 64K | multi_hop | Substring EM | 100 | 15.00 | 100 | 15.00 | 16.00 | 15.00 | 16.00 |
| FactConsolidation SH 262K | single_hop | Substring EM | 100 | 74.00 | 100 | 74.00 | 79.00 | 60.00 | 69.00 |
| FactConsolidation MH 262K | multi_hop | Substring EM | 100 | 9.00 | 100 | 9.00 | 8.00 | 8.00 | 8.00 |
| PersonaMem v2 32K | anti_stereotypical_pref | MCQ accuracy | 855 | 35.91 | 17 | 41.18 | 41.18 | 41.18 | 52.94 |
| PersonaMem v2 32K | ask_to_forget | MCQ accuracy | 1,048 | 76.15 | 21 | 71.43 | 80.95 | 80.95 | 66.67 |
| PersonaMem v2 32K | health_and_medical_conditions | MCQ accuracy | 568 | 23.24 | 11 | 36.36 | 36.36 | 36.36 | 36.36 |
| PersonaMem v2 32K | neutral_preferences | MCQ accuracy | 858 | 33.10 | 17 | 29.41 | 41.18 | 41.18 | 23.53 |
| PersonaMem v2 32K | sensitive_info | MCQ accuracy | 511 | 37.96 | 10 | 20.00 | 20.00 | 20.00 | 30.00 |
| PersonaMem v2 32K | stereotypical_pref | MCQ accuracy | 533 | 37.90 | 11 | 27.27 | 36.36 | 36.36 | 27.27 |
| PersonaMem v2 32K | therapy_background | MCQ accuracy | 627 | 52.31 | 13 | 69.23 | 69.23 | 69.23 | 53.85 |
| PersonaMem v2 128K | anti_stereotypical_pref | MCQ accuracy | 855 | 30.76 | 17 | 23.53 | 23.53 | 23.53 | 23.53 |
| PersonaMem v2 128K | ask_to_forget | MCQ accuracy | 1,048 | 76.05 | 21 | 80.95 | 80.95 | 80.95 | 80.95 |
| PersonaMem v2 128K | health_and_medical_conditions | MCQ accuracy | 568 | 19.89 | 11 | 18.18 | 18.18 | 18.18 | 27.27 |
| PersonaMem v2 128K | neutral_preferences | MCQ accuracy | 858 | 30.19 | 17 | 23.53 | 23.53 | 23.53 | 17.65 |
| PersonaMem v2 128K | sensitive_info | MCQ accuracy | 511 | 36.79 | 10 | 30.00 | 20.00 | 20.00 | 40.00 |
| PersonaMem v2 128K | stereotypical_pref | MCQ accuracy | 533 | 32.83 | 11 | 45.45 | 36.36 | 36.36 | 27.27 |
| PersonaMem v2 128K | therapy_background | MCQ accuracy | 627 | 48.96 | 13 | 46.15 | 53.85 | 46.15 | 61.54 |
| BEAM 100K | abstention | Rubric mean | 40 | 85.00 | 10 | 60.00 | 60.00 | 60.00 | 70.00 |
| BEAM 100K | contradiction_resolution | Rubric mean | 40 | 25.94 | 10 | 28.75 | 21.25 | 22.50 | 20.00 |
| BEAM 100K | event_ordering | Rubric mean | 40 | 2.77 | 10 | 0.00 | 2.00 | 2.00 | 2.50 |
| BEAM 100K | information_extraction | Rubric mean | 40 | 50.21 | 10 | 44.17 | 44.17 | 44.17 | 46.67 |
| BEAM 100K | instruction_following | Rubric mean | 40 | 28.75 | 10 | 35.00 | 45.00 | 35.00 | 32.50 |
| BEAM 100K | knowledge_update | Rubric mean | 40 | 52.50 | 10 | 90.00 | 80.00 | 85.00 | 80.00 |
| BEAM 100K | multi_session_reasoning | Rubric mean | 40 | 30.33 | 10 | 50.00 | 50.00 | 40.00 | 30.00 |
| BEAM 100K | preference_following | Rubric mean | 40 | 50.83 | 10 | 65.83 | 65.83 | 65.83 | 80.83 |
| BEAM 100K | summarization | Rubric mean | 40 | 1.43 | 10 | 1.46 | 1.46 | 1.46 | 3.50 |
| BEAM 100K | temporal_reasoning | Rubric mean | 40 | 36.25 | 10 | 15.00 | 10.00 | 15.00 | 15.00 |
| BEAM 500K | abstention | Rubric mean | 70 | 77.14 | 10 | 70.00 | 70.00 | 70.00 | 60.00 |
| BEAM 500K | contradiction_resolution | Rubric mean | 70 | 20.71 | 10 | 22.50 | 20.00 | 10.00 | 27.50 |
| BEAM 500K | event_ordering | Rubric mean | 70 | 0.84 | 10 | 0.00 | 0.00 | 0.00 | 0.00 |
| BEAM 500K | information_extraction | Rubric mean | 70 | 48.69 | 10 | 25.00 | 25.00 | 25.00 | 16.25 |
| BEAM 500K | instruction_following | Rubric mean | 70 | 23.81 | 10 | 27.50 | 17.50 | 17.50 | 12.50 |
| BEAM 500K | knowledge_update | Rubric mean | 70 | 40.24 | 10 | 30.00 | 40.00 | 40.00 | 50.00 |
| BEAM 500K | multi_session_reasoning | Rubric mean | 70 | 27.68 | 10 | 11.67 | 3.33 | 11.67 | 15.00 |
| BEAM 500K | preference_following | Rubric mean | 70 | 40.06 | 10 | 30.00 | 26.67 | 17.08 | 21.67 |
| BEAM 500K | summarization | Rubric mean | 70 | 6.18 | 10 | 1.00 | 4.63 | 0.00 | 6.00 |
| BEAM 500K | temporal_reasoning | Rubric mean | 70 | 26.43 | 10 | 50.00 | 40.00 | 50.00 | 65.00 |
| BEAM 1M | abstention | Rubric mean | 70 | 82.14 | 10 | 70.00 | 70.00 | 70.00 | 70.00 |
| BEAM 1M | contradiction_resolution | Rubric mean | 70 | 20.00 | 10 | 20.00 | 13.75 | 18.75 | 18.75 |
| BEAM 1M | event_ordering | Rubric mean | 70 | 1.12 | 10 | 0.00 | 1.11 | 0.00 | 1.00 |
| BEAM 1M | information_extraction | Rubric mean | 70 | 39.52 | 10 | 15.00 | 15.00 | 15.00 | 20.00 |
| BEAM 1M | instruction_following | Rubric mean | 70 | 25.48 | 10 | 11.67 | 31.67 | 35.83 | 32.50 |
| BEAM 1M | knowledge_update | Rubric mean | 70 | 47.86 | 10 | 55.00 | 50.00 | 40.00 | 55.00 |
| BEAM 1M | multi_session_reasoning | Rubric mean | 70 | 25.04 | 10 | 33.33 | 33.33 | 30.00 | 33.33 |
| BEAM 1M | preference_following | Rubric mean | 70 | 51.25 | 10 | 41.67 | 38.33 | 46.67 | 41.67 |
| BEAM 1M | summarization | Rubric mean | 70 | 2.01 | 10 | 0.00 | 0.00 | 2.00 | 0.00 |
| BEAM 1M | temporal_reasoning | Rubric mean | 70 | 22.50 | 10 | 20.00 | 22.50 | 22.50 | 17.50 |

## What the mechanisms actually did

Counts below are sums across prepared histories. Superseded facts are counted once per stored history, not once per question. Fallback counts are extraction pieces, not whole source units. Changed context compares B-A, C-B and D-C respectively.

| Test | Pieces | Fallback pieces | Facts | Compressed units | Superseded | Q hiding facts | B changed | C changed | D changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | 606 | 214 | 549 | 135 | 2 | 18 | 98 | 98 | 100 |
| MuSiQue | 739 | 409 | 3,111 | 330 | 1 | 76 | 99 | 57 | 97 |
| 2WikiMultiHopQA | 712 | 323 | 3,131 | 377 | 0 | 87 | 100 | 40 | 100 |
| LongMemEval S | 958 | 594 | 1,673 | 182 | 0 | 4 | 83 | 83 | 89 |
| LongMemEval oracle | 965 | 602 | 1,637 | 171 | 0 | 1 | 81 | 81 | 82 |
| MultiHop-RAG | 501 | 395 | 1,674 | 106 | 5 | 73 | 82 | 82 | 99 |
| FactConsolidation SH 6K | 26 | 0 | 416 | 26 | 67 | 100 | 100 | 100 | 100 |
| FactConsolidation MH 6K | 26 | 0 | 416 | 26 | 74 | 100 | 100 | 100 | 100 |
| FactConsolidation SH 32K | 134 | 3 | 2,095 | 131 | 385 | 100 | 100 | 100 | 99 |
| FactConsolidation MH 32K | 132 | 3 | 2,063 | 129 | 359 | 100 | 100 | 100 | 99 |
| FactConsolidation SH 64K | 239 | 4 | 3,757 | 235 | 656 | 100 | 100 | 100 | 100 |
| FactConsolidation MH 64K | 244 | 7 | 3,789 | 237 | 659 | 100 | 100 | 100 | 100 |
| FactConsolidation SH 262K | 490 | 3 | 7,792 | 487 | 799 | 100 | 100 | 100 | 99 |
| FactConsolidation MH 262K | 485 | 8 | 7,632 | 477 | 803 | 100 | 100 | 100 | 100 |
| PersonaMem v2 32K | 827 | 413 | 976 | 90 | 0 | 0 | 67 | 5 | 86 |
| PersonaMem v2 128K | 819 | 356 | 897 | 102 | 0 | 1 | 63 | 7 | 84 |
| BEAM 100K | 970 | 749 | 771 | 95 | 2 | 43 | 40 | 40 | 99 |
| BEAM 500K | 1,085 | 825 | 542 | 106 | 3 | 22 | 50 | 50 | 98 |
| BEAM 1M | 1,132 | 825 | 730 | 103 | 7 | 22 | 43 | 43 | 95 |

| Test | Topics | Communities | Entities | Typed edges | Shared fact setup |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | 35 | 500 | 554 | 3,561 | $0.0921 |
| MuSiQue | 13 | 551 | 3,176 | 14,273 | $0.4004 |
| 2WikiMultiHopQA | 13 | 427 | 2,918 | 14,204 | $0.3432 |
| LongMemEval S | 100 | 721 | 1,770 | 8,552 | $0.4606 |
| LongMemEval oracle | 100 | 707 | 1,749 | 8,374 | $0.4704 |
| MultiHop-RAG | 11 | 433 | 2,007 | 8,155 | $0.3983 |
| FactConsolidation SH 6K | 2 | 4 | 396 | 2,088 | $0.0177 |
| FactConsolidation MH 6K | 2 | 4 | 397 | 2,088 | $0.0173 |
| FactConsolidation SH 32K | 5 | 7 | 1,721 | 10,614 | $0.0959 |
| FactConsolidation MH 32K | 5 | 6 | 1,711 | 10,455 | $0.0951 |
| FactConsolidation SH 64K | 7 | 9 | 3,034 | 19,021 | $0.1699 |
| FactConsolidation MH 64K | 7 | 13 | 3,078 | 19,191 | $0.1749 |
| FactConsolidation SH 262K | 11 | 10 | 7,132 | 39,292 | $0.3546 |
| FactConsolidation MH 262K | 11 | 16 | 7,043 | 38,566 | $0.3498 |
| PersonaMem v2 32K | 91 | 764 | 1,112 | 5,628 | $0.3159 |
| PersonaMem v2 128K | 93 | 763 | 975 | 5,302 | $0.2592 |
| BEAM 100K | 45 | 569 | 758 | 4,450 | $0.5894 |
| BEAM 500K | 59 | 645 | 523 | 3,691 | $0.6474 |
| BEAM 1M | 60 | 656 | 719 | 4,477 | $0.6658 |

Temporal interval audit: end-before-start facts are invalid intervals in the stored timeline. This count is separate from supersession and includes intervals on events. C and D were measured without repairing them.

| Test | End before start |
| --- | ---: |
| LongMemEval S | 4 |
| LongMemEval oracle | 3 |
| MultiHop-RAG | 4 |
| BEAM 100K | 3 |
| BEAM 500K | 4 |
| BEAM 1M | 3 |

## Context, indexing and runtime

Corpus tokens below use the MiniLM WordPiece tokenizer. Published size labels such as 32K or 100K may use other tokenizers. Source totals sum all indexed histories, including related variants that share source material. Index minutes are saved build/load durations, not total campaign elapsed time. B/C/D reuse A indexes.

| Test | Groups | Source units | Source tokens | Mean/group | A index min | Index MB | BCD prep min | Graph sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | 10 | 5,882 | 196,490 | 19,649 | 0.53 | 8.39 | 1.02 | 0.22 |
| MuSiQue | 1 | 11,656 | 1,267,600 | 1,267,600 | 0.13 | 18.08 | 2.61 | 0.09 |
| 2WikiMultiHopQA | 1 | 6,119 | 624,234 | 624,234 | 1.89 | 10.23 | 2.31 | 0.09 |
| LongMemEval S | 500 | 246,930 | 54,701,150 | 109,402 | 233.43 | 645.70 | 8.88 | 1.46 |
| LongMemEval oracle | 500 | 10,960 | 3,029,881 | 6,060 | 11.51 | 33.34 | 5.76 | 1.34 |
| MultiHop-RAG | 1 | 3,376 | 1,423,563 | 1,423,563 | 5.71 | 13.32 | 2.60 | 0.17 |
| FactConsolidation SH 6K | 1 | 26 | 7,082 | 7,082 | 0.14 | 0.07 | 0.11 | 0.03 |
| FactConsolidation MH 6K | 1 | 26 | 7,082 | 7,082 | 0.14 | 0.07 | 0.16 | 0.02 |
| FactConsolidation SH 32K | 1 | 134 | 38,279 | 38,279 | 0.25 | 0.38 | 0.72 | 0.06 |
| FactConsolidation MH 32K | 1 | 134 | 38,279 | 38,279 | 0.31 | 0.38 | 0.64 | 0.05 |
| FactConsolidation SH 64K | 1 | 267 | 77,984 | 77,984 | 0.71 | 0.76 | 1.10 | 0.14 |
| FactConsolidation MH 64K | 1 | 267 | 77,984 | 77,984 | 0.48 | 0.76 | 1.06 | 0.08 |
| FactConsolidation SH 262K | 1 | 1,078 | 321,873 | 321,873 | 1.18 | 3.07 | 2.10 | 0.21 |
| FactConsolidation MH 262K | 1 | 1,078 | 321,873 | 321,873 | 1.15 | 3.07 | 2.11 | 0.13 |
| PersonaMem v2 32K | 200 | 46,632 | 6,718,637 | 33,593 | 33.04 | 94.46 | 6.44 | 1.21 |
| PersonaMem v2 128K | 200 | 199,038 | 23,001,920 | 115,010 | 122.65 | 355.71 | 6.39 | 1.19 |
| BEAM 100K | 20 | 5,732 | 2,845,367 | 142,268 | 11.42 | 27.56 | 3.95 | 0.71 |
| BEAM 500K | 35 | 38,058 | 19,317,471 | 551,928 | 73.51 | 187.10 | 5.96 | 0.96 |
| BEAM 1M | 35 | 74,630 | 39,717,906 | 1,134,797 | 148.98 | 387.06 | 5.62 | 1.03 |

Reader latency excludes cached answers, extraction, retrieval and judging. All token counts below are provider reader tokens except mean evidence tokens, which use MiniLM WordPiece. Hidden reasoning consumes the output allowance.

| Test | Arm | N | Input tokens | Output tokens | Mean evidence | Cached Q | Empty Q | Reader p50 sec | Reader p95 sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | A full | 1,540 | 7,379,130 | 755,790 | 3,997.04 | 12 | 0 | 3.60 | 5.71 |
| LoCoMo | A | 77 | 369,021 | 40,871 | 3,998.09 | 1 | 0 | 4.08 | 5.66 |
| LoCoMo | B | 77 | 370,482 | 39,681 | 3,998.64 | 3 | 0 | 3.79 | 5.84 |
| LoCoMo | C | 77 | 373,821 | 40,753 | 3,997.58 | 3 | 0 | 4.09 | 5.63 |
| LoCoMo | D | 77 | 377,366 | 41,491 | 3,997.82 | 0 | 0 | 4.09 | 5.90 |
| LoCoMo adversarial | A full | 446 | 2,133,233 | 272,476 | 3,996.53 | 1 | 0 | 4.44 | 5.86 |
| LoCoMo adversarial | A | 23 | 109,853 | 14,228 | 3,994.78 | 0 | 0 | 4.43 | 6.11 |
| LoCoMo adversarial | B | 23 | 110,274 | 14,557 | 3,994.74 | 0 | 0 | 4.36 | 5.86 |
| LoCoMo adversarial | C | 23 | 111,396 | 14,950 | 3,996.70 | 0 | 0 | 5.01 | 5.57 |
| LoCoMo adversarial | D | 23 | 112,152 | 14,105 | 3,995.96 | 0 | 0 | 4.27 | 5.77 |
| MuSiQue | A full | 1,000 | 4,613,245 | 638,607 | 3,998.92 | 12 | 0 | 4.58 | 5.72 |
| MuSiQue | A | 100 | 459,821 | 62,638 | 3,999.38 | 0 | 0 | 4.50 | 5.58 |
| MuSiQue | B | 100 | 462,172 | 63,806 | 3,998.49 | 1 | 0 | 4.85 | 6.02 |
| MuSiQue | C | 100 | 463,169 | 63,579 | 3,998.41 | 43 | 0 | 4.62 | 5.61 |
| MuSiQue | D | 100 | 464,025 | 63,331 | 3,998.89 | 3 | 0 | 4.69 | 5.66 |
| 2WikiMultiHopQA | A full | 1,000 | 4,651,974 | 526,324 | 3,998.94 | 1 | 3 | 4.06 | 5.48 |
| 2WikiMultiHopQA | A | 100 | 464,270 | 54,233 | 3,998.92 | 0 | 0 | 4.09 | 5.60 |
| 2WikiMultiHopQA | B | 100 | 466,861 | 55,695 | 3,999.33 | 0 | 0 | 4.37 | 5.70 |
| 2WikiMultiHopQA | C | 100 | 467,057 | 55,585 | 3,999.17 | 60 | 0 | 4.28 | 5.62 |
| 2WikiMultiHopQA | D | 100 | 469,945 | 57,384 | 3,999.20 | 0 | 0 | 4.11 | 5.36 |
| LongMemEval S | A full | 500 | 2,085,219 | 260,373 | 3,999.61 | 0 | 0 | 3.63 | 5.67 |
| LongMemEval S | A | 100 | 418,146 | 54,601 | 3,999.62 | 0 | 0 | 3.84 | 5.69 |
| LongMemEval S | B | 100 | 420,242 | 53,371 | 3,999.22 | 17 | 0 | 3.78 | 6.02 |
| LongMemEval S | C | 100 | 427,968 | 53,069 | 3,999.00 | 17 | 0 | 3.76 | 5.82 |
| LongMemEval S | D | 100 | 428,090 | 52,849 | 3,999.00 | 11 | 0 | 4.05 | 5.83 |
| LongMemEval oracle | A full | 500 | 1,878,119 | 247,642 | 3,623.85 | 2 | 0 | 3.56 | 5.82 |
| LongMemEval oracle | A | 100 | 379,490 | 50,829 | 3,641.52 | 1 | 0 | 3.73 | 5.80 |
| LongMemEval oracle | B | 100 | 376,596 | 49,906 | 3,594.62 | 19 | 0 | 3.81 | 6.01 |
| LongMemEval oracle | C | 100 | 391,339 | 50,577 | 3,656.81 | 19 | 0 | 3.86 | 5.88 |
| LongMemEval oracle | D | 100 | 391,364 | 51,338 | 3,656.81 | 18 | 0 | 4.03 | 5.97 |
| MultiHop-RAG | A full | 2,556 | 10,888,861 | 1,563,534 | 3,999.52 | 0 | 0 | 4.41 | 5.92 |
| MultiHop-RAG | A | 100 | 426,186 | 58,729 | 3,999.23 | 0 | 0 | 4.25 | 5.77 |
| MultiHop-RAG | B | 100 | 429,964 | 60,458 | 3,999.18 | 18 | 0 | 4.58 | 6.00 |
| MultiHop-RAG | C | 100 | 441,619 | 57,930 | 3,999.35 | 18 | 0 | 4.08 | 5.89 |
| MultiHop-RAG | D | 100 | 453,475 | 54,148 | 3,999.34 | 1 | 0 | 3.94 | 5.68 |
| FactConsolidation SH 6K | A full | 100 | 478,017 | 24,485 | 4,000.00 | 0 | 0 | 2.44 | 3.39 |
| FactConsolidation SH 6K | A | 100 | 478,017 | 24,485 | 4,000.00 | 0 | 0 | 2.44 | 3.39 |
| FactConsolidation SH 6K | B | 100 | 451,520 | 27,756 | 4,000.00 | 0 | 0 | 2.48 | 4.29 |
| FactConsolidation SH 6K | C | 100 | 478,039 | 27,194 | 3,999.03 | 0 | 0 | 2.49 | 4.61 |
| FactConsolidation SH 6K | D | 100 | 477,696 | 24,917 | 3,999.55 | 0 | 0 | 2.27 | 4.44 |
| FactConsolidation MH 6K | A full | 100 | 477,347 | 60,386 | 4,000.00 | 0 | 0 | 4.23 | 5.64 |
| FactConsolidation MH 6K | A | 100 | 477,347 | 60,386 | 4,000.00 | 0 | 0 | 4.23 | 5.64 |
| FactConsolidation MH 6K | B | 100 | 449,901 | 61,439 | 4,000.00 | 0 | 0 | 4.81 | 5.98 |
| FactConsolidation MH 6K | C | 100 | 477,068 | 57,921 | 3,999.16 | 0 | 0 | 4.13 | 5.78 |
| FactConsolidation MH 6K | D | 100 | 476,623 | 55,568 | 3,999.33 | 0 | 0 | 4.05 | 5.64 |
| FactConsolidation SH 32K | A full | 100 | 482,513 | 24,597 | 3,998.29 | 0 | 0 | 2.34 | 3.68 |
| FactConsolidation SH 32K | A | 100 | 482,513 | 24,597 | 3,998.29 | 0 | 0 | 2.34 | 3.68 |
| FactConsolidation SH 32K | B | 100 | 455,684 | 28,971 | 3,999.15 | 0 | 0 | 2.44 | 4.90 |
| FactConsolidation SH 32K | C | 100 | 493,382 | 30,748 | 3,999.31 | 0 | 0 | 2.35 | 4.90 |
| FactConsolidation SH 32K | D | 100 | 489,722 | 24,414 | 3,998.78 | 1 | 0 | 2.29 | 4.18 |
| FactConsolidation MH 32K | A full | 100 | 483,470 | 68,521 | 3,998.20 | 0 | 0 | 5.09 | 5.90 |
| FactConsolidation MH 32K | A | 100 | 483,470 | 68,521 | 3,998.20 | 0 | 0 | 5.09 | 5.90 |
| FactConsolidation MH 32K | B | 100 | 453,638 | 69,137 | 3,999.23 | 0 | 0 | 5.12 | 5.79 |
| FactConsolidation MH 32K | C | 100 | 490,694 | 72,203 | 3,998.92 | 0 | 0 | 4.96 | 5.75 |
| FactConsolidation MH 32K | D | 100 | 487,130 | 68,112 | 3,999.44 | 1 | 1 | 4.98 | 5.76 |
| FactConsolidation SH 64K | A full | 100 | 481,586 | 27,426 | 3,999.77 | 0 | 0 | 2.43 | 4.56 |
| FactConsolidation SH 64K | A | 100 | 481,586 | 27,426 | 3,999.77 | 0 | 0 | 2.43 | 4.56 |
| FactConsolidation SH 64K | B | 100 | 459,551 | 30,725 | 3,999.10 | 0 | 0 | 2.54 | 4.79 |
| FactConsolidation SH 64K | C | 100 | 486,174 | 31,081 | 3,999.07 | 0 | 0 | 2.27 | 4.86 |
| FactConsolidation SH 64K | D | 100 | 486,254 | 23,788 | 3,998.77 | 0 | 0 | 2.27 | 4.06 |
| FactConsolidation MH 64K | A full | 100 | 482,343 | 70,631 | 3,999.69 | 0 | 0 | 4.93 | 5.80 |
| FactConsolidation MH 64K | A | 100 | 482,343 | 70,631 | 3,999.69 | 0 | 0 | 4.93 | 5.80 |
| FactConsolidation MH 64K | B | 100 | 457,880 | 69,439 | 3,998.41 | 0 | 0 | 5.08 | 5.83 |
| FactConsolidation MH 64K | C | 100 | 484,436 | 71,728 | 3,999.00 | 0 | 0 | 4.93 | 5.89 |
| FactConsolidation MH 64K | D | 100 | 484,599 | 68,871 | 3,999.22 | 0 | 0 | 4.92 | 5.70 |
| FactConsolidation SH 262K | A full | 100 | 490,198 | 34,957 | 4,000.00 | 0 | 0 | 2.56 | 5.11 |
| FactConsolidation SH 262K | A | 100 | 490,198 | 34,957 | 4,000.00 | 0 | 0 | 2.56 | 5.11 |
| FactConsolidation SH 262K | B | 100 | 469,367 | 36,401 | 3,999.22 | 0 | 0 | 2.90 | 5.47 |
| FactConsolidation SH 262K | C | 100 | 493,475 | 38,003 | 3,999.62 | 0 | 0 | 2.60 | 4.97 |
| FactConsolidation SH 262K | D | 100 | 492,640 | 32,371 | 3,998.94 | 1 | 0 | 2.43 | 4.73 |
| FactConsolidation MH 262K | A full | 100 | 488,208 | 71,110 | 4,000.00 | 0 | 0 | 5.08 | 5.73 |
| FactConsolidation MH 262K | A | 100 | 488,208 | 71,110 | 4,000.00 | 0 | 0 | 5.08 | 5.73 |
| FactConsolidation MH 262K | B | 100 | 469,718 | 70,741 | 3,999.71 | 0 | 0 | 5.20 | 5.89 |
| FactConsolidation MH 262K | C | 100 | 493,358 | 71,896 | 3,999.31 | 0 | 1 | 4.89 | 5.70 |
| FactConsolidation MH 262K | D | 100 | 492,266 | 71,952 | 3,998.87 | 0 | 0 | 5.02 | 5.60 |
| PersonaMem v2 32K | A full | 5,000 | 22,433,679 | 3,347,598 | 3,999.13 | 0 | 0 | 4.75 | 6.05 |
| PersonaMem v2 32K | A | 100 | 449,493 | 67,397 | 3,999.38 | 0 | 0 | 4.96 | 5.91 |
| PersonaMem v2 32K | B | 100 | 450,874 | 67,319 | 3,999.08 | 33 | 0 | 4.47 | 6.35 |
| PersonaMem v2 32K | C | 100 | 450,889 | 67,640 | 3,999.08 | 95 | 0 | 4.13 | 5.07 |
| PersonaMem v2 32K | D | 100 | 451,000 | 68,068 | 3,999.39 | 14 | 0 | 4.74 | 5.92 |
| PersonaMem v2 128K | A full | 5,000 | 22,638,777 | 3,363,888 | 3,999.02 | 0 | 0 | 4.71 | 6.00 |
| PersonaMem v2 128K | A | 100 | 453,699 | 66,746 | 3,998.49 | 0 | 0 | 4.79 | 5.91 |
| PersonaMem v2 128K | B | 100 | 454,836 | 67,156 | 3,998.62 | 37 | 0 | 4.98 | 5.93 |
| PersonaMem v2 128K | C | 100 | 454,916 | 67,331 | 3,998.66 | 93 | 0 | 4.41 | 5.99 |
| PersonaMem v2 128K | D | 100 | 455,084 | 65,554 | 3,998.61 | 16 | 0 | 4.47 | 5.95 |
| BEAM 100K | A full | 400 | 1,613,564 | 254,904 | 3,999.74 | 1 | 0 | 4.55 | 6.03 |
| BEAM 100K | A | 100 | 399,640 | 62,317 | 3,999.60 | 0 | 0 | 4.39 | 5.88 |
| BEAM 100K | B | 100 | 400,280 | 63,184 | 3,999.80 | 65 | 0 | 4.56 | 6.24 |
| BEAM 100K | C | 100 | 402,832 | 62,382 | 3,999.47 | 65 | 0 | 4.30 | 5.96 |
| BEAM 100K | D | 100 | 412,751 | 63,095 | 3,999.53 | 2 | 0 | 4.63 | 5.93 |
| BEAM 500K | A full | 700 | 2,807,344 | 452,298 | 3,999.77 | 1 | 0 | 4.45 | 6.06 |
| BEAM 500K | A | 100 | 401,168 | 65,428 | 4,000.00 | 0 | 0 | 4.37 | 5.87 |
| BEAM 500K | B | 100 | 401,610 | 64,828 | 3,999.84 | 50 | 0 | 4.52 | 6.06 |
| BEAM 500K | C | 100 | 403,583 | 65,110 | 3,999.76 | 50 | 0 | 4.79 | 5.77 |
| BEAM 500K | D | 100 | 406,981 | 65,143 | 3,999.76 | 2 | 0 | 4.22 | 6.04 |
| BEAM 1M | A full | 700 | 2,775,958 | 448,983 | 3,999.74 | 1 | 0 | 4.55 | 6.09 |
| BEAM 1M | A | 100 | 393,618 | 65,362 | 3,999.58 | 1 | 0 | 4.46 | 6.07 |
| BEAM 1M | B | 100 | 394,214 | 65,238 | 3,999.50 | 57 | 0 | 4.70 | 6.27 |
| BEAM 1M | C | 100 | 396,456 | 65,491 | 3,999.53 | 57 | 0 | 4.45 | 6.22 |
| BEAM 1M | D | 100 | 402,179 | 64,456 | 3,999.53 | 6 | 0 | 4.63 | 6.25 |

## Version A retrieval diagnostics

These are evidence coverage diagnostics, not interchangeable vendor Hit@K scores. MultiHop-RAG coverage uses source/article evidence. N/A means no applicable gold evidence or metric.

| Test | Gold Q | Incomplete excluded | Recall@5 /100 | Recall@10 /100 | Recall@100 /100 | Recall in 4K /100 | All evidence /100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | 1,531 | 5 | 34.41 | 44.49 | 79.79 | 73.04 | 66.88 |
| LoCoMo adversarial | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| MuSiQue | 1,000 | 0 | 50.14 | 59.12 | 80.04 | 67.72 | 34.40 |
| 2WikiMultiHopQA | 1,000 | 0 | 67.58 | 71.80 | 80.40 | 75.48 | 48.00 |
| LongMemEval S | 479 | 0 | 64.60 | 80.45 | 97.72 | 84.38 | 75.99 |
| LongMemEval oracle | 479 | 0 | 71.70 | 86.97 | 100.00 | 88.70 | 81.84 |
| MultiHop-RAG | 2,255 | 0 | 69.53 | 83.14 | 99.41 | 70.28 | 41.55 |
| FactConsolidation SH 6K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| FactConsolidation MH 6K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| FactConsolidation SH 32K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| FactConsolidation MH 32K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| FactConsolidation SH 64K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| FactConsolidation MH 64K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| FactConsolidation SH 262K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| FactConsolidation MH 262K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| PersonaMem v2 32K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| PersonaMem v2 128K | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| BEAM 100K | 355 | 0 | 37.03 | 44.77 | 80.15 | 39.04 | 27.32 |
| BEAM 500K | 629 | 0 | 31.76 | 41.20 | 69.37 | 34.05 | 21.46 |
| BEAM 1M | 625 | 0 | 22.62 | 30.31 | 58.92 | 24.97 | 13.60 |

## Actual API accounting

Completed-request ledger: A **$114.9430**, B/C/D **$32.2851**, combined **$147.2281**. Outstanding reservations: **$0.0000**. Costs use recorded token usage and configured prices, not a reconciled cloud invoice. These campaign totals exclude earlier Part 1 work. The B/C/D total includes $0.38770325 of imported preflight requests. Cache reuse and recovered attempts mean actual spend differs from summed logical per-arm serving costs.

| Campaign | Role | Status | Model | Calls | USD | Input | Output |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A | preflight/smoke | done | gemini-2.5-flash-lite | 1 | $0.0000 | 7 | 1 |
| A | preflight/smoke | done | gemini-3.6-flash | 3 | $0.0101 | 9,528 | 797 |
| A | answer | done | gemini-3.6-flash | 20,129 | $114.1841 | 89,700,504 | 12,508,987 |
| A | judge | done | gemini-2.5-flash-lite | 13,554 | $0.7487 | 5,897,967 | 397,377 |
| B/C/D | answer | done | gemini-3.6-flash | 4,858 | $26.5129 | 22,257,204 | 2,618,674 |
| B/C/D | extract | done | gemini-2.5-flash-lite | 10,441 | $5.6084 | 5,991,740 | 12,523,041 |
| B/C/D | judge | done | gemini-2.5-flash-lite | 2,259 | $0.1638 | 1,266,652 | 92,816 |

## What is and is not established

| Topic | Finding or limit |
| --- | --- |
| Report scope | This report covers the current Version A and B/C/D campaigns. Older Part 1 variants, scores and spending are not mixed into these comparisons. |
| Evidence budget audit | MuSiQue / A full: 1 answers over 4,000 tokens, maximum 4001 |
| Completion | A: 20,142 questions across 19 active variants. B/C/D: 19/19 variants complete; 100 requested per variant per arm. BEAM 10M deferred. |
| Metric scale | Scores are on 0-100 display scale. F1 and BEAM rubric mean are not percentages of questions answered correctly. Correct counts use exact match for F1 datasets. |
| Scope | A full means the selected released benchmark variant. MuSiQue/2Wiki use HippoRAG 2 released 1,000-question samples. Only FactConsolidation B/C/D are full 100-question variants; others are screens. |
| Oracle | LongMemEval oracle supplies relevant sessions and is a diagnostic; it does not establish full-history retrieval performance. |
| Accounting | USD estimates use recorded provider token usage and fixed configured prices. Actual API spend counts each completed ledger request once, including preflights and recovered attempts. This is not an invoice reconciliation. |
| Logical cost | Per-arm serving cost = reader cost + shared fact setup allocation. Cache hits retain standalone logical cost. B/C/D share setup in the experiment; do not sum setup three times. Judges are separate evaluation overhead. |
| Quality-adjusted cost | Cost per correct = logical serving dollars / correct count. Uses exact match for MuSiQue/2Wiki, binary correctness elsewhere; undefined for BEAM. It includes setup amortized over this sample, not a proven production workload. |
| Runtime | Index minutes sum saved group build durations, including cached group loads; exclude lost interrupted work. Preparation minutes sum group extraction/graph durations. Neither is end-to-end campaign elapsed time or CPU core-hours. Cache reconciliation timestamps are not initial runtimes. |
| Latency | Reader latency quantiles exclude cached answers. They omit fact setup, retrieval, judging and production serving overhead. A retrieval latency is shown separately. |
| Local compute | Local embedding, graph compute, storage, energy, engineering and SaaS operations are not monetized. Zero ingestion LLM dollars does not mean zero total indexing cost. |
| Uncertainty | Paired 95% bootstrap intervals resample histories (2,000 draws, seed 13). Single-corpus datasets have no history-cluster interval. Results are exploratory, with no multiple-comparison correction or repeated model seeds. |
| Causal limits | B changes representation; C adds validity text, suppression and ranking replacement together; D adds all graph channels together. No B-plus-graph without C, topics-only, communities-only or relationships-only ablation has run. |
| Lazy scope | Facts are cached once per exact namespace/source/protocol identity. This is not one extraction forever for changing text or extraction versions. Warm discovery uses only source text selected by sample queries; no gold answers/options/rubrics enter extraction. |
| Validity limits | Temporal updates operate on discovered, validated facts with strict matching and ordering. No full entity resolution, exhaustive contradiction detection or production transaction-time API is established. |
| Fact validation limits | The parser checks normalized quote and object text against the source. It does not independently verify the subject/predicate interpretation, exclusivity classification or date meaning. Extracted dates are checked for ISO-shaped text, not source entailment or calendar validity. Those fields still depend on extractor accuracy. |
| Observed temporal defect | 21 stored facts have valid_to earlier than valid_from. Missing starts default to the source message date, and the code has no interval-order guard. C and D results retain this defect as run; no post-hoc correction or paid rerun is included. Its contribution to score changes has not been isolated. |
| Failures | Empty reader answers score zero. Invalid extraction JSON or grounding uses raw source fallback. BEAM invalid JSON escapes were repaired deterministically on cached text. No quality retries were used to improve answers. |
| Judge JSON recovery | 1 malformed BEAM reason strings were recovered using a separately recorded parser overlay. Only a sole leading numeric score in {0,0.5,1} is accepted; ambiguous/duplicate scores are rejected. Cached raw text and token usage are preserved. Recovery did not ask the model to judge again. |
| Retrieval metric | Evidence recall and all-evidence coverage are diagnostics. MultiHop-RAG uses source/article coverage, not the paper's fact Hit@K. Missing gold evidence is disclosed. Do not compare unlike Hit@K definitions. |
| External comparisons | This campaign compares our A/B/C/D. It does not rerun Chandan, Zep, Mem0 or other providers under the same models and costs; it does not establish industry-best quality or a total-cost advantage over those systems. |
| Remaining work | No BEAM 10M, full B/C/D expansion on larger variants, PersonaMem open-ended scoring, cold-start/online replay, production load test, independent judge audit or complete external-provider controlled comparison. |
| Validation | Saved validation: 429 passed / 1 skipped full suite; subsequent BCD targeted check 19 passed. These overlap and must not be added. Report values are independently recomputed from saved answers. |
| Lineage | A: metrics.json and answers.jsonl; B/C/D: sample.json, answers_A/B/C/D.jsonl, contexts.jsonl, groups/*.json, run.json; API accounting: calls.sqlite. Report audit JSON records exact source hashes. |
| Reproducibility scope | New A/B/C/D implementation is identified by saved working-tree code hashes. The report recomputes from local saved artifacts. A fresh clone rerun, all publication gates and public release have not been completed. |

## Dataset provenance and protocol differences

| Test | Variant | Source | Revision | Protocol differences |
| --- | --- | --- | --- | --- |
| LoCoMo | locomo10_all_categories | https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json | 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 | Version A retrieves raw turns and supplied BLIP captions, never generated observations or summaries. / Primary judge accuracy uses categories 1-4 (1540 questions); 446 category-5 abstention questions are separate. / Original LoCoMo reports category-specific stemmed F1; judge accuracy is a distinct metric. / Compound evidence labels are split when unambiguous; unresolvable source labels are recorded, not repaired. / Question-level retrieval recall with incomplete source evidence must be reported separately. |
| MuSiQue | HippoRAG 2 released 1,000-question sample; shared full sampled corpus | https://huggingface.co/datasets/osunlp/HippoRAG_2 | 5ec05b38deecc3318bb432c69865959c56058990 | source_evaluation: https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/main.py / corpus_unit: Whole title + newline + paragraph, unchanged from release / retrieval: Mean gold-passage Recall@5 and all-evidence coverage, answerable questions only / answering: Maximum normalized token F1 and exact match over answer and answer_aliases / reference_answer_context: Top five whole passages in the HippoRAG 2 evaluation / leakage_control: No supporting flags, answers, decompositions, or query-specific contexts in corpus; one global search pool / scoring_note: Our token budget, encoder, reader and judge must be disclosed separately from the released dataset |
| 2WikiMultiHopQA | HippoRAG 2 released 1,000-question sample; shared full sampled corpus | https://huggingface.co/datasets/osunlp/HippoRAG_2 | 5ec05b38deecc3318bb432c69865959c56058990 | source_evaluation: https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/main.py / corpus_unit: Whole title + newline + paragraph, unchanged from release / retrieval: Mean gold-passage Recall@5 and all-evidence coverage, answerable questions only / answering: Maximum normalized token F1 and exact match over answer and answer_aliases / reference_answer_context: Top five whole passages in the HippoRAG 2 evaluation / leakage_control: No supporting flags, answers, decompositions, or query-specific contexts in corpus; one global search pool / scoring_note: Our token budget, encoder, reader and judge must be disclosed separately from the released dataset |
| LongMemEval S | s | https://huggingface.co/datasets/xiaowu0162/longmemeval | See source manifest | Repeated session IDs retain each dated occurrence with a distinct unit ID; source IDs and positions are preserved. |
| LongMemEval oracle | oracle | https://huggingface.co/datasets/xiaowu0162/longmemeval | See source manifest | Oracle is derived by retaining marked evidence sessions only. |
| MultiHop-RAG | Full train split, including null questions | https://huggingface.co/datasets/yixuantt/MultiHopRAG | 71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82 | corpus_unit: Part 1 paragraph-aware 500-MiniLM-token chunks with no overlap; overlong sentences remain whole / parent_id: Original article ID; question evidence_ids refer to parents, so fold chunk ranking to distinct parents before scoring / retrieval: Article-coverage proxy: distinct gold-article recall; 2,255 nonnull questions. Nulls excluded from retrieval denominator / answering: Answer accuracy including 301 null questions; published paper top-six chunks differs from a fixed token budget / leakage_control: Evidence facts exist only in raw evaluation data; normalized corpus contains article text only / comparison_note: Retrieving one chunk does not establish that the supporting fact was returned. Parent recall and budget coverage are article proxies, not the paper's evidence-fact Hits@K |
| FactConsolidation SH 6K | factconsolidation_sh_6k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| FactConsolidation MH 6K | factconsolidation_mh_6k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| FactConsolidation SH 32K | factconsolidation_sh_32k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| FactConsolidation MH 32K | factconsolidation_mh_32k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| FactConsolidation SH 64K | factconsolidation_sh_64k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| FactConsolidation MH 64K | factconsolidation_mh_64k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| FactConsolidation SH 262K | factconsolidation_sh_262k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| FactConsolidation MH 262K | factconsolidation_mh_262k | [{"url": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench", "revision": "7ea066982b140a19337e17e60d45d4076e042faf", "file": "Conflict_Resolution-00000-of-00001.parquet", "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"}] | See source manifest | Full author-provided variant, 100 questions; no question subsampling. / Corpus contains only original context, including superseded and counterfactual facts in order. / 200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair. / No gold evidence IDs supplied. Answer containment is not evidence recall. / Official accuracy is maximum normalized substring match across reference answers. / Scoring source: HUST-AI-HYZ/MemoryAgentBench@fe1735de8cf8b9908e1e3d3b5612afc815698062, utils/eval_other_utils.py. |
| PersonaMem v2 32K | text_benchmark_32k_mcq | https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/ed956dea41521fc4499acbc63f966e0fd3c053ba/ | ed956dea41521fc4499acbc63f966e0fd3c053ba | All 5000 text benchmark questions; train, validation and multimodal variants excluded. / Each of 200 personas has one searchable history, reused for its questions. / Provided system-persona source message is retained, matching official inference.py history loading. / CSV preferences, answers and related snippets are evaluation metadata only, never corpus text. / MCQ order uses a stable SHA256-based seed; upstream uses a process-dependent Python hash. / Related snippets are preserved for open-ended judges but not converted into retrieval gold labels. / Open-ended answers are not measured by this MCQ variant. |
| PersonaMem v2 128K | text_benchmark_128k_mcq | https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/ed956dea41521fc4499acbc63f966e0fd3c053ba/ | ed956dea41521fc4499acbc63f966e0fd3c053ba | All 5000 text benchmark questions; train, validation and multimodal variants excluded. / Each of 200 personas has one searchable history, reused for its questions. / Provided system-persona source message is retained, matching official inference.py history loading. / CSV preferences, answers and related snippets are evaluation metadata only, never corpus text. / MCQ order uses a stable SHA256-based seed; upstream uses a process-dependent Python hash. / Related snippets are preserved for open-ended judges but not converted into retrieval gold labels. / Open-ended answers are not measured by this MCQ variant. |
| BEAM 100K | beam_100k | [{"url": "https://huggingface.co/datasets/Mohammadta/BEAM", "revision": "3205395e897e7318c7b094ef4e6047b9b82dbb03", "file": "100K-00000-of-00001.parquet", "sha256": "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace"}] | See source manifest | Full author-provided split; corpus contains only chat messages, never plans, profiles, questions or rubrics. / Each conversation is isolated. Messages retained intact in source order; time anchors carried forward. / The HF split is named 100K; some paper descriptions call it 128K. Do not silently relabel. / Reference responses can disagree with rubrics. Official rubric, not lexical reference overlap, controls judgement. / Official score averages independent {0,0.5,1} rubric-item judgements per question. No universal binary threshold. / Event ordering also reports Kendall tau-b times F1 after LLM alignment; keep separate from rubric mean. / Gold source IDs are mapped to messages for retrieval diagnostics; missing references are counted, never guessed. / Scoring source: mohammadtavakoli78/BEAM@b2da22eac88bb0874c64665f13457eb99835774a, src/evaluation/compute_metrics.py. |
| BEAM 500K | beam_500k | [{"url": "https://huggingface.co/datasets/Mohammadta/BEAM", "revision": "3205395e897e7318c7b094ef4e6047b9b82dbb03", "file": "500K-00000-of-00001.parquet", "sha256": "af05921c979355038e1761b7cde3d2dd713200dd3071b278de0200f6c7f30122"}] | See source manifest | Full author-provided split; corpus contains only chat messages, never plans, profiles, questions or rubrics. / Each conversation is isolated. Messages retained intact in source order; time anchors carried forward. / The HF split is named 100K; some paper descriptions call it 128K. Do not silently relabel. / Reference responses can disagree with rubrics. Official rubric, not lexical reference overlap, controls judgement. / Official score averages independent {0,0.5,1} rubric-item judgements per question. No universal binary threshold. / Event ordering also reports Kendall tau-b times F1 after LLM alignment; keep separate from rubric mean. / Gold source IDs are mapped to messages for retrieval diagnostics; missing references are counted, never guessed. / Scoring source: mohammadtavakoli78/BEAM@b2da22eac88bb0874c64665f13457eb99835774a, src/evaluation/compute_metrics.py. |
| BEAM 1M | beam_1m | [{"url": "https://huggingface.co/datasets/Mohammadta/BEAM", "revision": "3205395e897e7318c7b094ef4e6047b9b82dbb03", "file": "1M-00000-of-00001.parquet", "sha256": "41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533"}] | See source manifest | Full author-provided split; corpus contains only chat messages, never plans, profiles, questions or rubrics. / Each conversation is isolated. Messages retained intact in source order; time anchors carried forward. / The HF split is named 100K; some paper descriptions call it 128K. Do not silently relabel. / Reference responses can disagree with rubrics. Official rubric, not lexical reference overlap, controls judgement. / Official score averages independent {0,0.5,1} rubric-item judgements per question. No universal binary threshold. / Event ordering also reports Kendall tau-b times F1 after LLM alignment; keep separate from rubric mean. / Gold source IDs are mapped to messages for retrieval diagnostics; missing references are counted, never guessed. / Scoring source: mohammadtavakoli78/BEAM@b2da22eac88bb0874c64665f13457eb99835774a, src/evaluation/compute_metrics.py. |

## Recompute

This report is generated by `scripts/build_full_results_report.py` from saved answers, metrics, preparation artifacts and read-only API ledgers. Companion CSV tables and `AUDIT.json` contain unrounded numbers and source hashes. No model calls are needed to regenerate it.

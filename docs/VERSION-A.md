# Version A benchmark run

Authorized 2026-09-07: start the cheap baseline on every selected public dataset.

| Choice | Frozen setting |
|---|---|
| Ingestion | Raw source text only; no generative calls |
| Retrieval | BM25 + MiniLM local dense; equal reciprocal-rank fusion, k=60 |
| Encoder coverage | 200-token windows, 30 overlap, max score per source unit |
| Candidate depth | 100 per channel; ignore zero-score lexical matches |
| Evidence budget | 4,000 MiniLM WordPiece tokens, including labels/separators |
| Answer model | Existing Vertex proxy, gemini-3.6-flash, temperature 0 |
| Judge | gemini-2.5-flash-lite; benchmark rubric/prompts where available |
| Provider restriction | Version A benchmark API calls use Google Vertex AI only, project scout7ai; embeddings and storage are local. No other cloud provider fallback. |
| Non-LLM scoring | Official substring match for FactConsolidation, MCQ for PersonaMem, EM/F1 for HippoRAG samples |
| Reuse | Hash-keyed embedding and answer caches; identical reruns reuse results |
| Seed | 13 for retrieval; Persona options deterministic seed 42 |
| Runtime parallelism | Started at 4 QA workers; resumed at 16 after measured API waiting. Indexing increased from 1 to 2 local processes, each capped at 4 CPU threads. Prompts, data, models and token caps unchanged; latency measurements span these concurrency settings. |
| New campaign budget | USD 150 maximum; USD 40 maximum per full dataset variant |

The larger caps replace the repository's USD 10 default for this authorized
multi-dataset execution: each PersonaMem variant has 5,000 questions and BEAM
requires multiple rubric judgments per question. The USD 500 programme cap
still applies. Before launch the existing proxy ledger recorded USD 102.01;
Part 1's handover records approximately USD 25 additional bench spending.
The 20-variant campaign has 20,342 questions. Metered two-question preflight
cost USD 0.0098 in answers; a USD 150 cap allows completion plus rubric judges
without silently changing the reader. This is a ceiling, not a spending target.
The new campaign has a persistent pre-request reservation ledger, including
unresolved requests, so restarting a script does not reset the cap.

## Scope

Scope revision, 2026-09-08: BEAM 10M is deferred at the user's request until
the other datasets succeed. The active campaign is 19 variants and 20,142
questions. Its completed index caches are retained, and it will not resume
automatically. `results/version_a/SCOPE.json` records the active dataset list.
The original 20-variant protocol and dataset inventory below remain available.

| Family | Variants |
|---|---|
| LongMemEval | S, Oracle diagnostic |
| LoCoMo | 1,540 categories 1-4; 446 adversarial questions separately |
| MemoryAgentBench | FactConsolidation SH/MH at 6K,32K,64K,262K |
| Document multi-hop | Official HippoRAG 2 MuSiQue and 2Wiki samples |
| BEAM | 100K,500K,1M,10M |
| PersonaMem-v2 | 32K and128K MCQ; open-ended evaluation metadata preserved |
| Regression | Full MultiHop-RAG |

This is a new frozen cross-dataset baseline. It does not silently reuse Part 1
scores: the old speaker-specific filtering and rendering differed. Reader and
judge differ from published vendor runs. Published scores remain external
reference rows, not controlled head-to-head claims. Local hardware costs are
not priced as free; report CPU seconds and stored bytes beside API dollars.

## Reproduce

    uv run python -m multicard.version_a.conversation_data all
    uv run python -m multicard.version_a.document_data
    uv run python -m multicard.version_a.scale_data
    uv run python -m multicard.version_a.lme_data
    uv run python -m multicard.version_a.runner --datasets all --stage retrieve
    uv run python -m multicard.version_a.runner --datasets all --stage qa
    uv run python -m multicard.version_a.runner --stage report

Raw downloads and processed corpus files are gitignored. Each dataset manifest
records source URL, pinned revision/checksum and protocol differences. Run
identity includes hashes of corpus/questions and the full configuration.
Use a different output directory when changing the reader, data or limit.
Smoke runs must use a separate output directory and never count as full runs.

## Reporting

Generated SUMMARY.md and SUMMARY.csv contain the status table. Per-question
answers, usage and scores support independent recomputation. Serving cost
excludes judging, and incremental spend is separate from equivalent uncached
workload cost. F1 is not binary accuracy. BEAM rubric scores have no invented
pass threshold and no cost-per-correct figure. Unresolved source evidence is
excluded from retrieval recall and disclosed. Incomplete runs keep their full
denominator and are not reported as finished benchmark results.

Protocol amendment, 2026-09-07: the first empty reader response on 2Wiki
revealed that the runner stopped instead of scoring the failure. Empty answers
now receive zero score, with their original billed cost retained and no retry.
BEAM keeps binary correctness unset. Nonempty answers, prompts, token limits,
models and retrieval are unchanged. The original runner source and each prior
run identity are archived beside the amendment record; resumed answers reuse
the original call cache. Summary files count empty responses explicitly.

BEAM transport recovery, 2026-09-08: `recover_beam` records an execution overlay
in each affected dataset's `recovery_protocol.json`. It waits for identical
in-flight calls to finish instead of submitting duplicates, and makes illegal
JSON backslash escapes literal before parsing a judge response. Numeric score
validation remains strict. Original cache text and cost entries are unchanged;
each repaired response hash and score are logged in `json_escape_repairs.jsonl`.
This overlay is additional to the original run identity. The recovery controller
attempts each failed BEAM variant once and regenerates the campaign status only
after every dataset's metrics are complete. It preserves the USD 150/40 caps.

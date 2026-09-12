# B, C and D hypothesis screen

Declared 2026-09-08 before the B/C/D screening campaign. User authorized implementing
and starting all three variants on all active datasets. BEAM 10M stays deferred.

| Comparison | Hypothesis | Implementation |
|---|---|---|
| B versus A | Lazy fact compression improves useful evidence per token | Extract facts from detected source pieces once; cache across queries and arms; retain raw text when extraction fails |
| C versus B | Explicit temporal validity reduces stale answers | Close older exclusive-state facts on later conflicting facts, preserve history and provenance, expose validity intervals |
| D versus C | Topic, community and relation links improve evidence ranking | Separate topic/community indexes plus a typed fact/entity/source graph, weighted retrieval fusion |

This is our implementation of those ideas, not a reproduction of another
provider's entire system. LazyGraphRAG defers expensive analysis to query time;
it does not establish that every query-independent fact is extracted once.
Our content-addressed fact cache adds that explicit contract. Sources:
[Microsoft LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
and [Chandan Rajah's post-graph-rag](https://arxiv.org/abs/2608.24921).

## Fixed experimental controls

- Up to 100 proportionally stratified questions per dataset, seed 13. All eight
  100-question FactConsolidation variants are full runs; larger datasets are
  screening samples, never presented as full published benchmark results.
- Same question IDs for every arm. A uses its saved answers. B/C/D use the same
  reader, judge, prompts, 768-token answer allowance and 4,000-token evidence
  budget as A. Empty answers score zero. BEAM keeps its graded metric.
- B discovers at most 8 source units among the first 30 A candidates per query,
  with at most 12 fixed 500-token pieces. Units longer than 1,000 tokens retain
  raw text. Extraction uses Vertex Gemini 2.5 Flash Lite, at most 16 facts per
  piece and a 2,048-token output allowance. This is bounded compression, not
  exhaustive fact recall. Bad JSON or unsupported evidence falls back to raw.
- Extractors receive source text, speaker and time, never evaluation questions,
  options, reference answers, evidence labels or rubrics. Each fact must quote
  its source. Cache identity includes namespace, exact source piece and the
  extraction protocol; identical reruns do not make a second model request.
- Discovery is materialized for the fixed sample before answering. All arms
  share the same discovered facts. D is a graph of the discovered evidence,
  not an eagerly extracted graph of the entire corpus. This measures a warm
  shared memory; extraction setup cost and time are reported separately.
- C invalidates only conflicting exclusive states with matching entity,
  predicate and scope and a strict date/serial ordering. Events and multi-valued
  facts are retained. Historical queries keep superseded facts and their
  intervals; source order alone does not prove temporal precedence outside
  the benchmark that explicitly declares numbered facts newer.
- D discovers local TF-IDF topics and graph communities without model calls.
  Graph edges retain their types and source IDs. Fusion weights are fixed:
  base 1.0, topics 0.2, communities 0.2, relationships 0.4; RRF constant 60.
  No weights are tuned on these evaluation answers.
- Entity resolution is limited to normalized names and the history's user
  pronoun. Predicate normalization uses a small fixed synonym map. Unknown
  aliases and ambiguous dates remain unresolved rather than guessed. Both
  effective/source dates and extraction-recording times are retained, but this
  screen evaluates historical validity, not a production transaction-time API.
- Report paired score deltas and bootstrap intervals by dataset/cohort, plus
  actual and logical serving dollars, extraction dollars, judging overhead,
  source coverage and intervention counts. No mixed-metric overall average.
  Screening intervals are exploratory and do not prove industry superiority.

## Budget and execution

The screen has its own shared persistent meter, USD 60 total and USD 10 per
dataset/arm job. The cap includes extraction, answers and judging. This replaces
the default USD 10 run cap with a justified multi-dataset configuration. With
the reserved Version A ceiling of USD 150 and approximately USD 127 of prior
programme work, the programme stays below its USD 500 hard ceiling even if
both active campaign caps are fully spent. No other cloud provider is used.

Preparation reuses Version A rankings and original corpora. It does not rebuild
the expensive dense indexes. Runtime fingerprints, samples, fact caches,
graph artifacts, contexts, answers and paired CSV/Markdown tables are saved.
Full expansion is a separate decision after these screening measurements.

Separate preflights exercised numbered facts, LoCoMo and BEAM. Their completed
API cache entries seed the screen's meter, so identical calls are reused and
preflight spending counts inside the USD 60 cap. Their scores are not included
in the screening table. Parser, predicate-normalization and reporting fixes
were validated before the screen's code fingerprint was frozen.

Run or resume:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/python -m multicard.version_abcd.runner

The entrypoint reads the active Version A scope and rejects BEAM 10M. Results
are generated in `results/version_abcd/COMPARISON.md` and companion CSV files.

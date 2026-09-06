# Bench stage: dependencies and the MultiHop-RAG loader

Date: 2026-09-06. Repo: /Users/muralisid/github_other/multicard-bench. Nothing committed. No servers started by this stage. No LLM calls, spend USD 0.

## 1. Packages added (uv add)

Recorded in pyproject.toml and uv.lock. Installed into the repo venv (.venv, Python 3.11.16, arm64).

| Package | Version | Note |
|---|---|---|
| spacy | 3.8.16 | pulls thinc 8.3.13, blis 1.3.3, typer, rich and 20 other transitive packages |
| python-igraph | 1.0.0 | imports as `igraph` |
| leidenalg | 0.12.0 | binary arm64 wheel from PyPI, no build step was needed |
| en-core-web-sm | 3.8.0 | spaCy model, added as a URL dependency so `uv sync` keeps it |

The model source is pinned in pyproject under `[tool.uv.sources]`:
`https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl`

leiden_lib = leidenalg (the first choice worked; graspologic and networkx were not needed).

Verification run with `uv run python`:

- `import spacy, igraph, leidenalg` succeeded.
- spaCy on "The quick brown fox jumped over the lazy dog near the old river bank." gave noun chunks: `The quick brown fox`, `the lazy dog`, `the old river bank`.
- Leiden (ModularityVertexPartition, seed 13) on a 10-node graph made of two 5-node cliques joined by one edge returned 2 communities, membership `[0,0,0,0,0,1,1,1,1,1]`.

## 2. MultiHop-RAG loader

New file: /Users/muralisid/github_other/multicard-bench/src/multicard/data/multihoprag.py

Dataset: Hugging Face `yixuantt/MultiHopRAG`, revision pinned to `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`. Loaded with `datasets` 5.0.1.

Exact config and split names, discovered with `get_dataset_config_names` and `get_dataset_split_names`:

| Config name | Split | Source file in the repo | Rows |
|---|---|---|---|
| `MultiHopRAG` | `train` | MultiHopRAG.json | 2556 queries |
| `corpus` | `train` | corpus.json | 609 documents |

Features. corpus: title, author, source, published_at, category, url, body. MultiHopRAG: query, answer, question_type, evidence_list (a list of title, author, url, source, category, published_at, fact).

Raw cache under data/raw/multihoprag/ (gitignored):

- `corpus.jsonl` (609 lines, 6.6 MB) and `queries.jsonl` (2556 lines, 4.1 MB). The loader writes these on first use and reads only these afterwards, so later runs are offline.
- `hf_cache/` (9.5 MB): the datasets arrow cache, placed there by `cache_dir`.
- The raw JSON blobs downloaded from the Hub (11 MB) went to `~/.cache/huggingface/hub/datasets--yixuantt--MultiHopRAG/`. In datasets 5.x `cache_dir` only moves the arrow cache; Hub downloads use the default Hub cache. The JSONL files above are the copy the bench uses.
- `published_at` is `2023-11-27T08:45:59+00:00` in the Hub JSON. datasets parses it to a naive UTC datetime and the loader writes it back as ISO text without the offset, so the JSONL and the Document field read `2023-11-27T08:45:59`. All times are UTC.

API:

- `load_corpus(raw=RAW) -> list[Document]` with fields doc_id, title, source, category, published_at, body, url, author, and a `text` property (title plus body).
- `load_queries(raw=RAW, docs=None) -> list[Query]` with fields qid, query, answer, question_type, evidence (list of Evidence: title, fact, url, source, category, published_at, doc_id, resolved_by), and an `evidence_doc_ids` property giving the distinct resolved documents in first-mention order.
- `stratified(queries, n, seed=13) -> list[Query]`: shuffles each question type with `multicard.utils.seeds.rng(seed)`, then takes from the types round robin in sorted name order until n. Filter the input first to leave a type out.
- `summary(docs, queries) -> dict` with all the counts below.
- `python -m multicard.data.multihoprag` prints the summary.

Ids:

- doc_id is `mhr_` plus the first 12 hex digits of sha256(title + separator + url). Titles and urls are both unique across the 609 documents, so the id is content-derived and does not depend on file order. A repeat pair (not present on this revision) gets its file position appended.
- qid is `mhr_q0000` to `mhr_q2555`, the row position on the pinned revision. Query text is unique (2556 of 2556).

Evidence resolution: exact title match first. Fallbacks, in order: exact url match, then title match after case folding and whitespace collapsing. An entry that fails all three keeps doc_id None and counts as unresolved.

## 3. Counts (recomputed by summary() on the pinned revision)

- documents: 609
- queries: 2556
- queries per type: comparison_query 856, inference_query 816, temporal_query 583, null_query 301
- document categories: sports 211, technology 172, entertainment 114, business 81, science 21, health 10
- distinct sources: 49
- article dates: 2023-09-26 to 2023-12-25
- evidence entries: 6084, all 6084 resolved by exact title, 0 by url, 0 by normalised title
- unresolved evidence: 0
- evidence documents per query (distinct articles): 0 docs 301 queries, 2 docs 1169, 3 docs 774, 4 docs 312
- evidence entries per query: 0 entries 301, 2 entries 1079, 3 entries 778, 4 entries 398
- queries that cite the same article twice with two facts: 168 (this is why the two distributions differ)
- queries without evidence: 301, all null_query, all with answer "Insufficient information."
- documents with an empty author field: 68
- longest answer: 8 words. comparison answers are mostly Yes or no, temporal answers Yes, no or Consistent.

## 4. Tests

New file: /Users/muralisid/github_other/multicard-bench/tests/test_multihoprag.py. Seven tests on a three-document, six-query fixture written to tmp_path. No network: the download path is covered by monkeypatching `datasets.load_dataset`. They cover content-derived ids, the three resolution paths and the unresolved case, distinct-document ground truth for a repeated article, summary counts, the stratified sample (balanced, seeded, seed changes the sample, n above the total returns everything), and the first-load JSONL cache with dates as text.

`uv run pytest -q`: 73 passed, 4 warnings (pre-existing sklearn FutureWarning in test_cluster.py). Ran in 3.9 s.

Em-dash, en-dash, arrow, ellipsis and curly-quote lint over the new files and pyproject.toml: clean.

## 5. Files changed in the bench (uncommitted)

- modified: pyproject.toml (four dependencies and one uv source), uv.lock
- new: src/multicard/data/multihoprag.py, tests/test_multihoprag.py
- new data (gitignored): data/raw/multihoprag/

Also present and untracked, not created by this stage: docs/PART1-DESIGN.md. Left untouched.

## 6. Servers

None started. No start or stop commands apply to this stage.

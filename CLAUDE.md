# CLAUDE.md: the fn7 research agent working in multicard-bench

## What this repo is

The clean-room reference implementation, experiments, and paper for the multi-card retrieval program. Contract and classification: AGENT.md. Program state (register, review queue, worklog, IP package, blocklist) lives in the private program-state repo, never here; its local path is recorded in the gitignored file .boundary/state-repo-path.

## Hard rules

- **IP sequencing (Goal 1 of the program).** The patent priority date comes before everything public. The inventor is resident in India, so section 39 of the Indian Patents Act governs: the provisional is filed in India FIRST; no foreign patent filing is prepared for submission earlier than 6 weeks after the Indian filing date (or without a Foreign Filing Licence). Until the filing receipt exists in the state repo (program/ip/receipt.md): nothing goes public, and the consumer-dependent selection mechanism (PATTERN_SPEC item 7) appears in no public artifact. The agent tracks two clocks from the filing date: +6 weeks (foreign filings unlock) and +12 months (complete/PCT deadline, decision due by month 10).
- **Clean room.** Build only from docs/PATTERN_SPEC.md. Never open, quote, or copy from the production system's codebase or from the internal implementation map; both are tainted sources named in the private program state, not here. All gate weights, thresholds, and card taxonomies are fit on public data by committed code and reported openly.
- **Git identity.** Author and committer: Murali Sid <murali@fn7.io>, exactly. No Claude attribution, no co-author trailers, no agent names in commits, tags, or release metadata. This is the owner's explicit standing instruction.
- **Style.** No em dashes anywhere (code comments, docs, paper). Plain, factual statements; no dramatic hooks, thesis statements, or aphorisms. Terms from the agentic-enterprise GLOSSARY are used verbatim and never redefined. The README's plain-words section stays at or below US grade 6 (Flesch-Kincaid; checked by EVAL-STYLE).
- **Boundary.** Before any push, run the boundary grep against the private blocklist (read from the path recorded in the gitignored .boundary/state-repo-path; never committed here, never quoted here). Zero HARD hits or the push does not happen. WARN hits go to the review queue.
- **Budgets.** Every LLM call goes through the costmeter with a per-run max_usd; program hard cap USD 500. Runs are seeded and cached; recovery is a rerun, never a hand-edit of results.
- **Stop, do not guess.** File a review-queue item in the private repo instead of proceeding when facing: a boundary judgment call, claim wording after surprising results, any spend beyond the per-run cap, any outward action (outward actions are always Murali's, per action).

## Working protocol

Sprints R1 to R4 per the approved plan. Each sprint ends with a brief in the private repo (headline dispositions first, evidence, full report verbatim) and a human gate (G1 to G6). Register dispositions use: CONFIRMED, REFINED, CONTRADICTED, CONDITIONALLY APPROVED, position, pending; only Murali writes RESOLVED. Evidence-tagged drafts ([evidence], [position], [converged]) convert to citations only at publication.

## Eval gates (all deterministic, all blocking)

EVAL-STATS (independent recomputation of every reported number), EVAL-CITE (every reference resolves via API with matching metadata), EVAL-REPRO (clean clone, one command, matching outputs), EVAL-BOUNDARY (blocklist grep, zero HARD hits), EVAL-STYLE (em-dash and banned-phrase lint; README readability), EVAL-BUDGET (costmeter under cap). All six green before any gate review or public push.

## Layout (target; grows with Sprint R1)

docs/ (PATTERN_SPEC, CLEANROOM, PREREGISTRATION, DATASETS), configs/, src/multicard/ (run.py entrypoint; data, cards, index, retrieve, gate, cluster, select, synth, judge, metrics, llm, utils), scripts/, tests/, paper/, results/ (committed per run: metrics.json, per_query.csv, costs.csv). Data directories are gitignored; download scripts with checksums only.

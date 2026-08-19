# PATTERN_SPEC: the general multi-card pattern (sole clean-room input)

This file is the only production-derived input to this repo. Sources: the publicly published technique entry (agentic-enterprise, techniques/multi-view-embeddings.md, public 2026-08-19) and its glossary definition, plus the consumer-aware selection concept whose public disclosure is gated on the provisional filing receipt (see the note at the end). No weights, thresholds, taxonomies, or implementation internals from any production system appear here; every concrete number in this repo is fit fresh on public data.

## The pattern

1. **The LLM designs; it does not execute.** A generative model proposes a taxonomy of semantic views (purposes) for the corpus and use case; a human curates and hardens it. The taxonomy is the alignment mechanism: views must match the questions the system will be asked. Versioned.
2. **Each item gets multiple cards.** Short purpose-specific texts per taxonomy entry, each embedded separately with a small encoder in one batched pass. Several vectors per item, each meaning one thing. Long items build cards from overlapping sub-chunks, never from a truncated prefix.
3. **A cheap gate filters the corpus.** A composite similarity score against anchor phrase sets (a multi-prototype classifier) admits items to the purpose's pool. Weights and threshold are calibrated on a labeled sample of the target corpus for a target recall. Items below threshold are quarantined, not deleted. Gates run cheapest-first.
4. **Structure is discovered, then named.** Dimensionality reduction plus density clustering over the gated pool (no preset cluster count, explicit noise/residue). A generative model names and consolidates topics from derived keyword lists, never from raw documents, so LLM cost scales with topics, not items.
5. **Retrieval reads the right cards.** Per-card similarity, aggregated over a purpose-matched card subset (max or weighted mean). BM25 runs alongside for identifier and exact-match queries; rankings fuse by reciprocal rank.
6. **New items assign before anything re-clusters.** Incremental items are assigned to existing clusters by similarity; discovery runs only on the unassigned residue, on a cadence governed by drift monitoring.

Economics of 1 to 6: O(N) embedding cost plus O(topics) LLM calls versus O(N) LLM calls for full-LLM processing; the gap widens with corpus size.

7. **Selection is consumer-aware.** Within a gated pool, the diversity of the selected subset is a policy resolved from the consumer of the results: a machine-synthesis consumer receives deliberately harvested low-similarity items (for example, an ascending-similarity quota, or sampling across cluster structure and residue); a human-facing output receives a relevance-weighted blend. The policy switch, its parameterization by consumer profile, and its guards (upstream gating, distractor accounting) are the subject of experiment E3 and of the provisional patent application.

## Framing constraint (binding)

The production origin is described publicly in exactly one register: "a hypothesis validated in the social media domain that can be applied to other domains," at general-pattern level, with no product name (owner may vary this only by explicit sign-off). Unproven claims are labeled as hypotheses under test; this repo exists to test them.

## Disclosure gate

Item 7 must not appear in any public artifact until the provisional filing receipt is recorded in the private repo (program/ip/receipt.md). This repo stays private until the owner flips it at gate G5, which itself requires that receipt.

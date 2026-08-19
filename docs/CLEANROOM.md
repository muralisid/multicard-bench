# CLEANROOM: how this repo was built

This repo is a clean-room implementation. Its only production-derived input is docs/PATTERN_SPEC.md, which restates a publicly published pattern description. The production system that motivated the pattern (referred to publicly only as "a hypothesis validated in the social media domain") was not consulted while writing this code: its source, configuration, taxonomies, weights, and thresholds were never opened during development, and an automated audit greps every public artifact (tree, history, paper source, figures) against a private blocklist of production terms before any push.

Consequences a reader can rely on:

- Every gate weight and threshold here was fit on public data by code committed here, and is reported openly.
- Every card taxonomy here was freshly derived per public corpus (LLM-proposed, human-curated), with provenance notes committed alongside.
- Every number in the paper regenerates from this repo with one command per experiment, seeded.
- No production data, no production constants, no anonymized production metrics appear anywhere in this repo or the paper; the production system's only role is having motivated the hypotheses, which this repo tests on public corpora.

This protocol is part of the research method, not decoration: it separates "what the pattern is" (published, reproducible here) from "what any one production system does" (private), so the paper's claims rest entirely on evidence a reader can rerun.

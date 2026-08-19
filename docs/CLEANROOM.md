# CLEANROOM: how this repo was built

This repo is a clean-room implementation. Its only production-derived input is docs/PATTERN_SPEC.md, a general architectural description carrying no parameters. The production systems that motivated the pattern (referred to only as prior work in the social media domain) were not consulted while writing this code: their source, configuration, taxonomies, weights, and thresholds were never opened during development of anything in this repository, and an automated audit greps every artifact (tree, history, paper source, figures) against a private blocklist of production terms before any push.

For completeness, since the point of this document is candour rather than marketing: the same author reviewed those production systems separately, in a private context, to prepare a patent specification. That review informed no line of code here. What a reader of the paper needs to trust is narrower and is verifiable: no production constant, taxonomy, or threshold appears in this repository, every parameter reported was fit on public data by code committed here, and every number regenerates from a clean clone.

Consequences a reader can rely on:

- Every gate weight and threshold here was fit on public data by code committed here, and is reported openly.
- Every card taxonomy here was freshly derived per public corpus (LLM-proposed, human-curated), with provenance notes committed alongside.
- Every number in the paper regenerates from this repo with one command per experiment, seeded.
- No production data, no production constants, no anonymized production metrics appear anywhere in this repo or the paper; the production system's only role is having motivated the hypotheses, which this repo tests on public corpora.

This protocol is part of the research method, not decoration: it separates "what the pattern is" (published, reproducible here) from "what any one production system does" (private), so the paper's claims rest entirely on evidence a reader can rerun.

# Preregistration draft v2: use-case-specific multi-view embeddings

Status: APPROVED by the owner 2026-08-20 ("Lets run all the experiments"). Frozen; this commit timestamp is the receipt. Originally drafted to Murali's direction: hit hard on LLM-based, use-case-specific multi-view embeddings derived from corpus pre-processing, and show experimentally why the two occupied rings fall short: static multi-view (Ring 1: ACL 2022, KDD 2022, MC-indexing, named vectors) because it ignores both corpus and use case at design time, and CAMI-style agentic indexing because it conditions on generic retrieval quality rather than the use case. Awaiting approval; nothing runs and the sealed SciFact result stays sealed until then.

## The thesis, stated once

Which embedding views a corpus should carry is a function of TWO inputs: what the corpus contains, and what the user is trying to find out. Systems that fix the views in advance (Ring 1) get the corpus wrong; systems that derive views from the corpus alone under a generic quality target (CAMI) get the use case wrong; instruction-conditioned encoders tilt the vector but keep the structure fixed. The claim: an LLM that reads a corpus sample PLUS the research objective, and emits the view set from both, beats all three families at matched spend, and the advantage comes specifically from the objective input.

## The pipeline, including the stage the field leaves unspecified (added at Murali's direction, 2026-08-20)

Everyone who designs from a sample faces a question nobody states: WHICH sample? CAMI evaluates candidates on samples without specifying how the sample represents the corpus; a uniform random sample of a skewed enterprise corpus mostly shows the designer the majority material, so the designed views overfit the dominant topics and minority material becomes invisible to the index. The approach under test specifies the sampler:

1. Embed every item once with the cheap single-vector encoder (already required by the gate).
2. LLM-guided topic modelling over those vectors: dimensionality reduction plus HDBSCAN, generative naming and merging of the discovered topics from keyword lists, never from raw items. The noise bucket is kept as its own stratum, since outliers are precisely the minority material at risk.
3. Topic-stratified sampling: the designer's sample is drawn across all topics and the noise stratum, not uniformly at random, so the sample carries the corpus's diversity by construction.
4. The designer sees that diverse sample, the topic names as corpus context, and the research objective, and emits the view set. Multi-card build and per-view indexes follow as before.

Note for the paper's narrative: the production system used these same components in reverse order (gate, then cards, then clustering for consumption); here clustering runs first, in service of design. The machinery and its measured economics (O(N) encoder tokens plus O(topics) generative calls) are identical.

Sampling procedure, fixed in advance and amended 2026-08-20 before any H-DYN-6 result was produced: the topic stage uses BERTopic with its own defaults rather than a reimplementation, namely UMAP to 10 components (cosine, n_neighbors 15) followed by HDBSCAN with an explicit noise label, and c-TF-IDF for topic representation. min_cluster_size is swept over {25, 10, 5} and the first value yielding between 8 and 64 topics with at least 60 percent of items clustered is taken; if none qualifies, the best attempt is kept and its diagnostics reported. The k-means fallback declared in the first draft is withdrawn: k-means assigns every point and therefore has no residue, and the residue is the mechanism under test, so falling back to it would silently test something else. Stratified draw = round robin across topics plus a residue quota proportional to noise mass, capped at 25 percent of the sample. Topic names come from c-TF-IDF keywords in a single generative call.

Amendment recorded before results per the deviation rule; the reason is that the owner directed the use of BERTopic's established techniques rather than a bespoke pipeline, which also matches the production system this work derives from.

## The conditioning ladder (the experiment's spine)

Every arm shares the same encoder, the same partitioning card builder, the same indexes, gates and scoring. Only what the designer is shown differs. That isolates conditioning as the sole variable.

| Rung | Designer sees | Represents |
|---|---|---|
| L0 pooled | nothing (single vector) | the floor |
| L1 chunks | nothing (blind windows, matched units/doc) | capacity without semantics |
| L2 generic views | neither corpus nor objective: an LLM writes one universal taxonomy for "document retrieval in general", reused across all corpora | Ring 1, fixed named views (MC-indexing-class) |
| L3 hand-written views | a person's guess at the corpus | the static expert baseline (what this programme did first, and what lost) |
| L4 corpus-only views | a corpus sample, NO objective | CAMI-style conditioning: designed for the data, blind to the use case |
| L5 use-case views | corpus sample AND the research objective | the claim |
| I instruction embedding | no view structure; the objective as the encoder's instruction (instructor-base, fallback e5-small, named before running) | the strongest off-the-shelf competitor |

Honest labelling rule: L4 is "CAMI-style conditioning at matched cost", not a CAMI reimplementation; CAMI's actual enrichments (per-chunk synthetic queries, summaries) require per-item generative calls, which breaks matched spend and is the economics this approach exists to avoid. A faithful small-scale CAMI arm and trained multi-view baselines (MVR ACL 2022, MADRAL) need training or per-item generation and are declared venue-version work in the paper's limitations, not silently skipped.

## Hypotheses and frozen decision rules

**H-DYN-1 (gate). L5 beats I.** The designed view set must beat the instruction-conditioned single embedding given the same objective text, at matched total generative spend, same encoder. Primary metric nDCG@10, paired permutation, Holm. Supported only if L5 wins significantly on at least one corpus and never loses significantly. If I wins or ties everywhere, there is no paper and no patent argument worth money; pivot to defensive publication. Runs first.

**H-DYN-2 (the SciFact conjecture, sealed run). L5 beats L3 and at least ties L1.** Designer saw corpus sample plus training-split queries only. Supported if L5 significantly beats the hand-written taxonomy and the chunk delta's 95 percent interval includes zero or is positive. Partial if it beats L3 but trails L1. The sealed result is read on approval of this hypothesis.

**H-DYN-3 (the ladder, why CAMI-style conditioning is not enough).** Under objective-specific workloads, L5 beats L4, and L4's advantage over L2 is smaller than L5's advantage over L4. In plain terms: knowing the corpus helps some, knowing the use case helps more, and the gap between them is the measured refutation of objective-blind design. Supported if L5 minus L4 is positive and significant; the L4 minus L2 comparison is reported alongside as the decomposition. This is the direct "why CAMI-style selection falls short" evidence.

**H-DYN-4 (cross-over, why static views are not enough and the objective is doing the work).** Same corpus, two objectives A and B with disjoint ground truth. T_A (designed under A) must beat T_B under A's workload AND T_B must beat T_A under B's workload, both significant. Any static system, Ring 1 included, is one fixed point and cannot pass this test even in principle; passing it is the cleanest possible demonstration that view choice must be use-case-specific. A one-sided result does not count.

**H-DYN-5 (economics bookkeeping).** Design stays under 10 generative calls and under one percent of per-item LLM processing at the same model's prices (topic naming included in the count); compile and query costs reported per rung. No novelty claimed; this is the practitioner arithmetic, measured.

**H-DYN-6 (the sampler: why the diverse-representation step matters on crazy corporate data).** Setting: a SKEWED synthetic corpus whose topic masses fall roughly as 40, 25, 15, 8, 5, 3, 2, 1, 0.6, 0.4 percent, with ground-truth queries tagged by the mass of the topic they target; minority queries are those targeting topics under 5 percent. Arms, all at rung L5 (corpus sample plus objective) so conditioning is held fixed and only the sampler varies: (a) uniform random sample; (b) topic-stratified diverse sample per the procedure above; (c) relevance-biased sample, the items most similar to the objective, as the obvious alternative a practitioner would try. Decision rule, two parts, both required: on the skewed corpus, (b) beats (a) on minority-topic queries significantly, with overall quality no worse; and on the BALANCED version of the same corpus the (b) minus (a) gap's interval includes zero. The second part is the negative control that makes the first believable: the sampler should matter exactly when the corpus is skewed, and a mechanism that helps everywhere regardless of skew would be indistinguishable from noise-mining. Reported alongside: the fraction of minority topics that appear at all in each sampler's designer prompt, which is the mechanism made visible.

## Corpora

Round one, cheap and decisive: the corrected synthetic corpus (two workloads over the same documents for H-DYN-4; objective-specific ground truth by construction) and SciFact (real prose, human relevance labels; H-DYN-1, 2, 3). Round two, venue version, only if round one supports the claims: one enterprise-flavoured corpus (Enron, two objectives, pooled labelling with human audit) and the declared heavyweight baselines.

## Fixed in advance

Seed 13; byte-identical seeded reruns; stats-verifier recomputation before any disposition; no LLM judges anywhere (constructed or human ground truth only); Holm family = all primary comparisons in this file; no exclusions after unblinding; deviations logged in the worklog before the deviating run. Designer model gemini-2.5-flash at temperature 0; one design call per (corpus, condition); the L2 generic taxonomy is designed once and frozen for all corpora.

## Agreed meanings of outcomes

- H-DYN-1, 3 and 4 all supported: the paper is "use-case-specific multi-view embeddings", with the ladder as its central figure, Ring 1 and CAMI-style conditioning as measured baselines, and the earlier negative static results as honest framing. H-DYN-6 supported on top of that upgrades the contribution from a design method to a complete pipeline for heterogeneous enterprise corpora, with the sampler as its second measured differentiator against CAMI. The narrow patent residue gets its best available evidence.
- H-DYN-1 supported but 4 fails: views help but the objective is not doing the work; the claim collapses to corpus-conditioned design, which CAMI occupies; publish as a systems note, do not patent further.
- H-DYN-1 fails: instruction embeddings subsume the idea; defensive publication of mechanisms plus the established negative results; decline further patent spend.


## Encoder-choice record, 2026-08-20, before any H-DYN-1 evaluation

hkunlp/instructor-base cannot load on the study hardware: its weight format requires torch 2.6 or later, and the machine (Intel Mac) has no torch wheel beyond 2.2.2. Per the declared fallback rule, the instruction arm uses intfloat/e5-small-v2, with the objective text prepended to the query prefix, which is that model's supported form of task conditioning. Limitation recorded now: e5-small-v2 is a weaker instruction baseline than instructor-class or Promptriever-class models, so a win for the designed views over it is necessary but not sufficient against the strongest instruction-conditioned systems; the venue version must run those on capable hardware.

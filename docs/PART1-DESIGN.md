# Part 1 design: the head-to-head on LongMemEval_S and MultiHop-RAG

Version 4, 2026-09-06. Version 1 was reviewed by a three-lens panel
(statistics, fairness to the compared systems, implementability) before any
Part 1 code or index existed; version 2 applied its twenty kept findings;
version 3 applied the four-checker pass over version 2 (undefined terms,
unlabelled outcomes, the planner prompt, the pinned files); version 4 applies
what the environment smokes measured (section 13) before any Part 1 run. The
review records are in the state repo worklog. Decision rules, arms, metrics,
tests and predictions are fixed in this file. Anything added after a number is
read is labelled as such in the report.

The environment records (docs/part1/env/: pgr.md, pgr-schema.md, graphiti.md,
proxy.md, postgres.md, bench.md, models.json, copied from part1-tools/env
without the key file) are committed with this file. The subset script
scripts/part1_subsets.py and its output docs/part1/subsets.json are committed
before any Part 1 model call, and the hashes of both commits are recorded in
the report.

Owner's brief: prove that our index puts better evidence in front of the reader
than post-graph-rag (Chandan Rajah) and Graphiti (Zep) on the same questions,
at the same budget, under the same reader and the same official judge prompts.
If it does not, Part 2 does not start.

Owner's principle: use the latest inexpensive model, inherit what industry has
already solved, and be open about it. Every inherited component is named below
with its pinned version, and none of them is tuned.

Two decisions the review left to the owner, taken here by the research agent
under stated assumptions because the owner is away, and flagged for his review:

- D1. T1 outcomes: a positive and significant T1 (after Holm within Family A)
  is "shown". A positive T1 that misses significance is "not shown", reported
  with the CI and the wins, ties and losses; the Part 2 go decision on a "not
  shown" result is the owner's, in writing, with the CI in front of him. A T1
  whose point estimate is zero or below is "failed", whatever the CI.
- D2. T2 (the Graphiti comparison) runs on at most 150 questions and is low
  powered. T2 passes when its point estimate is positive. The 95 percent CI,
  wins, ties and losses are printed beside it, and the CI is expected to cross
  zero at this n. The stricter reading, positive and significant after Holm
  within Family A, is reported beside it. If Graphiti is dropped by the pilot
  or its run is partial, T2 is recorded as not run and the pass rule uses T1,
  T8a and T7 alone.

## 1. Question

At a fixed rendered-context budget, do topic and community channels added over
post-graph-rag's own extracted tables put a more complete set of the marked
evidence in front of the reader than post-graph-rag alone does, on the same
questions? Secondary: does that turn into more correct answers under the same
reader and judge? And, on a subset, does the same index beat Graphiti?

The comparison is additive by design: our arms read post-graph-rag's
entities, relations and aliases as inherited tables. An arm with those channels
switched off (S5_noPGR) is reported so the reader can see what our own layers
do alone. His build cost is charged to every arm that reads his tables.

## 2. Corpora, fixed subsets and order

LongMemEval_S (data/raw/longmemeval_s.json). 500 questions, 470 with marked
evidence turns, 30 abstention questions (they sit inside four of the six
types). Six types. The local set for the non-inferiority control is
single-session-user plus single-session-assistant (120 questions). Every
question has its own haystack of about 50 sessions; 19,829 unique sessions
fill 25,112 haystack slots.

MultiHop-RAG (Hugging Face yixuantt/MultiHopRAG, ODC-BY). 609 news articles,
2,556 queries with an evidence list of 2 to 4 documents each, every evidence
entry carrying the supporting fact excerpt. Four types: inference, comparison,
temporal, null. Null queries (301) have no evidence: they are excluded from
retrieval metrics and scored in answering only. Retrieval metrics on
MultiHop-RAG, including T8a and T8b, run on every non-null query (2,255),
processed in ORDER. MHRAG_ANSWER is for answering only.

Fixed subsets, all drawn with the bench's seeded rng, seed 13, by the committed
script scripts/part1_subsets.py, written to docs/part1/subsets.json before any
model call, the file's sha256 recorded in the report:

- ORDER: a seeded permutation of every question id per corpus. Every run
  processes questions in this order; a subset is processed in ORDER restricted
  to the subset; post-graph-rag spaces and Graphiti groups are built and
  queried one question at a time in that order, so a capped or partial run is
  a defined prefix. post-graph-rag shards are cut on question boundaries at
  about 2,000 sessions each.
- GRAPHITI_150: 150 LongMemEval questions, 25 per question type across the
  six types, drawn from the answerable questions only. The abstention
  questions are not in this subset, so T2 runs on all 150 less any refused
  ids. The Graphiti pilot is the first question of GRAPHITI_150 in ORDER and
  its first 20 haystack sessions in session-date order.
- CHANDAN_CAL_18: 18 questions, 3 per type, drawn from GRAPHITI_150. The
  calibration row of section 13 runs post-graph-rag with his own models on
  these questions only.
- MHRAG_ANSWER: 150 MultiHop-RAG queries per type (600).
- READER_B_MHRAG: 50 per type from MHRAG_ANSWER (200).
- JUDGE_AUDIT: 10 percent of every (arm, corpus, reader) answering cell,
  stratified by type.

Not in Part 1: AP News, ECT-QA, 20 Newsgroups, 2WikiMultiHopQA.

## 3. Units, scoping and canonical evidence

LongMemEval: sessions, turns (id "session#index", as in e5), sub-units (the
e5 split: sentences of user turns, items of assistant turns). Evidence is the
marked turn set per question. Every arm's candidate set is the question's own
haystack sessions. Corpus-wide structures (topics, communities, the noun-phrase
graph) only rank units inside that set. post-graph-rag and Graphiti are given
the same scoping through one space or group per question (section 4, item 6,
and section 5).

MultiHop-RAG: documents, paragraph-aware chunks of about 500 tokens with no
overlap, sentences. Every arm's candidate set is the whole corpus. Evidence is
the document set per query; a document hit is "at least one rendered chunk
contains the located fact excerpt for that document". Fact location:
normalise both texts (NFKC, lowercase, whitespace collapsed, leading and
trailing punctuation and quotes stripped) and find the excerpt as a substring
of the document; a located fact maps to exactly one chunk because overlap is
zero. Where a fact cannot be located the fallback is any chunk of the
document, and the fallback count is reported. The located share is the share
of non-null queries whose facts were all located; it is reported, and if it is
under 85 percent the report says the gate ran on a partial set. The share of
individual facts located is reported beside it.

Coverage rule for scoring, symmetric across arms: a marked LongMemEval turn is
covered when the rendered units together contain at least the lesser of half
of the turn's text and 2,000 characters of it (so our 2,000-character cut and
his 2,000-character chunk are credited alike), or contain a relation whose
provenance chunk does. A truncated evidence turn is flagged and counted. Of the
896 marked evidence turns, 11 are over 2,000 characters and one is over 4,000;
that one is listed in the report. For Graphiti, a fact covers every session
whose episode it cites (an edge can cite several episodes); an ENTITY summary
line covers no session, and this is stated. Graphiti is scored at session level
only.

Every retrievable unit in our index points back to one of these spans. Nothing
generated (topic, community, relation, proposal) is evidence on its own.

## 4. The frozen index, built once per corpus

Inherited, run with the authors' defaults, pinned, not tuned:

1. Dense vectors: sentence-transformers/all-MiniLM-L6-v2 (the bench Encoder)
   for sub-units, turns and chunks. One encoder for every vector in our index,
   including relation and prototype vectors.
2. Keyword search: BM25 over turns or chunks (the bench BM25).
3. Topics: BERTopic (version pinned in pyproject) fed the MiniLM embeddings of
   turns (LongMemEval, both roles) or chunks (MultiHop-RAG). UMAP(n_neighbors
   15, n_components 5, min_dist 0.0, metric cosine, random_state 13), HDBSCAN
   defaults, calculate_probabilities False. Seeding is reproducibility, not
   tuning. The per-unit topic distribution is the softmax of cosine similarity
   between the unit vector and topic_embeddings_, the top 5 kept sparse.
   Outlier status kept. Topic names from the top 10 c-TF-IDF terms. No LLM in
   the topic model. The fit is a documented separate step; topics.parquet is
   committed as a frozen artifact and the repro command loads it. A topic's
   members are the sub-units of the turns or chunks BERTopic labels with that
   topic; outlier units are members of no topic.
4. Noun-phrase graph: spaCy en_core_web_sm with ner disabled, noun chunks per
   sub-unit, normalised (lowercase, lemma, articles stripped), at most 50
   phrases per sub-unit by first occurrence. Nodes are phrases; edges are
   co-occurrence inside one sub-unit (the e5 sub-unit, never a whole turn),
   counted in numpy over int32 phrase ids. Weights and pruning as in the
   Microsoft graphrag NLP indexing path defaults (min_node_freq 2,
   min_edge_weight_pct 40). Declared hub rule: pronouns and the top 0.5
   percent of phrases by sub-unit frequency are dropped. Leiden through
   leidenalg.find_partition(G, ModularityVertexPartition, weights="weight",
   seed=13), one level. communities.parquet committed as a frozen artifact.
5. Communities in two versions from the same graph: plain, and topic-weighted,
   where each edge weight is multiplied by one plus the cosine similarity of
   the two phrases' topic vectors. A phrase's topic vector is the
   probability-weighted mean of the 384-dimensional topic_embeddings_ over
   the sub-units it appears in. Gamma equals 1, fixed. A sub-unit is a member
   of the community that holds the plurality of its surviving phrases, ties
   to the lower community id; a sub-unit with no surviving phrase is in no
   community. Community prototype vector is the mean of the member sub-units'
   vectors. Largest community share, edge density and topic entropy are
   reported for both versions.
6. Chandan's tables: post-graph-rag as shipped on PyPI, run the way his
   LongMemEval harness runs it: one space per question (space equals the
   question id) holding that question's haystack sessions, one document per
   session in session-date order, the session date written into the document
   body, queried with that space, and every query variant of section 5 run and
   saved to disk before the space is dropped. That is 25,112 session
   indexings, 27 percent more than the unique count, and it is the protocol
   his paper reports (section 8.1, every instance indexed from scratch).
   MultiHop-RAG is one space, documents in publication-date order, his default
   extraction prompt. His harness indexes through index_document, which never
   chunks: on LongMemEval his unit is the whole session document (mean about
   10,000 characters, the longest about 76,000), and on MultiHop-RAG the whole
   article. Build in shards (section 2); after the first shard the total is
   projected and the build stops before the second shard if the projection
   exceeds the cap. pgr.md records: the package version and its
   sha256; mode mix; top_k at the shipped default; token budgets None; the
   extraction model and the answer model; the embedding model and its
   dimension; gleaning passes; contradiction detection setting; whether
   communities are built; max_concurrent_chunks, and whether it differs from
   his run; the LongMemEval extraction prompt as in his run.py; how the
   question date is passed to the query as run.py passes it; the
   one-space-per-question rule; the endpoint used and how spend is read from
   its usage log; and, if our one-line-per-turn session document differs from
   the one his run.py builds, how. After each space's queries, its entities,
   relations (subject, predicate, object, negation, validity dates,
   confidence, provenance chunk), aliases, relation embeddings, and every
   chunk of the space with its text and its turn span are exported to
   parquet (pyarrow). Relation provenance is not returned by his query call;
   it is read from the relations table by edge id. Entities in his query
   output carry no id; they are joined on lower-cased name within the space.
   His session document is built by us as one line per turn with
   internal newlines replaced, with a turn-span table kept, so each returned
   chunk is located as an exact substring and mapped to turns; relations join
   to their source chunk through the exported store; the unmapped share is
   reported. These tables are read-only inputs to every arm that uses
   relations or entities.
7. Relation vectors, two renderings embedded with MiniLM: the canonical triple
   text, and the triple text prefixed with the c-TF-IDF name of the top topic
   of its provenance unit. His own relation embeddings are used only by his
   live arm.
8. The bridge: candidate pairs are topic-community pairs with at least 5
   shared sub-units, the topic not the outlier topic. Disagreement d(t, c)
   equals the shared count times one minus the Jaccard overlap of the two
   member sets. The flagged set is the top 20 percent by d, capped at 5,000
   pairs on LongMemEval and 500 on MultiHop-RAG.
9. Overlay proposals: one call per flagged pair to the chat model in
   part1-tools/env/models.json, showing at most 8 member sub-units of at most
   300 tokens each plus the topic's top 10 c-TF-IDF terms, asking whether the
   topic and the community describe the same subject and with what confidence
   in [0, 1]. Each proposal is a weighted link (t, c, w equals the confidence)
   with the units it cites and the model version. Stored separately; the topic
   model and the community assignment are never rewritten. The placebo set has
   the same number of links with the same degree distribution, targets
   shuffled with the seeded rng.

Built with no question, answer or evidence flag in reach. The overlay prompt
sees unit texts and topic terms only.

## 5. Arms and rendering

Rendering rule, the same for every arm: every hit renders its owner turn or
chunk (relations and entity seeds render their provenance turn or chunk),
deduplicated by unit id; units are taken in the arm's rank order until the
budget B is full; the kept units are then displayed oldest first, ties broken
by session order then turn index; each unit is prefixed with its date and
speaker or source; turns are cut at 2,000 characters as in e5. B is 4,000
tokens (primary) and 8,000 tokens (secondary), counted by the bench
TokenCounter. Rendered tokens per arm are reported beside every recall number.
Candidate-level joint recall (before the budget cut) is reported beside the
budgeted number for every arm.

Date key for competitor units: a post-graph-rag chunk takes its session date;
a relation takes its validity start date, else its provenance chunk's session
date; a Graphiti FACT takes its valid_at, else the reference_time of the
episode it cites; ENTITY lines and undated units keep their native position
ahead of dated units. The competitors' native block order is kept in the
chandan_full and chandan_full_uncut rows.

The e5 speaker rule is applied in every arm of ours (ours_cheap, S4_static,
S5_primary and every planner and overlay ablation, S5_noPGR, S2_lazy) except
the rows labelled without it.

Ours:

- ours_cheap: rrf(bm25_turn, turn dense) with the e5 speaker rule, that is
  e5's rrf_turn_route, rerun here so the rendering matches. The floor our
  index must beat. Also reported: ours_cheap without the rule (e5's rrf_turn),
  and rrf(bm25_turn, sentence dense) without the rule (e5's rrf_sentence, the
  best cheap arm).
- S4_static: every channel fused by RRF (k 60) with no model call at query
  time: sub-unit dense, BM25, topic prototypes then member units, community
  prototypes then member units (topic-weighted communities), relation vectors
  (both renderings), entity seeds from Chandan's aliases then their provenance
  units. Then lazy expansion: the top three sessions or documents have their
  remaining units re-scored by dense similarity and appended until the budget
  is full.
- S5_primary: S4 plus the planner (LLM) plus overlay R3. Also reported:
  S5_primary without the speaker rule, and S5_noPGR (relation and entity
  channels off, everything else identical; no model call).
- Planner ablations of S5: rules, and oracle (an upper bound, never a system).
- Overlay ablations of S5 with the planner held at LLM: R0 none, R2 every
  generated proposal regardless of confidence, R3 proposals with confidence
  at or above 0.7 (the primary), P0 placebo.
- S2_lazy: the LazyGraphRAG query pattern on its own. Rank topic-weighted
  communities by where the dense hits land; from the best community take up
  to 5 sub-units ranked by dense similarity and ask the chat model, with the
  e5 date-and-speaker prefix, the question and the unit text cut at 2,000
  characters, "Does this excerpt help answer the question? Answer yes or no."
  at temperature 0, cached by (question id, unit id); at most 20 tests per
  question; stop after three communities in a row with zero accepted; fill
  the budget with the accepted units then the fused ranking.

The planner. Channel weights in {0, 1, 2, 3} over (sub-unit dense, BM25,
topic, community, relation, entity) and an expansion depth in sessions or
documents, indexed by the question shape:

| shape | dense | bm25 | topic | community | relation | entity | depth |
|---|---|---|---|---|---|---|---|
| local | 2 | 2 | 1 | 1 | 1 | 1 | 1 |
| entity | 1 | 1 | 1 | 1 | 2 | 3 | 2 |
| thematic | 1 | 1 | 3 | 2 | 1 | 1 | 2 |
| cross-topic | 1 | 1 | 2 | 3 | 1 | 1 | 3 |
| multi-hop | 1 | 1 | 1 | 1 | 3 | 2 | 3 |
| temporal | 2 | 1 | 1 | 1 | 2 | 1 | 2 |
| lexical | 1 | 3 | 0 | 0 | 0 | 0 | 1 |
| unanswerable | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

Fused score of a unit equals the sum over channels of w_c / (60 + rank_c).
S(t) and S(c) are the cosine similarity between the query vector and the topic
or community prototype. Overlay links act before fusion: for a link (t, c, w),
S'(c) equals S(c) plus w times S(t) on the community-prototype channel and
S'(t) equals S(t) plus w times S(c) on the topic-prototype channel; both
updates read the original S values, not each other's updated value; lambda
and mu equal 1 as in pov-04 section 7, not swept. P0 applies the same rule
over the shuffled targets.

The LLM planner makes one call per question at temperature 0 and returns one
label; output cached. The prompt, fixed here and committed at
part1-tools/prompts/planner.txt, is the question followed by: "Label the
question with exactly one shape from this list and reply with the label only.
local: asks for one fact the user or assistant stated in one place. entity:
asks about a named person, pet, object, place or organisation. thematic: asks
for advice, ideas, plans or recommendations that fit what is known about the
user. cross-topic: needs facts from two or more separate conversations or
documents combined, counted or compared. multi-hop: needs one fact to find a
second fact, or asks which named thing did something. temporal: asks when,
how long, how long ago, what came first or last, or what is true now about
something that changed. lexical: quotes exact words, a code, an id or a long
number. unanswerable: asks for something the conversations or documents are
unlikely to contain." No example questions are given. A reply that is not one
of the eight labels is logged and treated as local. The definitions were
written with the same e5 knowledge as the table, and section 12 says so.

The rules planner assigns the first matching shape from this ordered list of
case-insensitive patterns, else local: unanswerable is never assigned by
rules; lexical when the question contains a quoted string, a token with
letters and digits, or a number of four or more digits; temporal when it
contains "how long", "how many (days|weeks|months|years)", "ago", "since",
"before", "after", "when", "date", "first time", "last time", "most recent",
"currently", "now"; cross-topic when it contains "in total", "altogether",
"combined", "all the", "across", "both", "compare", "same", "different",
"either", "neither"; multi-hop when it contains "which (company|organization|
organisation|person|team|country)", "who", "reported by", "according to";
thematic when it contains "recommend", "suggest", "ideas", "what should i",
"advice", "plan", "tips", "help me"; entity when it contains "my (dog|cat|
wife|husband|partner|son|daughter|boss|friend|sister|brother|mother|father|
car|house)" or a capitalised token that is not sentence-initial. The oracle
planner maps the benchmark's own labels: single-session-user and
single-session-assistant to local, single-session-preference to thematic,
multi-session to cross-topic, temporal-reasoning and knowledge-update to
temporal, abstention ids to unanswerable; MultiHop-RAG inference to
multi-hop, comparison to cross-topic, temporal to temporal, null to
unanswerable.

Chandan:

- chandan_live: post-graph-rag's query call in its default mix mode with the
  configuration in pgr.md, over his own per-question space. The reader sees
  his units as he renders them (his document units, which on LongMemEval are
  whole sessions; one relation per line with its validity dates; entity
  lines) taken in his rank order until the budget is full, then displayed
  oldest first by the date key above. His rank order is the order in which
  his own context assembly presents the units to his synthesis prompt. A unit
  larger than the remaining budget is truncated to fit, flagged and counted;
  coverage of a truncated unit follows the half rule of section 3. Run twice
  per question and saved before the space is dropped: at his shipped result
  limit, and with only the result limit raised until the rendered context
  reaches B. The gate uses whichever of the two scores higher on
  JointRecall@4k over the set. The turn mapping of section 3 is for scoring
  only.
- chandan_full: his assembled context in his native block order, cut to B,
  passed to our readers with READER_RULES verbatim (the wording mismatch
  about "who spoke" is noted in the report). Answering only.
- chandan_full_uncut: his own synthesis prompt and reader model on his uncut
  context. Not budget-matched. A reference line, never in a test.

Zep:

- graphiti: graphiti-core (graphiti-core, the Gemini client package, and the
  neo4j driver and server versions pinned in part1-tools/env/graphiti.md) on
  the GRAPHITI_150 subset, group_id equal to the question id, reference_time
  equal to the session date. The ingestion unit is decided by the pilot of
  section 2 before the run, among three variants: one text episode per
  session with the turns rendered as "role: text" lines; one message-type
  episode per session; one message episode per turn (Zep's own granularity).
  The pilot ingests the same 20 sessions three ways and records, per variant,
  calls, tokens, seconds and Vertex requests per minute, and evidence
  presence (for each marked evidence turn of the pilot question, whether any
  extracted fact cites its session or contains the normalised gold answer
  string). The variant chosen is the one with the highest evidence presence
  whose projected cost over GRAPHITI_150 fits the cap; ties go to Zep's
  per-turn granularity. The choice and the pilot table are written into
  graphiti.md before the run. Any variant other than per-turn is labelled as
  a deviation from Zep's ingestion. Search through the library's
  COMBINED_HYBRID_SEARCH_RRF recipe (edges and nodes; BM25, cosine and BFS;
  RRF; no cross-encoder, which needs OpenAI logprobs), limit 20 edges and 20
  nodes, rendered with Zep's own template (FACT lines with their date range,
  ENTITY_NAME: summary lines), taken in that order until the budget is full.
  Run twice as for chandan_live. Before a group is dropped, every edge in the
  group with its fact text, dates and episode ids is exported to parquet. The
  20-session pilot (section 2) measures calls, tokens and Vertex requests per
  minute first; the owner's rule "drop it if it cannot run on Vertex" is
  decided by that pilot. Neo4j runs locally with a 2 GB heap; no other index
  build runs during ingestion. Zep's paper does not name its reranker; the
  report says so.

Ceilings and floors:

- oracle_full: the evidence sessions or documents in full, as in e5.
- closed_book: the reader with no excerpts.

Missing output: an arm that returns nothing for a question scores 0 on every
retrieval metric and "wrong" on answering, and the count is reported per arm.
A question that an arm's index refused (post-graph-rag's harness evaluated 499
of 500 for this reason) is dropped from every pair involving that arm and its
id is listed. Tests on a partial run are labelled and never feed the pass
rule. If a cap binds, Reader B is withdrawn in this order: closed_book,
oracle_full, chandan_full, ours_cheap, graphiti; the head-to-head pair
(S5_primary, chandan_live) keeps Reader B last.

## 6. Readers and judges

Reader A: the chat model in part1-tools/env/models.json (the cheapest current
flash-class Gemini that works on the project). Reader B: Azure gpt-5.4. Both
see the identical rendered context and the e5 READER_RULES verbatim
(MultiHop-RAG reads "conversation excerpts" as "news excerpts" and "who spoke"
as "which outlet"). Reader A runs on every arm. Reader B runs at B equals
4,000 only, on S5_primary, chandan_live, chandan_full, graphiti, ours_cheap,
oracle_full and closed_book, on all 500 LongMemEval questions and on
READER_B_MHRAG.

Judges: the official LongMemEval prompts (e5 judge_prompt). Reader A's model
is the candidate primary judge; gpt-5.4 is the second judge. Before any
answering, JUDGE_AUDIT is drawn (section 2). Both judges score the audit
sample; agreement on the pooled sample (about 1,400 verdicts) decides the
primary judge before any test is run: if agreement is at or above 0.90 the
cheap model is primary, else gpt-5.4 is primary and its cost comes from the
answering cap. Every reported accuracy uses the primary judge; the second
judge is a separate column, never merged. The second judge also scores every
wrong answer on answerable questions for the head-to-head arms (S5_primary,
ours_cheap, chandan_live, chandan_full, graphiti) under both readers, so
bucket 5 is decidable there. Re-judging the questions that would flip a test
is a robustness table that never changes the primary judge; if a test's
outcome differs under it, both are reported. For MultiHop-RAG the standard
prompt is used with the gold answer; null queries use the abstention prompt.

## 7. Metrics

Primary: JointRecall@B, the share of questions for which every marked evidence
unit is covered (section 3) inside the rendered context at B equals 4,000
tokens. Reported also at 8,000 and at candidate level. On MultiHop-RAG the
primary is fact-level joint recall over the non-null queries whose facts were
all located; document-level joint recall is secondary.

Secondary retrieval: turn Recall@10 and nDCG@10, session Recall@5 (for
graphiti, sessions ordered by first appearance among the episodes of its
ranked facts); session-level JointRecall@4k (every evidence session has at
least one rendered unit), computed identically from the rendered context for
every arm; rendered tokens per query; duplicate share, the tokens of rendered
units already covered by an earlier rendered unit of the same turn divided by
rendered tokens; candidate list size per arm.

Answering: accuracy under each reader with the primary judge, on all
questions, on answerable questions, on abstention or null questions, by type.

Cost and time: index-time model calls and tokens per corpus (post-graph-rag's
and Graphiti's read from the proxy usage log, since the bench CostMeter does
not see them; his build cost charged to every arm that reads his tables),
query-time model calls and tokens per question, seconds per question, for
every arm. PRICES_USD_PER_MTOK gains embedding tiers and the chosen chat model
with a dated price.

## 8. Failure buckets, rules fixed now

Candidate list per arm, logged: ours, the fused top 100; chandan_live, the
full query output before the budget cut; graphiti, the 20 facts and 20
entities. Its size is reported beside bucket 2.

Bucket 5 is tested first. A wrong answer on an answerable question that the
two judges disagree on is bucket 5 and is not tested against buckets 1 to 4
(decidable for the head-to-head arms of section 6; reported as "on the
audited sample" for the others). Every other wrong answer is tested against
buckets 1 to 4 in order, first match wins:

1. Index failure: at least one evidence unit has no retrievable representation
   in the arm's index. Zero by construction for our arms. For chandan_live,
   counted when no exported chunk or relation provenance in the space covers
   the evidence turn. For graphiti, counted when no exported edge in the
   group cites the evidence session.
2. Retrieval failure: every evidence unit is represented, but at least one is
   absent from the candidate list.
3. Context assembly failure: every evidence unit is in the candidates, but at
   least one is outside the rendered context at B; or is inside but truncated
   at the 2,000-character cut and the normalised gold answer string appears
   in the cut-off tail (a proxy, named as such; MultiHop-RAG uses the located
   fact excerpt instead); or, for a knowledge-update question, the
   superseding and superseded turns share a date and the tie-break rendered
   them in the wrong order. The superseding turn is the evidence turn whose
   text contains the normalised gold answer string; if no evidence turn or
   more than one contains it, the clause does not fire and the case is
   counted.
4. Reader failure: every evidence unit is inside the rendered context at B,
   no bucket 3 clause fires, and the answer is still judged wrong by the
   primary judge. A truncated or half-covered evidence unit whose cut-off
   part does not contain the gold answer counts as inside here, and the
   number of such cases is reported.

## 9. Pre-declared tests

Paired at the question level, the bench's compare (10,000 permutations,
percentile bootstrap CI), alpha 0.05. Wins, ties and losses reported with
every test. Three families, Holm applied within each; families B and C never
feed the pass rule.

Family A, the gate:

- T1: S5_primary against chandan_live on JointRecall@4k, LongMemEval, n 470
  less any refused ids.
- T2: S5_primary against graphiti on session-level JointRecall@4k, on
  GRAPHITI_150 less any refused ids (at most 150).
- T8a: S5_primary against chandan_live on fact-level JointRecall@4k,
  MultiHop-RAG, on every non-null query whose facts were all located (up to
  2,255), not on MHRAG_ANSWER. Expected discordance is not known before the
  build and is reported with the result.
- T8b: S5_primary against ours_cheap on the same.

If Graphiti is dropped or partial, T2 is recorded as not run and Holm within
Family A runs over T1, T8a and T8b.

Family B, the mechanism, all on JointRecall@4k on LongMemEval, five
comparisons under Holm:

- T3: S5_primary against ours_cheap.
- T4: S5_primary against S4_static.
- T5: overlay R3 against R0, and R3 against P0, planner held at LLM.
- T6: S4_static against ours_cheap.

Family C, answers: McNemar (exact binomial on the discordant pairs, primary
judge): S5_primary against chandan_live and against graphiti under each reader
on LongMemEval, and against chandan_live under Reader A on MHRAG_ANSWER and
under Reader B on READER_B_MHRAG. Six tests, Holm within C.

Non-inferiority, T7, outside the families: on the 120 local questions at B
equals 4,000 on JointRecall@4k, S5_primary against ours_cheap; holds if losses
minus wins is at most 3. The CI is reported beside it and is not the rule.
Expected discordance on this set is about 7 of 120 (measured by the review on
the e5 rankings rendered at 4k: rrf_turn_route 1 win, 6 losses against
rrf_sentence), so the test resolves a net swing of three questions and
nothing finer.

Power, written down so the result is read correctly: on the e5 rankings
rendered at 4k the review measured 41 to 46 discordant pairs of 470 (about 9
percent), so the SE of a paired delta is about 0.014, the minimum detectable
effect about 0.04 before Holm and about 0.045 after Holm within Family A. The
section 10 prediction implies an overall T1 delta of about 0.02. A positive
T1 inside that band is "not shown" under D1.

T5 is read in four branches: if the CI of R3 minus R0 contains zero, "no
measurable effect"; else if P0 minus R0 is within 0.01 of R3 minus R0, "adds
material"; else if R3 exceeds P0 by more than 0.01 with a positive CI on the
R3 against P0 pair, "reconciles"; else "not resolved", with the three deltas
and their CIs shown. The number of questions whose rendered context differs at
all between R0, R3 and P0 is reported.

Pass rule. Part 1 passes when T1 is "shown" under D1, T8a is positive and
significant after Holm within Family A, T2 passes under D2 or is recorded as
not run, and T7 holds. T8b is reported. T1 is against the first build of his
index; his own paper reports development results swinging by 15 to 35 points
per category between builds at small n. A second build is run only if the
first build's metered cost leaves at least that amount unspent under the 500
hard total after every other cap is honoured; the report states the first
build's cost and whether the second build ran. T1 against the second build is
a robustness row; if the two differ in sign or significance the report says so
and the first build still decides.

## 10. Predictions, written before any build

Half-widths beside each per-type prediction come from the e5 discordance at
that n; a point estimate inside the half-width is "consistent with, not
confirmed".

- T1 overall positive, about 0.02 (band 0.03 either side).
- S5_primary beats chandan_live on JointRecall@4k by at least 0.05 on
  multi-session (n 121, half-width about 0.07) and at least 0.03 on
  temporal-reasoning (n 127, half-width about 0.08); within 0.02 on the local
  set. Note: on the e5 rankings rendered at 4k the review measured
  rrf_turn_route at 0.942 JointRecall@4k on the local set against 0.983 for
  rrf_sentence; rrf_sentence is the best cheap arm and rrf_turn is the
  no-rule counterpart of ours_cheap, so that 0.04 gap mixes the speaker rule
  with the sentence-versus-turn unit choice, and the no-rule rows separate
  them.
- S4_static beats ours_cheap by 0.02 to 0.04; most of the S5 over S4 gain
  comes from the planner on temporal and multi-session, not from the overlay.
- T5 lands in the "adds material" branch.
- The largest community share is above 50 percent without topic weighting and
  drops under it.
- graphiti session-level JointRecall@4k is below S5_primary on GRAPHITI_150 by
  0.02 to 0.06 (half-width about 0.04 at n 150, so likely "consistent with").
  If Graphiti is dropped this prediction is recorded as untested.
- MultiHop-RAG: S5_primary beats chandan_live on fact-level joint recall for
  comparison and inference queries by at least 0.05; closed_book accuracy on
  answerable queries exceeds the per-type majority-class rate by at least
  0.10 (the contamination floor).
- The S5_primary minus chandan_live accuracy difference (Family C) has the
  same sign under Reader A and Reader B, on LongMemEval and on MultiHop-RAG.

## 11. Budgets

CostMeter caps and proxy-metered caps, USD: post-graph-rag build and every
query-time call of chandan_live on LongMemEval with the study model 130
(25,112 session documents at two calls each plus embeddings, projected after
the first shard); the post-graph-rag calibration row on CHANDAN_CAL_18 with
his own models 70; post-graph-rag on MultiHop-RAG 10; Graphiti 60 on the
fixed subset (decided by the 20-session pilot); overlay generation 15; S2
relevance tests 25; planner 10; answering and judging 100 across both corpora,
both readers and the chandan_full_uncut reader, with Reader B restricted as in
section 6; verification reruns 20. Sum 440; hard total 500. A run that hits
its cap stops and is reported as partial, with the reduction order of
section 5. Spend by post-graph-rag and Graphiti is metered from the usage
field of each response inside the runner and cross-checked against the proxy
request log by time window and job tag, because the log is shared by every
caller on the machine.

## 12. What is deliberately not done, and what is disclosed

No component is tuned. No index is rebuilt except the conditional second
post-graph-rag build in section 9. One encoder, one seed (13). No question,
answer or evidence flag reaches the overlay prompt, the topic model, the
graph, or the planner table. The oracle planner is a bound, never a system.

Disclosed: the planner table, the planner prompt definitions and the rules
patterns were written with the e5 by-type results on all 500 LongMemEval
questions known to the author. The e5 speaker rule was written by someone who
knew the question shapes; it is kept in our arms as a declared component, not
applied to the competitors' units (his chunks span both roles and cutting them
is not "as shipped"; Graphiti facts carry no role), and its share is shown by
the no-rule rows. The post-graph-rag run reproduces the configuration of the
shipped package version recorded in pgr.md, with the models set by section
13; the numbers his repository reports (README 94.0 from his own judge panel,
marked not reportable in his result file; 78.2 under the official protocol
with a gpt-4o judge; paper v2 85.8) are printed beside our chandan_live
result, with the models each used. In his frozen configuration supersession
never fires (no exclusive predicate groups, contradiction detection off), so
the "later document closes earlier fact" mechanism is not part of his
LongMemEval result; the report says so. The Graphiti ingestion unit is the
one the pilot chose, and any deviation from Zep's per-message ingestion is
labelled.

## 13. Model policy and what the smokes measured, fixed before any run

Measured on 2026-09-06 (docs/part1/env): gemini-3.6-flash bills its
reasoning as output tokens and produced 2,500 to 5,600 output tokens per
extraction call on 1,150-character documents, 89 percent of the index cost;
post-graph-rag's LongMemEval harness makes two extraction calls per session
document and never chunks; at his models the per-question protocol over
25,112 session documents projects to well over USD 1,000, and even the
150-question subset to several hundred. His own result file names
gemini-3.7-flash as the index model and gemini-3.6-flash as the answer model;
gemini-3.7-flash serves on the global Vertex endpoint at the same price as
3.6. gemini-2.5-flash-lite is the cheapest current flash-class model on the
project (0.10 in, 0.40 out per million tokens, no reasoning tokens) and
passes the JSON-schema check.

Policy, under the owner's principle: one study model, gemini-2.5-flash-lite,
for every model call in Part 1 except Reader B and the second judge (Azure
gpt-5.4, applied to every arm alike). That covers Reader A, the candidate
primary judge, the planner, the S2 relevance tests, the overlay proposals,
post-graph-rag's extraction and gleaning and keyword and synthesis calls in
the main chandan rows, and Graphiti's extraction. Embeddings: each system's
own default (gemini-embedding-001 at 1,536 dimensions for post-graph-rag,
his value, and the largest pgvector will index; 3,072 for Graphiti in Neo4j;
MiniLM for ours). This differs from the models post-graph-rag's README run
used, and it is disclosed with a calibration row: chandan_live on
CHANDAN_CAL_18 indexed with gemini-3.7-flash and answered with
gemini-3.6-flash, exactly his result-file configuration, reported beside the
study-model row on the same 18 questions with JointRecall@4k and answer
accuracy, so the reader can see how much his retrieval changes with his own
extractor. The calibration row is a reported row, never in a test. If the
first shard of the main chandan build projects over its cap, the build
continues on GRAPHITI_150 only and T1 runs on that subset, labelled.

The Graphiti smoke with the study model extracted no speaker entity and lost
the pilot fact to a dropped self-edge under per-session text episodes; that
is why the ingestion unit is decided by the pilot (section 5) and not
assumed. The proxy's thinking-capable models return no text at small output
limits; every call in Part 1 sets an output limit of at least 64 tokens.

## 14. Changes made after a number was read

Listed here so the reader can weigh them. Each was made before any answering
run or any test in section 9 was computed.

1. 2026-09-06, after the first-pass LongMemEval retrieval on the arms that
   use none of post-graph-rag's tables (ours_cheap and its no-rule rows,
   S5_noPGR). The fused arms rendered 3,602 of 4,000 tokens and 4,369 of
   8,000 on average: the fused list was cut at the logged top 100 and the
   speaker rule then removed the assistant turns from it, so the rendering
   rule's "until the budget B is full" was not met. Change: the fused arms
   and S2_lazy fill the remaining budget from the fused list beyond the top
   100 (depth 400, after the speaker rule), after the lazy expansion units.
   The logged candidate list for bucket 2 stays the fused top 100. ours_cheap
   is untouched (it is e5's function and fills 4,000). The numbers read
   before the change, JointRecall@4k on 470 questions: ours_cheap 0.938,
   S5_noPGR 0.940, ours_cheap without the rule 0.800, sentence fusion without
   the rule 0.823; at 8,000: 0.957, 0.949, 0.868, 0.879. The report prints
   the pass after the change beside these.
2. 2026-09-06, the Graphiti pilot: no ingestion variant fitted the USD 60 cap
   (projections 69 to 174) and none could be measured for evidence presence
   because neither evidence session was among the first 20 by date. The arm
   is dropped by the section 5 rule and T2 is recorded as not run; the owner
   may decide otherwise later. Neither reading changed any other rule.
3. 2026-09-06, the judge audit: pooled agreement between the cheap judge and
   gpt-5.4 was 0.848 on 250 records, so gpt-5.4 became the primary judge by
   the section 6 rule. Section 6 did not foresee that the primary judge would
   then be the same model as Reader B. No rule changes; the report adds a
   cross-family column for the Reader B rows (the cheap judge on every
   Reader B record, not only the wrong ones) beside the primary verdicts, so
   the reader can see whether same-model judging moved those rows.

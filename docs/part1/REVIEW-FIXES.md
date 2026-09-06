# Review fixes, Part 1 code (design version 4)

Date: 2026-09-06. One entry per review finding: the design text it turns on,
the file, what changed, or why the code was left as it was. Nothing here is
committed by the agent that wrote it; the main session commits.

Test results and the smoke are at the end.

## render-score-units

### 1. score.py, truncated_evidence missed the 2,000-character cut (major)

Design, section 3: "A truncated evidence turn is flagged and counted. Of the
896 marked evidence turns, 11 are over 2,000 characters." Section 8 bucket 3:
"truncated at the 2,000-character cut".

Fixed. render.py: RenderedUnit has a second flag `cut` (a turn longer than
2,000 characters, cut there); render() sets it from the owner text. score.py:
TurnCoverage carries `cut`; QuestionScore.truncated_evidence counts evidence
turns with either flag and the new field `cut_evidence` counts the cut ones
alone; summarise() sums both. report.py prints both columns under the
LongMemEval table. RenderedContext.n_truncated stays the budget-only count,
so the chandan_live rule is unchanged. part1.py writes and reads the flag in
contexts.jsonl (an older row without it reads False).

### 2. units.py, locate_fact took the first occurrence even when it straddled (minor)

Design, section 3: "a located fact maps to exactly one chunk because overlap
is zero."

Fixed. locate_fact visits every occurrence and takes the first one that lies
whole inside one chunk; only when no occurrence does is the first occurrence
mapped by the most-overlap rule and flagged straddle (the design gap comment
stays on that fallback).

### 3. render.py, render_native broke date ties by native position (minor)

Design, section 5: "the kept units are then displayed oldest first, ties
broken by session order then turn index", the same for every arm.

Fixed. render_native takes an `order` mapping (container id to its position
in the question's haystack, the same mapping render() uses); dated units sort
by (date, session order, index of the first turn the unit spans or cites,
native position). part1.py passes the haystack order for the chandan, the
calibration and the graphiti rows. Entity lines stay undated and ahead.

### 4. render.py, _take truncates for every arm (minor)

Design, section 5: the general rule says "taken in the arm's rank order until
the budget B is full"; truncation appears under chandan_live only.

Kept, marked. A design gap comment at _take states that the chandan_live rule
is applied to every arm alike, and evaluate.DISCLOSURES carries the sentence
under "Added after reading the design", which the report prints.

## retrieve-planner-overlay

### 5. planner.py, whole-word matching (major)

Design, section 5: "assigns the first matching shape from this ordered list
of case-insensitive patterns ... when the question contains ...". Section 12
says the patterns were written and disclosed as fixed.

Fixed. Every pattern is a plain case-insensitive substring test; the design
gap comment is gone. tests/test_part1_retrieve.py asserts the substring
behaviour on the review's examples ("Is Chicago cold?" is temporal).

### 6. retrieve.py, S(c) was a dot product (major)

Design, section 5: "S(t) and S(c) are the cosine similarity between the query
vector and the topic or community prototype."

Fixed. retrieve.cosine_rows divides by the row norms and the query norm (a
zero row scores 0); QueryRun.group_scores uses it for both channels, so
S(c) is a cosine whatever the file holds. Test with scaled prototypes added.

### 7. overlay.py, P0 built from R2 (major)

Design, section 4 item 9: "The placebo set has the same number of links with
the same degree distribution, targets shuffled with the seeded rng."
Section 9, T5: "overlay R3 against R0, and R3 against P0" and "if P0 minus R0
is within 0.01 of R3 minus R0, adds material".

Fixed. Item 9 does not name the base set; the T5 reading only works when P0
is a control matched to the primary set, so link_sets builds P0 from R3 (same
count, degrees and weights). The placebo diagnostics record base_set R3. The
module docstring states the reading. Tests updated (P0 has R3's size).

### 8. planner.py, parse_reply normalised the reply (minor)

Design, section 5: "returns one label ... A reply that is not one of the
eight labels is logged and treated as local."

Fixed. parse_reply matches the reply with surrounding whitespace stripped
and case ignored, nothing else removed; "Temporal." or a second line is
off-list, logged and treated as local. The five smoke replies were bare
labels, so nothing in the smoke changes.

### 9. planner.py, the pronoun I was excluded from the entity rule (minor)

Design, section 5: "entity when it contains ... or a capitalised token that
is not sentence-initial."

Fixed to the design text: "I" is a capitalised token like any other. This
turns most LongMemEval questions that reach the entity rule into entity
under the rules planner (an ablation arm, never the primary). The docstring
says the effect is a matter for the design owner; the design is not amended
here. tests updated ("What did I say?" is entity).

### 10. retrieve.py, S5_noPGR makes a model call when run alone (minor)

Design, section 5: "S5_noPGR (relation and entity channels off, everything
else identical; no model call)."

Fixed. A comment at ARM_SPECS states the reading: the LLM planner is shared
and memoised per question. part1_retrieve decides the planner before the arm
loop when S5_noPGR is wanted and no LLM-planner arm runs before it, charges
that call to the pseudo-arm "planner" in query_cost, and raises if S5_noPGR
ever reports a model call. In the default arm order S5_primary runs first
and S5_noPGR reports zero calls.

## topics-graph

### 11. topics.py dropped the raw topic_embeddings_ (major)

Design, section 4 item 5: "A phrase's topic vector is the probability-weighted
mean of the 384-dimensional topic_embeddings_ over the sub-units it appears
in."

Fixed. topics.parquet's prototype column and TopicResult.prototypes now hold
the raw topic_embeddings_ rows (the plain mean BERTopic computes).
topic_distribution normalises its own copy, retrieve takes the cosine at
read time (finding 6), and graph.build receives the raw rows from part1.py.
Tests updated: the rows are not unit length.

### 12. graph.py, load_prototypes could not read topics.parquet (major)

Design, section 4 item 3: "topics.parquet is committed as a frozen artifact
and the repro command loads it."

Fixed. load_prototypes accepts the id column topic_id or topic and the vector
column prototype (what topics.save writes), embedding or vector, skips the
outlier row -1, and raises a clear error on any other layout. Test added
that saves a TopicResult through topics.save and loads it through
graph.load_prototypes.

### 13. graph.py, pronoun majority vote unmarked (minor)

Design, section 4 item 4: "pronouns ... are dropped", with no rule for when a
phrase is a pronoun.

Kept, marked. PRONOUN_RULE names the majority-vote reading; hub_rule carries
the design gap comment; diagnostics.json records pronoun_rule and part1_report
puts it in the disclosures.

### 14. graph.py, hub rule before graphrag pruning unmarked (minor)

Design, section 4 item 4 lists the graphrag defaults and the hub rule with no
order.

Kept, marked. HUB_RULE_ORDER states the order and that graphrag's one ego
node is then the highest-degree content phrase left after the hub rule; a
design gap comment sits at the call in build(); diagnostics.json records
hub_rule_order and the ego node's text, and part1_report puts both in the
disclosures.

### 15. graph.py, only leading articles stripped (minor)

Design, section 4 item 4: "normalised (lowercase, lemma, articles stripped)".

Fixed to the design text: every article token (a, an, the) is stripped
("the man of the hour" becomes "man of hour"). diagnostics.json records the
rule.

### 16. topics.py docstring called a range a pin (minor)

Design, section 4 item 3: "BERTopic (version pinned in pyproject)".

Docstring reworded: the exact version 0.17.4 is fixed by uv.lock and recorded
in diagnostics.json; pyproject holds the range bertopic>=0.16,<0.18, which is
not a pin. pyproject is outside this task's edit scope; the main session
should pin bertopic==0.17.4 there.

## evaluate-report-subsets

### 17. part1.py, graphiti's variant chosen on turn-level joint_recall (major)

Design, section 5: "The gate uses whichever of the two scores higher on
JointRecall@4k over the set"; section 3: Graphiti is scored at session level
only; section 9, T2 on session-level JointRecall@4k.

Fixed. CHOICE_METRIC picks session_joint_recall for graphiti and
joint_recall for the chandan arms; the metric and n are written beside the
means in chosen_variant.

### 18. part1.py, reader jobs over the whole corpus for every arm (major)

Design, section 2 (graphiti on GRAPHITI_150; the calibration row on
CHANDAN_CAL_18), section 5 (missing output counted per arm), section 13.

Fixed. arm_population() gives each arm its population inside the reader's:
graphiti on GRAPHITI_150, chandan_live_cal on CHANDAN_CAL_18, the chandan
arms on the subset their build was restricted to (finding 22), every other
arm on the reader's population. The missing rule applies inside that
population only. qa metrics record the per-arm counts.

### 19. part1.py, a Reader B withdrawal marked the whole arm partial (major)

Design, section 5: "If a cap binds, Reader B is withdrawn in this order ...";
section 9: T7 holds; tests on a partial run are labelled.

Fixed. part1_report keeps the withdrawal per (reader, arm) in
partial_answering; run_tests labels only the Family C tests under that reader
partial (with the withdrawn arms in the label) and leaves the retrieval tests
and T7 untouched. The retrieval partial dict is now filled from the retrieve
stage (finding 24). Test added.

### 20. part1.py, a cap in answering lost the run (major)

Design, section 11: "A run that hits its cap stops and is reported as
partial"; section 6: both judges score the audit before any test.

Fixed. answers.jsonl is written for every corpus before any judge call. One
meter serves the stage: the answering pool runs under the cap less a judging
reserve (JUDGING_RESERVE 0.15, a design gap: section 11 gives no split of
the 100 USD) and judging under the whole cap, so the total stays inside 100.
A cap in judging is caught, the verdicts made so far are kept, the pooled
audit is recomputed from them, "judging" is recorded as partial and the
report discloses it.

### 21. evaluate.py, the cut-off tail read the largest unit only (major)

Design, section 8 bucket 3; section 3 ("the rendered units together").

Fixed. _rendered_chars sums the turn's characters over the rendered units,
the same sum the coverage rule uses, so a turn rendered whole across two
chunks has no tail. The tail is still taken as the turn text past the
rendered characters; for a competitor chunk that starts inside the turn this
is the proxy section 8 names, and a comment says so. Test added.

### 22. part1.py, no path for T1 on GRAPHITI_150 (major)

Design, section 13: "If the first shard of the main chandan build projects
over its cap, the build continues on GRAPHITI_150 only and T1 runs on that
subset, labelled."

Fixed. competitors.pgr_build_subset reads the runner's run_<job_tag>.json
logs and returns the subsets.json key the build was launched with
(--subset GRAPHITI_150 per the RUNBOOK). part1_report then passes
overrides={"T1": ...} (labelled), restricts the chandan arms' retrieval
population and answering population to the subset (arm_populations), and
narrows every test that reads a chandan arm to it with a label. Test added
for the log reading and the per-arm populations.

### 23. report.py, no section 10 predictions (major)

Design, section 10.

Fixed. evaluate.predictions builds one row per prediction: the fixed number
and its band, the measured value or paired delta with CI and n, and one of
four labels: "consistent with, not confirmed" (inside the band), "not
confirmed" (outside on the predicted side), "contradicted" (wrong side of
zero, or outside a within band), "untested". Per-type deltas reuse
run_paired on the type's questions. The majority-class rate is the share of
the commonest normalised gold answer among the closed_book answers of the
type (design gap, named in the row). The community share reads the nodes
share with the units share beside it (design gap, named). The report has a
"Predictions (section 10)" section between the tests and the buckets.

### 24. part1.py, a cap in retrieve aborted the stage (minor)

Design, section 11 and section 5 (partial runs labelled).

Fixed. part1_retrieve catches BudgetExceeded around the question loop, closes
the writers, records stopped, questions_processed, questions_not_processed
and partial (every arm of the stage) in retrieve/metrics.json; part1_report
reads partial from there.

### 25. part1.py, the MultiHop-RAG variant choice used every non-null query (minor)

Design, section 7: the primary is fact-level joint recall over the queries
whose facts were all located.

Fixed. On MultiHop-RAG the choice averages fact_joint_recall over the rows
with all_located true.

### 26. evaluate.py, S2_lazy not charged his build cost (minor)

Design, section 1: "His build cost is charged to every arm that reads his
tables."

Fixed. ARMS_READING_PGR_TABLES is derived from retrieve.ARM_SPECS (every spec
with pgr True, S2_lazy included, since its fill is the fused ranking) plus
his three arms. Test updated.

### 27. evaluate.py, an invented explanation for null queries (minor)

Design, section 6: "null queries use the abstention prompt". e5 judge_prompt
passes the gold answer as the explanation.

Fixed. judge_prompt_for passes the query's own gold answer ("Insufficient
information.") to the abstention prompt, as e5 does. NULL_EXPLANATION is
gone.

### 28. part1.py, Reader A never answered at 8,000 by default (minor)

Design, section 7: reported also at 8,000; section 6: Reader B at 4,000 only.

Fixed. part1_qa defaults to budget "4000,8000"; Reader B jobs stay at 4,000.
The RUNBOOK sentence "add --budget 4000,8000 for the 8,000 contexts too" is
now stale (the file is outside this task's scope) and should be dropped.

### 29. report.py, the cross-check sentence was typed prose (minor)

Design, section 11.

Fixed. The cost section prints the cross-check figures only when the payload
carries a proxy window total (a key ending in _usd under cost.proxy);
otherwise it says the cross-check is pending and that the file holds the job
tags only.

### 30. evaluate.py, no second-build record (minor)

Design, section 9: "the report states the first build's cost and whether the
second build ran. T1 against the second build is a robustness row".

Fixed. The second build is read from data/part1/pgr_second/<corpus>/ as the
arm chandan_live_second (LongMemEval; rendered twice like chandan_live, its
own variant choice). run_tests adds T1_second_build (outside Holm, its
significance read at alpha on its own p, a design gap) and second_build
{ran, sign_differs, significance_differs, first_build_decides};
build_metrics adds first_build_usd from the pgr_build cost. The pass-rule
section prints both. Test added.

## runners

### 31. run_spaces.py, the raised ladder stopped at a character count (major)

Design, section 5: "with only the result limit raised until the rendered
context reaches B"; "B is 4,000 tokens (primary) and 8,000 tokens (secondary),
counted by the bench TokenCounter".

Fixed. BenchTokenCounter reproduces the bench TokenCounter (the
all-MiniLM-L6-v2 tokenizer from the Hugging Face cache, no special tokens)
through the tokenizers library, which was installed into the pgr venv
(tokenizers 0.22.2, the bench's version; the venv has no transformers).
rendered_units builds the units the bench renders for chandan_live (his
chunk lines, entity lines, relation lines, deduplicated by id, the same line
forms as multicard.part1.competitors) and rendered_tokens sums their tokens
as render_native does. raise_until raises top_k until that count reaches
4,000 and 8,000. Every query file records rendered_tokens, target_tokens
and budget_reached; context_chars stays beside it as a side figure.
tests/test_part1_pgr.py checks the counter equals the bench TokenCounter and
the line forms equal the bench's.

### 32. run_groups.py, the same character ladder (major)

Fixed the same way: BenchTokenCounter, rendered_tokens over the FACT and
ENTITY lines (deduplicated by uuid, the bench's line forms), search_ladder
on 4,000 and 8,000 tokens, context_chars beside it. tokenizers installed
into the graphiti venv.

### 33. run_groups.py, three groups at once (major)

Design, section 2: "post-graph-rag spaces and Graphiti groups are built and
queried one question at a time in that order, so a capped or partial run is
a defined prefix."

Fixed. --k defaults to 1 and the run processes ORDER one question at a time;
--k above 1 keeps the gather and run_summary.json records prefix_of_order
False with a note that the report must state the deviation. The RUNBOOK's
"--k 3" is now a deviation to be removed (file outside this task's scope).

### 34. run_groups.py, a hard-coded projection basis (major)

Design, section 5 (the pilot's projection over GRAPHITI_150) and section 2
(one group per question holding its haystack sessions).

Fixed. graphiti_basis reads GRAPHITI_150 from subsets.json at pilot time and
counts the haystack slots (7,581 in the committed file), the unique sessions
(7,032) and the rendered characters over the slots with the bench loader;
project() scales USD per character by that total. The basis is written into
pilot.json, every row and the Pilot section of graphiti.md. The old constants
are gone. Test added.

### 35. run_spaces.py, SHIPPED_TOP_K called 8 the shipped default (major)

Design, section 4 item 6: "top_k at the shipped default"; section 5: "at his
shipped result limit".

Not decided here; the finding asks the main session to decide in writing.
The comment at SHIPPED_TOP_K now names the three candidates with their
source lines (package default 5, reader_sweep.py 8, run.py 32), says 8 is
the published run's value from pgr.md section 2.1 and not the package
default, and the value is a command line option (--shipped-top-k) written
into every query file and meta.json with a note. The main session records
its decision in pgr.md and, if it is not 8, relaunches with the option.

### 36. run_spaces.py, which harness form of the question (minor)

Design, section 4 item 6: pgr.md records "how the question date is passed to
the query as run.py passes it".

Made explicit, not decided. --question-form reader_sweep (default, the
published run's "Today is {date}. {question}") or run_py (his original
harness, the three-rule answer-side block before that text, copied from
run.py lines 381-392). The form is recorded in every query file and
meta.json (question_form, with a note). The main session decides and records
it in pgr.md.

### 37. run_spaces.py, no default cap (minor)

Design, section 11.

Fixed. CAP_USD holds 130 (lme), 70 (calibration), 10 (mhrag); --cap-usd
overrides and the source of the cap in force is logged (cap_usd_source in
the run log and meta.json). Test added.

### 38. run_groups.py, the pilot's spend outside the 60 (minor)

Design, section 11: "Graphiti 60 on the fixed subset (decided by the
20-session pilot)".

Fixed. cmd_run takes the pilot's spend off the cap: --spent-usd, else the
usd_total of --pilot-json, else the pilot question's pilot.json under the
pilot out dir (derived.GRAPHITI_PILOT of subsets.json). run_summary.json
records pilot_spent_usd, run_cap_usd and total_usd_with_pilot. Test added.

### 39. run_groups.py, BFS in the design text (minor)

No code change, as the finding says. evaluate.DISCLOSURES carries the
design-text error: the COMBINED_HYBRID_SEARCH_RRF recipe as shipped in
graphiti-core 0.30.1 has no BFS method; it is used as shipped. The report
prints it.

### 40. run_spaces.py, concurrent extraction not recorded (minor)

Design, section 4 item 6: pgr.md records "max_concurrent_chunks, and whether
it differs from his run".

Fixed in the runner's records: meta.json carries max_concurrent_chunks and a
concurrency_note (8 sessions at a time with the writes in date order; his
harness one at a time, run.py line 164). pgr.md is outside this task's scope
and needs the same sentence.

### 41. both runners, the ladder ceilings (minor)

Design, section 5: raise until B is reached.

Fixed. The ceilings (128 for top_k, 80 for the search limit) carry design
gap comments; the pgr run log counts raised_not_reached per variant and the
Graphiti run_summary.json and meta.json do the same; part1_retrieve counts
per arm and budget the questions with a raised run whose top step did not
reach B (n_raised, n_raised_not_reached) and the report prints them under
the retrieval table beside the chandan_live and graphiti rows.

## Other changes made on the way

- evaluate.py imports retrieve.ARM_SPECS (finding 26); no import cycle.
- part1.py: contexts.jsonl rows carry `cut` and `reached`; retrieve
  metrics.json carries raised_not_reached, stopped, questions_not_processed,
  partial; qa metrics.json carries judging_reserve, answering_cap_usd,
  judging_stopped and per-arm populations; the report payload carries
  predictions, second_build, partial_answering, arm_populations.
- The runner venvs gained tokenizers 0.22.2 (part1-tools/pgr/.venv and
  part1-tools/graphiti/.venv), installed with uv pip, no sudo. The
  environment records (pgr.md, graphiti.md, outside this task's scope) should
  list it.

## Left for the main session

- Decide SHIPPED_TOP_K (finding 35) and the question form (finding 36) in
  writing in pgr.md; relaunch the runner with --shipped-top-k and
  --question-form if the decision differs from the defaults.
- Pin bertopic==0.17.4 in pyproject (finding 16).
- RUNBOOK: drop "--k 3" from the Graphiti run (finding 33) and the "add
  --budget 4000,8000" sentence (finding 28); add --subsets to the pilot
  command if the default path is not wanted.
- pgr.md: the concurrency sentence (finding 40) and the tokenizers install.
- The rules planner now labels most questions with "I" as entity (finding 9);
  if the design owner wants the exclusion, it needs a design change.
- The section 13 restriction (finding 22) is read from the runner's run logs
  under data/part1/pgr/<corpus>/; a build restricted by hand without
  --subset is not detected.

## Test results

    cd /Users/muralisid/github_other/multicard-bench && PYTHONHASHSEED=13 uv run pytest -q

319 passed, 6 warnings (umap n_jobs), 26 s, exit 0, on 2026-09-06 after every
change above. New or changed tests: test_part1_units (the cut flag and its
count, locate_fact over every occurrence, the native tie-break),
test_part1_retrieve (substring rules, the pronoun I, strict replies, cosine
prototypes), test_part1_overlay (P0 from R3), test_part1_topics (raw
prototype rows), test_part1_graph (every article, load_prototypes on
topics.parquet, the diagnostics keys), test_part1_evaluate (the null gold,
the summed tail, partial answering, arm_ids, the second build, the
predictions and their labels, the S2_lazy charge, the report lines),
test_part1_pgr (the bench-equal token counter, the rendered units, the token
ladder, the cap defaults and question forms), test_part1_graphiti (the
projection basis, the token lines, the pilot spend), test_part1_integration
(the cut flag round trip, pgr_build_subset, arm_population).

## The smoke

The RUNBOOK's six commands on the same 5 LongMemEval and 5 MultiHop-RAG
questions (tag smoke, --sample), the earlier outputs moved to
results/part1_smoke_before_review_fixes/ first so the index stage exercised
the new topics and graph code. Caps 0.15, 0.15, 0.15, 0.15, 0.40 (sum 1.00).
Every stage exited 0:

| stage | wall | USD |
|---|---|---|
| part1_index lme (topics refitted, graph rebuilt, overlay 64 pairs, P0 31 links = R3) | 1.4 min | 0.0069 |
| part1_index mhrag | 0.6 min | 0.0031 |
| part1_retrieve lme (S5_noPGR 0 calls, chandan_live raised 5 of 5 reached B) | 0.6 min | 0.0005 |
| part1_retrieve mhrag | 0.4 min | 0.0004 |
| part1_qa all, reader_a, budgets 4000 and 8000 (audit n 13, agreement 0.923) | 1.3 min | 0.0914 |
| part1_report | 5 s | 0 |

Total USD 0.102. results/part1_smoke/REPORT.md carries the predictions
section (15 rows, labels from the four), the cut column beside the truncated
count, the raised-not-reached table, the second-build line, the pending
cross-check sentence, the per-reader withdrawal line and the graph
disclosures; no banned character. Not a study result.

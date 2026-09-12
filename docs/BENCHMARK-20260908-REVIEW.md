# Why the 2026-09-08 benchmark numbers came out low

Written 2026-09-08 against `outputs/benchmark-report-20260908/FULL-REPORT.md` and
the saved run artifacts under `results/version_a/` and `results/version_abcd/`.
Every number here is read from a file on disk or computed from one. Nothing in
the report was edited.

## The short version

The run is a new implementation. It does not reuse the Part 1 retrieval code,
the Part 1 render rule, or the Part 1 Vertex client. Four things differ from
Part 1, and three of them push the score down.

1. The reader's whole output allowance is 768 tokens, and hidden reasoning is
   spent inside it. 17 percent of LongMemEval answers and 49 percent of
   MultiHop-RAG answers were cut off mid sentence. This is the largest cause.
2. The LongMemEval speaker rule is off. Part 1 measured that rule at plus 13.8
   points of joint recall on the same corpus with everything else identical.
3. The judge is gemini-2.5-flash-lite. The Part 1 judge audit measured it at
   84.3 percent agreement with gpt-5.4 over 1,200 judgments and demoted it.
4. The reader is gemini-3.6-flash, not the gpt-5.4 that produced the Part 1
   headline of 0.844. This is a choice, not a defect, but it is most of the
   remaining difference and the two numbers must not be compared directly.

One separate defect runs the other way. On LongMemEval S the reader is shown
the session id of every excerpt, and the gold sessions are the only ones whose
id starts with `answer_`. Closing that leak on 250 questions costs 3.2 points,
which is inside the noise, so it is a design fault that must be fixed rather
than a proven inflation.

The controlled fix for the first item is measured below. Raising the reader's
output allowance from 768 to 3,000 tokens, and changing nothing else, moves
LongMemEval S from **0.7820 to 0.8340**.

## 1. The reader was cut off mid sentence

`src/multicard/version_a/runner.py:93`

    a = reader.generate(reader_prompt(q, context["context"]), 768, role="answer")

`src/multicard/version_a/client.py` sets `reasoning_effort: "none"` only for
gemini-2.5-flash-lite. The reader, gemini-3.6-flash, keeps its default
reasoning, and reasoning tokens count inside `max_tokens`. Nothing in the
client, the runner or the tests reads `finish_reason`, so a cut-off answer is
scored as if it were a finished one.

Part 1 hit this exact failure and fixed it. `src/multicard/llm/vertex.py:123`:

    if self.disable_thinking:
        # Reasoning models spend the output allowance on hidden thinking
        # before writing anything, which returned ten-word stubs where a
        # two-hundred-word summary was asked for.
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

The new client does not go through that file.

Observed output tokens, from `results/version_a/*/answers.jsonl`. The provider
ceiling shows up at 764.

| test | n | mean out | at 700+ | at 760+ | ends without terminal punctuation, at 760+ |
|---|---:|---:|---:|---:|---:|
| LongMemEval S | 500 | 520.7 | 149 | 83 | 90% |
| LongMemEval oracle | 500 | 495.3 | 131 | 85 | 90% |
| MultiHop-RAG | 2,556 | 611.7 | 1,241 | 545 | 93% |
| FactConsolidation MH 262K | 100 | 711.1 | 89 | 9 | - |

Sample cut-offs from LongMemEval S, quoted whole:

    'Based on the evidence provided:\n\n* **1 Gbps** (as stated on May 30, 2023)\n*'
    'you spent a total of **15 hours** driving to your three past road trip destinations:\n\n* **Washington'
    'the only mentioned bike-related expense is **$40** for a new set of'

Accuracy against output length, holding evidence presence fixed. `all evidence
in 4K` is the run's own `all_evidence_at_budget`.

| test | evidence all in 4K | not truncated | truncated (700+) |
|---|---|---|---|
| LongMemEval S | yes | 0.979 (n 285) | 0.468 (n 79) |
| LongMemEval S | no | 0.673 (n 49) | 0.364 (n 66) |
| LongMemEval oracle | yes | 0.922 (n 309) | 0.578 (n 83) |
| MultiHop-RAG | yes | 0.749 (n 430) | 0.237 (n 507) |
| MultiHop-RAG | no | 0.654 (n 662) | 0.168 (n 656) |

The evidence is in the context and the answer still fails. On MultiHop-RAG the
split inside a single question type is starker: inference_query scores 0.989 on
816 questions when the answer is short and 0.286 when it runs to 700 tokens.

Two readings are possible and both are partly true. Hard questions make the
model reason longer, and long reasoning gets cut off. The cut-off is not a
guess: 90 percent of the capped answers end mid word.

Size of the effect on MultiHop-RAG, as an upper bound. If the 1,241 truncated
answers scored like the 1,315 that finished, the test would read 0.741 instead
of the reported 0.4953.

### The controlled test

The saved contexts make a clean paired test possible: same question, same
rendered context, same judge, only the output allowance changed from 768 to
3,000. Script at `scratchpad/trunc/rerun.py`, run over the 149 LongMemEval S
questions whose answers used 700 or more tokens.

The result, on all 149 questions:

| | 768-token allowance | 3,000-token allowance |
|---|---:|---:|
| accuracy on the 149 | 0.423 | 0.597 |

Difference plus 0.174, bootstrap 95 percent interval [+0.094, +0.255], 32 wins
to 6 losses, exact McNemar p 0.00002. The reader used a mean of 1,105 output
tokens once it had room; 148 of the 149 finished inside 3,000.

Carried back to the whole test, holding every other question at its recorded
score, LongMemEval S moves from **0.7820 to 0.8340** on the same 500 questions.
That is 5.2 points from one configuration line.

By question type, on the 149:

| question type | n | 768 cap | 3,000 cap |
|---|---:|---:|---:|
| multi-session | 62 | 0.516 | 0.661 |
| temporal-reasoning | 49 | 0.408 | 0.612 |
| single-session-preference | 18 | 0.167 | 0.333 |
| knowledge-update | 15 | 0.467 | 0.667 |
| single-session-user | 3 | 0.333 | 0.333 |
| single-session-assistant | 2 | 0.000 | 0.500 |

The types that lose most are the ones that need reasoning across sessions and
dates, which is exactly where the reader spends its allowance thinking.

## 2. The speaker rule is off

`src/multicard/version_a/core.py:26` sets `"speaker_filter": False`.

Part 1 measured the rule in isolation. `ours_cheap` and `ours_cheap_norule`
differ on that one flag and on nothing else:

| arm | joint recall at 4,000 tokens | candidates |
|---|---:|---:|
| ours_cheap (rule on) | 0.938 | 62.2 |
| ours_cheap_norule (rule off) | 0.800 | 100 |

The new run's `all_evidence_at_budget` on LongMemEval S is 0.760 on 479
questions. That sits next to the Part 1 no-rule figure of 0.800, not next to
0.938. The residual four points is explained by the new run keeping every
repeated session occurrence, which grows the haystack from 19,829 containers
to 25,112 and adds 24 percent more distractors.

The rule is not free of judgement. Part 1's own disclosure says so: "The e5
speaker rule was written by someone who knew the question shapes." It is a
declared component, not a neutral default. Leaving it out is defensible. It
just has to be stated, because it is worth 13.8 points on this corpus.

## 3. The judge is the one Part 1 rejected

The new run judges everything with gemini-2.5-flash-lite at a 64-token
allowance. Part 1 ran a judge audit over 1,200 judgments before any test: the
cheap judge agreed with gpt-5.4 84.3 percent of the time, below the 0.90
threshold fixed in advance, so gpt-5.4 became the primary judge.

The effect is not one-directional. On the Part 1 Reader B rows both judges
scored every record: ours_cheap 0.844 under gpt-5.4 and 0.834 under the cheap
judge; chandan_live 0.624 under gpt-5.4 and 0.666 under the cheap judge. So it
is noise of a few points in both directions, not a systematic penalty. It is
not the reason the numbers are low. It does mean no cross-arm difference of
two or three points in this report can be trusted.

Judge truncation was checked and is not a problem. Mean judge output is 29
tokens, and every short judge reply parsed.

## 4. The reader model is different

Part 1's 0.844 on LongMemEval S was read by gpt-5.4 and judged by gpt-5.4. The
same retrieval read by gemini-2.5-flash-lite scored 0.558. The reader moved
that arm by 28.6 points. The new run's reader, gemini-3.6-flash, sits between
those two, and the new run's 78.20 sits between those two.

This is the point Part 1 already recorded: accuracy on these benchmarks is not
comparable across runs unless the reader and judge are held fixed.

## 5. A leak that pushes the other way

`src/multicard/version_a/core.py:151` writes the unit id into every excerpt
header the reader sees:

    [answer_280352e9#4; 2023/05/20; user; order=]

In LongMemEval S the gold sessions are the only ones whose id starts with
`answer_`. Measured over the prepared data:

- 10,960 of 246,930 corpus units carry the prefix, 4.4 percent.
- For all 479 questions with gold evidence, every evidence unit carries it.
- A haystack holds 47.8 sessions on average, of which 1.9 carry the prefix.
- For 438 of 479 questions, 91 percent, the prefixed sessions in the haystack
  are exactly the evidence set.

So the reader is being told which passages are the answer. Part 1's render
writes only `[date, speaker]` and has no such channel. The oracle variant keeps
only answer sessions, so it is unaffected. No other dataset in the run has this
shape: on LoCoMo, MuSiQue, 2Wiki, MultiHop-RAG and BEAM the prefix covers the
whole corpus.

### Does the leak move the score

Tested directly. 250 LongMemEval S questions drawn with seed 13, same rendered
context, same 768 allowance, same judge, with each container id replaced by a
fixed-length pseudonym so the `answer_` prefix is gone. Script at
`scratchpad/leak/rerun.py`.

| | ids shown | ids pseudonymised |
|---|---:|---:|
| accuracy on the 250 | 0.824 | 0.792 |

Difference minus 0.032, bootstrap 95 percent interval [-0.068, +0.004], 6 wins
to 14 losses, exact McNemar p 0.115. So the channel is there and it is 91
percent precise, but at this sample size the leak cannot be shown to have moved
the score, and the honest statement is that its effect is at most a few points.

The split is the interesting part. Where the answer was not truncated the leak
is worth nothing, 0.947 against 0.941 on 187 questions. Where the answer was
truncated it is worth 11 points, 0.460 against 0.349 on 63. The reader leans on
the id shortcut only when it has run out of room to reason. That is the same
defect showing up again from a second direction.

The header should still be fixed. A reader that has the gold sessions marked
for it cannot be used to argue anything about retrieval.

## 6. The B/C/D screen cannot support its own conclusions

Of the 100 paired comparisons in `paired.csv`, 3 have a bootstrap interval that
excludes zero:

- LongMemEval oracle, C minus B, plus 0.06, 6 wins 0 losses 94 ties
- PersonaMem v2 32K, B minus A, plus 0.05, 5 wins 0 losses 95 ties
- PersonaMem v2 32K, C minus A, plus 0.05, 5 wins 0 losses 95 ties

The screen is 100 questions per variant. At an accuracy near 0.75 the standard
error on 100 questions is about 4.3 points, and almost every B/C/D delta in the
report is smaller than that. The report's own summary line, "fact compression,
temporal validity and graph ranking each help some cases and hurt others", is
consistent with all three doing nothing at all.

The size of the sampling noise is visible inside the report itself. On
LongMemEval S the same arm A scores 78.20 on 500 questions and 72.00 on the
matched 100.

## 7. What the report does not contain

The report has no post-graph-rag arm, no Zep arm and no Graphiti arm. Its own
caveat says so: "It does not rerun Chandan, Zep, Mem0 or other providers under
the same models and costs." So nothing in it can be read as beating or losing
to Chandan. The only measured comparison against his system is Part 1's, and
that stands unchanged.

For the record, the comparison as it actually stands on LongMemEval S at a
4,000-token budget:

| system | all evidence in budget | answer accuracy | reader | judge |
|---|---:|---:|---|---|
| Part 1 ours_cheap | 0.938 | 0.844 | gpt-5.4 | gpt-5.4 |
| Part 1 ours_cheap | 0.938 | 0.558 | gemini-2.5-flash-lite | gpt-5.4 |
| Part 1 ours_cheap_norule | 0.800 | 0.532 | gemini-2.5-flash-lite | gpt-5.4 |
| Part 1 chandan_live | 0.574 | 0.624 | gpt-5.4 | gpt-5.4 |
| this run, Version A | 0.760 | 0.782 | gemini-3.6-flash | gemini-2.5-flash-lite |

On LongMemEval oracle this run reads 80.00. The published post-graph-rag
numbers for that split are 0.858 in his paper, 0.940 in his repository README
under his own judge panel, and 0.782 when his own repository regrades that run
under the official protocol with a gpt-4o judge. Part 1 recorded all three, and
that spread is why accuracy is never compared across papers here.

## What to fix before this report is used

1. Give the reader an output allowance large enough for its reasoning, or turn
   reasoning off. Then check `finish_reason` on every call and treat a
   truncated answer as a failed call, not a wrong answer.
2. Take the unit id out of the excerpt header on LongMemEval, or pseudonymise
   it, and rerun that dataset.
3. State the speaker rule decision, with the 0.938 against 0.800 measurement
   beside it, so the retrieval number is read correctly.
4. Either audit the cheap judge on this run's own answers, or stop reading
   differences of a few points.
5. Raise the B/C/D sample, or say plainly that 97 of 100 paired comparisons are
   consistent with no effect.

The good news is that none of this touches the indexing claim. Zero ingestion
model calls across 19 variants is measured and stands, and the truncation
defect is on the answering side, where a rerun is cheap.

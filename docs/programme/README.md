# The programme record

These are the working documents of the research, published verbatim rather than tidied up.
They are the primary sources behind the write-ups at
[agenticarchitectureskills.com/patterns](https://www.agenticarchitectureskills.com/patterns).

If you only read one, read `FINDINGS.md`.

## What each file is

| File | What it is | Read it if |
|---|---|---|
| `FINDINGS.md` | The single summary of what was asked, what held, and what did not | You want the whole programme in fifteen minutes |
| `hypothesis-register.md` | Every hypothesis with its verdict: confirmed, refined, contradicted, not supported | You want the scorecard, including the five that failed |
| `adversarial-review-2026-08-20.md` | An independent review told to reject the work, which scored the first sprint 1.5 out of 5 | You want to see what an unfriendly reader found |
| `preregistration-dynamic-taxonomy.md` | Hypotheses and decision rules, frozen before the experiments ran | You want to check the rules were not written to fit the answers |
| `sprint-r1-dilution.md` | The first sprint's working diary | You want the day-by-day, with the caveats below |
| `sprint-r2-dynamic.md` | The second sprint's working diary | The same |

## A warning about the two sprint diaries

They are append-only logs. **They still contain conclusions that were later withdrawn**, kept
that way on purpose so the record shows what was believed and when. They are a good way to
see how a result gets overturned and a bad way to learn what is true. For what is actually
true, use `FINDINGS.md` and the register.

## What is not here

The patent working files, the prior-art analyses and the commercial context stay private.
They are not part of the evidence for any published claim. Nothing in the write-ups depends
on them.

## The short version

The headline idea was that giving a document several purpose-specific embeddings beats giving
it one blended embedding. That is true, but mostly for a duller reason than we proposed:
having more vectors per document is what buys the improvement, and the clever purpose framing
adds something only when the questions target one aspect of a document. Where questions
concern a whole document, the clever version loses to plain chunking.

Two supporting claims shrank under measurement. A cheap pre-filter works but saves a linear
factor rather than an order of magnitude. The cost advantage of processing per discovered
topic is a large constant, not the widening scaling law we claimed, and the apparent widening
turned out to be our own summaries getting coarser as the corpus grew.

The most novel claim, that the right amount of variety in a result set depends on whether a
model or a person will read it, did not survive. The effect shrank as the judging model moved
further from the generator's own family and vanished across companies, which is the signature
of a judge marking its own homework.

One result nobody predicted turned out to be the most durable: harvesting varied results
reduced unsupported statements in generated summaries, and that held under judges from every
family tried.

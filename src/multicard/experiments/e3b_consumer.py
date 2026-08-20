"""E3b: does the right amount of diversity depend on who consumes the results?

This is the claim the whole programme exists to test, and the only one that
cannot be settled without a generative model. The retrieval half is already
measured: diversity policies buy subtopic coverage, and the ascending-similarity
harvest buys it more cheaply than MMR or a determinantal point process. What
remains is whether that coverage is worth anything, and to whom.

Design. The same gated pool, the same query, and the same synthesis model are
used throughout; the only thing that varies is the selection policy and the
consumer the output is written for. Two consumers:

  machine   the selection is assembled as evidence for a model that must
            synthesise an account of everything the material covers
  human     the same selection is rendered as a short brief for a person who
            will read it once and act on it

The primary endpoint is objective and needs no judge: the corpus is constructed,
so each task's ground-truth subtopics are known, and coverage of them in the
generated text can be counted by checking whether each subtopic's distinguishing
terms appear. A secondary endpoint counts unsupported content, material in the
output that no selected document contains, which is where a diversity policy
would be expected to do harm.

The prediction under test is an interaction, not a main effect: harvesting should
raise coverage for the machine consumer by more than it raises it for the human
consumer, and should cost more in focus for the human. A main effect in either
direction would not support the claim.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from ..index.encoder import Encoder
from ..llm.costmeter import CostMeter
from ..llm.vertex import GenerativeClient
from ..metrics.stats import compare, holm
from ..select.policies import outlier_harvest, top_k
from ..utils.seeds import set_seed
from .e3_diversity import build_diverse_task

# A consumer is defined by its budget as well as its purpose. A person reads a
# short brief once; a downstream model ingests a long context and can afford to
# be told everything. Earlier versions gave both the same word limit, which tested
# only the framing and found, unsurprisingly, no difference between them.
CONSUMER_SPECS = {
    "machine": {
        "words": 600,
        "max_output_tokens": 1600,
        "prompt": (
            "You are assembling an evidence summary that another system will use "
            "to answer questions about this material. Cover every distinct matter "
            "that appears in the excerpts, including minor ones. Be comprehensive "
            "rather than selective, and do not omit anything.\n\n"
            "Excerpts:\n{items}\n\nQuestion: {query}\n\n"
            "Write the summary as plain prose, up to {words} words."
        ),
    },
    "human": {
        "words": 100,
        "max_output_tokens": 400,
        "prompt": (
            "You are writing a short brief for a busy colleague who will read it "
            "once and act on it. Lead with what matters most and leave out "
            "anything peripheral.\n\n"
            "Excerpts:\n{items}\n\nQuestion: {query}\n\n"
            "Write the brief as plain prose, at most {words} words."
        ),
    },
}

POLICIES = ("relevance", "harvest")
CONSUMERS = ("machine", "human")

# Phrase matching is a blunt instrument in both directions: a bag-of-words rule
# fires on shared topic vocabulary without evidence, and a strict phrase rule
# misses ordinary paraphrase. This asks a model which facts a passage actually
# states. It is an extraction task, not a preference judgement, and the judge
# never learns which policy produced the text, so the usual self-preference and
# position biases do not apply.
JUDGE_PROMPT = (
    "Below is a summary, followed by a numbered list of candidate facts.\n\n"
    "For each fact, decide whether the summary actually states it. Paraphrase "
    "counts. Merely touching the same general topic does not count; the summary "
    "must convey that specific fact.\n\n"
    "Summary:\n{summary}\n\nCandidate facts:\n{facts}\n\n"
    "Reply with only the numbers of the facts the summary states, comma "
    "separated, or the word NONE."
)


def judged_coverage(gen, summary: str, facts: list[tuple[str, str, str]]) -> set[int]:
    """Which of the candidate facts does this summary actually state?"""
    if not summary.strip():
        return set()
    listing = "\n".join(f"{i + 1}. {a}, and the decision to {b}"
                         for i, (_, a, b) in enumerate(facts))
    r = gen.generate(JUDGE_PROMPT.format(summary=summary, facts=listing),
                     max_output_tokens=120)
    out = set()
    for tok in re.findall(r"\d+", r.text or ""):
        i = int(tok) - 1
        if 0 <= i < len(facts):
            out.add(i)
    return out


def subtopic_terms(fact: tuple[str, str, str]) -> tuple[str, str]:
    """The attribute and action phrases that identify one subtopic."""
    _, a, b = fact
    return (a.lower(), b.lower())


def covered(text: str, terms: tuple[str, str]) -> bool:
    """A subtopic counts as covered when both of its phrases appear together.

    An earlier version counted a majority of the individual words, which was far
    too loose: every document in a task comes from one topic pool and shares its
    vocabulary, so a summary of any excerpt from that pool used enough of those
    words to trigger a match. The measured consequence was absurd, outputs
    "covering" two and a half times as many subtopics as their own selection
    contained, which can only mean the metric was firing without evidence.

    Requiring both the attribute and the action phrase, as phrases rather than
    bags of words, ties a match to the specific fact rather than to the topic.
    """
    if not terms:
        return False
    low = text.lower()
    a, b = terms
    return a in low and b in low


def run(n_docs: int = 0, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gen_model: str = "gemini-2.5-flash", judge_model: str = "gemini-2.5-flash",
        n_tasks: int = 60, budget: int = 10,
        harvest_fraction: float = 0.4, max_usd: float = 5.0,
        out_dir: str = "results/e3b_consumer") -> dict:
    set_seed(seed)
    enc = Encoder(model_name=model)
    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model=gen_model, meter=meter, tier="vertex-flash")
    # A separate client for judging so the model can differ from the synthesiser
    # if wanted; blinded either way, since the judge sees only text and facts.
    # Judges may come from a different vendor entirely. Cross-family judging is
    # the check that matters most here, because two judges sharing a training
    # lineage can agree for reasons unrelated to the text they are scoring.
    if judge_model.startswith("azure:"):
        from ..llm.azure import AzureClient
        judge = AzureClient(model=judge_model.split(":", 1)[1] or None,
                            meter=meter, tier="azure-gpt54")
    else:
        judge = GenerativeClient(model=judge_model, meter=meter,
                                 tier=("vertex-pro" if "pro" in judge_model
                                       else "vertex-flash"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tasks = build_diverse_task(seed=seed)[:n_tasks]
    rows = []

    try:
        for t in tasks:
            ids = [d for d, _ in t["docs"]]
            texts = {d: x for d, x in t["docs"]}
            vecs = enc.encode([x for _, x in t["docs"]]).astype(np.float64)
            qv = enc.encode([t["query"]]).astype(np.float64)[0]
            rel = vecs @ qv

            keep = np.argsort(-rel)[: max(budget * 3, len(ids) // 2)]
            pool_ids = [ids[i] for i in keep]
            pool_rel = rel[keep]

            selections = {
                "relevance": top_k(pool_rel, budget),
                "harvest": outlier_harvest(pool_rel, budget,
                                           fraction=harvest_fraction),
            }

            # Ground truth: each subtopic's distinguishing terms are the words
            # of the (attribute, action) pair its documents were generated from.
            by_sub = {i: subtopic_terms(f) for i, f in enumerate(t["facts"])}

            for policy, sel in selections.items():
                chosen = [pool_ids[i] for i in sel]
                items = "\n".join(f"- {texts[d]}" for d in chosen)
                sel_vocab = set(re.findall(r"[a-z]{4,}",
                                           " ".join(texts[d] for d in chosen).lower()))
                sel_subs = set()
                for d in chosen:
                    sel_subs |= t["subtopics"].get(d, set())

                for consumer in CONSUMERS:
                    spec = CONSUMER_SPECS[consumer]
                    prompt = spec["prompt"].format(
                        items=items, query=t["query"], words=spec["words"])
                    r = gen.generate(prompt,
                                     max_output_tokens=spec["max_output_tokens"])
                    text = r.text

                    covered_subs = {s for s in t["all_subtopics"]
                                    if by_sub.get(s) and covered(text, by_sub[s])}
                    judged_subs = judged_coverage(judge, text, t["facts"])
                    out_words = set(re.findall(r"[a-z]{4,}", text.lower()))
                    unsupported = len(out_words - sel_vocab) / max(1, len(out_words))

                    rows.append({
                        "task": t["qid"], "policy": policy, "consumer": consumer,
                        "subtopics_total": len(t["all_subtopics"]),
                        "subtopics_in_selection": len(sel_subs),
                        "subtopics_covered_in_output": len(covered_subs),
                        "coverage_rate": round(
                            len(covered_subs) / max(1, len(t["all_subtopics"])), 6),
                        "judged_covered": len(judged_subs),
                        "judged_coverage_rate": round(
                            len(judged_subs) / max(1, len(t["all_subtopics"])), 6),
                        "judged_beyond_selection": len(judged_subs - sel_subs),
                        "unsupported_word_rate": round(unsupported, 6),
                        "output_words": len(text.split()),
                        "cached": int(r.cached),
                    })
    finally:
        gen.close()
        judge.close()

    # The claim is an interaction: does harvesting help the machine consumer more
    # than the human one?
    def series(policy, consumer, field):
        return [r[field] for r in rows if r["policy"] == policy
                and r["consumer"] == consumer]

    tests, pvals = {}, {}
    for consumer in CONSUMERS:
        for field in ("coverage_rate", "judged_coverage_rate",
                      "unsupported_word_rate"):
            a = series("harvest", consumer, field)
            b = series("relevance", consumer, field)
            if a and b and len(a) == len(b):
                t = compare(a, b)
                tests[f"{consumer}::{field}"] = t.as_dict()
                pvals[f"{consumer}::{field}"] = t.p_value
    sig = holm(pvals) if pvals else {}
    for k in tests:
        tests[k]["significant_holm_0.05"] = sig.get(k, False)

    interaction = None
    if all(f"{c}::judged_coverage_rate" in tests for c in CONSUMERS):
        m = tests["machine::judged_coverage_rate"]["mean_delta"]
        h = tests["human::judged_coverage_rate"]["mean_delta"]
        per_task_m = np.array(series("harvest", "machine", "judged_coverage_rate")) - \
            np.array(series("relevance", "machine", "judged_coverage_rate"))
        per_task_h = np.array(series("harvest", "human", "judged_coverage_rate")) - \
            np.array(series("relevance", "human", "judged_coverage_rate"))
        it = compare(per_task_m, per_task_h)
        interaction = {"machine_gain": m, "human_gain": h, **it.as_dict()}

    result = {
        "experiment": "e3b_consumer",
        "gen_model": gen_model,
        "encoder": model,
        "seed": seed,
        "n_tasks": len(tasks),
        "budget": budget,
        "harvest_fraction": harvest_fraction,
        "cost": meter.as_dict(),
        "wall_seconds": round(time.time() - t0, 1),
        "cells": {f"{p}/{c}": {
            "coverage_rate": float(np.mean(series(p, c, "coverage_rate"))),
            "judged_coverage_rate": float(np.mean(series(p, c, "judged_coverage_rate"))),
            "judged_beyond_selection": float(np.mean(series(p, c, "judged_beyond_selection"))),
            "unsupported_word_rate": float(np.mean(series(p, c, "unsupported_word_rate"))),
            "output_words": float(np.mean(series(p, c, "output_words"))),
        } for p in POLICIES for c in CONSUMERS},
        "tests": tests,
        "interaction": interaction,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(tasks)} tasks, {len(rows)} generations, "
          f"${meter.total_usd():.4f} of credit, {meter.total_calls()} calls\n")
    print(f"{'cell':<22}{'phrase':>9}{'judged':>9}{'beyond':>8}"
          f"{'unsup':>8}{'words':>7}")
    for p in POLICIES:
        for c in CONSUMERS:
            d = result["cells"][f"{p}/{c}"]
            print(f"{p + '/' + c:<22}{d['coverage_rate']:>9.3f}"
                  f"{d['judged_coverage_rate']:>9.3f}"
                  f"{d['judged_beyond_selection']:>8.2f}"
                  f"{d['unsupported_word_rate']:>8.3f}{d['output_words']:>7.0f}")
    print("\nharvest minus relevance, per consumer:")
    for k, t in tests.items():
        print(f"  {k:<34} {t['mean_delta']:+.3f} "
              f"[{t['ci_low']:+.3f},{t['ci_high']:+.3f}] p={t['p_value']:.4f}")
    if interaction:
        print(f"\ninteraction (machine gain minus human gain): "
              f"{interaction['mean_delta']:+.3f} "
              f"[{interaction['ci_low']:+.3f},{interaction['ci_high']:+.3f}] "
              f"p={interaction['p_value']:.4f}")
    return result

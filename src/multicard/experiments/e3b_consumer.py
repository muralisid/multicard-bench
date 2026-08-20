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

CONSUMER_PROMPTS = {
    "machine": (
        "You are assembling an evidence summary that another system will use to "
        "answer questions about this material. Cover every distinct matter that "
        "appears in the excerpts, including minor ones. Be comprehensive rather "
        "than selective.\n\nExcerpts:\n{items}\n\nQuestion: {query}\n\n"
        "Write the summary as plain prose, at most 200 words."
    ),
    "human": (
        "You are writing a short brief for a busy colleague who will read it once "
        "and act on it. Lead with what matters most and leave out anything "
        "peripheral.\n\nExcerpts:\n{items}\n\nQuestion: {query}\n\n"
        "Write the brief as plain prose, at most 200 words."
    ),
}

POLICIES = ("relevance", "harvest")
CONSUMERS = ("machine", "human")


def subtopic_terms(fact: tuple[str, str, str]) -> list[str]:
    """The words that distinguish one subtopic from its siblings."""
    _, a, b = fact
    return [w.lower() for w in re.findall(r"[a-z]{4,}", f"{a} {b}", re.I)]


def covered(text: str, terms: list[str]) -> bool:
    """A subtopic counts as covered when most of its distinguishing terms appear.

    A strict all-terms rule punishes ordinary paraphrase; a single-term rule fires
    on coincidence. Requiring a majority is the compromise, and it is applied
    identically to every condition so no policy is advantaged by the choice.
    """
    if not terms:
        return False
    low = text.lower()
    hits = sum(1 for t in terms if t in low)
    return hits >= max(1, (len(terms) + 1) // 2)


def run(n_docs: int = 0, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gen_model: str = "gemini-2.5-flash", n_tasks: int = 60, budget: int = 10,
        harvest_fraction: float = 0.4, max_usd: float = 5.0,
        out_dir: str = "results/e3b_consumer") -> dict:
    set_seed(seed)
    enc = Encoder(model_name=model)
    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model=gen_model, meter=meter, tier="vertex-flash")
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
                    prompt = CONSUMER_PROMPTS[consumer].format(
                        items=items, query=t["query"])
                    r = gen.generate(prompt, max_output_tokens=1200)
                    text = r.text

                    covered_subs = {s for s in t["all_subtopics"]
                                    if by_sub.get(s) and covered(text, by_sub[s])}
                    out_words = set(re.findall(r"[a-z]{4,}", text.lower()))
                    unsupported = len(out_words - sel_vocab) / max(1, len(out_words))

                    rows.append({
                        "task": t["qid"], "policy": policy, "consumer": consumer,
                        "subtopics_total": len(t["all_subtopics"]),
                        "subtopics_in_selection": len(sel_subs),
                        "subtopics_covered_in_output": len(covered_subs),
                        "coverage_rate": round(
                            len(covered_subs) / max(1, len(t["all_subtopics"])), 6),
                        "unsupported_word_rate": round(unsupported, 6),
                        "output_words": len(text.split()),
                        "cached": int(r.cached),
                    })
    finally:
        gen.close()

    # The claim is an interaction: does harvesting help the machine consumer more
    # than the human one?
    def series(policy, consumer, field):
        return [r[field] for r in rows if r["policy"] == policy
                and r["consumer"] == consumer]

    tests, pvals = {}, {}
    for consumer in CONSUMERS:
        for field in ("coverage_rate", "unsupported_word_rate"):
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
    if all(f"{c}::coverage_rate" in tests for c in CONSUMERS):
        m = tests["machine::coverage_rate"]["mean_delta"]
        h = tests["human::coverage_rate"]["mean_delta"]
        per_task_m = np.array(series("harvest", "machine", "coverage_rate")) - \
            np.array(series("relevance", "machine", "coverage_rate"))
        per_task_h = np.array(series("harvest", "human", "coverage_rate")) - \
            np.array(series("relevance", "human", "coverage_rate"))
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
    print(f"{'cell':<22}{'coverage':>10}{'unsupported':>13}{'words':>8}")
    for p in POLICIES:
        for c in CONSUMERS:
            d = result["cells"][f"{p}/{c}"]
            print(f"{p + '/' + c:<22}{d['coverage_rate']:>10.3f}"
                  f"{d['unsupported_word_rate']:>13.3f}{d['output_words']:>8.0f}")
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

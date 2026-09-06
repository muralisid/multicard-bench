"""Fixed question subsets for Part 1.

Implements docs/PART1-DESIGN.md (version 4) section 2, the fixed subsets, and
the answering cells of sections 5 and 6 that JUDGE_AUDIT is drawn over. Every
draw uses the bench generator, multicard.utils.seeds.rng, at seed 13. The
functions take plain (id, type) records so the tests run on synthetic ids;
scripts/part1_subsets.py feeds them the real loaders and writes
docs/part1/subsets.json.

Draw method, the same for every subset: the ids of one type are sorted, then
permuted with the generator, and the first quota ids are taken; types are
visited in sorted name order. This is the method of
multicard.data.multihoprag.stratified with a per-type quota in place of the
round robin. Each named subset is drawn with a fresh generator at the seed.
JUDGE_AUDIT is one draw over all its cells with a single generator, cells in
the fixed order reader, corpus, arm, so different cells get different
questions. Every subset list is stored in ORDER, so the first element of a
subset is its first question in ORDER.

The subsets:

- ORDER: a seeded permutation of every question id per corpus. The base list
  is the sorted ids, so the permutation does not depend on file order.
- GRAPHITI_150: 25 LongMemEval questions per type from the answerable
  questions (abstention ids excluded).
- CHANDAN_CAL_18: 3 per type from GRAPHITI_150.
- MHRAG_ANSWER: 150 MultiHop-RAG queries per type (600).
- READER_B_MHRAG: 50 per type from MHRAG_ANSWER (200).
- JUDGE_AUDIT: 10 percent of every (arm, corpus, reader) answering cell,
  stratified by type. LongMemEval cells are over all 500 questions, the
  graphiti cells over GRAPHITI_150. MultiHop-RAG cells are over MHRAG_ANSWER
  for Reader A and READER_B_MHRAG for Reader B.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ..utils.seeds import rng

SEED = 13

LONGMEMEVAL = "longmemeval"
MULTIHOPRAG = "multihoprag"
CORPORA = (LONGMEMEVAL, MULTIHOPRAG)

# Quotas fixed by design section 2.
GRAPHITI_PER_TYPE = 25
CAL_PER_TYPE = 3
MHRAG_ANSWER_PER_TYPE = 150
READER_B_MHRAG_PER_TYPE = 50
AUDIT_SHARE = 0.10

# Population labels of the answering cells.
POP_LONGMEMEVAL_ALL = "LONGMEMEVAL_ALL"
POP_GRAPHITI_150 = "GRAPHITI_150"
POP_MHRAG_ANSWER = "MHRAG_ANSWER"
POP_READER_B_MHRAG = "READER_B_MHRAG"

# Readers of design section 6. Reader A is the study chat model, Reader B is
# Azure gpt-5.4. chandan_full_uncut is read by post-graph-rag's own reader
# model (section 5) and is an answering cell of its own; it is not a Reader A
# row. design gap: the design gives that reader no label; "chandan_own" here.
READER_A = "reader_a"
READER_B = "reader_b"
CHANDAN_OWN = "chandan_own"

# Arms of design section 5 that have an answering row under Reader A, in the
# order section 5 lists them, each with the corpora it runs on. graphiti runs
# on LongMemEval only (GRAPHITI_150). The three "without the speaker rule"
# rows are LongMemEval only: the e5 rule quarantines assistant turns, and
# MultiHop-RAG has no speaker. design gap: section 5 leaves these rows and the
# planner and overlay ablations unnamed; the names below are used across
# Part 1. S5_overlay_R3 is S5_primary and is not listed twice.
READER_A_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ours_cheap", CORPORA),
    ("ours_cheap_norule", (LONGMEMEVAL,)),
    ("ours_sentence_norule", (LONGMEMEVAL,)),
    ("S4_static", CORPORA),
    ("S5_primary", CORPORA),
    ("S5_primary_norule", (LONGMEMEVAL,)),
    ("S5_noPGR", CORPORA),
    ("S5_planner_rules", CORPORA),
    ("S5_planner_oracle", CORPORA),
    ("S5_overlay_R0", CORPORA),
    ("S5_overlay_R2", CORPORA),
    ("S5_overlay_P0", CORPORA),
    ("S2_lazy", CORPORA),
    ("chandan_live", CORPORA),
    ("chandan_full", CORPORA),
    ("graphiti", (LONGMEMEVAL,)),
    ("oracle_full", CORPORA),
    ("closed_book", CORPORA),
)

# Reader B arms, design section 6, at B equals 4,000 only.
READER_B_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("S5_primary", CORPORA),
    ("chandan_live", CORPORA),
    ("chandan_full", CORPORA),
    ("graphiti", (LONGMEMEVAL,)),
    ("ours_cheap", CORPORA),
    ("oracle_full", CORPORA),
    ("closed_book", CORPORA),
)

# The reference line of section 5, answered by post-graph-rag's own reader.
CHANDAN_OWN_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("chandan_full_uncut", CORPORA),
)

READERS: tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...] = (
    (READER_A, READER_A_ARMS),
    (READER_B, READER_B_ARMS),
    (CHANDAN_OWN, CHANDAN_OWN_ARMS),
)


# ----------------------------------------------------------------------------
# Draw primitives
# ----------------------------------------------------------------------------

def seeded_order(ids: list[str], seed: int = SEED) -> list[str]:
    """ORDER for one corpus: the sorted ids permuted with rng(seed)."""
    base = sorted(ids)
    if len(base) != len(set(base)):
        raise ValueError("question ids are not unique")
    r_ = rng(seed)
    return [base[i] for i in r_.permutation(len(base))]


def by_type(records: list[tuple[str, str]]) -> dict[str, list[str]]:
    """{type: sorted ids}, types in sorted name order."""
    groups: dict[str, list[str]] = {}
    for qid, qtype in records:
        groups.setdefault(qtype, []).append(qid)
    return {t: sorted(groups[t]) for t in sorted(groups)}


def draw_by_type(records: list[tuple[str, str]], quota: dict[str, int],
                 r_: np.random.Generator) -> list[str]:
    """Per type, in sorted name order: permute the sorted ids with r_ and take
    the first quota[type]. Raises when a type has fewer ids than its quota."""
    groups = by_type(records)
    chosen: list[str] = []
    for qtype in sorted(quota):
        ids = groups.get(qtype, [])
        if quota[qtype] > len(ids):
            raise ValueError(f"type {qtype!r}: quota {quota[qtype]} but only {len(ids)} ids")
        perm = r_.permutation(len(ids))
        chosen.extend(ids[i] for i in perm[:quota[qtype]])
    return chosen


def proportional_quota(counts: dict[str, int], share: float) -> dict[str, int]:
    """A per-type quota that sums to round(share * total).

    Largest remainder: each type gets floor(share * count), and the remaining
    places go to the types with the largest fractional part, ties broken by
    type name. design gap: the design says 10 percent stratified by type and
    does not say how to round.
    """
    total = int(math.floor(share * sum(counts.values()) + 0.5))
    exact = {t: share * n for t, n in counts.items()}
    quota = {t: int(math.floor(x)) for t, x in exact.items()}
    rest = total - sum(quota.values())
    for t in sorted(counts, key=lambda t: (-(exact[t] - quota[t]), t))[:rest]:
        quota[t] += 1
    return quota


def in_order(ids: list[str], order: list[str]) -> list[str]:
    """The ids sorted by their position in ORDER."""
    pos = {q: i for i, q in enumerate(order)}
    return sorted(ids, key=pos.__getitem__)


# ----------------------------------------------------------------------------
# The subsets
# ----------------------------------------------------------------------------

def _population(arm: str, corpus: str, reader: str, lme: list[tuple[str, str]],
                graphiti: list[tuple[str, str]], mh_answer: list[tuple[str, str]],
                reader_b_mhrag: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    """The answering population of one cell (design sections 2, 5 and 6)."""
    if corpus == LONGMEMEVAL:
        if arm == "graphiti":
            return POP_GRAPHITI_150, graphiti
        return POP_LONGMEMEVAL_ALL, lme
    if reader == READER_B:
        return POP_READER_B_MHRAG, reader_b_mhrag
    return POP_MHRAG_ANSWER, mh_answer


def judge_audit(lme: list[tuple[str, str]], graphiti: list[tuple[str, str]],
                mh_answer: list[tuple[str, str]], reader_b_mhrag: list[tuple[str, str]],
                order: dict[str, list[str]], seed: int = SEED) -> dict:
    """JUDGE_AUDIT: 10 percent of every answering cell, stratified by type.

    One generator at the seed serves every cell; cells are visited in the
    order reader, corpus, arm as the constants list them.
    """
    r_ = rng(seed)
    cells: list[dict] = []
    for reader, arms in READERS:
        for corpus in CORPORA:
            for arm, corpora in arms:
                if corpus not in corpora:
                    continue
                label, pairs = _population(arm, corpus, reader, lme, graphiti,
                                           mh_answer, reader_b_mhrag)
                counts = dict(Counter(t for _, t in pairs))
                quota = proportional_quota(counts, AUDIT_SHARE)
                ids = in_order(draw_by_type(pairs, quota, r_), order[corpus])
                cells.append({
                    "arm": arm, "corpus": corpus, "reader": reader,
                    "population": label, "n_population": len(pairs),
                    "n": len(ids), "ids": ids,
                })
    return {
        "share": AUDIT_SHARE,
        "arms": {reader: [arm for arm, _ in arms] for reader, arms in READERS},
        "cells": cells,
    }


def build(lme: list[tuple[str, str, bool]], mh: list[tuple[str, str]],
          seed: int = SEED) -> dict:
    """Every subset of design section 2 from (id, type[, abstention]) records.

    lme: (question id, question type, abstention flag) per LongMemEval question.
    mh: (query id, question type) per MultiHop-RAG query.
    Returns the JSON payload written to docs/part1/subsets.json.
    """
    lme_pairs = [(q, t) for q, t, _ in lme]
    lme_types = dict(lme_pairs)
    mh_types = dict(mh)
    order = {
        LONGMEMEVAL: seeded_order(list(lme_types), seed),
        MULTIHOPRAG: seeded_order(list(mh_types), seed),
    }
    abstention = sorted(q for q, _, a in lme if a)
    answerable = [(q, t) for q, t, a in lme if not a]
    types_l = sorted(set(lme_types.values()))
    types_m = sorted(set(mh_types.values()))

    graphiti = in_order(
        draw_by_type(answerable, {t: GRAPHITI_PER_TYPE for t in types_l}, rng(seed)),
        order[LONGMEMEVAL])
    graphiti_pairs = [(q, lme_types[q]) for q in graphiti]
    cal = in_order(
        draw_by_type(graphiti_pairs, {t: CAL_PER_TYPE for t in types_l}, rng(seed)),
        order[LONGMEMEVAL])
    mh_answer = in_order(
        draw_by_type(mh, {t: MHRAG_ANSWER_PER_TYPE for t in types_m}, rng(seed)),
        order[MULTIHOPRAG])
    mh_answer_pairs = [(q, mh_types[q]) for q in mh_answer]
    reader_b = in_order(
        draw_by_type(mh_answer_pairs, {t: READER_B_MHRAG_PER_TYPE for t in types_m}, rng(seed)),
        order[MULTIHOPRAG])
    reader_b_pairs = [(q, mh_types[q]) for q in reader_b]
    audit = judge_audit(lme_pairs, graphiti_pairs, mh_answer_pairs, reader_b_pairs, order, seed)

    # The local set of design sections 2 and 9: single-session-user plus
    # single-session-assistant, answerable questions only. Derived, not drawn.
    local = in_order([q for q, t in answerable
                      if t in ("single-session-user", "single-session-assistant")],
                     order[LONGMEMEVAL])

    return {
        "meta": {
            "design": "docs/PART1-DESIGN.md version 4, section 2",
            "seed": seed,
            "script": "scripts/part1_subsets.py",
            "module": "multicard.part1.subsets",
            "draw": ("per type, in sorted type order: sorted ids permuted with "
                     "multicard.utils.seeds.rng(seed), first quota taken; a fresh "
                     "generator per named subset, one generator for all JUDGE_AUDIT "
                     "cells; every subset list is stored in ORDER"),
            "corpora": {
                LONGMEMEVAL: {"n": len(lme), "n_abstention": len(abstention),
                              "per_type": dict(sorted(Counter(lme_types.values()).items()))},
                MULTIHOPRAG: {"n": len(mh),
                              "per_type": dict(sorted(Counter(mh_types.values()).items()))},
            },
        },
        "types": {LONGMEMEVAL: dict(sorted(lme_types.items())),
                  MULTIHOPRAG: dict(sorted(mh_types.items()))},
        "abstention": {LONGMEMEVAL: abstention},
        "ORDER": order,
        "GRAPHITI_150": graphiti,
        "CHANDAN_CAL_18": cal,
        "MHRAG_ANSWER": mh_answer,
        "READER_B_MHRAG": reader_b,
        "JUDGE_AUDIT": audit,
        "derived": {
            # The Graphiti pilot question (section 2): first of GRAPHITI_150 in ORDER.
            "GRAPHITI_PILOT": graphiti[0] if graphiti else None,
            "LOCAL_120": local,
        },
    }


# ----------------------------------------------------------------------------
# Serialisation and loading
# ----------------------------------------------------------------------------

def serialise(payload: dict) -> bytes:
    """Canonical bytes: sorted keys, one-space indent, ASCII, trailing newline."""
    return (json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def counts(payload: dict) -> dict:
    """The sizes the script prints: per subset, per type, and the audit cells."""
    types = payload["types"]
    out: dict = {}
    for key, corpus in (("GRAPHITI_150", LONGMEMEVAL), ("CHANDAN_CAL_18", LONGMEMEVAL),
                        ("MHRAG_ANSWER", MULTIHOPRAG), ("READER_B_MHRAG", MULTIHOPRAG)):
        ids = payload[key]
        out[key] = {"n": len(ids),
                    "per_type": dict(sorted(Counter(types[corpus][q] for q in ids).items()))}
    out["ORDER"] = {c: len(v) for c, v in payload["ORDER"].items()}
    cells = payload["JUDGE_AUDIT"]["cells"]
    per_reader = Counter()
    for c in cells:
        per_reader[c["reader"]] += c["n"]
    out["JUDGE_AUDIT"] = {"n_cells": len(cells), "n_ids": sum(c["n"] for c in cells),
                          "per_reader": dict(sorted(per_reader.items()))}
    return out


def longmemeval_records(path: str | Path) -> list[tuple[str, str, bool]]:
    """(question id, question type, abstention) from the LongMemEval loader."""
    from ..data.longmemeval import load_longmemeval

    data = load_longmemeval(path)
    return [(x.qid, x.qtype, x.abstention) for x in data["instances"]]


def multihoprag_records(raw: str | Path) -> list[tuple[str, str]]:
    """(query id, question type) from the MultiHop-RAG loader."""
    from ..data import multihoprag as mh

    return [(q.qid, q.question_type) for q in mh.load_queries(Path(raw))]

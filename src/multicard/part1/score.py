"""Scoring for Part 1 (design sections 3 and 7).

Coverage rule, symmetric across arms: a marked LongMemEval turn is covered
when the rendered units together contain at least the lesser of half of the
turn's text and 2,000 characters of it, or contain a relation whose
provenance chunk does. A competitor chunk covers a turn through its turn
span (RenderedUnit.turn_chars, clipped by any truncation). A truncated
evidence turn is flagged and counted (section 3): truncated_evidence counts
evidence turns whose unit was truncated to fit the budget or cut at the
2,000-character limit, and cut_evidence the second kind alone.

JointRecall@B: every marked evidence turn covered inside the rendered
context. Candidate-level joint recall: the same rule over the candidate list
before the cut, on the units' full texts. Session-level JointRecall: every
evidence session has at least one rendered unit. Turn Recall@10, nDCG@10 and
session Recall@5 come from the bench metrics.ranking functions.

MultiHop-RAG: a document is hit when a rendered chunk contains its located
fact excerpt (fallback facts: any rendered chunk of the document). Fact-level
joint recall is every fact contained; document-level is every evidence
document hit.

Missing output: an empty ranking scores 0 on every metric and is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..experiments.e5_longmemeval import session_ranking
from ..metrics.ranking import ndcg_at_k, recall_at_k
from .render import RenderedContext, RenderedUnit, candidate_units_from_ids
from .units import DocTable, FactLocation, SessionTable, normalise

COVER_CHARS = 2000


def cover_threshold(turn_len: int) -> float:
    """The lesser of half of the turn's text and 2,000 characters."""
    return min(turn_len / 2.0, float(COVER_CHARS))


@dataclass
class TurnCoverage:
    turn_id: str
    covered: bool
    chars: int            # characters of the turn inside the units, summed
    via_relation: bool    # covered only through a relation's provenance chunk
    present: bool         # some unit holds part of the turn
    truncated: bool       # a unit holding the turn was truncated to fit the budget
    cut: bool = False     # a unit holding the turn was cut at the 2,000-character limit (section 3)


def turn_coverage(turn_id: str, units: list[RenderedUnit], turn_len: int) -> TurnCoverage:
    threshold = cover_threshold(turn_len)
    chars = sum(u.turn_chars.get(turn_id, 0) for u in units)
    present = any(turn_id in u.turn_chars or turn_id in u.provenance_chars for u in units)
    truncated = any(u.truncated and turn_id in u.turn_chars for u in units)
    cut = any(u.cut and turn_id in u.turn_chars for u in units)
    if present and chars >= threshold:
        return TurnCoverage(turn_id, True, chars, False, present, truncated, cut)
    via = any(u.provenance_chars.get(turn_id, 0) >= threshold and turn_id in u.provenance_chars
              for u in units)
    return TurnCoverage(turn_id, via, chars, via, present, truncated, cut)


def turn_lengths(tables: dict, turn_ids) -> dict[str, int]:
    """Full text length of each turn id from the session tables."""
    out = {}
    for t in turn_ids:
        sid, i = t.rsplit("#", 1)
        out[t] = len(tables[sid].turns[int(i)].text)
    return out


def joint_recall(evidence_turns, units: list[RenderedUnit],
                 lengths: dict[str, int]) -> tuple[float, list[TurnCoverage]]:
    """1.0 when every evidence turn is covered, else 0.0, with the per-turn detail."""
    cov = [turn_coverage(t, units, lengths[t]) for t in sorted(evidence_turns)]
    if not cov:
        return 0.0, cov
    return (1.0 if all(c.covered for c in cov) else 0.0), cov


def session_joint_recall(evidence_sessions, units: list[RenderedUnit]) -> float:
    """1.0 when every evidence session has at least one rendered unit."""
    if not evidence_sessions:
        return 0.0
    have = {s for u in units for s in u.sessions}
    return 1.0 if set(evidence_sessions) <= have else 0.0


def turn_ranking_from_units(units: list[RenderedUnit], lengths: dict[str, int] | None = None) -> list[str]:
    """Turn ids in first-appearance order over the units.

    Our units contribute their own turn. A competitor unit contributes the
    turns its full text covers under the coverage rule (lengths needed for
    the threshold; without lengths every spanned turn counts).
    """
    out: list[str] = []
    seen: set[str] = set()
    for u in units:
        # turn_spans are given in turn order by the caller, so dict order is turn order
        for t in list(u.turn_chars) + list(u.provenance_chars):
            n = max(u.turn_chars.get(t, 0), u.provenance_chars.get(t, 0))
            if lengths is not None and t in lengths and n < cover_threshold(lengths[t]):
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def session_ranking_from_units(units: list[RenderedUnit]) -> list[str]:
    """Sessions by first appearance among the units (for Graphiti, among the episodes its ranked facts cite)."""
    out: list[str] = []
    seen: set[str] = set()
    for u in units:
        for s in u.sessions:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


@dataclass
class QuestionScore:
    qid: str
    joint_recall: float
    candidate_joint_recall: float
    session_joint_recall: float
    turn_r10: float | None
    turn_ndcg10: float | None
    sess_r5: float
    rendered_tokens: int
    duplicate_share: float
    n_candidates: int
    n_rendered: int
    n_truncated: int
    truncated_evidence: int          # evidence turns rendered truncated (budget) or cut at 2,000 characters (section 3)
    uncovered: list[str] = field(default_factory=list)
    covered_via_relation: list[str] = field(default_factory=list)
    missing: bool = False
    cut_evidence: int = 0            # of those, evidence turns over 2,000 characters cut at the section 3 limit


def _empty_score(qid: str, session_level_only: bool, n_candidates: int = 0) -> QuestionScore:
    return QuestionScore(qid=qid, joint_recall=0.0, candidate_joint_recall=0.0, session_joint_recall=0.0,
                         turn_r10=None if session_level_only else 0.0,
                         turn_ndcg10=None if session_level_only else 0.0, sess_r5=0.0,
                         rendered_tokens=0, duplicate_share=0.0, n_candidates=n_candidates, n_rendered=0,
                         n_truncated=0, truncated_evidence=0, missing=True)


def score_longmemeval(qid: str, evidence_turns, evidence_sessions, rendered: RenderedContext,
                      tables: dict, turn_ranking: list[str] | None = None,
                      session_rank: list[str] | None = None,
                      session_level_only: bool = False) -> QuestionScore:
    """Every LongMemEval retrieval metric for one question and one arm.

    rendered comes from render() (our arms) or render_native() (competitors).
    tables holds the question's haystack SessionTables (no vectors needed).
    turn_ranking defaults to the candidate units' turns in rank order;
    session_rank defaults to the sessions of that ranking. session_level_only
    (Graphiti) leaves the turn metrics None.
    """
    if not rendered.candidates:
        return _empty_score(qid, session_level_only)
    lengths = turn_lengths(tables, evidence_turns)
    cands = rendered.candidate_units
    if cands is None:
        cands = candidate_units_from_ids(rendered.candidates, tables)
    jr, cov = joint_recall(evidence_turns, rendered.units, lengths)
    cjr, _ = joint_recall(evidence_turns, cands, lengths)
    rel_t = {t: 1.0 for t in evidence_turns}
    rel_s = {s: 1.0 for s in evidence_sessions}
    if turn_ranking is None:
        turn_ranking = turn_ranking_from_units(cands, lengths if rendered.candidate_units is not None else None)
    if session_rank is None:
        session_rank = session_ranking_from_units(cands) if rendered.candidate_units is not None \
            else session_ranking(turn_ranking)
    return QuestionScore(
        qid=qid, joint_recall=jr, candidate_joint_recall=cjr,
        session_joint_recall=session_joint_recall(evidence_sessions, rendered.units),
        turn_r10=None if session_level_only else recall_at_k(turn_ranking, rel_t, 10),
        turn_ndcg10=None if session_level_only else ndcg_at_k(turn_ranking, rel_t, 10),
        sess_r5=recall_at_k(session_rank, rel_s, 5),
        rendered_tokens=rendered.tokens, duplicate_share=rendered.duplicate_share,
        n_candidates=rendered.n_candidates, n_rendered=len(rendered.units), n_truncated=rendered.n_truncated,
        truncated_evidence=sum(1 for c in cov if c.truncated or c.cut),
        uncovered=[c.turn_id for c in cov if not c.covered],
        covered_via_relation=[c.turn_id for c in cov if c.via_relation],
        missing=False, cut_evidence=sum(1 for c in cov if c.cut))


# ----------------------------------------------------------------------------
# MultiHop-RAG
# ----------------------------------------------------------------------------
def _doc_text(units: list[RenderedUnit], doc_id: str) -> str:
    """Rendered chunk bodies of one document joined in chunk order, normalised."""
    # design gap: containment is checked on the document's rendered chunks
    # joined in chunk order, so a fact that crosses a chunk boundary counts
    # when both chunks are rendered
    parts = sorted((u for u in units if u.kind == "chunk" and u.container == doc_id),
                   key=lambda u: u.index)
    return normalise(" ".join(u.body for u in parts), strip_ends=False)


def fact_contained(loc: FactLocation, units: list[RenderedUnit]) -> bool:
    """A located fact is contained when the rendered chunks of its document
    together contain the normalised excerpt. A fallback fact is contained
    when any chunk of the document is rendered."""
    if loc.doc_id is None:
        return False
    text = _doc_text(units, loc.doc_id)
    if not text:
        return False
    if loc.fallback:
        return True
    return normalise(loc.fact) in text


@dataclass
class MhrQuestionScore:
    qid: str
    fact_joint_recall: float
    doc_joint_recall: float
    candidate_fact_joint_recall: float
    candidate_doc_joint_recall: float
    all_located: bool
    n_facts: int
    n_contained: int
    n_fallback: int
    doc_hits: dict[str, bool]
    rendered_tokens: int
    duplicate_share: float
    n_candidates: int
    n_rendered: int
    n_truncated: int
    missing: bool = False


def _fact_scores(locations: list[FactLocation], units: list[RenderedUnit]) -> tuple[float, float, dict, int]:
    contained = [fact_contained(l, units) for l in locations]
    docs = list(dict.fromkeys(l.doc_id for l in locations))
    # design gap: a document cited with two facts is hit when either fact is contained
    hits = {d: any(c for l, c in zip(locations, contained) if l.doc_id == d) for d in docs}
    fact_jr = 1.0 if locations and all(contained) else 0.0
    doc_jr = 1.0 if docs and all(hits.values()) else 0.0
    return fact_jr, doc_jr, hits, sum(contained)


def score_multihoprag(qid: str, locations: list[FactLocation], rendered: RenderedContext,
                      tables: dict[str, DocTable]) -> MhrQuestionScore:
    """Document hit and fact-level joint recall for one non-null query and one arm.

    locations comes from units.locate_query. tables holds the DocTables (used
    for our candidate list). all_located says whether the query belongs to
    the fact-level primary set.
    """
    n_fb = sum(1 for l in locations if l.fallback)
    all_located = bool(locations) and all(l.located for l in locations)
    if not rendered.candidates:
        return MhrQuestionScore(qid=qid, fact_joint_recall=0.0, doc_joint_recall=0.0,
                                candidate_fact_joint_recall=0.0, candidate_doc_joint_recall=0.0,
                                all_located=all_located, n_facts=len(locations), n_contained=0, n_fallback=n_fb,
                                doc_hits={d: False for d in dict.fromkeys(l.doc_id for l in locations)},
                                rendered_tokens=0, duplicate_share=0.0, n_candidates=0, n_rendered=0,
                                n_truncated=0, missing=True)
    cands = rendered.candidate_units
    if cands is None:
        cands = candidate_units_from_ids(rendered.candidates, tables)
    fact_jr, doc_jr, hits, n_contained = _fact_scores(locations, rendered.units)
    c_fact_jr, c_doc_jr, _, _ = _fact_scores(locations, cands)
    return MhrQuestionScore(
        qid=qid, fact_joint_recall=fact_jr, doc_joint_recall=doc_jr,
        candidate_fact_joint_recall=c_fact_jr, candidate_doc_joint_recall=c_doc_jr,
        all_located=all_located, n_facts=len(locations), n_contained=n_contained, n_fallback=n_fb,
        doc_hits=hits, rendered_tokens=rendered.tokens, duplicate_share=rendered.duplicate_share,
        n_candidates=rendered.n_candidates, n_rendered=len(rendered.units),
        n_truncated=rendered.n_truncated, missing=False)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
LME_FIELDS = ("joint_recall", "candidate_joint_recall", "session_joint_recall", "turn_r10",
              "turn_ndcg10", "sess_r5", "rendered_tokens", "duplicate_share", "n_candidates")
MHR_FIELDS = ("fact_joint_recall", "doc_joint_recall", "candidate_fact_joint_recall",
              "candidate_doc_joint_recall", "rendered_tokens", "duplicate_share", "n_candidates")


def summarise(scores: list, fields=LME_FIELDS) -> dict:
    """Means per field over the scores (None values skipped), with n and the missing count.

    A missing question scores 0 and is inside every mean, per section 5.
    """
    out = {"n": len(scores), "n_missing": sum(1 for s in scores if s.missing)}
    for f in fields:
        vals = [getattr(s, f) for s in scores if getattr(s, f) is not None]
        out[f] = float(np.mean(vals)) if vals else None
    if scores and hasattr(scores[0], "all_located"):
        out["n_all_located"] = sum(1 for s in scores if s.all_located)
        primary = [s.fact_joint_recall for s in scores if s.all_located]
        out["fact_joint_recall_all_located"] = float(np.mean(primary)) if primary else None
    if scores and hasattr(scores[0], "truncated_evidence"):
        out["truncated_evidence"] = sum(s.truncated_evidence for s in scores)
        out["cut_evidence"] = sum(getattr(s, "cut_evidence", 0) for s in scores)
    return out

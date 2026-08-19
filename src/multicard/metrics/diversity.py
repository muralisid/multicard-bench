"""Diversity metrics.

Sprint 2 of the literature review established the measurement rule this study
follows: a diversity benefit is measured with alpha-nDCG or subtopic coverage,
never with plain recall@k, because plain recall cannot see redundancy.

Subtopics are supplied by the corpus, not inferred: article sections for the
encyclopaedic corpus, source documents for the synthesis corpus.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def alpha_dcg(ranking: Sequence[str], subtopics: dict[str, set],
              alpha: float = 0.5, k: int = 20) -> float:
    """alpha-DCG of Clarke et al., SIGIR 2008.

    A document's gain for a subtopic is discounted by (1-alpha)^m, where m counts
    how many already-ranked documents covered that subtopic, so redundant
    coverage is worth progressively less.
    """
    seen: dict[object, int] = {}
    total = 0.0
    for rank, doc in enumerate(ranking[:k]):
        gain = 0.0
        for s in subtopics.get(doc, ()):  # missing document contributes nothing
            m = seen.get(s, 0)
            gain += (1 - alpha) ** m
            seen[s] = m + 1
        total += gain / math.log2(rank + 2)
    return total


def _ideal_alpha_dcg(candidates: Sequence[str], subtopics: dict[str, set],
                     alpha: float, k: int) -> float:
    """Greedy ideal ranking. Exact optimisation is NP-hard, and the greedy
    construction is the convention used by TREC; the paper states this."""
    seen: dict[object, int] = {}
    remaining = list(candidates)
    total = 0.0
    for rank in range(min(k, len(remaining))):
        best, best_gain = None, -1.0
        for doc in remaining:
            g = sum((1 - alpha) ** seen.get(s, 0) for s in subtopics.get(doc, ()))
            if g > best_gain:
                best, best_gain = doc, g
        if best is None:
            break
        for s in subtopics.get(best, ()):
            seen[s] = seen.get(s, 0) + 1
        total += best_gain / math.log2(rank + 2)
        remaining.remove(best)
    return total


def alpha_ndcg(ranking: Sequence[str], subtopics: dict[str, set],
               alpha: float = 0.5, k: int = 20,
               candidates: Sequence[str] | None = None) -> float:
    ideal_pool = candidates if candidates is not None else list(subtopics)
    ideal = _ideal_alpha_dcg(ideal_pool, subtopics, alpha, k)
    if ideal <= 0:
        return 0.0
    return alpha_dcg(ranking, subtopics, alpha, k) / ideal


def subtopic_recall(ranking: Sequence[str], subtopics: dict[str, set],
                    all_subtopics: set, k: int = 20) -> float:
    """S-recall: the fraction of the query's subtopics touched by the top k."""
    if not all_subtopics:
        return 0.0
    covered: set = set()
    for doc in ranking[:k]:
        covered |= set(subtopics.get(doc, ()))
    return len(covered & all_subtopics) / len(all_subtopics)


def intra_list_distance(ranking: Sequence[str], vectors: dict[str, "object"],
                        k: int = 20) -> float:
    """Mean pairwise cosine distance within the selection, a policy-side check
    that a diversity policy actually diversified. Not a quality metric."""
    import numpy as np

    vs = [vectors[d] for d in ranking[:k] if d in vectors]
    if len(vs) < 2:
        return 0.0
    m = np.asarray(vs, dtype=np.float64)
    sims = m @ m.T
    iu = np.triu_indices(len(vs), k=1)
    return float(1.0 - sims[iu].mean())

"""Ranking metrics.

Implemented here rather than imported so the paper can state exactly what was
computed, and cross-checked against hand-worked fixtures in tests/. Conventions:

- Rankings are lists of doc ids, best first, already deduplicated to document
  level (a card or chunk hit is folded into its parent document by max score
  before anything here is called).
- Relevance is a dict doc_id -> gain. Unjudged documents count as zero, which is
  why every results table also reports Judged@k.
"""

from __future__ import annotations

import math


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: list[str], rel: dict[str, float], k: int) -> float:
    gains = [float(rel.get(d, 0.0)) for d in ranking[:k]]
    ideal = sorted((float(v) for v in rel.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: list[str], rel: dict[str, float], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: list[str], rel: dict[str, float], k: int) -> float:
    for i, d in enumerate(ranking[:k]):
        if rel.get(d, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def judged_at_k(ranking: list[str], rel: dict[str, float], k: int) -> float:
    """Fraction of the top k that carries any judgement. Guards pooling bias."""
    if k == 0:
        return 0.0
    return sum(1 for d in ranking[:k] if d in rel) / min(k, len(ranking)) if ranking else 0.0


def success_at_k(ranking: list[str], rel: dict[str, float], k: int) -> float:
    return 1.0 if any(rel.get(d, 0.0) > 0 for d in ranking[:k]) else 0.0


def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[str]:
    """Reciprocal rank fusion. A document missing from a list contributes zero,
    not a rank of infinity, which is the convention the paper states."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for w, r in zip(weights, rankings):
        for i, d in enumerate(r):
            scores[d] = scores.get(d, 0.0) + w / (k + i + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]

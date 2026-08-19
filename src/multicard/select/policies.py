"""Selection policies: how a result subset is chosen from a gated candidate pool.

This module is the study's novel claim in code. Every policy takes the same pool
and the same budget and differs only in how it trades relevance against coverage.
The experiment then varies one further thing that the literature does not: who
consumes the result.

  top_k             pure relevance. The default everywhere.
  mmr               maximal marginal relevance, Carbonell and Goldstein 1998.
  dpp_greedy        fast greedy MAP inference for a determinantal point process,
                    Chen et al. 2018.
  cluster_round_robin  one item per cluster in turn, using structure already
                    computed by the second pass.
  outlier_harvest   a relevance head plus a quota drawn from the low-similarity
                    tail of the gated pool.
  coverage_stratified  partition by an attribute, walk each partition with a
                    stride rather than taking its head, and guarantee every
                    contrastive class a place.

All policies are deterministic given their inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_arrays(rel: Sequence[float], vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rel, dtype=np.float64)
    v = np.asarray(vecs, dtype=np.float64)
    if v.shape[0] != r.shape[0]:
        raise ValueError(f"relevance and vectors disagree: {r.shape} vs {v.shape}")
    return r, v


def top_k(rel: Sequence[float], n: int, **_) -> list[int]:
    r = np.asarray(rel, dtype=np.float64)
    return [int(i) for i in np.argsort(-r, kind="stable")[:n]]


def mmr(rel: Sequence[float], vecs: np.ndarray, n: int, lambda_: float = 0.5,
        **_) -> list[int]:
    """Iteratively pick argmax of lambda*rel - (1-lambda)*max_similarity_to_chosen.

    lambda_=1 reduces exactly to top_k, which the tests assert.
    """
    r, v = _as_arrays(rel, vecs)
    n = min(n, r.size)
    chosen: list[int] = []
    remaining = list(range(r.size))
    sim = v @ v.T
    while remaining and len(chosen) < n:
        if not chosen:
            best = max(remaining, key=lambda i: (r[i], -i))
        else:
            def score(i: int) -> float:
                return lambda_ * r[i] - (1 - lambda_) * max(sim[i, j] for j in chosen)
            best = max(remaining, key=lambda i: (score(i), -i))
        chosen.append(best)
        remaining.remove(best)
    return chosen


def dpp_greedy(rel: Sequence[float], vecs: np.ndarray, n: int, theta: float = 1.0,
               epsilon: float = 1e-10, **_) -> list[int]:
    """Fast greedy MAP for a DPP with kernel L = diag(q) S diag(q).

    q = exp(theta * rel / 2) so that L_ii = exp(theta * rel_i) * S_ii, and S is
    the Gram matrix of the (normalised) item vectors, which is positive
    semi-definite by construction. Implements the Cholesky-style incremental
    update of Chen et al. rather than recomputing determinants.
    """
    r, v = _as_arrays(rel, vecs)
    n = min(n, r.size)
    if n == 0:
        return []
    q = np.exp(theta * r / 2.0)
    S = v @ v.T
    L = (q[:, None] * S) * q[None, :]

    cis = np.zeros((n, r.size))
    di2 = np.copy(np.diag(L))
    selected: list[int] = []
    j = int(np.argmax(di2))
    selected.append(j)
    for it in range(1, n):
        ci_opt = cis[:it, j]
        di_opt = np.sqrt(max(di2[j], epsilon))
        eis = (L[j, :] - ci_opt @ cis[:it, :]) / di_opt
        cis[it, :] = eis
        di2 = di2 - np.square(eis)
        di2[selected] = -np.inf
        j = int(np.argmax(di2))
        if di2[j] < epsilon:
            break
        selected.append(j)
    return selected


def cluster_round_robin(rel: Sequence[float], labels: Sequence[int], n: int,
                        **_) -> list[int]:
    """Take the best remaining item from each cluster in turn.

    Clusters are visited in order of their best item's relevance, so the head of
    the selection still looks sensible, and the noise label (-1) is visited last
    rather than dropped.
    """
    r = np.asarray(rel, dtype=np.float64)
    lab = np.asarray(labels)
    buckets: dict[int, list[int]] = {}
    for i in np.argsort(-r, kind="stable"):
        buckets.setdefault(int(lab[i]), []).append(int(i))
    order = sorted(buckets, key=lambda c: (c == -1, -r[buckets[c][0]]))
    out: list[int] = []
    while len(out) < min(n, r.size):
        progressed = False
        for c in order:
            if buckets[c]:
                out.append(buckets[c].pop(0))
                progressed = True
                if len(out) >= min(n, r.size):
                    break
        if not progressed:
            break
    return out


def outlier_harvest(rel: Sequence[float], n: int, fraction: float = 0.3,
                    floor: float | None = None, **_) -> list[int]:
    """A relevance head plus a quota taken from the least similar of the pool.

    This is the policy the study proposes for a machine consumer that must
    synthesise: the head supplies the answer, the tail supplies coverage the head
    would never surface. The pool is assumed already gated, so even the tail is
    on topic; `floor` optionally enforces that explicitly.
    """
    r = np.asarray(rel, dtype=np.float64)
    n = min(n, r.size)
    n_out = int(round(fraction * n))
    n_head = n - n_out
    order_desc = list(np.argsort(-r, kind="stable"))
    head = [int(i) for i in order_desc[:n_head]]
    taken = set(head)
    tail_pool = [int(i) for i in reversed(order_desc) if int(i) not in taken]
    if floor is not None:
        tail_pool = [i for i in tail_pool if r[i] >= floor]
    return head + tail_pool[:n_out]


def coverage_stratified(rel: Sequence[float], strata: Sequence, n: int,
                        must_cover: Sequence | None = None, **_) -> list[int]:
    """Partition, stride through each partition, and guarantee contrastive classes.

    Within a stratum the pool is walked with a stride instead of taking its head,
    so the selection spreads across the stratum rather than concentrating at its
    top. Used where a machine consumer must generalise from a sample.
    """
    r = np.asarray(rel, dtype=np.float64)
    st = list(strata)
    n = min(n, r.size)
    buckets: dict[object, list[int]] = {}
    for i in np.argsort(-r, kind="stable"):
        buckets.setdefault(st[int(i)], []).append(int(i))

    out: list[int] = []
    if must_cover:
        for cls in must_cover:
            if cls in buckets and buckets[cls]:
                out.append(buckets[cls].pop(0))

    keys = sorted(buckets, key=lambda c: (-len(buckets[c]), str(c)))
    per = max(1, (n - len(out)) // max(1, len([k for k in keys if buckets[k]])))
    for key in keys:
        pool = buckets[key]
        if not pool:
            continue
        step = max(1, len(pool) // per)
        for i in range(0, len(pool), step):
            if len(out) >= n:
                break
            if pool[i] not in out:
                out.append(pool[i])
        if len(out) >= n:
            break
    # Top up by relevance if the strides did not fill the budget.
    for i in np.argsort(-r, kind="stable"):
        if len(out) >= n:
            break
        if int(i) not in out:
            out.append(int(i))
    return out[:n]


POLICIES = {
    "top_k": top_k,
    "mmr": mmr,
    "dpp": dpp_greedy,
    "cluster_rr": cluster_round_robin,
    "outlier_harvest": outlier_harvest,
    "coverage_stratified": coverage_stratified,
}

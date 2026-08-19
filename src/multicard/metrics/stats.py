"""Paired significance testing.

Every comparative claim in the paper passes through here. Pairing is at the query
level: the same query scored under two systems. Randomness is seeded, so a rerun
reproduces the reported p-value exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    ties: int
    losses: int
    effect_size: float  # standardised mean difference of the paired deltas

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def paired_permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10000,
                         seed: int = 13) -> float:
    """Two-sided paired permutation test on the per-query deltas.

    Under the null the sign of each delta is exchangeable, so the reference
    distribution is built by flipping signs at random.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(d.mean())
    if observed == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    # Add-one correction keeps the p-value strictly positive.
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                        alpha: float = 0.05, seed: int = 13) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean paired delta."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def compare(a, b, n_perm: int = 10000, n_boot: int = 10000, seed: int = 13) -> PairedResult:
    """Compare system a against system b over paired per-query scores."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    d = a - b
    lo, hi = paired_bootstrap_ci(a, b, n_boot=n_boot, seed=seed)
    sd = d.std(ddof=1)
    return PairedResult(
        n=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_delta=float(d.mean()),
        ci_low=lo,
        ci_high=hi,
        p_value=paired_permutation_p(a, b, n_perm=n_perm, seed=seed),
        wins=int(np.sum(d > 0)),
        ties=int(np.sum(d == 0)),
        losses=int(np.sum(d < 0)),
        effect_size=float(d.mean() / sd) if sd > 0 else 0.0,
    )


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni correction. Returns name -> significant at alpha."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out, rejected_all = {}, True
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        if rejected_all and p <= threshold:
            out[name] = True
        else:
            rejected_all = False
            out[name] = False
    return out

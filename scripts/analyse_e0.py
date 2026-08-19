#!/usr/bin/env python3
"""Significance testing and the dilution figure for S0.

Reads results/e0_dilution/per_query.csv, runs paired tests per k, writes
results/e0_dilution/significance.json and figures/fig1_dilution.pdf.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from multicard.metrics.stats import compare, holm

RES = Path("results/e0_dilution")
FIG = Path("paper/figures")


def load() -> dict[int, dict[str, list[float]]]:
    by_k: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(RES / "per_query.csv") as fh:
        for row in csv.DictReader(fh):
            k = int(row["k"])
            for col in ("sim_pooled", "sim_maxcard", "ndcg10_pooled", "ndcg10_maxcard",
                        "recall10_pooled", "recall10_maxcard", "n_relevant"):
                by_k[k][col].append(float(row[col]))
    return by_k


def main() -> None:
    by_k = load()
    summary = json.loads((RES / "metrics.json").read_text())
    ks = sorted(by_k)

    tests, pvals = {}, {}
    for k in ks:
        d = by_k[k]
        for metric in ("ndcg10", "recall10", "sim"):
            r = compare(d[f"{metric}_maxcard"], d[f"{metric}_pooled"])
            name = f"k{k}_{metric}"
            tests[name] = r.as_dict()
            pvals[name] = r.p_value
    sig = holm(pvals)
    for name in tests:
        tests[name]["significant_holm_0.05"] = sig[name]

    (RES / "significance.json").write_text(json.dumps(tests, indent=2))

    print(f"{'k':>3} {'nDCG@10 pooled':>15} {'maxcard':>9} {'delta':>8} "
          f"{'95% CI':>18} {'p':>9} {'win/tie/loss':>14} {'rel/query':>10}")
    for k in ks:
        t = tests[f"k{k}_ndcg10"]
        print(f"{k:>3} {t['mean_b']:>15.3f} {t['mean_a']:>9.3f} {t['mean_delta']:>+8.3f} "
              f"[{t['ci_low']:+.3f},{t['ci_high']:+.3f}] {t['p_value']:>9.4f} "
              f"{t['wins']:>4}/{t['ties']}/{t['losses']:<5} "
              f"{np.mean(by_k[k]['n_relevant']):>10.1f}")

    # ---- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {r["k"]: r for r in summary["rows"]}
    sim_p = [rows[k]["sim_pooled"] for k in ks]
    sim_m = [rows[k]["sim_maxcard"] for k in ks]
    ratio = [rows[k]["ratio_measured"] for k in ks]
    theory = [1 / np.sqrt(k) for k in ks]
    nd_p = [rows[k]["ndcg10_pooled"] for k in ks]
    nd_m = [rows[k]["ndcg10_maxcard"] for k in ks]

    # Fit the measured decay exponent: ratio ~ k^-alpha
    lk = np.log(np.array(ks[1:], dtype=float))
    lr = np.log(np.array(ratio[1:]))
    alpha = float(-np.polyfit(lk, lr, 1)[0])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    plt.rcParams.update({"font.size": 9})

    ax = axes[0]
    ax.plot(ks, sim_m, "o-", color="#1b6ca8", label="max over cards")
    ax.plot(ks, sim_p, "s--", color="#c0392b", label="pooled single vector")
    ax.set_xlabel("aspects per document, k")
    ax.set_ylabel("mean cosine to relevant documents")
    ax.set_title("(a) aspect signal")
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ks, ratio, "o-", color="#c0392b", label="measured")
    ax.plot(ks, theory, "k:", label=r"theory $1/\sqrt{k}$")
    ax.plot(ks, [k ** -alpha for k in ks], "--", color="#7f8c8d",
            label=rf"fit $k^{{-{alpha:.2f}}}$")
    ax.set_xlabel("aspects per document, k")
    ax.set_ylabel("pooled / max-card similarity")
    ax.set_title("(b) measured dilution vs theory")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ks, nd_m, "o-", color="#1b6ca8", label="max over cards")
    ax.plot(ks, nd_p, "s--", color="#c0392b", label="pooled single vector")
    ax.set_xlabel("aspects per document, k")
    ax.set_ylabel("nDCG@10")
    ax.set_title("(c) retrieval consequence")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_dilution.{ext}", dpi=200, bbox_inches="tight")

    print(f"\nfitted decay exponent alpha = {alpha:.3f} (theory 0.5)")
    print(f"figure written to {FIG}/fig1_dilution.pdf")


if __name__ == "__main__":
    main()

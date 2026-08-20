"""E2a: can a cheap gate separate on-purpose from off-purpose material?

Register row P3 asks whether the pass-one gate, which the whole economic argument
depends on, actually discriminates on a corpus it was not designed for. A gate
that cannot separate is not a filter, it is a random sample, and the two-pass
design collapses.

Labels here are structural rather than annotated, which keeps the experiment free
and free of annotator noise: on-purpose items are Enron business email, and
off-purpose items are Usenet posts from unrelated newsgroups. The provenance of
each item is known with certainty, so the labels are exact.

The limitation of that choice is stated rather than hidden. Separating corporate
email from Usenet is an easier problem than separating on-purpose from off-purpose
material inside one corpus, which is the situation a deployed gate faces. This
experiment therefore establishes an upper bound and a floor: if the gate failed
here it would be worthless, and passing here is necessary but not sufficient. The
within-corpus question needs a human-labelled sample and is deferred to the
labelling gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..data.enron import parse as parse_enron
from ..gate.composite import (ENRON_PURPOSE_ANCHORS_CANDIDATE, apply_gate,
                              component_scores, evaluate_gate, feature_matrix,
                              fit_gate)
from ..index.encoder import Encoder
from ..utils.seeds import set_seed

OFF_DOMAIN_GROUPS = [
    "rec.sport.baseball", "sci.space", "talk.politics.mideast",
    "comp.graphics", "rec.autos", "soc.religion.christian",
]


def run(n_docs: int = 4000, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        target_recall: float = 0.95, out_dir: str = "results/e2_gate") -> dict:
    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from sklearn.datasets import fetch_20newsgroups

    on = [m.text for m in parse_enron(target=20000)][:n_docs]
    news = fetch_20newsgroups(subset="all", categories=OFF_DOMAIN_GROUPS,
                              remove=("headers", "footers", "quotes"),
                              random_state=seed)
    off = [t.strip() for t in news.data if len(t.split()) >= 50][:n_docs]
    n = min(len(on), len(off))
    on, off = on[:n], off[:n]
    texts = on + off
    labels = np.array([1] * len(on) + [0] * len(off))
    print(f"on-purpose {len(on)}, off-purpose {len(off)}")

    enc = Encoder(model_name=model)
    vecs = enc.encode(texts).astype(np.float64)
    anchors = {a.key: enc.encode(a.phrases).astype(np.float64)
               for a in ENRON_PURPOSE_ANCHORS_CANDIDATE}
    top_k = {a.key: a.top_k for a in ENRON_PURPOSE_ANCHORS_CANDIDATE}
    order = [a.key for a in ENRON_PURPOSE_ANCHORS_CANDIDATE]

    comps = component_scores(vecs, anchors, top_k)
    feats = feature_matrix(comps, order)

    # Fit on one half, report on the held-out half. Fitting and evaluating on the
    # same data would inflate every number here.
    rs = np.random.default_rng(seed)
    idx = rs.permutation(len(texts))
    split = len(idx) // 2
    tr, te = idx[:split], idx[split:]

    fit = fit_gate(feats[tr], labels[tr], order, target_recall=target_recall, seed=seed)
    scores_te = apply_gate(feats[te], fit, order)
    metrics = evaluate_gate(scores_te, labels[te], fit)

    # An unweighted gate is the honest baseline: does fitting the weights help?
    flat = type(fit)(weights={k: 1.0 / len(order) for k in order}, bias=0.0,
                     threshold=0.0, target_recall=target_recall)
    flat_scores_tr = apply_gate(feats[tr], flat, order)
    flat.threshold = float(np.quantile(flat_scores_tr[labels[tr] == 1],
                                       1 - target_recall))
    flat_metrics = evaluate_gate(apply_gate(feats[te], flat, order), labels[te], flat)

    # How much of the corpus survives at each recall target? The economic
    # argument for a two-pass design is proportional to what the gate discards,
    # so this trade is reported rather than left implicit at one operating point.
    sweep = []
    for tr_recall in (0.80, 0.90, 0.95, 0.98, 0.99):
        f = fit_gate(feats[tr], labels[tr], order, target_recall=tr_recall, seed=seed)
        m = evaluate_gate(apply_gate(feats[te], f, order), labels[te], f)
        sweep.append({
            "target_recall": tr_recall,
            "recall": round(m["recall_at_threshold"], 4),
            "precision": round(m["precision_at_threshold"], 4),
            "survival_rate": round(m["survival_rate"], 4),
            "discarded_fraction": round(1 - m["survival_rate"], 4),
        })

    result = {
        "experiment": "e2_gate",
        "recall_cost_sweep": sweep,
        "model": model,
        "seed": seed,
        "n_on_purpose": len(on),
        "n_off_purpose": len(off),
        "target_recall": target_recall,
        "anchor_sets": order,
        "fitted_weights": fit.weights,
        "fitted_bias": fit.bias,
        "threshold": fit.threshold,
        "held_out": metrics,
        "unweighted_baseline": flat_metrics,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    import csv
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_index", "label", "gate_score", "passed"])
        for i, (li, sc) in enumerate(zip(labels[te], scores_te)):
            w.writerow([int(te[i]), int(li), round(float(sc), 6),
                        int(sc >= fit.threshold)])

    print(f"\nheld-out ROC-AUC {metrics['roc_auc']:.3f}  PR-AUC {metrics['pr_auc']:.3f}")
    print(f"at the recall-{target_recall:.2f} threshold: "
          f"recall {metrics['recall_at_threshold']:.3f}, "
          f"precision {metrics['precision_at_threshold']:.3f}, "
          f"survival {metrics['survival_rate']:.3f}")
    print(f"unweighted baseline ROC-AUC {flat_metrics['roc_auc']:.3f}")
    print("\nrecall target -> what the gate actually discards:")
    for r in sweep:
        print(f"  target {r['target_recall']:.2f}: recall {r['recall']:.3f}, "
              f"precision {r['precision']:.3f}, discards {r['discarded_fraction']:.1%}")
    print("fitted weights: " + ", ".join(f"{k}={v:+.3f}" for k, v in fit.weights.items()))
    return result

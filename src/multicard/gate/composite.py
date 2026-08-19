"""The pass-one relevancy gate.

A cheap, high-recall filter that decides which items enter a purpose's candidate
pool before anything expensive touches them. Structurally it is a multi-prototype
classifier: several anchor sets, each describing one facet of the purpose, and a
score that combines them.

Two properties matter for the study. It must be cheap, because it runs over every
item in the corpus, so it uses only embeddings already computed. And its weights
and threshold must be fitted and reported rather than asserted, because a gate
whose constants come from somewhere else is not reproducible. Items below the
threshold are quarantined, never deleted, so a later gate revision can readmit
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnchorSet:
    key: str
    phrases: list[str]
    top_k: int = 3   # soft max-pool: mean of the k best anchor similarities


@dataclass
class GateFit:
    weights: dict[str, float]
    bias: float
    threshold: float
    target_recall: float
    metrics: dict = field(default_factory=dict)


def component_scores(item_vecs: np.ndarray, anchors: dict[str, np.ndarray],
                     top_k: dict[str, int]) -> dict[str, np.ndarray]:
    """One score per anchor set per item: the mean of its top-k similarities.

    Taking a few best anchors rather than the single best is deliberate: one
    lucky anchor match is noise, while agreement among several is signal.
    """
    out = {}
    for key, mat in anchors.items():
        sims = item_vecs @ mat.T                       # (n_items, n_anchors)
        k = min(top_k.get(key, 3), sims.shape[1])
        part = np.partition(sims, -k, axis=1)[:, -k:]
        out[key] = part.mean(axis=1)
    return out


def feature_matrix(components: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    return np.column_stack([components[k] for k in order])


def fit_gate(features: np.ndarray, labels: np.ndarray, order: list[str],
             target_recall: float = 0.95, seed: int = 13) -> GateFit:
    """Fit weights by logistic regression and pick the threshold by recall.

    The threshold is not tuned for accuracy. A pass-one gate exists to lose as
    little relevant material as possible while discarding most of the corpus, so
    it is set at the lowest score that still retains `target_recall` of the
    positives on the fitting data, and its cost is reported as the fraction of
    the corpus that survives.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(features, labels)
    scores = clf.decision_function(features)

    pos = scores[labels == 1]
    if pos.size == 0:
        raise ValueError("cannot fit a gate with no positive examples")
    threshold = float(np.quantile(pos, 1.0 - target_recall))

    return GateFit(
        weights={k: float(w) for k, w in zip(order, clf.coef_[0])},
        bias=float(clf.intercept_[0]),
        threshold=threshold,
        target_recall=target_recall,
    )


def apply_gate(features: np.ndarray, fit: GateFit, order: list[str]) -> np.ndarray:
    w = np.array([fit.weights[k] for k in order], dtype=np.float64)
    return features @ w + fit.bias


def evaluate_gate(scores: np.ndarray, labels: np.ndarray, fit: GateFit) -> dict:
    """Report the numbers a practitioner needs before trusting a gate."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    passed = scores >= fit.threshold
    tp = int(np.sum(passed & (labels == 1)))
    fn = int(np.sum(~passed & (labels == 1)))
    fp = int(np.sum(passed & (labels == 0)))
    tn = int(np.sum(~passed & (labels == 0)))
    n_pos = max(1, tp + fn)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "threshold": fit.threshold,
        "recall_at_threshold": tp / n_pos,
        "precision_at_threshold": tp / max(1, tp + fp),
        "survival_rate": float(np.mean(passed)),
        "quarantined": int(np.sum(~passed)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


# Candidate anchor sets for the Enron purpose "commercial energy trading and
# operations". Curation at gate G1; nothing published depends on these until a
# human has reviewed them against a sample.
ENRON_PURPOSE_ANCHORS_CANDIDATE = [
    AnchorSet("domain", [
        "natural gas pipeline capacity and transportation",
        "power generation and electricity markets",
        "energy trading desk activity",
        "wholesale commodity markets",
    ]),
    AnchorSet("objective", [
        "closing a commercial transaction with a counterparty",
        "managing exposure and positions on the book",
        "negotiating contract terms and pricing",
        "operational scheduling of deliveries",
    ]),
    AnchorSet("intent", [
        "requesting approval or a decision",
        "reporting a result or a status",
        "raising a problem that needs resolution",
        "coordinating work between teams",
    ]),
]

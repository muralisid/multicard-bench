"""Tests for the pass-one relevancy gate.

The gate's job is high recall at low cost, so the tests check that its threshold
honours the recall target and that its reporting is honest about the cost.
"""

import numpy as np
import pytest

from multicard.gate.composite import (GateFit, apply_gate, component_scores,
                                      evaluate_gate, feature_matrix, fit_gate)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_component_scores_soft_pool_over_anchors():
    items = unit([[1.0, 0.0], [0.0, 1.0]])
    anchors = {"d": unit([[1.0, 0.0], [0.9, 0.1]])}
    out = component_scores(items, anchors, {"d": 2})
    assert out["d"][0] > out["d"][1]
    # mean of the two anchor similarities, not the max
    assert out["d"][0] < 1.0


def test_component_scores_respects_top_k():
    items = unit([[1.0, 0.0]])
    # one strong anchor, two weak: top-1 is high, top-3 is dragged down
    anchors = {"d": unit([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])}
    top1 = component_scores(items, anchors, {"d": 1})["d"][0]
    top3 = component_scores(items, anchors, {"d": 3})["d"][0]
    assert top1 > top3


def _separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.6, 0.15, size=(n, 3))
    neg = rng.normal(0.2, 0.15, size=(n, 3))
    X = np.vstack([pos, neg])
    y = np.array([1] * n + [0] * n)
    return X, y


def test_fit_gate_hits_its_recall_target():
    X, y = _separable()
    order = ["domain", "objective", "intent"]
    fit = fit_gate(X, y, order, target_recall=0.95)
    scores = apply_gate(X, fit, order)
    m = evaluate_gate(scores, y, fit)
    assert m["recall_at_threshold"] >= 0.94
    assert m["roc_auc"] > 0.9


def test_gate_reports_the_cost_of_its_recall():
    X, y = _separable()
    order = ["domain", "objective", "intent"]
    m = evaluate_gate(apply_gate(X, fit_gate(X, y, order), order), y,
                      fit_gate(X, y, order))
    # survival rate is the fraction of the corpus that still needs processing
    assert 0.0 < m["survival_rate"] < 1.0
    assert m["quarantined"] > 0
    assert sum(m["confusion"].values()) == len(y)


def test_higher_recall_target_lowers_the_threshold():
    X, y = _separable()
    order = ["domain", "objective", "intent"]
    lo = fit_gate(X, y, order, target_recall=0.80)
    hi = fit_gate(X, y, order, target_recall=0.99)
    assert hi.threshold < lo.threshold


def test_fit_gate_refuses_a_corpus_with_no_positives():
    X = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValueError):
        fit_gate(X, np.zeros(10, dtype=int), ["a", "b", "c"])


def test_feature_matrix_column_order_is_the_declared_order():
    comps = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    M = feature_matrix(comps, ["b", "a"])
    assert list(M[0]) == [3.0, 1.0]

"""Golden-value tests for the metric implementations.

Values are worked by hand in the comments so a reader can check them without
running anything. These guard the numbers that end up in the paper.
"""

import math

import numpy as np
import pytest

from multicard.metrics.ranking import (dcg, judged_at_k, mrr_at_k, ndcg_at_k,
                                       recall_at_k, rrf, success_at_k)
from multicard.metrics.stats import compare, holm, paired_permutation_p


def test_dcg_hand_worked():
    # gains [1, 0, 1] -> 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5
    assert dcg([1, 0, 1]) == pytest.approx(1.5)


def test_ndcg_hand_worked():
    # ranking a,b,c with a and c relevant.
    # DCG  = 1 + 0 + 1/2                     = 1.5
    # IDCG = 1 + 1/log2(3) = 1 + 0.6309...   = 1.63093
    ranking, rel = ["a", "b", "c"], {"a": 1.0, "c": 1.0}
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, rel, 10) == pytest.approx(expected)
    assert expected == pytest.approx(0.91972, abs=1e-5)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["a"], {}, 5) == 0.0
    assert ndcg_at_k([], {"a": 1.0}, 5) == 0.0


def test_recall_counts_unretrieved_relevant():
    # relevant {a, e}; top-2 of [a,b,c,d] is {a,b}; intersection {a} -> 0.5
    assert recall_at_k(["a", "b", "c", "d"], {"a": 1.0, "e": 1.0}, 2) == pytest.approx(0.5)


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(["x", "y", "a"], {"a": 1.0}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y"], {"a": 1.0}, 10) == 0.0


def test_judged_and_success():
    # a and b carry judgements (b judged non-relevant), c does not -> 2/3
    assert judged_at_k(["a", "b", "c"], {"a": 1.0, "b": 0.0}, 3) == pytest.approx(2 / 3)
    assert success_at_k(["a", "b"], {"b": 1.0}, 2) == 1.0
    assert success_at_k(["a", "b"], {"c": 1.0}, 2) == 0.0


def test_rrf_hand_worked():
    # k=60. a: 1/61 + 1/63 = 0.0322664; c: 1/63 + 1/61 = same; b: 2/62 = 0.0322581
    # so a and c tie above b, and the id tiebreak puts a first.
    assert rrf([["a", "b", "c"], ["c", "b", "a"]], k=60) == ["a", "c", "b"]


def test_rrf_missing_document_contributes_zero():
    # d appears in only one list and must still rank below documents in both.
    out = rrf([["a", "b"], ["a", "d"]], k=60)
    assert out[0] == "a"
    assert set(out) == {"a", "b", "d"}


def test_rrf_is_invariant_to_list_order():
    r1, r2 = ["a", "b", "c"], ["c", "a", "b"]
    assert rrf([r1, r2], k=60) == rrf([r2, r1], k=60)


def test_permutation_p_is_one_for_identical_systems():
    x = np.array([0.1, 0.5, 0.9, 0.3])
    assert paired_permutation_p(x, x.copy()) == 1.0


def test_permutation_p_is_small_for_a_consistent_win():
    a = np.array([0.9] * 30)
    b = np.array([0.1] * 30)
    assert paired_permutation_p(a, b, n_perm=2000, seed=13) < 0.01


def test_compare_reports_direction_and_counts():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.4, 0.6, 0.9])
    r = compare(a, b, n_perm=1000, n_boot=1000)
    assert r.n == 3
    assert r.wins == 1 and r.ties == 1 and r.losses == 1
    assert r.mean_delta == pytest.approx((0.1 + 0.0 - 0.2) / 3)


def test_compare_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compare(np.zeros(3), np.zeros(4))


def test_holm_step_down():
    # m=3. 0.001 <= 0.05/3 rejected; 0.04 > 0.05/2 stops the chain.
    out = holm({"x": 0.001, "y": 0.04, "z": 0.5})
    assert out == {"x": True, "y": False, "z": False}

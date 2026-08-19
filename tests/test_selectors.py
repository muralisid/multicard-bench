"""Property tests for the selection policies.

These guard the claims the paper makes about what each policy does, and catch the
classic implementation errors (a lambda convention flipped, a DPP kernel that is
not PSD, a round robin that silently drops the noise cluster).
"""

import numpy as np
import pytest

from multicard.metrics.diversity import (alpha_dcg, alpha_ndcg, intra_list_distance,
                                         subtopic_recall)
from multicard.select.policies import (cluster_round_robin, coverage_stratified,
                                       dpp_greedy, mmr, outlier_harvest, top_k)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_top_k_is_relevance_order():
    assert top_k([0.1, 0.9, 0.5], 2) == [1, 2]


def test_mmr_with_lambda_one_equals_top_k():
    rel = [0.2, 0.9, 0.5, 0.7]
    vecs = unit(np.eye(4))
    assert mmr(rel, vecs, 3, lambda_=1.0) == top_k(rel, 3)


def test_mmr_with_low_lambda_avoids_a_near_duplicate():
    # items 0 and 1 are identical; 2 is orthogonal and slightly less relevant.
    vecs = unit([[1, 0], [1, 0], [0, 1]])
    rel = [0.9, 0.85, 0.8]
    assert mmr(rel, vecs, 2, lambda_=1.0) == [0, 1]      # relevance only
    assert mmr(rel, vecs, 2, lambda_=0.2) == [0, 2]      # diversity kicks in


def test_dpp_with_orthogonal_items_follows_relevance():
    vecs = unit(np.eye(3))
    assert dpp_greedy([0.1, 0.9, 0.5], vecs, 3)[0] == 1


def test_dpp_avoids_duplicates():
    vecs = unit([[1, 0], [1, 0], [0, 1]])
    out = dpp_greedy([0.9, 0.9, 0.5], vecs, 2)
    assert 2 in out, "a DPP must prefer the dissimilar item over a duplicate"


def test_dpp_returns_at_most_n_unique_items():
    vecs = unit(np.random.default_rng(0).normal(size=(8, 4)))
    out = dpp_greedy(np.linspace(0.1, 0.9, 8), vecs, 4)
    assert len(out) <= 4 and len(set(out)) == len(out)


def test_cluster_round_robin_spreads_across_clusters():
    rel = [0.9, 0.85, 0.8, 0.4]
    labels = [0, 0, 1, 2]
    out = cluster_round_robin(rel, labels, 3)
    assert len(set(labels[i] for i in out)) == 3


def test_cluster_round_robin_visits_noise_last_but_does_not_drop_it():
    rel = [0.9, 0.8, 0.7]
    labels = [-1, 0, 1]
    out = cluster_round_robin(rel, labels, 3)
    assert out[-1] == 0  # index 0 carries the noise label
    assert set(out) == {0, 1, 2}


def test_outlier_harvest_mixes_head_and_tail():
    rel = [0.9, 0.8, 0.7, 0.2, 0.1]
    out = outlier_harvest(rel, n=4, fraction=0.5)
    assert out[:2] == [0, 1]          # relevance head
    assert set(out[2:]) == {4, 3}     # least similar of the remainder


def test_outlier_harvest_with_zero_fraction_is_top_k():
    rel = [0.9, 0.8, 0.7, 0.2]
    assert outlier_harvest(rel, n=3, fraction=0.0) == top_k(rel, 3)


def test_outlier_harvest_respects_a_floor():
    rel = [0.9, 0.8, 0.05, 0.02]
    out = outlier_harvest(rel, n=3, fraction=0.34, floor=0.5)
    assert all(rel[i] >= 0.5 for i in out[:2])


def test_coverage_stratified_covers_every_stratum():
    rel = [0.9, 0.85, 0.8, 0.75, 0.2]
    strata = ["a", "a", "a", "b", "c"]
    out = coverage_stratified(rel, strata, 3)
    assert len(set(strata[i] for i in out)) >= 2


def test_coverage_stratified_guarantees_must_cover_classes():
    rel = [0.9, 0.9, 0.9, 0.05]
    strata = ["common", "common", "common", "rare"]
    out = coverage_stratified(rel, strata, 3, must_cover=["rare"])
    assert 3 in out, "the guaranteed class must appear despite low relevance"


# ---- diversity metrics ----------------------------------------------------

def test_alpha_dcg_discounts_redundant_coverage():
    subs = {"a": {1}, "b": {1}, "c": {2}}
    # covering two distinct subtopics beats covering one twice
    assert alpha_dcg(["a", "c"], subs, alpha=0.5, k=2) > alpha_dcg(["a", "b"], subs, 0.5, 2)


def test_alpha_dcg_first_position_is_undiscounted():
    subs = {"a": {1, 2}}
    assert alpha_dcg(["a"], subs, alpha=0.5, k=1) == pytest.approx(2.0)


def test_alpha_ndcg_perfect_ranking_is_one():
    subs = {"a": {1}, "b": {2}}
    assert alpha_ndcg(["a", "b"], subs, k=2, candidates=["a", "b"]) == pytest.approx(1.0)


def test_subtopic_recall_counts_distinct_subtopics():
    subs = {"a": {1}, "b": {1}, "c": {2}}
    assert subtopic_recall(["a", "b"], subs, {1, 2}, k=2) == pytest.approx(0.5)
    assert subtopic_recall(["a", "c"], subs, {1, 2}, k=2) == pytest.approx(1.0)


def test_intra_list_distance_is_zero_for_identical_items():
    vecs = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    assert intra_list_distance(["a", "b"], vecs, k=2) == pytest.approx(0.0, abs=1e-9)

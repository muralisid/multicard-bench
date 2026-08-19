"""Tests for the two-pass clustering.

The properties that matter: structure is discovered without being told how many
clusters to find, items that fit nothing are left as residue rather than forced
somewhere, and the naming step never sees a document.
"""

import numpy as np

from multicard.cluster.twopass import (assign, discover, keywords,
                                       summarise_for_naming)


def three_blobs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    X = np.vstack([c + rng.normal(0, 0.05, size=(n, 3)) for c in centres])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_discover_finds_structure_without_being_told_how_many():
    clusters = discover(three_blobs(), min_cluster_size=5, reduce=False)
    assert 2 <= len(clusters) <= 4
    assert sum(len(c.members) for c in clusters) > 100


def test_centroids_are_unit_length():
    for c in discover(three_blobs(), min_cluster_size=5, reduce=False):
        assert np.isclose(np.linalg.norm(c.centroid), 1.0, atol=1e-6)


def test_assign_routes_a_near_item_and_leaves_a_far_one_as_residue():
    clusters = discover(three_blobs(), min_cluster_size=5, reduce=False)
    near = clusters[0].centroid.reshape(1, -1)
    far = np.array([[-1.0, -1.0, -1.0]])
    far = far / np.linalg.norm(far)
    labels, unassigned = assign(np.vstack([near, far]), clusters, threshold=0.5)
    assert labels[0] != -1, "an item at a centroid must be assigned to it"
    assert 1 in unassigned, "an item unlike every cluster must stay in the residue"


def test_assign_with_no_clusters_leaves_everything_unassigned():
    labels, unassigned = assign(np.eye(3), [], threshold=0.4)
    assert list(labels) == [-1, -1, -1] and unassigned == [0, 1, 2]


def test_keywords_drop_stopwords_and_rank_by_frequency():
    kw = keywords(["the pipeline capacity was reduced",
                   "pipeline capacity and pipeline scheduling"], top_n=3)
    assert kw[0] == "pipeline"
    assert "the" not in kw and "and" not in kw


def test_naming_payload_carries_keywords_not_documents():
    X = three_blobs()
    texts = [f"document number {i} about pipeline capacity" for i in range(len(X))]
    clusters = discover(X, min_cluster_size=5, reduce=False)
    payload = summarise_for_naming(clusters, texts, examples=2)
    assert payload, "expected at least one cluster to summarise"
    for p in payload:
        assert p["keywords"], "a naming request must carry keywords"
        assert len(p["examples"]) <= 2
        assert all(len(e) <= 160 for e in p["examples"]), "excerpts must stay short"
        assert "size" in p

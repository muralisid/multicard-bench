"""Tests for part1.topics on a synthetic 2,000-unit sample.

No encoder is loaded. The vectors are Gaussian clusters on the unit sphere
and the texts are cluster-specific words, so the fit (UMAP, HDBSCAN,
c-TF-IDF, all real) recovers the clusters and the terms are checkable.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multicard.data.multihoprag import Document
from multicard.part1 import topics as T
from multicard.part1 import units as U

N_UNITS = 2000
N_CLUSTERS = 6
FILLER = ["the", "and", "of"]


def vocab(c):
    return [f"w{c}{j}" for j in range(8)]


def synthetic(n=N_UNITS, n_clusters=N_CLUSTERS, seed=13):
    """Unit vectors around n_clusters random centres, with matching texts."""
    r = np.random.default_rng(seed)
    centres = r.normal(size=(n_clusters, 384)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    truth = r.integers(0, n_clusters, size=n)
    vecs = centres[truth] + 0.08 * r.normal(size=(n, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    texts = [" ".join(list(r.choice(vocab(c), 6)) + list(r.choice(FILLER, 2))) for c in truth]
    ids = [f"s{i // 4}#{i % 4}" for i in range(n)]
    return vecs.astype(np.float32), texts, ids, truth


@pytest.fixture(scope="module")
def sample():
    return synthetic()


@pytest.fixture(scope="module")
def fitted(sample, tmp_path_factory):
    vecs, texts, ids, _ = sample
    out = tmp_path_factory.mktemp("topics")
    return T.fit(vecs, texts, ids, out), out


def majority(truth, mask):
    vals, counts = np.unique(truth[mask], return_counts=True)
    return int(vals[counts.argmax()]), int(counts.max())


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def test_fit_recovers_the_clusters(fitted, sample):
    r, _ = fitted
    _, _, _, truth = sample
    assert 2 <= r.n_topics <= 12
    real = [t for t in r.topic_ids if t != -1]
    pure = 0
    for t in real:
        c, n_major = majority(truth, r.labels == t)
        pure += n_major
        row = r.row_of(t)
        # the c-TF-IDF terms of a topic come from its cluster's words
        assert sum(w in vocab(c) for w in r.terms[row]) >= 6
    assert pure / (r.labels != -1).sum() >= 0.95


def test_topic_tables_are_consistent(fitted):
    r, _ = fitted
    n = len(r.unit_ids)
    assert list(r.topic_ids) == sorted(r.topic_ids)
    assert np.array_equal(r.is_outlier, r.topic_ids == -1)
    assert r.sizes.sum() == n
    for t, s in zip(r.topic_ids, r.sizes):
        assert s == (r.labels == t).sum()
    assert r.prototypes.shape == (len(r.topic_ids), 384)
    assert r.prototypes.dtype == np.float32
    # the raw topic_embeddings_ rows (means of unit vectors): not unit length, at most 1
    norms = np.linalg.norm(r.prototypes, axis=1)
    assert (norms > 0).all() and (norms <= 1.0 + 1e-5).all()
    assert not np.allclose(norms, 1.0, atol=1e-3)
    for ws, name in zip(r.terms, r.names):
        assert 1 <= len(ws) <= 10 and all(ws)
        assert name == " ".join(ws)
    assert r.labels.dtype == np.int32 and r.labels.shape == (n,)
    assert set(r.labels.tolist()) == set(r.topic_ids.tolist())


def test_distribution_is_the_softmax_of_cosine(fitted, sample):
    r, _ = fitted
    vecs, _, _, _ = sample
    k = min(5, len(r.topic_ids))
    assert r.top_ids.shape == (len(vecs), k) and r.top_probs.shape == (len(vecs), k)
    assert r.top_ids.dtype == np.int32 and r.top_probs.dtype == np.float32
    protos = r.prototypes / np.linalg.norm(r.prototypes, axis=1, keepdims=True)
    for i in np.random.default_rng(0).choice(len(vecs), 20, replace=False):
        v = vecs[i] / np.linalg.norm(vecs[i])
        cos = protos @ v
        p = np.exp(cos - cos.max())
        p /= p.sum()
        order = np.argsort(-p, kind="stable")[:k]
        assert list(r.top_ids[i]) == [int(r.topic_ids[j]) for j in order]
        assert np.allclose(r.top_probs[i], p[order], atol=1e-5)
        assert all(r.top_probs[i][:-1] >= r.top_probs[i][1:])
    assert (r.top_probs > 0).all() and (r.top_probs.sum(axis=1) <= 1.0 + 1e-5).all()


def test_fit_is_deterministic(fitted, sample):
    r, _ = fitted
    vecs, texts, ids, _ = sample
    again = T.fit(vecs, texts, ids)
    assert np.array_equal(again.labels, r.labels)
    assert np.array_equal(again.topic_ids, r.topic_ids)
    assert np.array_equal(again.top_ids, r.top_ids)
    assert np.allclose(again.prototypes, r.prototypes, atol=1e-6)
    assert np.allclose(again.top_probs, r.top_probs, atol=1e-6)
    assert again.terms == r.terms and again.names == r.names


def test_diagnostics(fitted):
    r, _ = fitted
    d = r.diagnostics
    n = len(r.unit_ids)
    real = r.sizes[~r.is_outlier]
    assert d["n_units"] == n == N_UNITS
    assert d["n_topics"] == r.n_topics == len(real)
    assert d["outlier_share"] == pytest.approx((r.labels == -1).mean())
    assert d["largest_topic_share"] == pytest.approx(real.max() / n)
    assert d["seconds"] > 0
    assert d["seed"] == 13 and d["top_k"] == 5 and d["top_terms"] == 10
    assert d["umap"] == {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0,
                         "metric": "cosine", "random_state": 13}
    assert d["hdbscan"]["min_cluster_size"] == 10
    assert d["hdbscan"]["metric"] == "euclidean"
    assert d["hdbscan"]["cluster_selection_method"] == "eom"
    assert d["calculate_probabilities"] is False
    assert d["versions"]["bertopic"] == "0.17.4"


def test_fit_rejects_bad_input(sample):
    vecs, texts, ids, _ = sample
    with pytest.raises(ValueError):
        T.fit(vecs[:10], texts[:9], ids[:10])
    with pytest.raises(ValueError):
        T.fit(vecs[:10, :100], texts[:10], ids[:10])
    with pytest.raises(ValueError):
        T.fit(vecs[:0], [], [])


# ----------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------
def test_parquet_schemas(fitted):
    r, out = fitted
    topics = pq.read_schema(out / "topics.parquet")
    assert topics.names == ["topic_id", "terms", "name", "prototype", "size", "is_outlier"]
    assert topics.field("topic_id").type == pa.int32()
    assert topics.field("terms").type == pa.list_(pa.string())
    assert topics.field("name").type == pa.string()
    assert topics.field("prototype").type == pa.list_(pa.float32())
    assert topics.field("size").type == pa.int32()
    assert topics.field("is_outlier").type == pa.bool_()
    ut = pq.read_schema(out / "unit_topics.parquet")
    assert ut.names == ["unit_id", "topic_id", "prob"]
    assert ut.field("unit_id").type == pa.string()
    assert ut.field("topic_id").type == pa.int32()
    assert ut.field("prob").type == pa.float32()
    ul = pq.read_schema(out / "unit_labels.parquet")
    assert ul.names == ["unit_id", "label", "is_outlier"]
    assert ul.field("label").type == pa.int32()
    assert pq.read_metadata(out / "unit_topics.parquet").num_rows == len(r.unit_ids) * r.top_ids.shape[1]
    assert (out / "diagnostics.json").exists()


def test_save_and_load_round_trip(fitted):
    r, out = fitted
    back = T.load(out)
    assert np.array_equal(back.topic_ids, r.topic_ids)
    assert back.terms == r.terms and back.names == r.names
    assert np.allclose(back.prototypes, r.prototypes)
    assert np.array_equal(back.sizes, r.sizes)
    assert np.array_equal(back.is_outlier, r.is_outlier)
    assert back.unit_ids == r.unit_ids
    assert np.array_equal(back.labels, r.labels)
    assert np.array_equal(back.top_ids, r.top_ids)
    assert np.allclose(back.top_probs, r.top_probs)
    assert back.diagnostics == r.diagnostics
    assert back.unit_labels() == r.unit_labels()
    assert back.unit_distribution()["s0#0"] == r.unit_distribution()["s0#0"]


def hand_built():
    """A result with an outlier topic, without a fit."""
    protos = T._unit_norm(np.eye(3, 384, dtype=np.float32) + 0.01)
    labels = np.array([0, -1, 1, 0, -1], dtype=np.int32)
    ids = ["s0#0", "s0#1", "s1#0", "s1#1", "d0#0"]
    vecs = protos[[1, 0, 2, 1, 0]]
    rows, probs = T.topic_distribution(vecs, protos, k=5)
    topic_ids = np.array([-1, 0, 1], dtype=np.int32)
    return T.TopicResult(
        topic_ids=topic_ids, terms=[["noise"], ["a", "b"], ["c"]], names=["noise", "a b", "c"],
        prototypes=protos, sizes=np.array([2, 2, 1], dtype=np.int32),
        is_outlier=topic_ids == -1, unit_ids=ids, labels=labels,
        top_ids=topic_ids[rows].astype(np.int32), top_probs=probs, diagnostics={"n_units": 5})


def test_outlier_topic_row_survives_the_files(tmp_path):
    r = hand_built()
    assert r.n_topics == 2 and r.row_of(-1) == 0
    T.save(r, tmp_path)
    topics = pq.read_table(tmp_path / "topics.parquet").to_pydict()
    assert topics["topic_id"] == [-1, 0, 1] and topics["is_outlier"] == [True, False, False]
    labels = pq.read_table(tmp_path / "unit_labels.parquet").to_pydict()
    assert labels["is_outlier"] == [False, True, False, False, True]
    ut = pq.read_table(tmp_path / "unit_topics.parquet").to_pydict()
    # the outlier row is part of the softmax, so -1 appears in the top k
    assert -1 in ut["topic_id"]
    assert ut["unit_id"][:3] == ["s0#0"] * 3
    back = T.load(tmp_path)
    assert np.array_equal(back.labels, r.labels) and np.array_equal(back.top_ids, r.top_ids)
    assert back.diagnostics == {"n_units": 5}


def test_load_rejects_a_broken_unit_topics_table(tmp_path):
    r = hand_built()
    T.save(r, tmp_path)
    ut = pq.read_table(tmp_path / "unit_topics.parquet")
    pq.write_table(ut.slice(0, ut.num_rows - 1), tmp_path / "unit_topics.parquet")
    with pytest.raises(ValueError):
        T.load(tmp_path)


# ----------------------------------------------------------------------------
# distribution
# ----------------------------------------------------------------------------
def test_topic_distribution_small_k_and_ties():
    protos = np.zeros((3, 384), dtype=np.float32)
    protos[0, 0] = 1.0
    protos[1, 1] = 1.0
    protos[2, 1] = 1.0            # a copy of row 1: an exact tie
    vecs = np.zeros((4, 384), dtype=np.float32)
    vecs[:, 1] = 1.0
    vecs[3, 0] = 1.0              # equal similarity to every row
    rows, probs = T.topic_distribution(vecs, protos, k=5)
    assert rows.shape == (4, 3)   # k is capped at the number of rows
    assert list(rows[0]) == [1, 2, 0]
    assert list(rows[3]) == [0, 1, 2] or np.allclose(probs[3], probs[3][0])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    rows2, probs2 = T.topic_distribution(vecs, protos, k=2, block=3)
    assert rows2.shape == (4, 2) and list(rows2[0]) == [1, 2]
    assert np.allclose(probs2, probs[:, :2])
    # a non-contiguous view of the same rows and an unnormalised prototype give the same answer
    view = np.repeat(vecs, 2, axis=0)[::2]
    assert not view.flags["C_CONTIGUOUS"] and np.array_equal(view, vecs)
    rows3, probs3 = T.topic_distribution(view, protos * 3.0, k=2)
    assert np.array_equal(rows3, rows2) and np.allclose(probs3, probs2)
    empty_rows, empty_probs = T.topic_distribution(vecs[:0], protos)
    assert empty_rows.shape == (0, 3) and empty_probs.shape == (0, 3)


# ----------------------------------------------------------------------------
# members
# ----------------------------------------------------------------------------
SESSIONS = {
    "s0": {"date": "2023-05-20", "turns": [
        {"role": "user", "content": "I bought a red bike yesterday. It cost two hundred dollars."},
        {"role": "assistant", "content": "Great choice for the city.\nEnjoy the ride every day."},
    ]},
    "s1": {"date": "2023-05-18", "turns": [
        {"role": "user", "content": "My cat is called Tom and he is three years old."},
        {"role": "assistant", "content": "Nice."},
    ]},
}


class Words:
    def count(self, text):
        return len(text.split())


def test_topic_members_are_the_sub_units_of_labelled_units():
    tables = U.build_tables(SESSIONS, ["s0", "s1"])
    doc = Document(doc_id="d0", title="Title", source="src", category="c", published_at="2023-10-01T00:00:00",
                   body="First sentence here. Second sentence here.\n\nThird one here.")
    tables.update(U.build_doc_tables([doc], Words(), max_tokens=500))
    r = hand_built()
    members = T.topic_members(r, tables)
    assert set(members) == {0, 1}
    assert members[0] == [s.unit_id for s in tables["s0"].subs[0]] + [s.unit_id for s in tables["s1"].subs[1]]
    assert members[1] == [s.unit_id for s in tables["s1"].subs[0]]
    # the outlier units s0#1 and d0#0 are members of no topic
    all_members = {m for ms in members.values() for m in ms}
    assert not any(m.startswith("s0#1/") or m.startswith("d0#0/") for m in all_members)
    r.labels[4] = 1
    members = T.topic_members(r, tables)
    assert members[1][1:] == [s.unit_id for s in tables["d0"].sentences[0]]
    with pytest.raises(ValueError):
        T.topic_members(r, {"s0": tables["s0"]})


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def test_cli_writes_the_tables(monkeypatch, tmp_path):
    vecs, texts, ids, _ = synthetic(n=500, seed=7)
    seen = {}

    def fake_corpus_units(corpus, enc=None, limit=0, data=None, raw=None):
        seen.update(corpus=corpus, limit=limit)
        return ids, texts, vecs

    monkeypatch.setattr(T, "corpus_units", fake_corpus_units)
    diag = T.main(["--corpus", "mhrag", "--out", str(tmp_path / "o"), "--limit", "500"])
    assert seen == {"corpus": "mhrag", "limit": 500}
    assert diag["n_units"] == 500
    for name in ("topics.parquet", "unit_topics.parquet", "unit_labels.parquet", "diagnostics.json"):
        assert (tmp_path / "o" / name).exists()
    assert T.load(tmp_path / "o").unit_ids == ids


def test_corpus_units_rejects_an_unknown_corpus():
    with pytest.raises(ValueError):
        T.corpus_units("newsgroups", enc=object())

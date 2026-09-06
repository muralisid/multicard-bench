"""Tests for part1.graph on tiny fixtures (design section 4, items 4 and 5).

spaCy en_core_web_sm is loaded once per session for the extraction tests;
every other test builds Occurrences by hand so the numpy rules are checked
against hand-computed values.
"""

import json

import numpy as np
import pytest

from multicard.part1 import graph as G


@pytest.fixture(scope="module")
def nlp():
    return G.load_nlp()


def occ_from(units: list[list[tuple[str, bool]]]) -> G.Occurrences:
    """Occurrences from per-unit lists of (phrase, is_pron)."""
    vocab, ids, offsets, phrase, pron = [], {}, [0], [], []
    for group in units:
        for text, p in group:
            if text not in ids:
                ids[text] = len(vocab)
                vocab.append(text)
            phrase.append(ids[text])
            pron.append(p)
        offsets.append(len(phrase))
    return G.Occurrences(vocab=vocab, offsets=np.asarray(offsets, dtype=np.int64),
                         phrase=np.asarray(phrase, dtype=np.int32), pron=np.asarray(pron, dtype=bool))


# ----------------------------------------------------------------------------
# Phrases
# ----------------------------------------------------------------------------
def test_phrase_text_strips_every_article_and_lowercases():
    assert G.phrase_text("The Quick  Brown fox") == "quick brown fox"
    assert G.phrase_text("an  old bike") == "old bike"
    assert G.phrase_text("A") == ""
    assert G.phrase_text("my friend's dog") == "my friend's dog"
    # section 4 item 4: articles stripped, not only the leading one
    assert G.phrase_text("the man of the hour") == "man of hour"
    assert G.phrase_text("an apple a day") == "apple day"
    assert G.phrase_text("the a an") == ""


def test_extract_phrases_lemmas_pronouns_and_order(nlp):
    texts = ["The quick brown foxes jumped over the lazy dogs.",
             "She loves them and an old bike.",
             "The foxes saw the foxes again.",
             ""]
    occ = G.extract_phrases(texts, n_process=1, nlp=nlp)
    assert occ.n_units == 4
    assert occ.vocab[:2] == ["quick brown fox", "lazy dog"]
    assert occ.offsets.tolist() == [0, 2, 5, 6, 6]
    # pronoun roots are flagged, not dropped, at extraction time
    second = [(occ.vocab[p], bool(f)) for p, f in zip(occ.phrase[2:5], occ.pron[2:5])]
    assert second == [("she", True), ("they", True), ("old bike", False)]
    # a phrase repeated inside one sub-unit is one occurrence, so frequency counts sub-units
    assert occ.vocab[occ.phrase[5]] == "fox" and occ.freq()[occ.phrase[5]] == 1
    assert occ.seconds > 0 and occ.n_process == 1


def test_extract_phrases_caps_at_fifty_per_unit(nlp):
    text = " ".join(f"The item{i} and the thing{i}." for i in range(40))
    occ = G.extract_phrases([text], n_process=1, nlp=nlp)
    assert occ.n_units == 1
    assert occ.n_occurrences == G.MAX_PHRASES_PER_UNIT
    assert occ.n_phrases == G.MAX_PHRASES_PER_UNIT


# ----------------------------------------------------------------------------
# Hub rule
# ----------------------------------------------------------------------------
def test_hub_rule_drops_pronouns_and_top_half_percent():
    # 400 phrases: p0 in 10 units, p1 in 9, p2 in 8, the rest in 1 unit; "it" is a pronoun in 3 units
    units = [[("p0", False), ("p1", False), ("p2", False)] for _ in range(8)]
    units += [[("p0", False), ("p1", False)], [("p0", False)]]
    units += [[(f"q{i}", False)] for i in range(397)]
    units += [[("it", True)], [("it", True)], [("it", False)]]
    occ = occ_from(units)
    assert occ.n_phrases == 401
    h = G.hub_rule(occ)
    assert h.n_hub == 2                                  # int(0.005 * 401)
    assert [occ.vocab[i] for i in np.flatnonzero(h.is_hub)] == ["p0", "p1"]
    assert [occ.vocab[i] for i in np.flatnonzero(h.is_pronoun)] == ["it"]
    assert h.dropped.sum() == 3


def test_hub_rule_pronoun_is_a_majority_vote():
    occ = occ_from([[("one", True)], [("one", False)], [("one", False)], [("x", False)] * 1])
    assert not G.hub_rule(occ).is_pronoun.any()


# ----------------------------------------------------------------------------
# Edges, PMI, pruning
# ----------------------------------------------------------------------------
def test_cooccurrence_counts_pairs_inside_one_unit_only():
    units = [[("a", False), ("b", False), ("c", False)],
             [("a", False), ("b", False)],
             [("c", False)],
             [("d", False), ("a", False)]]
    occ = occ_from(units)
    e = G.cooccurrence(occ, np.ones(occ.n_phrases, dtype=bool))
    pairs = {(occ.vocab[x], occ.vocab[y]): int(c) for x, y, c in zip(e.a, e.b, e.count)}
    assert pairs == {("a", "b"): 2, ("a", "c"): 1, ("b", "c"): 1, ("a", "d"): 1}
    assert e.n_pairs == 5
    assert (e.a < e.b).all()
    # a dropped phrase takes its pairs with it
    keep = np.ones(occ.n_phrases, dtype=bool)
    keep[occ.vocab.index("a")] = False
    e2 = G.cooccurrence(occ, keep)
    assert {(occ.vocab[x], occ.vocab[y]) for x, y in zip(e2.a, e2.b)} == {("b", "c")}
    e3 = G.cooccurrence(occ, np.zeros(occ.n_phrases, dtype=bool))
    assert len(e3) == 0 and e3.n_pairs == 0


def test_pmi_matches_the_graphrag_formula():
    occ = occ_from([[("a", False), ("b", False), ("c", False)], [("a", False), ("b", False)], [("c", False)]])
    freq = occ.freq()
    mask = np.ones(occ.n_phrases, dtype=bool)
    e = G.cooccurrence(occ, mask)
    w = G.pmi_weights(e, freq, mask)
    total_edge, total_freq = e.count.sum(), freq.sum()
    for x, y, c, got in zip(e.a, e.b, e.count, w):
        p_xy = c / total_edge
        p_x, p_y = freq[x] / total_freq, freq[y] / total_freq
        assert got == pytest.approx(p_xy * np.log2(p_xy / (p_x * p_y)))
    # a pair of two frequent phrases seen once gets a negative weight
    ab = [i for i, (x, y) in enumerate(zip(e.a, e.b)) if {occ.vocab[x], occ.vocab[y]} == {"a", "b"}][0]
    assert w[ab] > 0
    assert G.pmi_weights(G.cooccurrence(occ, np.zeros(occ.n_phrases, dtype=bool)), freq, mask).shape == (0,)


def test_prune_applies_the_graphrag_defaults_in_order():
    # h is the ego node (degree 4); s is seen once (freq 1); z has no edge
    units = [[("h", False), ("a", False), ("b", False)],
             [("h", False), ("a", False), ("b", False)],
             [("h", False), ("c", False)],
             [("h", False), ("s", False)],
             [("a", False), ("b", False)],
             [("a", False), ("c", False)],
             [("b", False), ("c", False)],
             [("c", False)], [("z", False)], [("z", False)]]
    occ = occ_from(units)
    freq = occ.freq()
    mask = np.ones(occ.n_phrases, dtype=bool)
    e = G.cooccurrence(occ, mask)
    w = G.pmi_weights(e, freq, mask)
    p = G.prune(e, w, freq, mask)
    v = occ.vocab
    assert v[p.ego] == "h"
    assert p.n_removed_degree == 1 and not p.node_kept[v.index("z")]
    assert p.n_removed_freq == 1 and not p.node_kept[v.index("s")]
    assert sorted(v[i] for i in np.flatnonzero(p.node_kept)) == ["a", "b", "c"]
    # edges between survivors: ab, ac, bc; the 40th percentile of their weights is the cut
    survivors = e.a[p.edge_kept], e.b[p.edge_kept]
    kept_pairs = {(v[x], v[y]) for x, y in zip(*survivors)}
    inner = np.array([wi for x, y, wi in zip(e.a, e.b, w) if v[x] in "abc" and v[y] in "abc"])
    assert p.min_weight == pytest.approx(np.percentile(inner, 40.0))
    expected = {(v[x], v[y]) for x, y, wi in zip(e.a, e.b, w)
                if v[x] in "abc" and v[y] in "abc" and wi >= p.min_weight and wi > 0}
    assert kept_pairs == expected
    assert p.n_removed_edges_nodes == int(sum(1 for x, y in zip(e.a, e.b) if v[x] not in "abc" or v[y] not in "abc"))


def test_prune_drops_nonpositive_weights_for_leiden():
    e = G.Edges(a=np.array([0, 0, 1], np.int32), b=np.array([1, 2, 2], np.int32),
                count=np.array([1, 1, 1], np.int64), n_pairs=3)
    freq = np.array([2, 2, 2, 5])
    mask = np.array([True, True, True, False])
    w = np.array([1.0, -0.5, 0.0])
    old = G.REMOVE_EGO_NODES
    G.REMOVE_EGO_NODES = False
    try:
        p = G.prune(e, w, freq, mask)
    finally:
        G.REMOVE_EGO_NODES = old
    # percentile 40 of [1, -0.5, 0] is -0.2: the negative edge goes, the zero edge goes as non-positive
    assert p.edge_kept.tolist() == [True, False, False]
    assert p.n_removed_edges_pct == 1 and p.n_removed_nonpositive == 1


# ----------------------------------------------------------------------------
# Leiden, topics, membership, prototypes
# ----------------------------------------------------------------------------
def two_cliques():
    la = [i for i in range(5) for j in range(i + 1, 5)] + [i for i in range(5, 10) for j in range(i + 1, 10)] + [4]
    lb = [j for i in range(5) for j in range(i + 1, 5)] + [j for i in range(5, 10) for j in range(i + 1, 10)] + [5]
    return np.array(la, np.int64), np.array(lb, np.int64)


def test_leiden_two_cliques_seeded():
    la, lb = two_cliques()
    w = np.ones(len(la))
    m1, q1 = G.leiden(10, la, lb, w, seed=13)
    m2, _ = G.leiden(10, la, lb, w, seed=13)
    assert m1.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] and m1.tolist() == m2.tolist()
    assert 0.4 < q1 < 0.5
    m0, q0 = G.leiden(0, la[:0], lb[:0], w[:0])
    assert len(m0) == 0


def test_topic_inputs_direct_and_owner_rows():
    ids = ["s1#0/0", "s1#0/1", "s1#1/0"]
    protos = {0: np.ones(384), 1: np.zeros(384)}
    rows = [("s1#0/0", 0, 0.7), ("s1#0/0", 1, 0.3), ("s1#1", 0, 1.0), ("nope", 0, 1.0), ("s1#0/1", 9, 1.0)]
    t = G.topic_inputs(ids, rows, protos)
    assert t.n_topics == 2 and t.topic_ids == [0, 1]
    R = t.R.toarray()
    assert R[0].tolist() == [0.7, 0.3]
    assert R[1].tolist() == [0.0, 0.0]        # its only row named an unknown topic
    assert R[2].tolist() == [1.0, 0.0]        # inherited from the owner turn
    assert t.n_ignored_units == 1 and t.n_ignored_topics == 1 and t.n_rows == 5
    empty = G.topic_inputs(ids, None, None)
    assert empty.n_topics == 0 and empty.R.shape == (3, 0)


def test_phrase_topic_vectors_and_factor():
    ids = ["u0", "u1", "u2"]
    occ = occ_from([[("a", False), ("b", False)], [("a", False), ("c", False)], [("b", False)]])
    r = np.random.default_rng(13)
    protos = {0: r.standard_normal(384), 1: r.standard_normal(384)}
    rows = [("u0", 0, 0.6), ("u0", 1, 0.4), ("u1", 1, 1.0)]
    topics = G.topic_inputs(ids, rows, protos)
    nodes = np.array([0, 1, 2], np.int32)
    vec = G.phrase_topic_vectors(occ, nodes, topics)
    # a: units u0 and u1 -> (0.6 p0 + 0.4 p1 + 1.0 p1) / 2.0
    expect_a = (0.6 * protos[0] + 1.4 * protos[1]) / 2.0
    assert np.allclose(vec[0], expect_a)
    assert np.allclose(vec[1], 0.6 * protos[0] + 0.4 * protos[1])   # b: u0 only, u2 has no topic
    assert np.allclose(vec[2], protos[1])                           # c: u1 only
    la, lb = np.array([0, 1]), np.array([1, 2])
    f = G.topic_factor(vec, la, lb)
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert f[0] == pytest.approx(1 + cos(vec[0], vec[1]))
    assert f[1] == pytest.approx(1 + cos(vec[1], vec[2]))
    # zero vectors give factor 1
    assert G.topic_factor(np.zeros((2, 384)), np.array([0]), np.array([1])).tolist() == [1.0]
    assert G.phrase_topic_vectors(occ, nodes, G.topic_inputs(ids, None, None)).shape == (3, 384)


def test_unit_membership_plurality_ties_and_none():
    occ = occ_from([[("a", False), ("b", False), ("c", False)],   # a,b in 0; c in 1 -> 0
                    [("c", False), ("d", False), ("a", False)],   # c,d in 1; a in 0 -> 1
                    [("a", False), ("c", False)],                 # tie 0 vs 1 -> 0
                    [("z", False)],                               # no surviving phrase
                    [("e", False), ("f", False), ("a", False)]])  # e,f in 2 -> 2
    comm_of = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 2, "f": 2, "z": G.NO_COMMUNITY}
    node_comm = np.array([comm_of[t] for t in occ.vocab], np.int32)
    m = G.unit_membership(occ, node_comm, 3)
    assert m.tolist() == [0, 1, 0, G.NO_COMMUNITY, 2]
    assert G.unit_membership(occ, np.full(7, G.NO_COMMUNITY, np.int32), 0).tolist() == [G.NO_COMMUNITY] * 5


def test_community_prototypes_are_member_means():
    vec = np.arange(12, dtype=np.float32).reshape(4, 3)
    member = np.array([0, 1, 0, G.NO_COMMUNITY], np.int32)
    p = G.community_prototypes(member, vec, 3)
    assert p.shape == (3, 3)
    assert np.allclose(p[0], (vec[0] + vec[2]) / 2) and np.allclose(p[1], vec[1]) and np.allclose(p[2], 0)


def test_topic_entropy_within_communities():
    ids = ["u0", "u1", "u2"]
    protos = {0: np.zeros(384), 1: np.zeros(384)}
    rows = [("u0", 0, 0.5), ("u0", 1, 0.5), ("u1", 0, 1.0), ("u2", 1, 1.0)]
    topics = G.topic_inputs(ids, rows, protos)
    # community 0 holds u0 (half and half) and u1 -> mass (1.5, 0.5); community 1 holds u2 -> mass (0, 1)
    member = np.array([0, 0, 1], np.int32)
    e = G.topic_entropy(member, topics, 2)
    h0 = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert e["mean_bits"] == pytest.approx(h0 / 2)
    assert e["member_weighted_bits"] == pytest.approx((h0 * 2 + 0) / 3)
    assert e["n_communities_with_topics"] == 2


# ----------------------------------------------------------------------------
# build and load
# ----------------------------------------------------------------------------
def corpus():
    """Two subjects (bikes, cats) with shared phrases; enough repeats to pass min_node_freq 2."""
    bikes = ["The red bike has a new chain and a bell.", "My red bike needs a new chain.",
             "The bell on the red bike is loud.", "A new chain and a bell for the bike."]
    cats = ["The old cat sleeps on the warm sofa.", "My old cat likes the warm sofa.",
            "The sofa is where the cat sleeps.", "A warm sofa for an old cat."]
    texts = bikes + cats
    ids = [f"s{i // 4}#{i % 4}/0" for i in range(len(texts))]
    return ids, texts


def test_build_writes_files_and_load_reads_them_back(tmp_path, nlp):
    ids, texts = corpus()
    r = np.random.default_rng(13)
    vecs = r.standard_normal((len(ids), 384)).astype(np.float32)
    protos = {0: r.standard_normal(384), 1: r.standard_normal(384)}
    rows = [(u, 0 if u.startswith("s0") else 1, 1.0) for u in ids]
    g = G.build(texts, ids, vecs, rows, protos, tmp_path, n_process=1, nlp=nlp)
    for name in (G.PHRASE_FILE, G.EDGE_FILE, G.COMMUNITY_FILE, G.UNIT_FILE, G.DIAG_FILE):
        assert (tmp_path / name).exists()
    assert list(g.phrases.columns) == ["phrase_id", "text", "freq", "pron_count", "is_pronoun", "is_hub",
                                       "dropped_by_hub_rule", "kept"]
    assert list(g.edges.columns) == ["a", "b", "count", "weight", "weight_topic"]
    assert list(g.communities.columns) == ["variant", "community_id", "phrase_ids", "member_unit_ids",
                                           "prototype", "size", "n_members"]
    assert list(g.units.columns) == ["unit_id", "n_phrases", "n_kept_phrases", "community_plain", "community_topic"]
    assert g.units["unit_id"].tolist() == ids
    # the hub rule is on: "my" is not a chunk root, so no pronoun phrase in this corpus; 0.5 percent of a
    # few dozen phrases floors to zero hubs
    assert g.diagnostics["n_hub_phrases"] == 0
    # the two design gaps of the hub rule are written into the diagnostics for the report
    assert g.diagnostics["pronoun_rule"] == G.PRONOUN_RULE and g.diagnostics["hub_rule_order"] == G.HUB_RULE_ORDER
    assert "article" in g.diagnostics["article_rule"]
    # kept phrases are exactly the nodes of the surviving edges
    kept = set(g.phrases.loc[g.phrases["kept"], "phrase_id"])
    assert kept == set(g.edges["a"]) | set(g.edges["b"])
    # every edge weight is positive and the topic weight is weight times a factor in [0, 2]
    assert (g.edges["weight"] > 0).all()
    ratio = g.edges["weight_topic"] / g.edges["weight"]
    assert ((ratio >= 0) & (ratio <= 2)).all()
    for variant in G.VARIANTS:
        sub = g.communities[g.communities["variant"] == variant]
        assert sub["community_id"].tolist() == list(range(len(sub)))
        assert sub["size"].sum() == len(kept)
        members = [u for m in sub["member_unit_ids"] for u in m]
        col = g.units[f"community_{variant}"]
        assert sorted(members) == sorted(g.units.loc[col >= 0, "unit_id"])
        # a prototype is the mean of its members' vectors
        for _, row in sub.iterrows():
            if row["n_members"]:
                idx = [ids.index(u) for u in row["member_unit_ids"]]
                assert np.allclose(row["prototype"], vecs[idx].mean(axis=0), atol=1e-5)
        d = g.diagnostics["variants"][variant]
        assert d["n_communities"] == len(sub)
        assert 0 < d["largest_community_share_nodes"] <= 1
        assert d["topic_entropy"]["n_communities_with_topics"] >= 1
    # bikes and cats separate: the two subjects land in different plain communities
    plain = g.units["community_plain"].tolist()
    assert len({c for c in plain[:4] if c >= 0}) >= 1 and set(plain[:4]) != set(plain[4:])
    loaded = G.load(tmp_path)
    assert loaded.phrases.equals(g.phrases)
    assert np.allclose(loaded.edges["weight_topic"], g.edges["weight_topic"])
    assert loaded.units.equals(g.units)
    assert len(loaded.communities) == len(g.communities)
    assert loaded.communities["member_unit_ids"].tolist() == g.communities["member_unit_ids"].tolist()
    assert np.allclose(np.stack(loaded.communities["prototype"]), np.stack(g.communities["prototype"]))
    assert loaded.diagnostics == json.loads(json.dumps(g.diagnostics, default=G._json_default))
    cids, pvecs = loaded.prototypes("topic")
    assert pvecs.shape == (len(cids), 384)
    assert loaded.membership("plain") == {u: c for u, c in zip(ids, plain) if c >= 0}


def test_build_is_deterministic(tmp_path, nlp):
    ids, texts = corpus()
    r = np.random.default_rng(7)
    vecs = r.standard_normal((len(ids), 384)).astype(np.float32)
    a = G.build(texts, ids, vecs, None, None, None, n_process=1, nlp=nlp)
    b = G.build(texts, ids, vecs, None, None, None, n_process=1, nlp=nlp)
    assert a.units.equals(b.units) and a.edges.equals(b.edges)
    # without topics the two variants are the same partition
    assert a.units["community_plain"].tolist() == a.units["community_topic"].tolist()
    assert a.diagnostics["topics"]["n_topics"] == 0


def test_build_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        G.build(["a"], ["u0", "u1"], np.zeros((2, 384)), None, None, None, n_process=1)
    with pytest.raises(ValueError):
        G.build(["a"], ["u0"], np.zeros((2, 384)), None, None, None, n_process=1)


def test_prototype_loaders(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"unit_id": ["u"], "topic_id": [1], "prob": [0.9]}), tmp_path / "ut.parquet")
    np.savez(tmp_path / "p.npz", topic_ids=np.array([3, 5, -1]), vectors=np.ones((3, 384), np.float32))
    p = G.load_prototypes(str(tmp_path / "p.npz"))
    assert set(p) == {3, 5} and p[3].shape == (384,)
    pq.write_table(pa.table({"topic_id": pa.array([1], pa.int64()),
                             "vector": pa.array([[0.5] * 384], pa.list_(pa.float32()))}), tmp_path / "p.parquet")
    p2 = G.load_prototypes(str(tmp_path / "p.parquet"))
    assert set(p2) == {1} and p2[1][0] == pytest.approx(0.5)
    # the frozen topics.parquet of part1.topics (column prototype, the outlier row -1 skipped)
    from multicard.part1 import topics as T

    protos = np.eye(3, 384, dtype=np.float32) * 0.6
    topic_ids = np.array([-1, 0, 1], dtype=np.int32)
    labels = np.array([0, -1, 1], dtype=np.int32)
    rows, probs = T.topic_distribution(protos[[1, 0, 2]], protos, k=3)
    result = T.TopicResult(topic_ids=topic_ids, terms=[["n"], ["a"], ["b"]], names=["n", "a", "b"],
                           prototypes=protos, sizes=np.array([1, 1, 1], dtype=np.int32), is_outlier=topic_ids == -1,
                           unit_ids=["s0#0", "s0#1", "s1#0"], labels=labels,
                           top_ids=topic_ids[rows].astype(np.int32), top_probs=probs, diagnostics={})
    T.save(result, tmp_path / "topics")
    p3 = G.load_prototypes(str(tmp_path / "topics" / "topics.parquet"))
    assert set(p3) == {0, 1} and p3[0][1] == pytest.approx(0.6) and p3[1][2] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        G.load_prototypes(str(tmp_path / "ut.parquet"))
    ut = G.load_unit_topics(str(tmp_path / "ut.parquet"))
    assert ut.columns.tolist() == ["unit_id", "topic_id", "prob"]
    pq.write_table(pa.table({"unit_id": ["u"]}), tmp_path / "bad.parquet")
    with pytest.raises(ValueError):
        G.load_unit_topics(str(tmp_path / "bad.parquet"))
    assert G.load_prototypes(None) == {} and G.load_unit_topics(None) is None

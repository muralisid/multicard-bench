"""Noun-phrase graph and communities for Part 1 (design section 4, items 4 and 5).

Nodes are noun phrases from spaCy en_core_web_sm with ner disabled, taken from
doc.noun_chunks of each e5 sub-unit, normalised to the lowercase lemma with
every article token (a, an, the) stripped, at most 50 distinct phrases per
sub-unit by first occurrence. Edges are co-occurrence inside one sub-unit
(never a whole turn), counted in numpy over int32 phrase ids and int64 pair
codes with np.unique.

Weights and pruning follow the Microsoft graphrag NLP indexing path defaults
(graphrag 3.1.2, commit f40e9a26ce62ba0b3fef8837d24aafdcc6e6c704, files under
packages/graphrag/graphrag/, cited at each function): the raw weight is the
number of sub-units where both phrases occur, normalised to PMI, then
min_node_freq 2, min_node_degree 1, one ego node removed, and edges under the
40th weight percentile removed. The declared hub rule of the design drops
pronoun phrases (a phrase whose chunk root is PRON in the majority of its
occurrences; design gap, see hub_rule) and the top 0.5 percent of phrases by
sub-unit frequency. Design gap: the design fixes no order between the hub
rule and graphrag's pruning; here the hub rule runs first, before edges are
counted, so graphrag's ego-node removal then takes the highest-degree content
phrase left after the hub rule (one phrase; named in diagnostics.json as
ego_node). Both readings are written into diagnostics.json (pronoun_rule,
hub_rule_order) and the runner puts them in the report's disclosures.

Communities: leidenalg.find_partition(G, ModularityVertexPartition,
weights="weight", seed=13), one level, in two versions over the same pruned
graph: plain, and topic-weighted, where each weight is multiplied by one plus
the cosine similarity of the two phrases' topic vectors. A phrase's topic
vector is the probability-weighted mean of the 384-dimensional topic prototype
vectors over the sub-units it appears in. A sub-unit belongs to the community
holding the plurality of its surviving phrases, ties to the lower community
id; a sub-unit with no surviving phrase is in no community. A community
prototype is the mean of its member sub-units' vectors.

Files written by build() to out_dir (all parquet through pyarrow):
  phrases.parquet      phrase_id, text, freq, pron_count, is_pronoun, is_hub,
                       dropped_by_hub_rule, kept (a node of the pruned graph)
  edges.parquet        a, b, count, weight, weight_topic (surviving edges only)
  communities.parquet  variant (plain|topic), community_id, phrase_ids,
                       member_unit_ids, prototype (384 floats), size (phrases),
                       n_members (sub-units)
  units.parquet        unit_id, n_phrases, n_kept_phrases, community_plain,
                       community_topic (-1 = no community)
  diagnostics.json     counts before and after pruning, per-variant largest
                       community share, edge density, topic entropy, seconds

Sizing for LongMemEval (998,042 sub-units, mean 195 characters): measured on
a 2,000 sub-unit sample drawn with the bench rng at seed 13, a sub-unit
yields about 8.5 distinct phrases and about 31 phrase pairs; the largest had
47 phrases, so no sub-unit reached the 50-phrase cap, which bounds any
sub-unit at 1,225 pairs. Over the full corpus that is about 8.5 million
occurrences (int32 phrase id plus a bool flag and an int64 unit index, about
120 MB) and about 31 million raw pair codes (int64, 250 MB, about three times
that inside np.unique), so the edge stage fits in a few GB. The spaCy pass is
the long stage: the sample ran at about 230 texts per second with one process
on this machine (Apple Silicon, en_core_web_sm, batch 256), about 70 minutes
for the full corpus at one process. On 2,000 texts four processes gave no
gain (about 215 per second) because each worker loads the model first; the
steady-state rate at four processes is recorded by the full build in
diagnostics.json (phrase_throughput_per_s).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from .units import owner_id

SEED = 13
DIM = 384
SPACY_MODEL = "en_core_web_sm"
MAX_PHRASES_PER_UNIT = 50
ARTICLES = ("a", "an", "the")
HUB_SHARE = 0.005
# graphrag config/defaults.py lines 285 to 291 (PruneGraphDefaults) and line
# 187 (ExtractGraphNLPDefaults.normalize_edge_weights = True).
MIN_NODE_FREQ = 2
MIN_NODE_DEGREE = 1
MIN_EDGE_WEIGHT_PCT = 40.0
REMOVE_EGO_NODES = True
NORMALIZE_EDGE_WEIGHTS = True
GRAPHRAG_REF = ("microsoft/graphrag 3.1.2, commit f40e9a26ce62ba0b3fef8837d24aafdcc6e6c704, "
                "packages/graphrag/graphrag/")
VARIANTS = ("plain", "topic")
NO_COMMUNITY = -1
ROW_CHUNK = 100000       # rows per block in the dense matrix products
PHRASE_FILE, EDGE_FILE, COMMUNITY_FILE, UNIT_FILE, DIAG_FILE = (
    "phrases.parquet", "edges.parquet", "communities.parquet", "units.parquet", "diagnostics.json")


def _log(msg: str) -> None:
    print(f"  [graph] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Phrase extraction (section 4 item 4: nodes)
# ----------------------------------------------------------------------------
def load_nlp():
    """spaCy en_core_web_sm with ner excluded; the parser stays for noun_chunks."""
    import spacy

    return spacy.load(SPACY_MODEL, exclude=["ner"])


def phrase_text(lemma: str) -> str:
    """Lowercase lemma, whitespace collapsed, every article token stripped
    (section 4 item 4: "articles stripped", so "man of the hour" becomes
    "man of hour"). Empty when nothing is left or no letter or digit remains."""
    words = [w for w in lemma.lower().split() if w not in ARTICLES]
    text = " ".join(words)
    # design gap: a chunk made only of punctuation (markdown bullets such as
    # "*" parse as noun chunks) is not a phrase
    return text if any(c.isalnum() for c in text) else ""


def doc_phrases(doc) -> list[tuple[str, bool]]:
    """(phrase, root is PRON) per distinct phrase of one sub-unit, first
    occurrence order, at most MAX_PHRASES_PER_UNIT."""
    out: dict[str, bool] = {}
    for chunk in doc.noun_chunks:
        text = phrase_text(chunk.lemma_)
        if not text or text in out:
            continue
        out[text] = chunk.root.pos_ == "PRON"
        if len(out) >= MAX_PHRASES_PER_UNIT:
            break
    return list(out.items())


@dataclass
class Occurrences:
    """Phrase occurrences per sub-unit: unit u owns rows offsets[u]:offsets[u+1]
    of phrase (int32 ids) and pron (chunk root was PRON), in first occurrence
    order, one row per distinct phrase of the sub-unit."""
    vocab: list[str]
    offsets: np.ndarray
    phrase: np.ndarray
    pron: np.ndarray
    seconds: float = 0.0
    n_process: int = 1

    @property
    def n_units(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_phrases(self) -> int:
        return len(self.vocab)

    @property
    def n_occurrences(self) -> int:
        return len(self.phrase)

    def unit_of_occurrence(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_units, dtype=np.int64), np.diff(self.offsets))

    def freq(self) -> np.ndarray:
        """Sub-unit frequency per phrase: the number of sub-units it occurs in."""
        return np.bincount(self.phrase, minlength=self.n_phrases).astype(np.int64)

    def pron_count(self) -> np.ndarray:
        return np.bincount(self.phrase, weights=self.pron.astype(np.float64),
                           minlength=self.n_phrases).astype(np.int64)

    def phrases_per_unit(self) -> np.ndarray:
        return np.diff(self.offsets)


def extract_phrases(texts: list[str], n_process: int = 4, batch_size: int = 256, nlp=None,
                    log_every: int = 100000) -> Occurrences:
    """Noun phrases of every text through nlp.pipe with n_process workers.

    Phrase ids are assigned in corpus first-occurrence order. Throughput is
    recorded in seconds (texts per second = len(texts) / seconds).
    """
    t0 = time.time()
    ids: dict[str, int] = {}
    vocab: list[str] = []
    offsets = [0]
    phrase: list[int] = []
    pron: list[bool] = []
    if texts:
        nlp = nlp or load_nlp()
        for n, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), 1):
            for text, is_pron in doc_phrases(doc):
                pid = ids.get(text)
                if pid is None:
                    pid = len(vocab)
                    ids[text] = pid
                    vocab.append(text)
                phrase.append(pid)
                pron.append(is_pron)
            offsets.append(len(phrase))
            if log_every and n % log_every == 0:
                _log(f"{n}/{len(texts)} sub-units parsed, {len(vocab)} phrases ({time.time() - t0:.0f}s)")
    return Occurrences(vocab=vocab, offsets=np.asarray(offsets, dtype=np.int64),
                       phrase=np.asarray(phrase, dtype=np.int32), pron=np.asarray(pron, dtype=bool),
                       seconds=time.time() - t0, n_process=n_process)


# ----------------------------------------------------------------------------
# Hub rule (section 4 item 4, declared)
# ----------------------------------------------------------------------------
@dataclass
class HubRule:
    is_pronoun: np.ndarray    # the majority of the phrase's occurrences have a PRON root
    is_hub: np.ndarray        # top HUB_SHARE of phrases by sub-unit frequency
    n_hub: int

    @property
    def dropped(self) -> np.ndarray:
        return self.is_pronoun | self.is_hub


PRONOUN_RULE = ("a phrase is a pronoun when its chunk root is PRON in more than half of its "
                "occurrences (majority vote over occurrences)")
HUB_RULE_ORDER = ("the hub rule runs before graphrag's counting and pruning: co-occurrence, PMI and the "
                  "node and edge rules see the surviving phrases only, and graphrag's one ego node is "
                  "the highest-degree content phrase left after the hub rule")


def hub_rule(occ: Occurrences, share: float = HUB_SHARE) -> HubRule:
    """Pronoun phrases and the top `share` of phrases by sub-unit frequency.

    The share is taken over every extracted phrase, pronouns included, and the
    two sets are unioned. Ties at the boundary go to the lower phrase id.

    design gap: the design says "pronouns ... are dropped" and does not say
    how a phrase becomes a pronoun. A phrase is a pronoun here when the
    majority of its occurrences have a PRON chunk root (PRONOUN_RULE); a
    phrase flagged PRON in fewer than half of its occurrences is kept, those
    occurrences included.
    """
    freq = occ.freq()
    is_pronoun = occ.pron_count() * 2 > freq
    n_hub = int(share * occ.n_phrases)   # design gap: the count is floored
    order = np.lexsort((np.arange(occ.n_phrases), -freq))
    is_hub = np.zeros(occ.n_phrases, dtype=bool)
    is_hub[order[:n_hub]] = True
    return HubRule(is_pronoun=is_pronoun, is_hub=is_hub, n_hub=n_hub)


# ----------------------------------------------------------------------------
# Edges (section 4 item 4: co-occurrence inside one sub-unit)
# ----------------------------------------------------------------------------
@dataclass
class Edges:
    a: np.ndarray            # int32 phrase id, a < b
    b: np.ndarray
    count: np.ndarray        # int64 number of sub-units holding both phrases
    n_pairs: int = 0         # raw pair codes before np.unique

    def __len__(self) -> int:
        return len(self.a)


def cooccurrence(occ: Occurrences, keep_phrase: np.ndarray) -> Edges:
    """Pairs of kept phrases inside one sub-unit, counted over int64 codes.

    Matches graphrag build_noun_graph.py lines 114 to 134: every unordered
    pair of distinct phrases of a text unit is an edge whose raw weight is the
    number of text units holding the pair.
    """
    n = occ.n_phrases
    unit = occ.unit_of_occurrence()
    mask = keep_phrase[occ.phrase]
    unit, ph = unit[mask], occ.phrase[mask].astype(np.int64)
    order = np.lexsort((ph, unit))
    unit, ph = unit[order], ph[order]
    k = np.bincount(unit, minlength=occ.n_units)
    starts = np.concatenate([[0], np.cumsum(k)[:-1]])
    codes = []
    for size in np.unique(k):
        if size < 2:
            continue
        us = np.flatnonzero(k == size)
        mat = ph[starts[us][:, None] + np.arange(size)[None, :]]
        iu, ju = np.triu_indices(int(size), 1)
        codes.append(mat[:, iu].ravel() * n + mat[:, ju].ravel())
    if not codes:
        return Edges(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64), 0)
    codes = np.concatenate(codes)
    uniq, cnt = np.unique(codes, return_counts=True)
    return Edges(a=(uniq // n).astype(np.int32), b=(uniq % n).astype(np.int32),
                 count=cnt.astype(np.int64), n_pairs=int(len(codes)))


def pmi_weights(edges: Edges, freq: np.ndarray, node_mask: np.ndarray) -> np.ndarray:
    """PMI edge weights as graphrag computes them by default.

    graphrag graphs/edge_weights.py lines 10 to 62 (calculate_pmi_edge_weights),
    called from build_noun_graph.py lines 140 to 141 when
    normalize_edge_weights is true (config/defaults.py line 187):
      p(x,y) = weight(x,y) / sum of all edge weights          (lines 28, 35)
      p(x)   = frequency(x) / sum of all node frequencies    (lines 29 to 32)
      weight = p(x,y) * log2(p(x,y) / (p(x) * p(y)))          (lines 58 to 60)
    Node frequency is the number of text units holding the phrase
    (build_noun_graph.py lines 33 to 42). The node set here is every phrase
    left after the hub rule, isolated phrases included, as nodes_df holds every
    extracted phrase.
    """
    if not len(edges):
        return np.zeros(0, dtype=np.float64)
    total_edge = float(edges.count.sum())
    total_freq = float(freq[node_mask].sum())
    p_xy = edges.count / total_edge
    p_x = freq[edges.a] / total_freq
    p_y = freq[edges.b] / total_freq
    return p_xy * np.log2(p_xy / (p_x * p_y))


@dataclass
class Pruned:
    node_kept: np.ndarray     # survived the node rules (ego, degree, frequency)
    edge_kept: np.ndarray     # survived the node rules and the weight percentile
    ego: int | None
    min_weight: float
    n_removed_degree: int
    n_removed_freq: int
    n_removed_edges_nodes: int
    n_removed_edges_pct: int
    n_removed_nonpositive: int


def prune(edges: Edges, weight: np.ndarray, freq: np.ndarray, node_mask: np.ndarray) -> Pruned:
    """graphrag index/operations/prune_graph.py lines 14 to 106 at the
    defaults of config/defaults.py lines 285 to 291.

    Order as in that file: degree from the full edge list (line 30), the one
    highest-degree node removed (lines 45 to 47, remove_ego_nodes True), nodes
    with degree under min_node_degree 1 removed (lines 50 to 52), nodes with
    frequency under min_node_freq 2 removed among the rest (lines 65 to 70),
    edges kept only between surviving nodes (lines 81 to 84), then
    min_weight = np.percentile(weights, 40) over those edges and edges with
    weight >= min_weight kept (lines 87 to 95). max_node_freq_std,
    max_node_degree_std and lcc_only are None or False by default and do not
    fire. Ties for the ego node go to the lower phrase id; graphrag takes the
    first maximum in its dict order.
    """
    n = len(freq)
    degree = np.bincount(edges.a, minlength=n) + np.bincount(edges.b, minlength=n)
    kept = node_mask.copy()
    ego = None
    if REMOVE_EGO_NODES and len(edges) and kept.any():
        ego = int(np.argmax(np.where(kept, degree, -1)))
        kept[ego] = False
    low_degree = kept & (degree < MIN_NODE_DEGREE)
    kept &= ~low_degree
    low_freq = kept & (freq < MIN_NODE_FREQ)
    kept &= ~low_freq
    edge_kept = kept[edges.a] & kept[edges.b]
    n_edges_nodes = int((~edge_kept).sum())
    min_weight = float("nan")
    n_pct = 0
    if edge_kept.any() and MIN_EDGE_WEIGHT_PCT > 0:
        min_weight = float(np.percentile(weight[edge_kept], MIN_EDGE_WEIGHT_PCT))
        below = edge_kept & (weight < min_weight)
        n_pct = int(below.sum())
        edge_kept &= ~below
    # design gap: leidenalg refuses negative weights and a zero weight adds
    # nothing to modularity, so surviving edges with weight <= 0 are dropped
    nonpos = edge_kept & (weight <= 0)
    edge_kept &= ~nonpos
    return Pruned(node_kept=kept, edge_kept=edge_kept, ego=ego, min_weight=min_weight,
                  n_removed_degree=int(low_degree.sum()), n_removed_freq=int(low_freq.sum()),
                  n_removed_edges_nodes=n_edges_nodes, n_removed_edges_pct=n_pct,
                  n_removed_nonpositive=int(nonpos.sum()))


# ----------------------------------------------------------------------------
# Leiden (section 4 item 4)
# ----------------------------------------------------------------------------
def leiden(n_nodes: int, la: np.ndarray, lb: np.ndarray, weight: np.ndarray,
           seed: int = SEED) -> tuple[np.ndarray, float]:
    """One-level Leiden over local node indices. Returns (membership, modularity).

    leidenalg numbers communities by size, largest first, so the lower
    community id of the membership tie rule is the larger community.
    """
    import igraph
    import leidenalg

    g = igraph.Graph(n=n_nodes, edges=np.stack([la, lb], axis=1) if len(la) else [])
    g.es["weight"] = [float(w) for w in weight]
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", seed=seed)
    return np.asarray(part.membership, dtype=np.int32), float(part.modularity)


# ----------------------------------------------------------------------------
# Topics (section 4 item 5: topic-weighted variant)
# ----------------------------------------------------------------------------
@dataclass
class Topics:
    """unit_topics rows folded to a sparse (n_units, n_topics) probability
    matrix R over the sub-units, and the prototype matrix in the same topic
    order. A row keyed by an owner id (turn or chunk) applies to every sub-unit
    of that owner."""
    topic_ids: list
    proto: np.ndarray
    R: sp.csr_matrix
    n_rows: int
    n_ignored_units: int
    n_ignored_topics: int

    @property
    def n_topics(self) -> int:
        return len(self.topic_ids)


def _topic_frame(unit_topics) -> pd.DataFrame:
    if unit_topics is None:
        return pd.DataFrame({"unit_id": pd.Series([], dtype=object), "topic_id": pd.Series([], dtype=object),
                             "prob": pd.Series([], dtype=np.float64)})
    if isinstance(unit_topics, pd.DataFrame):
        return unit_topics[["unit_id", "topic_id", "prob"]].reset_index(drop=True)
    return pd.DataFrame(list(unit_topics), columns=["unit_id", "topic_id", "prob"])


def topic_inputs(unit_ids: list[str], unit_topics, prototypes) -> Topics:
    """Fold unit_topics rows (unit_id, topic_id, prob) and a {topic_id: vector}
    map into the matrices the topic weighting needs. Rows whose topic has no
    prototype or whose unit is neither a sub-unit nor an owner are ignored and
    counted."""
    proto_map = dict(prototypes or {})
    topic_ids = list(proto_map)
    proto = (np.stack([np.asarray(proto_map[t], dtype=np.float64) for t in topic_ids])
             if topic_ids else np.zeros((0, DIM), dtype=np.float64))
    df = _topic_frame(unit_topics)
    n_rows = len(df)
    col = df["topic_id"].map({t: i for i, t in enumerate(topic_ids)})
    bad_topic = col.isna()
    df = df[~bad_topic]
    col = col[~bad_topic].astype(np.int64)
    direct = pd.Series(np.arange(len(unit_ids), dtype=np.int64), index=pd.Index(unit_ids, dtype=object))
    row = df["unit_id"].map(direct)
    is_direct = row.notna()
    owners = pd.DataFrame({"owner": [owner_id(u) for u in unit_ids], "idx": np.arange(len(unit_ids), dtype=np.int64)})
    rest = df[~is_direct].assign(col=col[~is_direct].to_numpy())
    n_bad_unit = int((~rest["unit_id"].isin(owners["owner"])).sum())
    rest = rest.merge(owners, left_on="unit_id", right_on="owner", how="inner")
    rows = np.concatenate([row[is_direct].to_numpy(dtype=np.int64), rest["idx"].to_numpy(dtype=np.int64)])
    cols = np.concatenate([col[is_direct].to_numpy(dtype=np.int64), rest["col"].to_numpy(dtype=np.int64)])
    vals = np.concatenate([df.loc[is_direct, "prob"].to_numpy(dtype=np.float64),
                           rest["prob"].to_numpy(dtype=np.float64)])
    R = sp.csr_matrix((vals, (rows, cols)), shape=(len(unit_ids), len(topic_ids)))
    R.sum_duplicates()
    return Topics(topic_ids=topic_ids, proto=proto, R=R, n_rows=n_rows,
                  n_ignored_units=n_bad_unit, n_ignored_topics=int(bad_topic.sum()))


def phrase_topic_vectors(occ: Occurrences, nodes: np.ndarray, topics: Topics) -> np.ndarray:
    """(len(nodes), 384) probability-weighted mean of the prototypes over the
    sub-units each phrase appears in. Zero where no sub-unit carries a topic."""
    if topics.n_topics == 0 or not len(nodes):
        return np.zeros((len(nodes), DIM), dtype=np.float64)
    local = np.full(occ.n_phrases, -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    row = local[occ.phrase]
    m = row >= 0
    unit = occ.unit_of_occurrence()[m]
    incidence = sp.csr_matrix((np.ones(int(m.sum()), dtype=np.float64), (row[m], unit)),
                              shape=(len(nodes), occ.n_units))
    pt = (incidence @ topics.R).tocsr()             # (nodes, topics) probability mass
    mass = np.asarray(pt.sum(axis=1)).ravel()
    vec = np.zeros((len(nodes), topics.proto.shape[1]), dtype=np.float32)
    for lo in range(0, len(nodes), ROW_CHUNK):        # chunked so the float64 product stays small
        hi = min(lo + ROW_CHUNK, len(nodes))
        vec[lo:hi] = np.asarray(pt[lo:hi] @ topics.proto)
    nz = mass > 0
    vec[nz] /= mass[nz][:, None].astype(np.float32)
    return vec


def topic_factor(vec: np.ndarray, la: np.ndarray, lb: np.ndarray, chunk: int = 200000) -> np.ndarray:
    """One plus the cosine similarity of the endpoint topic vectors, per edge.
    The cosine of a zero vector is taken as 0, so the factor is 1."""
    unit = np.array(vec, dtype=np.float32, copy=True)
    norms = np.linalg.norm(unit, axis=1)
    nz = norms > 0
    unit[nz] /= norms[nz][:, None]
    out = np.empty(len(la), dtype=np.float64)
    for lo in range(0, len(la), chunk):
        hi = min(lo + chunk, len(la))
        out[lo:hi] = np.einsum("ij,ij->i", unit[la[lo:hi]], unit[lb[lo:hi]])
    return 1.0 + np.clip(out, -1.0, 1.0)


# ----------------------------------------------------------------------------
# Membership and prototypes (section 4 item 5)
# ----------------------------------------------------------------------------
def unit_membership(occ: Occurrences, node_community: np.ndarray, n_communities: int) -> np.ndarray:
    """Community per sub-unit: the plurality of its surviving phrases, ties to
    the lower community id, NO_COMMUNITY when no phrase survived.
    node_community holds a community id per phrase id, NO_COMMUNITY for a
    phrase outside the pruned graph."""
    member = np.full(occ.n_units, NO_COMMUNITY, dtype=np.int32)
    if n_communities == 0 or not occ.n_occurrences:
        return member
    comm = node_community[occ.phrase]
    m = comm >= 0
    if not m.any():
        return member
    unit = occ.unit_of_occurrence()[m]
    codes = unit * n_communities + comm[m]
    uniq, cnt = np.unique(codes, return_counts=True)
    u, c = uniq // n_communities, uniq % n_communities
    order = np.lexsort((c, -cnt, u))
    u, c = u[order], c[order]
    first = np.concatenate([[True], u[1:] != u[:-1]])
    member[u[first]] = c[first]
    return member


def community_prototypes(member: np.ndarray, vectors: np.ndarray, n_communities: int) -> np.ndarray:
    """(n_communities, 384) mean of the member sub-units' vectors; zeros for a
    community with no member sub-unit."""
    out = np.zeros((n_communities, vectors.shape[1] if vectors.ndim == 2 else DIM), dtype=np.float32)
    m = member >= 0
    if n_communities == 0 or not m.any():
        return out
    idx = np.flatnonzero(m)
    C = sp.csr_matrix((np.ones(len(idx)), (member[idx].astype(np.int64), idx)),
                      shape=(n_communities, len(member))).tocsc()
    sums = np.zeros((n_communities, out.shape[1]), dtype=np.float64)
    for lo in range(0, len(member), ROW_CHUNK):       # chunked so the float64 copy of vectors stays small
        hi = min(lo + ROW_CHUNK, len(member))
        sums += np.asarray(C[:, lo:hi] @ vectors[lo:hi].astype(np.float64))
    counts = np.bincount(member[idx], minlength=n_communities)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz][:, None]).astype(np.float32)
    return out


def topic_entropy(member: np.ndarray, topics: Topics, n_communities: int) -> dict:
    """Entropy in bits of the topic mass of each community's member sub-units
    (unit_topics probabilities summed per topic, normalised). Reported as the
    mean over communities with any topic mass and the member-weighted mean."""
    empty = {"mean_bits": None, "member_weighted_bits": None, "n_communities_with_topics": 0}
    m = member >= 0
    if n_communities == 0 or topics.n_topics == 0 or not m.any():
        return empty
    idx = np.flatnonzero(m)
    C = sp.csr_matrix((np.ones(len(idx)), (member[idx].astype(np.int64), idx)),
                      shape=(n_communities, len(member)))
    M = (C @ topics.R).tocsr()
    M.sum_duplicates()
    row_sum = np.asarray(M.sum(axis=1)).ravel()
    rows = np.repeat(np.arange(n_communities), np.diff(M.indptr))
    p = M.data / row_sum[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log2(p), 0.0)
    h = -np.bincount(rows, weights=plogp, minlength=n_communities)
    has = row_sum > 0
    if not has.any():
        return empty
    members = np.bincount(member[idx], minlength=n_communities).astype(np.float64)
    return {"mean_bits": float(h[has].mean()),
            "member_weighted_bits": float((h[has] * members[has]).sum() / members[has].sum()),
            "n_communities_with_topics": int(has.sum())}


def variant_diagnostics(membership: np.ndarray, la: np.ndarray, lb: np.ndarray, weight: np.ndarray,
                        member: np.ndarray, topics: Topics, modularity: float) -> dict:
    """Largest community share, edge density and topic entropy for one variant
    (section 4 item 5). Shares are given over nodes (phrases in the pruned
    graph) and over sub-units with a community."""
    n_nodes = len(membership)
    n_comm = int(membership.max()) + 1 if n_nodes else 0
    sizes = np.bincount(membership, minlength=n_comm) if n_comm else np.zeros(0, dtype=np.int64)
    has_member = member >= 0
    members = np.bincount(member[has_member], minlength=n_comm) if n_comm else np.zeros(0, dtype=np.int64)
    same = membership[la] == membership[lb] if len(la) else np.zeros(0, dtype=bool)
    # design gap: "edge density" is reported three ways, the pruned graph's
    # density, the share of edges inside a community, and the mean density
    # inside communities of two or more nodes
    pairs_inside = (sizes.astype(np.float64) * (sizes - 1) / 2) if n_comm else np.zeros(0)
    inside = np.bincount(membership[la][same], minlength=n_comm) if len(la) else np.zeros(n_comm)
    multi = sizes >= 2
    within = float((inside[multi] / pairs_inside[multi]).mean()) if multi.any() else None
    return {
        "n_communities": n_comm,
        "modularity": modularity,
        "largest_community_share_nodes": float(sizes.max() / n_nodes) if n_nodes else None,
        "largest_community_share_units": (float(members.max() / has_member.sum())
                                          if has_member.any() else None),
        "n_units_with_community": int(has_member.sum()),
        "n_units_without_community": int((~has_member).sum()),
        "graph_density": (float(2 * len(la) / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else None),
        "intra_community_edge_share": float(same.mean()) if len(la) else None,
        "intra_community_weight_share": (float(weight[same].sum() / weight.sum())
                                         if len(la) and weight.sum() > 0 else None),
        "within_community_density_mean": within,
        "community_size_mean_nodes": float(sizes.mean()) if n_comm else None,
        "community_members_mean_units": float(members.mean()) if n_comm else None,
        "topic_entropy": topic_entropy(member, topics, n_comm),
    }


# ----------------------------------------------------------------------------
# Build, write, load
# ----------------------------------------------------------------------------
@dataclass
class GraphBuild:
    phrases: pd.DataFrame
    edges: pd.DataFrame
    communities: pd.DataFrame
    units: pd.DataFrame
    diagnostics: dict

    def membership(self, variant: str) -> dict[str, int]:
        """{unit_id: community_id} over the sub-units that have a community."""
        col = self.units[f"community_{variant}"].to_numpy()
        ids = self.units["unit_id"].to_numpy()
        return {str(u): int(c) for u, c in zip(ids[col >= 0], col[col >= 0])}

    def prototypes(self, variant: str) -> tuple[np.ndarray, np.ndarray]:
        """(community ids, (n, 384) prototype matrix) for one variant."""
        sub = self.communities[self.communities["variant"] == variant]
        ids = sub["community_id"].to_numpy().astype(np.int32)
        vecs = (np.stack([np.asarray(v, dtype=np.float32) for v in sub["prototype"]])
                if len(sub) else np.zeros((0, DIM), dtype=np.float32))
        return ids, vecs


def build(subunit_texts: list[str], subunit_ids: list[str], subunit_vectors: np.ndarray,
          unit_topics, prototypes, out_dir: str | Path | None, n_process: int = 4,
          seed: int = SEED, nlp=None, batch_size: int = 256) -> GraphBuild:
    """The whole of section 4 items 4 and 5 over the given sub-units.

    subunit_vectors is (n_units, 384). unit_topics holds rows (unit_id,
    topic_id, prob) as a DataFrame or an iterable of tuples; prototypes maps
    topic_id to a 384-vector. Both may be empty, in which case every topic
    factor is 1 and the topic variant equals the plain one. Writes the files
    named in the module docstring when out_dir is given.
    """
    if len(subunit_texts) != len(subunit_ids):
        raise ValueError(f"{len(subunit_texts)} texts, {len(subunit_ids)} ids")
    vectors = np.asarray(subunit_vectors, dtype=np.float32)
    if vectors.shape[0] != len(subunit_ids):
        raise ValueError(f"{vectors.shape[0]} vectors, {len(subunit_ids)} ids")
    t_start = time.time()
    seconds: dict[str, float] = {}

    occ = extract_phrases(subunit_texts, n_process=n_process, batch_size=batch_size, nlp=nlp)
    seconds["phrases"] = occ.seconds
    _log(f"{occ.n_units} sub-units, {occ.n_phrases} phrases, {occ.n_occurrences} occurrences "
         f"({occ.seconds:.0f}s, {occ.n_process} processes)")

    t0 = time.time()
    freq = occ.freq()
    # design gap: the design fixes no order between the declared hub rule and
    # graphrag's pruning (HUB_RULE_ORDER); the hub rule runs first here.
    hubs = hub_rule(occ)
    node_mask = ~hubs.dropped
    edges = cooccurrence(occ, node_mask)
    weight = pmi_weights(edges, freq, node_mask) if NORMALIZE_EDGE_WEIGHTS else edges.count.astype(np.float64)
    pruned = prune(edges, weight, freq, node_mask)
    seconds["edges_and_pruning"] = time.time() - t0
    _log(f"hub rule dropped {int(hubs.is_pronoun.sum())} pronoun and {hubs.n_hub} hub phrases; "
         f"{len(edges)} edges from {edges.n_pairs} pairs; {int(pruned.edge_kept.sum())} edges kept")

    # Nodes of the Leiden graph: phrases with a surviving edge. graphrag's
    # cluster_graph.py lines 58 to 88 builds the Leiden input from the edge
    # list alone, so an entity without a surviving edge is in no community.
    ea, eb = edges.a[pruned.edge_kept], edges.b[pruned.edge_kept]
    ew = weight[pruned.edge_kept]
    ec = edges.count[pruned.edge_kept]
    nodes = np.unique(np.concatenate([ea, eb])).astype(np.int32)
    local = np.full(occ.n_phrases, -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    la, lb = local[ea], local[eb]

    t0 = time.time()
    topics = topic_inputs(subunit_ids, unit_topics, prototypes)
    tvec = phrase_topic_vectors(occ, nodes, topics)
    ew_topic = ew * topic_factor(tvec, la, lb)
    seconds["topic_weights"] = time.time() - t0

    per_variant = {}
    members = {}
    protos = {}
    node_comm = {}
    for variant, w in (("plain", ew), ("topic", ew_topic)):
        t0 = time.time()
        membership, modularity = leiden(len(nodes), la, lb, w, seed=seed)
        n_comm = int(membership.max()) + 1 if len(membership) else 0
        nc = np.full(occ.n_phrases, NO_COMMUNITY, dtype=np.int32)
        nc[nodes] = membership
        member = unit_membership(occ, nc, n_comm)
        proto = community_prototypes(member, vectors, n_comm)
        seconds[f"leiden_{variant}"] = time.time() - t0
        per_variant[variant] = variant_diagnostics(membership, la, lb, w, member, topics, modularity)
        members[variant], protos[variant], node_comm[variant] = member, proto, (membership, n_comm)
        _log(f"{variant}: {n_comm} communities, modularity {modularity:.3f}, "
             f"largest share (nodes) {per_variant[variant]['largest_community_share_nodes']}")

    kept = np.zeros(occ.n_phrases, dtype=bool)
    kept[nodes] = True
    phrases = pd.DataFrame({
        "phrase_id": np.arange(occ.n_phrases, dtype=np.int32), "text": occ.vocab, "freq": freq,
        "pron_count": occ.pron_count(), "is_pronoun": hubs.is_pronoun, "is_hub": hubs.is_hub,
        "dropped_by_hub_rule": hubs.dropped, "kept": kept})
    edges_df = pd.DataFrame({"a": ea, "b": eb, "count": ec, "weight": ew, "weight_topic": ew_topic})
    rows = []
    for variant in VARIANTS:
        membership, n_comm = node_comm[variant]
        member = members[variant]
        by_comm_nodes = _group(membership, nodes, n_comm)
        by_comm_units = _group(member, np.arange(occ.n_units), n_comm)
        for c in range(n_comm):
            rows.append({"variant": variant, "community_id": c,
                         "phrase_ids": by_comm_nodes[c].astype(np.int32),
                         "member_unit_ids": [subunit_ids[i] for i in by_comm_units[c]],
                         "prototype": protos[variant][c], "size": int(len(by_comm_nodes[c])),
                         "n_members": int(len(by_comm_units[c]))})
    communities = pd.DataFrame(rows, columns=["variant", "community_id", "phrase_ids", "member_unit_ids",
                                              "prototype", "size", "n_members"])
    n_kept_per_unit = np.bincount(occ.unit_of_occurrence()[kept[occ.phrase]], minlength=occ.n_units)
    units = pd.DataFrame({"unit_id": list(subunit_ids), "n_phrases": occ.phrases_per_unit().astype(np.int32),
                          "n_kept_phrases": n_kept_per_unit.astype(np.int32),
                          "community_plain": members["plain"], "community_topic": members["topic"]})
    seconds["total"] = time.time() - t_start
    diagnostics = {
        "graphrag_reference": GRAPHRAG_REF,
        "spacy_model": SPACY_MODEL, "seed": seed, "n_process": occ.n_process,
        "hub_share": HUB_SHARE, "max_phrases_per_unit": MAX_PHRASES_PER_UNIT,
        "pronoun_rule": PRONOUN_RULE, "hub_rule_order": HUB_RULE_ORDER,
        "article_rule": "every article token (a, an, the) stripped from the lemma",
        "prune": {"min_node_freq": MIN_NODE_FREQ, "min_node_degree": MIN_NODE_DEGREE,
                  "min_edge_weight_pct": MIN_EDGE_WEIGHT_PCT, "remove_ego_nodes": REMOVE_EGO_NODES,
                  "normalize_edge_weights": NORMALIZE_EDGE_WEIGHTS},
        "n_units": occ.n_units, "n_occurrences": occ.n_occurrences,
        "phrases_per_unit_mean": float(occ.phrases_per_unit().mean()) if occ.n_units else 0.0,
        "units_at_phrase_cap": int((occ.phrases_per_unit() >= MAX_PHRASES_PER_UNIT).sum()),
        "phrase_throughput_per_s": (occ.n_units / occ.seconds) if occ.seconds > 0 else None,
        "n_phrases": occ.n_phrases,
        "n_pronoun_phrases": int(hubs.is_pronoun.sum()), "n_hub_phrases": hubs.n_hub,
        "n_dropped_by_hub_rule": int(hubs.dropped.sum()),
        "n_nodes_after_hub_rule": int(node_mask.sum()),
        "n_pairs_raw": edges.n_pairs, "n_edges_before_pruning": len(edges),
        "ego_node": (occ.vocab[pruned.ego] if pruned.ego is not None else None),
        "n_removed_by_degree": pruned.n_removed_degree, "n_removed_by_freq": pruned.n_removed_freq,
        "n_nodes_after_node_rules": int(pruned.node_kept.sum()),
        "n_edges_removed_by_node_rules": pruned.n_removed_edges_nodes,
        "min_edge_weight": None if np.isnan(pruned.min_weight) else pruned.min_weight,
        "n_edges_removed_by_percentile": pruned.n_removed_edges_pct,
        "n_edges_removed_nonpositive": pruned.n_removed_nonpositive,
        "n_nodes_after_pruning": int(len(nodes)), "n_edges_after_pruning": int(len(ea)),
        "topics": {"n_topics": topics.n_topics, "n_rows": topics.n_rows,
                   "n_rows_ignored_unknown_unit": topics.n_ignored_units,
                   "n_rows_ignored_unknown_topic": topics.n_ignored_topics,
                   "n_units_with_topic_mass": int((np.asarray(topics.R.sum(axis=1)).ravel() > 0).sum()),
                   "n_nodes_with_topic_vector": int((np.linalg.norm(tvec, axis=1) > 0).sum())},
        "variants": per_variant,
        "seconds": seconds,
    }
    out = GraphBuild(phrases=phrases, edges=edges_df, communities=communities, units=units,
                     diagnostics=diagnostics)
    if out_dir is not None:
        write(out, out_dir)
    return out


def _group(labels: np.ndarray, items: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """items split by label (labels < 0 ignored), each group in item order."""
    m = labels >= 0
    order = np.argsort(labels[m], kind="stable")
    sorted_items = items[m][order]
    counts = np.bincount(labels[m], minlength=n_groups)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [sorted_items[bounds[g]:bounds[g + 1]] for g in range(n_groups)]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write(g: GraphBuild, out_dir: str | Path) -> Path:
    """Write the five files of the module docstring."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(g.phrases, preserve_index=False), out / PHRASE_FILE)
    pq.write_table(pa.Table.from_pandas(g.edges, preserve_index=False), out / EDGE_FILE)
    comm = pa.table({
        "variant": pa.array(g.communities["variant"].tolist(), pa.string()),
        "community_id": pa.array(g.communities["community_id"].tolist(), pa.int32()),
        "phrase_ids": pa.array([list(map(int, v)) for v in g.communities["phrase_ids"]],
                               pa.list_(pa.int32())),
        "member_unit_ids": pa.array([list(map(str, v)) for v in g.communities["member_unit_ids"]],
                                    pa.list_(pa.string())),
        "prototype": pa.array([np.asarray(v, dtype=np.float32).tolist() for v in g.communities["prototype"]],
                              pa.list_(pa.float32())),
        "size": pa.array(g.communities["size"].tolist(), pa.int32()),
        "n_members": pa.array(g.communities["n_members"].tolist(), pa.int32()),
    })
    pq.write_table(comm, out / COMMUNITY_FILE)
    pq.write_table(pa.Table.from_pandas(g.units, preserve_index=False), out / UNIT_FILE)
    (out / DIAG_FILE).write_text(json.dumps(g.diagnostics, indent=2, default=_json_default))
    return out


def load(out_dir: str | Path) -> GraphBuild:
    """Read the files build() wrote. List columns come back as numpy arrays
    (phrase_ids, prototype) or lists of str (member_unit_ids)."""
    out = Path(out_dir)
    comm = pq.read_table(out / COMMUNITY_FILE).to_pandas()
    if len(comm):
        comm["phrase_ids"] = [np.asarray(v, dtype=np.int32) for v in comm["phrase_ids"]]
        comm["member_unit_ids"] = [list(v) for v in comm["member_unit_ids"]]
        comm["prototype"] = [np.asarray(v, dtype=np.float32) for v in comm["prototype"]]
    return GraphBuild(
        phrases=pq.read_table(out / PHRASE_FILE).to_pandas(),
        edges=pq.read_table(out / EDGE_FILE).to_pandas(),
        communities=comm,
        units=pq.read_table(out / UNIT_FILE).to_pandas(),
        diagnostics=json.loads((out / DIAG_FILE).read_text()),
    )


# ----------------------------------------------------------------------------
# CLI: the full build per corpus
# ----------------------------------------------------------------------------
def load_unit_topics(path: str | None) -> pd.DataFrame | None:
    """Parquet or CSV with columns unit_id, topic_id, prob."""
    if not path:
        return None
    p = Path(path)
    df = pd.read_csv(p) if p.suffix == ".csv" else pq.read_table(p).to_pandas()
    missing = {"unit_id", "topic_id", "prob"} - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing columns {sorted(missing)}")
    return df[["unit_id", "topic_id", "prob"]]


PROTOTYPE_ID_COLUMNS = ("topic_id", "topic")
PROTOTYPE_VECTOR_COLUMNS = ("prototype", "embedding", "vector")


def load_prototypes(path: str | None) -> dict:
    """{topic_id: vector} from a parquet or an npz (arrays topic_ids and vectors).

    A parquet needs an id column (topic_id or topic) and a vector column
    (prototype, the column part1.topics.save writes, or embedding or vector).
    The outlier topic -1 is skipped: it has no prototype of its own in the
    design and BERTopic's row for it is not a topic."""
    if not path:
        return {}
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p)
        return {_scalar(t): np.asarray(v, dtype=np.float32) for t, v in zip(z["topic_ids"], z["vectors"])
                if _scalar(t) != -1}
    df = pq.read_table(p).to_pandas()
    id_col = next((c for c in PROTOTYPE_ID_COLUMNS if c in df.columns), None)
    vec_col = next((c for c in PROTOTYPE_VECTOR_COLUMNS if c in df.columns), None)
    if id_col is None or vec_col is None:
        raise ValueError(f"{p}: needs one of {PROTOTYPE_ID_COLUMNS} and one of {PROTOTYPE_VECTOR_COLUMNS}, "
                         f"has {list(df.columns)}")
    return {_scalar(t): np.asarray(v, dtype=np.float32) for t, v in zip(df[id_col], df[vec_col])
            if _scalar(t) != -1}


def _scalar(t):
    return int(t) if isinstance(t, (np.integer, int)) else t


def corpus_units(corpus: str, longmemeval: str, multihoprag: str, limit: int = 0,
                 encoder=None) -> tuple[list[str], list[str], np.ndarray]:
    """(sub-unit ids, texts, vectors) for lme or mhrag through part1.units.

    lme: every session of the LongMemEval file in sorted session-id order (the
    e5 order); vectors through encode_tables so the Encoder cache keys are the
    e5 ones. mhrag: the corpus in loader order through build_doc_tables with
    the bench TokenCounter and encode_doc_tables. limit > 0 keeps the first
    `limit` sub-units and encodes just those (a smoke, its own cache entry).
    """
    from ..index.encoder import Encoder
    from . import units as U

    enc = encoder or Encoder()
    if corpus == "lme":
        from ..data.longmemeval import load_longmemeval

        ds = load_longmemeval(longmemeval)
        sids = sorted({s for x in ds["instances"] for s in x.session_ids})
        tables = U.build_tables(ds["sessions"], sids)
        ids, texts = U.sub_unit_texts(tables)
        if limit:
            ids, texts = ids[:limit], texts[:limit]
            return ids, texts, enc.encode(texts)
        U.encode_tables(enc, ds["sessions"], sids, tables)
        vecs = np.vstack([t.sub_matrix() for t in tables.values()]) if tables else np.zeros((0, DIM), np.float32)
        return ids, texts, vecs
    if corpus == "mhrag":
        from ..data.multihoprag import load_corpus
        from .render import default_counter

        docs = load_corpus(Path(multihoprag))
        tables = U.build_doc_tables(docs, default_counter())
        ids, texts = [], []
        for t in tables.values():
            for s in t.sentence_units:
                ids.append(s.unit_id)
                texts.append(s.text)
        if limit:
            ids, texts = ids[:limit], texts[:limit]
            return ids, texts, enc.encode(texts)
        U.encode_doc_tables(enc, tables)
        vecs = [np.vstack(t.sentence_vecs) for t in tables.values() if t.sentence_vecs]
        return ids, texts, (np.vstack(vecs) if vecs else np.zeros((0, DIM), np.float32))
    raise ValueError(f"unknown corpus {corpus}; use lme or mhrag")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Noun-phrase graph and communities (design section 4, items 4 and 5)")
    ap.add_argument("--corpus", required=True, choices=["lme", "mhrag"])
    ap.add_argument("--out", default=None, help="output directory (default data/part1/graph/<corpus>)")
    ap.add_argument("--longmemeval", default="data/raw/longmemeval_s.json")
    ap.add_argument("--multihoprag", default="data/raw/multihoprag")
    ap.add_argument("--unit-topics", default=None, help="parquet or csv: unit_id, topic_id, prob")
    ap.add_argument("--prototypes", default=None,
                    help="parquet (topic_id, prototype; part1.topics topics.parquet) or npz (topic_ids, vectors)")
    ap.add_argument("--n-process", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N sub-units only (smoke)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else Path("data/part1/graph") / args.corpus
    t0 = time.time()
    ids, texts, vecs = corpus_units(args.corpus, args.longmemeval, args.multihoprag, args.limit)
    _log(f"{args.corpus}: {len(ids)} sub-units loaded and encoded ({time.time() - t0:.0f}s)")
    unit_topics = load_unit_topics(args.unit_topics)
    prototypes = load_prototypes(args.prototypes)
    if unit_topics is None or not prototypes:
        _log("no unit topics or prototypes given: the topic variant equals the plain one")
    g = build(texts, ids, vecs, unit_topics, prototypes, out, n_process=args.n_process,
              seed=args.seed, batch_size=args.batch_size)
    _log(f"written to {out}")
    print(json.dumps({k: v for k, v in g.diagnostics.items() if k != "variants"}, indent=1, default=_json_default))
    print(json.dumps(g.diagnostics["variants"], indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())

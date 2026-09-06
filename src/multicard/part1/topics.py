"""Topic model for Part 1 (design section 4, item 3).

BERTopic 0.17.4 (the exact version is fixed by uv.lock and recorded in
diagnostics.json; pyproject holds the range bertopic>=0.16,<0.18, which is
not a pin) over the precomputed MiniLM vectors and texts of the owner units:
turns on LongMemEval
(both roles), chunks on MultiHop-RAG. Settings copied from the design:
UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
random_state=13), HDBSCAN at BERTopic's defaults, calculate_probabilities
False, no LLM anywhere in the model. The seed is for reproducibility only.

What comes out:

- topics.parquet: one row per BERTopic topic, the outlier topic (-1)
  included when HDBSCAN produced outliers. Columns topic_id (int32), terms
  (list<string>, the top 10 c-TF-IDF terms), name (string, the terms joined
  by a space), prototype (list<float32>, the topic_embeddings_ row as
  BERTopic computed it, the plain mean of the member vectors, not unit
  length: section 4 item 5 takes the probability-weighted mean of these raw
  rows for a phrase's topic vector, and retrieve takes the cosine at read
  time, which the norm does not change), size (int32, owner units labelled
  with the topic), is_outlier (bool).
- unit_topics.parquet: the sparse per-unit topic distribution. Columns
  unit_id (string), topic_id (int32), prob (float32). The distribution is
  the softmax of the cosine similarity between the unit vector and every row
  of topic_embeddings_, and the top 5 rows are kept per unit, best first.
  The outlier row is part of that softmax, so topic_id -1 can appear here.
- unit_labels.parquet: the hard label per unit. Columns unit_id (string),
  label (int32, the HDBSCAN topic, -1 for an outlier), is_outlier (bool).
  This is the "outlier status kept" of the design and the label that
  topic_members() uses. A topic's members are the sub-units of the owner
  units labelled with it; outlier units are members of no topic.
- diagnostics.json: n_units, n_topics (outlier topic not counted),
  outlier_share, largest_topic_share (largest real topic over all units),
  seconds, the settings and the library versions.

Memory. The vector matrix is held once: fit() makes it float32 and
C-contiguous (a copy only when it is not already), UMAP's check_array then
keeps a reference, BERTopic slices one topic at a time for the topic vectors,
and the unit-to-topic similarity runs in blocks of 20,000 rows. For 200,000
units by 384 the matrix is 307 MB; the similarity block is 20,000 by T
float32 (80 MB at T = 1,000); UMAP's neighbour graph and HDBSCAN's tree add
some hundreds of MB. Expected peak: under about 2 GB, an estimate, not a
measurement. The full LongMemEval fit is a separate documented step run by
the main session through the CLI below; this module is tested on samples of
at most 2,000 units.

CLI: python -m multicard.part1.topics --corpus lme|mhrag --out DIR
[--limit N] [--data PATH] [--raw DIR]. Loads the owner texts through the
units module (build_tables over the sorted list of every haystack session,
or build_doc_tables over the corpus) and encodes them through the bench
Encoder in e5's blocks of 20,000, so the cache keys are the ones
encode_tables and encode_doc_tables write and a full cache means no encoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..data.longmemeval import load_longmemeval
from ..data.multihoprag import RAW as MHR_RAW
from ..data.multihoprag import load_corpus
from ..experiments.e5_longmemeval import DATA as LME_DATA
from ..experiments.e5_longmemeval import _encode_chunked
from ..index.encoder import Encoder
from ..utils.seeds import set_seed
from . import units as U
from .render import default_counter

SEED = 13
TOP_K = 5
TOP_TERMS = 10
DIM = 384
BLOCK = 20000
UMAP_KWARGS = {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0,
               "metric": "cosine", "random_state": SEED}

TOPICS_FILE = "topics.parquet"
UNIT_TOPICS_FILE = "unit_topics.parquet"
UNIT_LABELS_FILE = "unit_labels.parquet"
DIAG_FILE = "diagnostics.json"


@dataclass
class TopicResult:
    """A fitted topic model as the frozen tables describe it.

    Topic arrays are aligned by row and sorted by topic id, so the outlier
    topic, when present, is row 0. Unit arrays are aligned with unit_ids.
    top_ids and top_probs are (n_units, k) with k = min(5, number of topic
    rows), best topic first in every row.
    """
    topic_ids: np.ndarray      # (T,) int32
    terms: list[list[str]]     # per topic, at most 10
    names: list[str]           # per topic, the terms joined by a space
    prototypes: np.ndarray     # (T, 384) float32, the raw topic_embeddings_ rows
    sizes: np.ndarray          # (T,) int32
    is_outlier: np.ndarray     # (T,) bool
    unit_ids: list[str]        # (n,)
    labels: np.ndarray         # (n,) int32, -1 for an outlier
    top_ids: np.ndarray        # (n, k) int32, topic ids
    top_probs: np.ndarray      # (n, k) float32
    diagnostics: dict

    @property
    def n_topics(self) -> int:
        """Number of real topics; the outlier topic is not counted."""
        return int((~self.is_outlier).sum())

    def row_of(self, topic_id: int) -> int:
        """Row index of a topic id in the topic arrays."""
        rows = np.nonzero(self.topic_ids == topic_id)[0]
        if len(rows) == 0:
            raise KeyError(topic_id)
        return int(rows[0])

    def unit_labels(self) -> dict[str, int]:
        return {u: int(l) for u, l in zip(self.unit_ids, self.labels)}

    def unit_distribution(self) -> dict[str, list[tuple[int, float]]]:
        """unit id -> [(topic id, prob)] best first."""
        return {u: [(int(t), float(p)) for t, p in zip(ts, ps)]
                for u, ts, ps in zip(self.unit_ids, self.top_ids, self.top_probs)}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model():
    """BERTopic with the design's UMAP and BERTopic's own default HDBSCAN.

    calculate_probabilities is False, top_n_words is 10 (the c-TF-IDF terms
    the name is built from), and no embedding model is set because every
    vector comes precomputed from the bench Encoder.
    """
    from bertopic import BERTopic
    from umap import UMAP

    # design gap: "HDBSCAN defaults" is read as BERTopic's default hdbscan
    # model (hdbscan.HDBSCAN with min_cluster_size 10 from min_topic_size,
    # metric euclidean, cluster_selection_method eom, prediction_data True),
    # the same reading that gives the design's UMAP arguments; hdbscan.HDBSCAN()
    # on its own would use min_cluster_size 5.
    return BERTopic(umap_model=UMAP(**UMAP_KWARGS), calculate_probabilities=False,
                    top_n_words=TOP_TERMS, embedding_model=None, verbose=False)


def _unit_norm(m: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; a zero row stays zero. Returns a copy."""
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def topic_distribution(vectors: np.ndarray, prototypes: np.ndarray, k: int = TOP_K,
                       block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Softmax of cosine similarity against every prototype row, top k kept.

    Returns (top_rows, top_probs), each (n, min(k, T)): top_rows are row
    indices into `prototypes`, best first; top_probs the softmax values.
    Works in blocks of `block` rows so only one block of similarities is in
    memory at a time. Ties keep the lower prototype row first.
    """
    protos = _unit_norm(prototypes)
    n, t = len(vectors), len(protos)
    kk = min(k, t)
    top_rows = np.zeros((n, kk), dtype=np.int32)
    top_probs = np.zeros((n, kk), dtype=np.float32)
    if n == 0 or t == 0:
        return top_rows, top_probs
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        b = _unit_norm(vectors[lo:hi])
        sims = b @ protos.T
        sims -= sims.max(axis=1, keepdims=True)
        np.exp(sims, out=sims)
        sims /= sims.sum(axis=1, keepdims=True)
        if kk < t:
            # the kept set, then sorted by value with the lower row first on a tie
            part = np.sort(np.argpartition(-sims, kk - 1, axis=1)[:, :kk], axis=1)
            vals = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-vals, axis=1, kind="stable")
            rows = np.take_along_axis(part, order, axis=1)
            vals = np.take_along_axis(vals, order, axis=1)
        else:
            rows = np.argsort(-sims, axis=1, kind="stable")[:, :kk]
            vals = np.take_along_axis(sims, rows, axis=1)
        top_rows[lo:hi] = rows
        top_probs[lo:hi] = vals
    return top_rows, top_probs


def _versions() -> dict:
    out = {}
    for name in ("bertopic", "umap-learn", "hdbscan", "numba", "scikit-learn", "numpy"):
        try:
            out[name] = _pkg_version(name)
        except Exception:
            out[name] = None
    return out


def fit(vectors: np.ndarray, texts: list[str], unit_ids: list[str],
        out_dir: str | Path | None = None) -> TopicResult:
    """Fit the topic model and, when out_dir is given, write the frozen tables.

    vectors: (n, 384) MiniLM unit vectors in unit_ids order. texts: the unit
    texts in the same order (c-TF-IDF reads them). Returns the TopicResult;
    the same input gives the same labels on the same machine.
    """
    n = len(unit_ids)
    if len(texts) != n or len(vectors) != n:
        raise ValueError(f"{n} unit ids, {len(texts)} texts, {len(vectors)} vectors")
    if n == 0:
        raise ValueError("no units to fit")
    # One copy at most: only when the input is not already float32 and C order.
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != DIM:
        raise ValueError(f"vectors must be (n, {DIM}), got {vectors.shape}")
    docs = [str(t) for t in texts]

    t0 = time.time()
    model = build_model()
    model.fit_transform(docs, embeddings=vectors)
    labels = np.asarray(model.topics_, dtype=np.int32)

    topic_ids = np.asarray(sorted(int(t) for t in model.get_topics()), dtype=np.int32)
    embeddings = np.asarray(model.topic_embeddings_, dtype=np.float32)
    if embeddings.shape[0] != len(topic_ids):
        raise RuntimeError(f"{embeddings.shape[0]} topic vectors for {len(topic_ids)} topics")
    if set(int(l) for l in np.unique(labels)) != set(int(t) for t in topic_ids):
        raise RuntimeError("topic ids and unit labels disagree")
    terms = []
    for t in topic_ids:
        words = [w for w, _ in (model.get_topic(int(t)) or []) if w]
        terms.append(words[:TOP_TERMS])
    # design gap: the join of the name string is not fixed; a single space
    names = [" ".join(ws) for ws in terms]
    # The raw topic_embeddings_ rows are kept (section 4 item 5 reads them);
    # topic_distribution normalises its own copy for the cosine.
    prototypes = np.ascontiguousarray(embeddings, dtype=np.float32)
    sizes = np.asarray([int((labels == t).sum()) for t in topic_ids], dtype=np.int32)
    is_outlier = topic_ids == -1

    top_rows, top_probs = topic_distribution(vectors, prototypes, TOP_K)
    top_ids = topic_ids[top_rows].astype(np.int32)
    seconds = time.time() - t0

    real = sizes[~is_outlier]
    hdb = model.hdbscan_model
    diagnostics = {
        "n_units": int(n),
        "n_topics": int(len(real)),
        "outlier_share": float((labels == -1).mean()),
        "largest_topic_share": float(real.max() / n) if len(real) else 0.0,
        "seconds": round(seconds, 3),
        "seed": SEED,
        "top_k": TOP_K,
        "top_terms": TOP_TERMS,
        "umap": dict(UMAP_KWARGS),
        "hdbscan": {k: v for k, v in hdb.get_params().items()
                    if k in ("min_cluster_size", "min_samples", "metric",
                             "cluster_selection_method", "prediction_data")},
        "calculate_probabilities": False,
        "versions": _versions(),
    }
    result = TopicResult(topic_ids=topic_ids, terms=terms, names=names, prototypes=prototypes,
                         sizes=sizes, is_outlier=is_outlier, unit_ids=list(unit_ids),
                         labels=labels, top_ids=top_ids, top_probs=top_probs,
                         diagnostics=diagnostics)
    if out_dir is not None:
        save(result, out_dir)
    return result


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def save(result: TopicResult, out_dir: str | Path) -> Path:
    """Write topics.parquet, unit_topics.parquet, unit_labels.parquet and
    diagnostics.json to out_dir. Returns out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    topics = pa.table({
        "topic_id": pa.array(result.topic_ids, type=pa.int32()),
        "terms": pa.array(result.terms, type=pa.list_(pa.string())),
        "name": pa.array(result.names, type=pa.string()),
        "prototype": pa.array(result.prototypes.tolist(), type=pa.list_(pa.float32())),
        "size": pa.array(result.sizes, type=pa.int32()),
        "is_outlier": pa.array(result.is_outlier, type=pa.bool_()),
    })
    pq.write_table(topics, out / TOPICS_FILE)

    n, k = result.top_ids.shape
    unit_topics = pa.table({
        "unit_id": pa.array([u for u in result.unit_ids for _ in range(k)], type=pa.string()),
        "topic_id": pa.array(result.top_ids.reshape(-1), type=pa.int32()),
        "prob": pa.array(result.top_probs.reshape(-1), type=pa.float32()),
    })
    pq.write_table(unit_topics, out / UNIT_TOPICS_FILE)

    unit_labels = pa.table({
        "unit_id": pa.array(result.unit_ids, type=pa.string()),
        "label": pa.array(result.labels, type=pa.int32()),
        "is_outlier": pa.array(result.labels == -1, type=pa.bool_()),
    })
    pq.write_table(unit_labels, out / UNIT_LABELS_FILE)
    (out / DIAG_FILE).write_text(json.dumps(result.diagnostics, indent=2))
    return out


def load(out_dir: str | Path) -> TopicResult:
    """Read the tables written by save() back into a TopicResult."""
    out = Path(out_dir)
    topics = pq.read_table(out / TOPICS_FILE)
    topic_ids = np.asarray(topics["topic_id"].to_pylist(), dtype=np.int32)
    terms = [list(ws) for ws in topics["terms"].to_pylist()]
    names = topics["name"].to_pylist()
    prototypes = np.asarray(topics["prototype"].to_pylist(), dtype=np.float32).reshape(len(topic_ids), -1)
    sizes = np.asarray(topics["size"].to_pylist(), dtype=np.int32)
    is_outlier = np.asarray(topics["is_outlier"].to_pylist(), dtype=bool)

    labels_t = pq.read_table(out / UNIT_LABELS_FILE)
    unit_ids = labels_t["unit_id"].to_pylist()
    labels = np.asarray(labels_t["label"].to_pylist(), dtype=np.int32)

    ut = pq.read_table(out / UNIT_TOPICS_FILE)
    n = len(unit_ids)
    rows = ut.num_rows
    if n == 0:
        k = 0
    elif rows % n:
        raise ValueError(f"{rows} unit_topics rows for {n} units")
    else:
        k = rows // n
    if ut["unit_id"].to_pylist() != [u for u in unit_ids for _ in range(k)]:
        raise ValueError("unit_topics rows are not in unit_labels order")
    top_ids = np.asarray(ut["topic_id"].to_pylist(), dtype=np.int32).reshape(n, k)
    top_probs = np.asarray(ut["prob"].to_pylist(), dtype=np.float32).reshape(n, k)

    diag_path = out / DIAG_FILE
    diagnostics = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    return TopicResult(topic_ids=topic_ids, terms=terms, names=names, prototypes=prototypes,
                       sizes=sizes, is_outlier=is_outlier, unit_ids=unit_ids, labels=labels,
                       top_ids=top_ids, top_probs=top_probs, diagnostics=diagnostics)


# ----------------------------------------------------------------------------
# Members
# ----------------------------------------------------------------------------
def topic_members(result: TopicResult, tables: dict) -> dict[int, list[str]]:
    """topic id -> sub-unit ids of the owner units labelled with it.

    tables maps a container id (session or document) to its SessionTable or
    DocTable from the units module; the sub-units of turn "sid#i" are
    tables[sid].subs[i], of chunk "doc#k" tables[doc].sentences[k]. Outlier
    units (label -1) are in no topic. Order inside a topic is unit order,
    then sub-unit position.
    """
    members: dict[int, list[str]] = {int(t): [] for t in result.topic_ids if t != -1}
    for uid, label in zip(result.unit_ids, result.labels):
        if label == -1:
            continue
        container, index = U.container_of(uid), U.index_of(uid)
        if container not in tables:
            raise ValueError(f"no table for container {container!r} of unit {uid!r}")
        table = tables[container]
        groups = table.subs if isinstance(table, U.SessionTable) else table.sentences
        members[int(label)].extend(s.unit_id for s in groups[index])
    return members


# ----------------------------------------------------------------------------
# Corpus loading for the CLI
# ----------------------------------------------------------------------------
def corpus_units(corpus: str, enc: Encoder | None = None, limit: int = 0,
                 data: str | Path = LME_DATA, raw: str | Path = MHR_RAW
                 ) -> tuple[list[str], list[str], np.ndarray]:
    """(unit ids, texts, vectors) of the owner units of a corpus.

    lme: turns of every haystack session of every question, sessions in
    sorted id order (the e5 list); mhrag: chunks of every article in loader
    order, chunked with the bench TokenCounter as build_doc_tables does.
    limit > 0 keeps the first `limit` units before encoding, for smokes.
    Vectors go through e5 _encode_chunked, so the Encoder cache keys are the
    ones encode_tables and encode_doc_tables use.
    """
    enc = enc or Encoder()
    if corpus == "lme":
        ds = load_longmemeval(data)
        sids = sorted({s for x in ds["instances"] for s in x.session_ids})
        ids, texts = U.turn_texts(U.build_tables(ds["sessions"], sids))
        label = "turns"
    elif corpus == "mhrag":
        tables = U.build_doc_tables(load_corpus(Path(raw)), default_counter())
        ids, texts = U.chunk_texts(tables)
        label = "chunks"
    else:
        raise ValueError(f"unknown corpus {corpus!r}; use lme or mhrag")
    if limit:
        ids, texts = ids[:limit], texts[:limit]
    vectors = _encode_chunked(enc, texts, label)
    return ids, texts, vectors


def main(argv: list[str] | None = None) -> dict:
    """python -m multicard.part1.topics --corpus lme|mhrag --out DIR [--limit N]"""
    p = argparse.ArgumentParser(description="Fit the Part 1 topic model (design section 4, item 3).")
    p.add_argument("--corpus", required=True, choices=("lme", "mhrag"))
    p.add_argument("--out", required=True, help="directory for topics.parquet and its companions")
    p.add_argument("--limit", type=int, default=0, help="fit on the first N units only (smoke)")
    p.add_argument("--data", default=LME_DATA, help="LongMemEval_S json path")
    p.add_argument("--raw", default=str(MHR_RAW), help="MultiHop-RAG raw directory")
    a = p.parse_args(argv)
    set_seed(SEED)
    t0 = time.time()
    ids, texts, vectors = corpus_units(a.corpus, limit=a.limit, data=a.data, raw=a.raw)
    print(f"[topics] {a.corpus}: {len(ids)} units, vectors {vectors.shape}, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)
    result = fit(vectors, texts, ids, a.out)
    print(json.dumps(result.diagnostics, indent=2), flush=True)
    return result.diagnostics


if __name__ == "__main__":
    main(sys.argv[1:])

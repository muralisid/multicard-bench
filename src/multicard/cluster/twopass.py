"""Pass two: discover structure among the items the gate admitted.

Reduction followed by density clustering, with no preset number of clusters and
an explicit noise label, then naming from derived keyword lists rather than from
the items themselves. The naming step is what keeps generative cost proportional
to the number of clusters instead of the number of items, which is the economic
claim the study measures.

Incremental arrivals are assigned to existing clusters by similarity to their
centroids, and discovery re-runs only over the residue that no cluster claims.
That is what makes the pattern usable on a corpus that keeps growing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    label: int
    members: list[int]
    centroid: np.ndarray
    keywords: list[str] = field(default_factory=list)
    name: str = ""


def reduce_dims(vecs: np.ndarray, n_components: int = 10, seed: int = 13) -> np.ndarray:
    """Reduce before clustering, to blunt distance concentration in high
    dimensions. UMAP where available, otherwise PCA, which keeps the pipeline
    runnable without the optional dependency."""
    n_components = min(n_components, vecs.shape[1], max(2, vecs.shape[0] - 1))
    try:
        import umap

        return umap.UMAP(n_components=n_components, metric="cosine",
                         random_state=seed).fit_transform(vecs)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=seed).fit_transform(vecs)


def discover(vecs: np.ndarray, min_cluster_size: int | None = None,
             seed: int = 13, reduce: bool = True) -> list[Cluster]:
    """Density clustering over (optionally reduced) embeddings."""
    from sklearn.cluster import HDBSCAN

    n = vecs.shape[0]
    if min_cluster_size is None:
        min_cluster_size = max(5, min(25, n // 100))
    X = reduce_dims(vecs, seed=seed) if reduce and n > 50 else vecs
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1,
                     metric="euclidean").fit_predict(X)

    clusters: list[Cluster] = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        members = [int(i) for i in np.where(labels == lab)[0]]
        c = vecs[members].mean(axis=0)
        norm = np.linalg.norm(c)
        clusters.append(Cluster(label=int(lab), members=members,
                                centroid=c / norm if norm > 0 else c))
    return clusters


def residue(labels: np.ndarray) -> list[int]:
    return [int(i) for i in np.where(np.asarray(labels) == -1)[0]]


def assign(vecs: np.ndarray, clusters: list[Cluster],
           threshold: float = 0.4) -> tuple[np.ndarray, list[int]]:
    """Assign new items to existing clusters, returning (labels, unassigned).

    Items whose best centroid similarity falls below the threshold are left for
    the next discovery pass rather than forced into a cluster they do not belong
    to, which is what keeps the structure honest as a corpus drifts.
    """
    if not clusters:
        return np.full(len(vecs), -1), list(range(len(vecs)))
    C = np.vstack([c.centroid for c in clusters])
    sims = np.asarray(vecs, dtype=np.float64) @ C.T
    best = sims.argmax(axis=1)
    bestsim = sims.max(axis=1)
    labels = np.array([clusters[b].label if s >= threshold else -1
                       for b, s in zip(best, bestsim)])
    return labels, [int(i) for i in np.where(labels == -1)[0]]


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_STOP = set("""the a an and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its it's we our you your they their he she his her
have has had do does did not no can could would should will shall may might must about into over
under more most other some such only own same so too very just also there here when where which
who whom what how why all any both each few many much per via""".split())


def keywords(texts: list[str], top_n: int = 12) -> list[str]:
    """Distinctive terms for a cluster, by frequency after stopword removal.

    Deliberately simple and free: the naming model sees these, never the items,
    so the quality bar is that a human could recognise the topic from the list.
    """
    counts = Counter()
    for t in texts:
        counts.update(w.lower() for w in _WORD.findall(t) if w.lower() not in _STOP)
    return [w for w, _ in counts.most_common(top_n)]


def summarise_for_naming(clusters: list[Cluster], texts: list[str],
                         examples: int = 3, top_n: int = 12) -> list[dict]:
    """Build the compact per-cluster description a naming model would receive.

    Returning it rather than calling a model keeps this module free to run and
    makes the cost of the naming step explicit and countable: one request per
    cluster, each carrying a keyword list and a few short excerpts, never the
    cluster's documents.
    """
    out = []
    for c in clusters:
        member_texts = [texts[i] for i in c.members]
        c.keywords = keywords(member_texts, top_n)
        out.append({
            "label": c.label,
            "size": len(c.members),
            "keywords": c.keywords,
            "examples": [t[:160] for t in member_texts[:examples]],
        })
    return out

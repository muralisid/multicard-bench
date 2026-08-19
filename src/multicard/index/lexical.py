"""BM25 lexical retrieval.

Pure Python via bm25s, chosen over Pyserini so the repository has no Java
dependency and a reader can reproduce results with one install command.
"""

from __future__ import annotations

import numpy as np


class BM25:
    def __init__(self, doc_ids: list[str], texts: list[str], stopwords: str = "en"):
        import bm25s

        self.doc_ids = list(doc_ids)
        self.stopwords = stopwords
        tokens = bm25s.tokenize(texts, stopwords=stopwords, show_progress=False)
        self._r = bm25s.BM25()
        self._r.index(tokens, show_progress=False)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        import bm25s

        q = bm25s.tokenize([query], stopwords=self.stopwords, show_progress=False)
        k = min(k, len(self.doc_ids))
        idx, scores = self._r.retrieve(q, k=k, show_progress=False)
        return [(self.doc_ids[int(i)], float(s)) for i, s in zip(idx[0], scores[0])]

    def rank(self, query: str, k: int = 100) -> list[str]:
        return [d for d, _ in self.search(query, k)]

    def score_matrix(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        return [self.search(q, k) for q in queries]


def fold_to_docs(hits: list[tuple[str, float]], owner: dict[str, str]) -> list[str]:
    """Fold card or chunk hits up to their parent document by best score.

    Every system in the study is scored at document level, so this runs before
    any metric is computed. Ties keep first-seen order, which is the higher score
    because the input is already sorted.
    """
    best: dict[str, float] = {}
    for unit_id, score in hits:
        doc = owner.get(unit_id, unit_id)
        if doc not in best or score > best[doc]:
            best[doc] = score
    return [d for d, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]


def dense_rank(sim_row: np.ndarray, ids: list[str], k: int = 100) -> list[str]:
    """Top-k ids by similarity, deterministic under ties via a stable sort."""
    k = min(k, len(ids))
    order = np.argsort(-sim_row, kind="stable")[:k]
    return [ids[int(i)] for i in order]

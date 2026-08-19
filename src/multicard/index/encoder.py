"""Encoder with an on-disk cache.

Every embedding the programme computes passes through here, so a rerun is a cache
hit rather than recomputation, and two seeded runs produce identical vectors.
The cache key covers the model name and the exact text, so changing either
invalidates only what it should.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(os.environ.get("MCB_CACHE", "data/cache/embeddings"))


class Encoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 256,
                 device: str | None = None, cache: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.cache = cache
        self._model = None
        self._mem: dict[str, np.ndarray] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _key(self, texts: list[str]) -> str:
        h = hashlib.sha256(self.model_name.encode())
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8"))
        return h.hexdigest()[:32]

    def encode(self, texts: list[str], normalise: bool = True) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised embeddings."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        key = self._key(texts)
        if key in self._mem:
            return self._mem[key]
        path = CACHE_DIR / f"{key}.npz"
        if self.cache and path.exists():
            v = np.load(path)["v"]
            self._mem[key] = v
            return v
        v = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalise,
            show_progress_bar=len(texts) > 5000,
        ).astype(np.float32)
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, v=v)
        self._mem[key] = v
        return v

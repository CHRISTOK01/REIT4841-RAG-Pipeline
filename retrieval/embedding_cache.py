"""Disk-backed cache for text embeddings, keyed by (model, text).

Both chunk and query embeddings flow through this cache so a given piece of
text is only ever sent to the OpenAI embeddings API once, across all runs --
this matters for API cost and for exact reproducibility of a sweep.

A cached vector is validated every time it is read (whole float32s, finite,
same dimension as its neighbours). A damaged row raises
`CorruptEmbeddingError` rather than being fed into the index.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

_SQLITE_MAX_VARIABLES_PER_QUERY = 900  # stay under SQLite's default ~999 parameter limit
_FLOAT32_BYTES = 4


class CorruptEmbeddingError(RuntimeError):
    """A cached or freshly returned embedding is unusable (truncated, non-finite,
    wrong dimension, or the wrong number of vectors came back)."""


def _validated(vector: np.ndarray, source: str) -> np.ndarray:
    if vector.ndim != 1 or vector.size == 0:
        raise CorruptEmbeddingError(f"{source}: embedding is empty or not a flat vector")
    if not np.all(np.isfinite(vector)):
        raise CorruptEmbeddingError(f"{source}: embedding contains NaN/inf")
    return vector


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  key TEXT PRIMARY KEY,"
            "  model TEXT NOT NULL,"
            "  vector BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def cache_key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Returns {cache_key: vector} for whichever of `texts` are already cached."""
        keys = [self.cache_key(model, t) for t in texts]
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), _SQLITE_MAX_VARIABLES_PER_QUERY):
            batch = keys[start : start + _SQLITE_MAX_VARIABLES_PER_QUERY]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", batch
            ).fetchall()
            for key, blob in rows:
                if len(blob) % _FLOAT32_BYTES != 0:
                    raise CorruptEmbeddingError(
                        f"{self.path}: cached embedding {key[:12]} is truncated ({len(blob)} bytes)"
                    )
                found[key] = _validated(np.frombuffer(blob, dtype=np.float32), f"{self.path} row {key[:12]}")
        return found

    def put_many(self, model: str, texts: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
        if len(texts) != len(vectors):
            raise CorruptEmbeddingError(f"{len(texts)} texts but {len(vectors)} vectors to cache")
        rows = [
            (self.cache_key(model, t), model, np.asarray(v, dtype=np.float32).tobytes())
            for t, v in zip(texts, vectors)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, model, vector) VALUES (?, ?, ?)", rows
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def get_or_compute_embeddings(
    cache: EmbeddingCache, model: str, texts: Sequence[str], embed_fn: EmbedFn
) -> np.ndarray:
    """Returns an (len(texts), dim) array, fetching from cache and only calling
    embed_fn for texts not already cached (each distinct text at most once)."""
    keys = [cache.cache_key(model, t) for t in texts]
    cached = cache.get_many(model, texts)

    missing: dict[str, str] = {}  # key -> text, first occurrence only
    for key, text in zip(keys, texts):
        if key not in cached and key not in missing:
            missing[key] = text
    if missing:
        missing_texts = list(missing.values())
        new_vectors = embed_fn(missing_texts)
        if len(new_vectors) != len(missing_texts):
            raise CorruptEmbeddingError(
                f"embed_fn returned {len(new_vectors)} vectors for {len(missing_texts)} texts"
            )
        new_arrays = [
            _validated(np.asarray(v, dtype=np.float32), "embed_fn result") for v in new_vectors
        ]
        cache.put_many(model, missing_texts, new_arrays)
        cached.update(zip(missing.keys(), new_arrays))

    dims = {cached[key].shape[0] for key in keys}
    if len(dims) > 1:
        raise CorruptEmbeddingError(f"embeddings for model {model!r} have mixed dimensions: {sorted(dims)}")
    return np.stack([cached[key] for key in keys])

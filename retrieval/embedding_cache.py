"""Disk-backed cache for text embeddings, keyed by (model, text).

Both chunk and query embeddings flow through this cache so a given piece of
text is only ever sent to the OpenAI embeddings API once, across all runs --
this matters for API cost and for exact reproducibility of a sweep.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

_SQLITE_MAX_VARIABLES_PER_QUERY = 900  # stay under SQLite's default ~999 parameter limit


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
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, model: str, texts: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
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
    embed_fn for texts not already cached."""
    keys = [cache.cache_key(model, t) for t in texts]
    cached = cache.get_many(model, texts)

    missing_positions = [i for i, key in enumerate(keys) if key not in cached]
    if missing_positions:
        missing_texts = [texts[i] for i in missing_positions]
        new_vectors = embed_fn(missing_texts)
        cache.put_many(model, missing_texts, new_vectors)
        for i, vector in zip(missing_positions, new_vectors):
            cached[keys[i]] = np.asarray(vector, dtype=np.float32)

    return np.stack([cached[key] for key in keys])

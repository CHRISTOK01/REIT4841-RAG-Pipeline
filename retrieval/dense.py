"""Dense retrieval: OpenAI embeddings + FAISS cosine similarity search.

`embed_fn` is injected rather than called directly against the OpenAI SDK here,
so this module -- and its tests -- have no network/API-key dependency. The real
OpenAI-backed embed_fn lives in generation/client.py and is passed in at
construction time.
"""

from __future__ import annotations

import numpy as np
import faiss

from data.schemas import Chunk
from retrieval.base import RetrievedChunk
from retrieval.embedding_cache import EmbedFn, EmbeddingCache, get_or_compute_embeddings

# Extra candidates fetched beyond k so that a run of tied scores straddling the
# k-th position is ordered by our own tie rule rather than FAISS's internal one.
# Ties are real in this corpus: MedQuAD repeats passages verbatim.
_TIE_MARGIN = 32


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        index: faiss.Index,
        model: str,
        embed_fn: EmbedFn,
        cache: EmbeddingCache,
    ):
        if chunks and index.ntotal != len(chunks):
            raise ValueError(
                f"FAISS index holds {index.ntotal} vectors but {len(chunks)} chunks were given"
            )
        self.chunks = chunks
        self.index = index
        self.model = model
        self._embed_fn = embed_fn
        self._cache = cache

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embed_fn: EmbedFn,
        model: str,
        cache: EmbeddingCache,
    ) -> "DenseRetriever":
        """Embeds every chunk (via cache) and builds a cosine-similarity FAISS
        index (exact inner product over L2-normalized vectors)."""
        if not chunks:
            index = faiss.IndexFlatIP(1)
            return cls(chunks, index, model, embed_fn, cache)

        vectors = get_or_compute_embeddings(cache, model, [c.text for c in chunks], embed_fn)
        vectors = l2_normalize(vectors.astype(np.float32))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(chunks, index, model, embed_fn, cache)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity, highest first; ties broken by
        corpus position (earlier chunk first)."""
        if not self.chunks:
            return []
        query_vec = get_or_compute_embeddings(self._cache, self.model, [query], self._embed_fn)
        query_vec = l2_normalize(query_vec.astype(np.float32))

        n_candidates = min(k + _TIE_MARGIN, len(self.chunks))
        scores, indices = self.index.search(query_vec, n_candidates)
        candidates = [(float(s), int(i)) for s, i in zip(scores[0], indices[0]) if i != -1]
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))

        return [
            RetrievedChunk(
                chunk_id=self.chunks[idx].chunk_id,
                doc_id=self.chunks[idx].doc_id,
                text=self.chunks[idx].text,
                score=score,
                rank=rank,
            )
            for rank, (score, idx) in enumerate(candidates[:k], start=1)
        ]

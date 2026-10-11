"""BM25 lexical retrieval over the chunk corpus."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from data.schemas import Chunk
from retrieval.base import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi | None = None):
        self.chunks = chunks
        if bm25 is not None:
            self.bm25 = bm25
        elif chunks:
            self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        else:
            self.bm25 = None  # BM25Okapi divides by corpus size and errors on an empty corpus

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "LexicalRetriever":
        return cls(chunks)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Top-k chunks by BM25 score, highest first.

        Only chunks that share at least one term with the query are returned,
        so the result can be shorter than k: a chunk with no lexical match was
        not retrieved, and padding the list with arbitrary zero-score chunks
        would hand them rank-based credit in hybrid fusion.
        Ties are broken by corpus position (earlier chunk first).
        """
        if not self.chunks:
            return []
        query_terms = tokenize(query)
        scores = np.asarray(self.bm25.get_scores(query_terms), dtype=np.float64)
        unique_terms = set(query_terms)
        matching = [
            i for i, term_freqs in enumerate(self.bm25.doc_freqs) if not unique_terms.isdisjoint(term_freqs)
        ]
        matching.sort(key=lambda i: (-scores[i], i))
        top_indices = matching[:k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[idx].chunk_id,
                doc_id=self.chunks[idx].doc_id,
                text=self.chunks[idx].text,
                score=float(scores[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]

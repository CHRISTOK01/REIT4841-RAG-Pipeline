"""BM25 lexical retrieval over the chunk corpus."""

from __future__ import annotations

import re

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
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
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

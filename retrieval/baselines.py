"""Baseline "retrievers" that bracket the real ones.

- `OracleRetriever`: returns the query's own gold (strict-relevant) chunks.
  Upper bound: the best context this KB variant can possibly supply.
- `RandomRetriever`: returns k chunks drawn uniformly at random. Lower bound.

Neither looks at the query text, so they are built per query by the harness
rather than implementing the text-in `Retriever` protocol. The third baseline,
no retrieval at all, needs no retriever (see `closed_book_v1` in
generation/prompts.py).
"""

from __future__ import annotations

import random

from data.schemas import Chunk
from retrieval.base import RetrievedChunk


def _as_retrieved(chunks: list[Chunk], score: float) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=score, rank=rank)
        for rank, c in enumerate(chunks, start=1)
    ]


def oracle_retrieve(chunks_by_id: dict[str, Chunk], relevant_chunk_ids: set[str], k: int) -> list[RetrievedChunk]:
    """The first k gold chunks in document order. Fewer than k (possibly none)
    when the KB variant holds fewer gold chunks: the list is never padded."""
    gold = sorted((chunks_by_id[cid] for cid in relevant_chunk_ids), key=lambda c: (c.doc_id, c.chunk_index))
    return _as_retrieved(gold[:k], score=1.0)


def random_retrieve(chunks: list[Chunk], k: int, rng: random.Random) -> list[RetrievedChunk]:
    return _as_retrieved(rng.sample(chunks, min(k, len(chunks))), score=0.0)

"""Hybrid retrieval: fuse rankings from multiple retrievers.

The fusion strategy is swappable -- pass any callable matching `FusionFn` to
`HybridRetriever`. `reciprocal_rank_fusion` (RRF) is the default.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from retrieval.base import RetrievedChunk, Retriever

FusionFn = Callable[[list[list[RetrievedChunk]]], list[RetrievedChunk]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k_constant: int = 60
) -> list[RetrievedChunk]:
    """score(chunk) = sum over lists of 1 / (k_constant + rank_in_that_list).

    Chunks absent from a given list simply don't contribute a term for it.
    k_constant=60 is the standard RRF default (Cormack et al., 2009).
    """
    fused_scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, RetrievedChunk] = {}
    for ranked_list in ranked_lists:
        for item in ranked_list:
            fused_scores[item.chunk_id] += 1.0 / (k_constant + item.rank)
            first_seen.setdefault(item.chunk_id, item)

    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    fused: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered_ids, start=1):
        base = first_seen[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                text=base.text,
                score=fused_scores[chunk_id],
                rank=rank,
            )
        )
    return fused


class HybridRetriever:
    def __init__(
        self,
        retrievers: list[Retriever],
        k_per_retriever: int = 50,
        fusion_fn: FusionFn = reciprocal_rank_fusion,
    ):
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever to fuse")
        self.retrievers = retrievers
        self.k_per_retriever = k_per_retriever
        self.fusion_fn = fusion_fn

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        ranked_lists = [
            retriever.retrieve(query, max(k, self.k_per_retriever)) for retriever in self.retrievers
        ]
        return self.fusion_fn(ranked_lists)[:k]

"""Hybrid retrieval: fuse rankings from multiple retrievers.

The fusion strategy is swappable -- pass any callable matching `FusionFn` to
`HybridRetriever`. `reciprocal_rank_fusion` (RRF) is the default.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Callable

from retrieval.base import RetrievedChunk, Retriever

FusionFn = Callable[[list[list[RetrievedChunk]]], list[RetrievedChunk]]

DEFAULT_RRF_K = 60  # Cormack, Clarke & Buettcher (2009)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k_constant: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """score(chunk) = sum over lists of 1 / (k_constant + rank_in_that_list).

    rank_in_that_list is the chunk's 1-based position in the list. Only ranks
    are fused, so the retrievers' raw scores (BM25 vs cosine, different scales)
    need no normalisation. Chunks absent from a list contribute no term for it.
    Ties in fused score are broken by chunk_id, ascending, so the order never
    depends on which retriever happened to be listed first.
    """
    fused_scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, RetrievedChunk] = {}
    for ranked_list in ranked_lists:
        ids_in_list = [item.chunk_id for item in ranked_list]
        if len(set(ids_in_list)) != len(ids_in_list):
            raise ValueError("a ranked list passed to RRF contains duplicate chunk ids")
        for position, item in enumerate(ranked_list, start=1):
            fused_scores[item.chunk_id] += 1.0 / (k_constant + position)
            first_seen.setdefault(item.chunk_id, item)

    ordered_ids = sorted(fused_scores, key=lambda cid: (-fused_scores[cid], cid))
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
        fusion_fn: FusionFn | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever to fuse")
        self.retrievers = retrievers
        self.k_per_retriever = k_per_retriever
        self.rrf_k = rrf_k
        self.fusion_fn = fusion_fn or partial(reciprocal_rank_fusion, k_constant=rrf_k)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        ranked_lists = [
            retriever.retrieve(query, max(k, self.k_per_retriever)) for retriever in self.retrievers
        ]
        return self.fusion_fn(ranked_lists)[:k]

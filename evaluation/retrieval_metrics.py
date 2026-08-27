"""Standard IR metrics for a single query's ranked retrieval result.

Every metric here can be *undefined* for a query -- e.g. Recall@k, MRR, and
NDCG@k are all 0/0 when the query has zero relevant chunks in the KB variant
being evaluated (see evaluation/relevance_judgments.py: this happens for real
under the `reduced` KB variant, when a query's own source topic was dropped).
Undefined values are returned as float('nan'), not 0.0 -- 0.0 would silently
conflate "retrieval found nothing relevant" with "there was nothing relevant
to find", which are different findings for the thesis. Aggregate with
nan-aware statistics (e.g. pandas .mean(skipna=True)) downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from retrieval.base import RetrievedChunk


def precision_at_k(retrieved: list[RetrievedChunk], relevant_chunk_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if c.chunk_id in relevant_chunk_ids)
    return hits / k


def recall_at_k(retrieved: list[RetrievedChunk], relevant_chunk_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_chunk_ids:
        return math.nan
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if c.chunk_id in relevant_chunk_ids)
    return hits / len(relevant_chunk_ids)


def reciprocal_rank(retrieved: list[RetrievedChunk], relevant_chunk_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_chunk_ids:
        return math.nan
    for chunk in retrieved[:k]:
        if chunk.chunk_id in relevant_chunk_ids:
            return 1.0 / chunk.rank
    return 0.0


def ndcg_at_k(retrieved: list[RetrievedChunk], relevant_chunk_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG@k."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_chunk_ids:
        return math.nan

    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(i + 1) for i, c in enumerate(top_k, start=1) if c.chunk_id in relevant_chunk_ids
    )
    ideal_hits = min(k, len(relevant_chunk_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return math.nan
    return dcg / idcg


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def compute_retrieval_metrics(
    retrieved: list[RetrievedChunk], relevant_chunk_ids: set[str], k: int
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved, relevant_chunk_ids, k),
        recall_at_k=recall_at_k(retrieved, relevant_chunk_ids, k),
        mrr=reciprocal_rank(retrieved, relevant_chunk_ids, k),
        ndcg_at_k=ndcg_at_k(retrieved, relevant_chunk_ids, k),
    )

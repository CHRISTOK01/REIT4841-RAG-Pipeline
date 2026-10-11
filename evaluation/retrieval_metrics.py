"""Standard IR metrics for a single query's ranked retrieval result, binary relevance.

Rank is always the 1-based *position in the list passed in* -- never a `rank`
field carried on the items -- so every metric here agrees about what "rank 3"
means even if a caller sliced or filtered the list.

Queries with no relevant chunk in the KB variant (this happens for real under
`reduced`): Recall@k, MRR and NDCG@k are 0/0 and are returned as float('nan'),
and Precision@k is returned as 0.0. Per-query values are logged exactly like
that. The analysis layer then reports every metric two ways:

- `<metric>_all`: over every query, counting an unanswerable query as 0. This
  is the headline number: a KB that cannot answer a query has failed it.
- `<metric>` (answerable-only): over queries that have a relevant chunk in that
  KB variant. This isolates ranking quality from coverage.

Graded relevance is not implemented: the labels are binary (see
evaluation/relevance_judgments.py), and binary NDCG is the same number under
the linear and exponential gain conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from retrieval.base import RetrievedChunk


def _ranked_ids(retrieved: Sequence[RetrievedChunk | str]) -> list[str]:
    ids = [item if isinstance(item, str) else item.chunk_id for item in retrieved]
    if len(set(ids)) != len(ids):
        # A duplicate would be counted as two hits (recall could exceed 1).
        raise ValueError("retrieved list contains duplicate chunk ids")
    return ids


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(retrieved: Sequence[RetrievedChunk | str], relevant_chunk_ids: set[str], k: int) -> float:
    """hits in the top k, divided by k (not by the number actually returned)."""
    _check_k(k)
    hits = sum(1 for cid in _ranked_ids(retrieved)[:k] if cid in relevant_chunk_ids)
    return hits / k


def recall_at_k(retrieved: Sequence[RetrievedChunk | str], relevant_chunk_ids: set[str], k: int) -> float:
    _check_k(k)
    ids = _ranked_ids(retrieved)
    if not relevant_chunk_ids:
        return math.nan
    hits = sum(1 for cid in ids[:k] if cid in relevant_chunk_ids)
    return hits / len(relevant_chunk_ids)


def reciprocal_rank(retrieved: Sequence[RetrievedChunk | str], relevant_chunk_ids: set[str], k: int) -> float:
    """1 / position of the first relevant chunk within the top k, else 0 (MRR@k)."""
    _check_k(k)
    ids = _ranked_ids(retrieved)
    if not relevant_chunk_ids:
        return math.nan
    for position, cid in enumerate(ids[:k], start=1):
        if cid in relevant_chunk_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[RetrievedChunk | str], relevant_chunk_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG@k: DCG = sum over relevant positions i<=k of
    1/log2(i+1); IDCG = the same sum for min(k, #relevant) hits at the top."""
    _check_k(k)
    ids = _ranked_ids(retrieved)
    if not relevant_chunk_ids:
        return math.nan
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, cid in enumerate(ids[:k], start=1)
        if cid in relevant_chunk_ids
    )
    ideal_hits = min(k, len(relevant_chunk_ids))
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def compute_retrieval_metrics(
    retrieved: Sequence[RetrievedChunk | str], relevant_chunk_ids: set[str], k: int
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved, relevant_chunk_ids, k),
        recall_at_k=recall_at_k(retrieved, relevant_chunk_ids, k),
        mrr=reciprocal_rank(retrieved, relevant_chunk_ids, k),
        ndcg_at_k=ndcg_at_k(retrieved, relevant_chunk_ids, k),
    )

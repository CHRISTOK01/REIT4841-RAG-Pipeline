import math

import pytest

from evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from retrieval.base import RetrievedChunk


def rc(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text="t", score=1.0, rank=rank)


RETRIEVED = [rc("a", 1), rc("b", 2), rc("c", 3), rc("d", 4), rc("e", 5)]


def test_precision_at_k_counts_hits_over_k():
    relevant = {"b", "d", "z"}  # z not retrieved
    assert precision_at_k(RETRIEVED, relevant, k=4) == 2 / 4  # b, d in top 4


def test_precision_at_k_zero_when_no_hits():
    assert precision_at_k(RETRIEVED, {"z"}, k=3) == 0.0


def test_recall_at_k_divides_by_total_relevant():
    relevant = {"a", "c", "z"}  # z not retrieved at all
    assert recall_at_k(RETRIEVED, relevant, k=3) == 2 / 3  # a, c found; z never found


def test_recall_at_k_nan_when_no_relevant_chunks_exist():
    assert math.isnan(recall_at_k(RETRIEVED, set(), k=3))


def test_reciprocal_rank_uses_rank_of_first_hit():
    assert reciprocal_rank(RETRIEVED, {"c"}, k=5) == 1 / 3
    assert reciprocal_rank(RETRIEVED, {"a", "c"}, k=5) == 1 / 1  # earliest hit wins


def test_reciprocal_rank_zero_when_relevant_exists_but_not_retrieved_in_k():
    assert reciprocal_rank(RETRIEVED, {"z"}, k=5) == 0.0


def test_reciprocal_rank_nan_when_no_relevant_chunks_exist():
    assert math.isnan(reciprocal_rank(RETRIEVED, set(), k=5))


def test_ndcg_perfect_when_relevant_chunks_rank_first():
    relevant = {"a", "b"}
    assert ndcg_at_k(RETRIEVED, relevant, k=2) == pytest.approx(1.0)


def test_ndcg_lower_when_relevant_chunks_rank_later():
    perfect = ndcg_at_k(RETRIEVED, {"a", "b"}, k=5)
    worse = ndcg_at_k(RETRIEVED, {"d", "e"}, k=5)
    assert worse < perfect


def test_ndcg_nan_when_no_relevant_chunks_exist():
    assert math.isnan(ndcg_at_k(RETRIEVED, set(), k=5))


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k])
def test_k_must_be_positive(fn):
    with pytest.raises(ValueError):
        fn(RETRIEVED, {"a"}, k=0)


def test_compute_retrieval_metrics_bundles_all_four():
    metrics = compute_retrieval_metrics(RETRIEVED, {"a"}, k=3)
    assert metrics.precision_at_k == 1 / 3
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_k == pytest.approx(1.0)

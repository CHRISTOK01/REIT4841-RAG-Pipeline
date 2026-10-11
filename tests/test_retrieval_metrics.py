"""Hand-worked examples for P@k, R@k, MRR@k and NDCG@k.

Every expected value below is written as a literal worked out by hand in the
comment beside it -- none is produced by the code under test.

Useful constants:  1/log2(2) = 1        1/log2(3) = 0.630930
                   1/log2(4) = 0.5      1/log2(5) = 0.430677
                   1/log2(6) = 0.386853
"""

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

RANKED = ["a", "b", "c", "d", "e"]
ALL_FOUR = [precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k]


def test_worked_example_two_relevant_at_ranks_2_and_4():
    relevant = {"b", "d", "z"}  # z is relevant but never retrieved
    # top 5 = a b c d e -> hits at ranks 2 and 4
    assert precision_at_k(RANKED, relevant, k=5) == pytest.approx(0.4)  # 2 / 5
    assert recall_at_k(RANKED, relevant, k=5) == pytest.approx(0.666667, abs=1e-6)  # 2 / 3
    assert reciprocal_rank(RANKED, relevant, k=5) == pytest.approx(0.5)  # first hit at rank 2
    # DCG  = 1/log2(3) + 1/log2(5)            = 0.630930 + 0.430677 = 1.061607
    # IDCG = 3 hits at ranks 1,2,3            = 1 + 0.630930 + 0.5  = 2.130930
    assert ndcg_at_k(RANKED, relevant, k=5) == pytest.approx(0.498189, abs=1e-6)  # 1.061607 / 2.130930


def test_worked_example_cutoff_excludes_hits_below_k():
    relevant = {"b", "d"}
    # top 3 = a b c -> only b (rank 2) counts; d at rank 4 is past the cutoff
    assert precision_at_k(RANKED, relevant, k=3) == pytest.approx(0.333333, abs=1e-6)  # 1 / 3
    assert recall_at_k(RANKED, relevant, k=3) == pytest.approx(0.5)  # 1 / 2
    assert reciprocal_rank(RANKED, relevant, k=3) == pytest.approx(0.5)
    # DCG = 1/log2(3) = 0.630930 ; IDCG = min(3, 2) = 2 hits = 1 + 0.630930 = 1.630930
    assert ndcg_at_k(RANKED, relevant, k=3) == pytest.approx(0.386853, abs=1e-6)


def test_first_relevant_just_past_k_scores_zero_not_one_over_rank():
    # the only relevant chunk sits at rank 4; with k = 3 it was not retrieved
    assert reciprocal_rank(RANKED, {"d"}, k=3) == 0.0
    assert recall_at_k(RANKED, {"d"}, k=3) == 0.0
    assert ndcg_at_k(RANKED, {"d"}, k=3) == 0.0
    # ...and with k = 4 it is, at rank 4
    assert reciprocal_rank(RANKED, {"d"}, k=4) == pytest.approx(0.25)
    assert ndcg_at_k(RANKED, {"d"}, k=4) == pytest.approx(0.430677, abs=1e-6)  # (1/log2(5)) / 1


def test_fewer_than_k_results_still_divides_precision_by_k():
    retrieved = ["a", "b"]  # only 2 results came back, k = 5
    relevant = {"b", "z"}
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.2)  # 1 / 5, not 1 / 2
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)  # 1 / 2
    assert reciprocal_rank(retrieved, relevant, k=5) == pytest.approx(0.5)
    # DCG = 1/log2(3) = 0.630930 ; IDCG = 2 hits = 1.630930
    assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(0.386853, abs=1e-6)


def test_k_larger_than_everything_changes_nothing_but_the_precision_denominator():
    relevant = {"a", "e"}
    assert precision_at_k(RANKED, relevant, k=100) == pytest.approx(0.02)  # 2 / 100
    assert recall_at_k(RANKED, relevant, k=100) == 1.0
    assert reciprocal_rank(RANKED, relevant, k=100) == 1.0
    # DCG = 1 + 1/log2(6) = 1.386853 ; IDCG = 1 + 0.630930 = 1.630930
    assert ndcg_at_k(RANKED, relevant, k=100) == pytest.approx(0.850344, abs=1e-6)


def test_more_relevant_chunks_than_k_caps_the_ideal_at_k():
    relevant = {"a", "b", "x", "y", "z", "w"}  # 6 relevant, only 2 slots
    assert precision_at_k(RANKED, relevant, k=2) == 1.0
    assert recall_at_k(RANKED, relevant, k=2) == pytest.approx(0.333333, abs=1e-6)  # 2 / 6
    assert ndcg_at_k(RANKED, relevant, k=2) == pytest.approx(1.0)  # best possible for k = 2


def test_perfect_ranking_scores_one_on_everything():
    relevant = {"a", "b", "c"}
    metrics = compute_retrieval_metrics(RANKED, relevant, k=3)
    assert (metrics.precision_at_k, metrics.recall_at_k, metrics.mrr) == (1.0, 1.0, 1.0)
    assert metrics.ndcg_at_k == pytest.approx(1.0)


def test_empty_retrieval_with_relevant_chunks_is_a_real_zero():
    """Nothing came back although something relevant exists: a failure (0), not undefined."""
    for fn in ALL_FOUR:
        assert fn([], {"a"}, k=5) == 0.0


def test_query_with_no_relevant_chunks_is_undefined_not_zero_or_one():
    """0/0 for recall, MRR and NDCG -> NaN (the analysis layer decides how to
    count it); precision is 0 hits / k = 0."""
    assert precision_at_k(RANKED, set(), k=5) == 0.0
    assert math.isnan(recall_at_k(RANKED, set(), k=5))
    assert math.isnan(reciprocal_rank(RANKED, set(), k=5))
    assert math.isnan(ndcg_at_k(RANKED, set(), k=5))
    assert math.isnan(recall_at_k([], set(), k=5))


@pytest.mark.parametrize("fn", ALL_FOUR)
@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(fn, k):
    with pytest.raises(ValueError):
        fn(RANKED, {"a"}, k=k)


@pytest.mark.parametrize("fn", ALL_FOUR)
def test_duplicate_chunk_ids_are_rejected_not_double_counted(fn):
    # counted naively, ["a", "a"] against {"a"} would give recall 2/1 = 2.0
    with pytest.raises(ValueError, match="duplicate"):
        fn(["a", "a", "b"], {"a"}, k=3)


def test_rank_means_list_position_not_the_rank_field():
    """Items carry a `rank` field, but a sliced or re-ordered list may leave it
    stale. Every metric must use where the item actually is in the list."""
    stale = [
        RetrievedChunk(chunk_id="x", doc_id="x", text="", score=0.9, rank=7),
        RetrievedChunk(chunk_id="y", doc_id="y", text="", score=0.8, rank=1),
    ]
    assert reciprocal_rank(stale, {"y"}, k=5) == pytest.approx(0.5)  # y is 2nd in the list
    assert ndcg_at_k(stale, {"y"}, k=5) == pytest.approx(0.630930, abs=1e-6)


def test_ties_in_score_do_not_matter_only_the_order_given():
    """Metrics never look at scores, so two chunks with equal scores are simply
    scored in the order the retriever returned them."""
    tied = [RetrievedChunk(chunk_id=c, doc_id=c, text="", score=1.0, rank=i) for i, c in enumerate("ab", start=1)]
    assert reciprocal_rank(tied, {"b"}, k=2) == pytest.approx(0.5)
    assert reciprocal_rank(list(reversed(tied)), {"b"}, k=2) == 1.0


def test_compute_retrieval_metrics_bundles_the_four_functions():
    relevant = {"b", "d", "z"}
    metrics = compute_retrieval_metrics(RANKED, relevant, k=5)
    assert metrics.precision_at_k == pytest.approx(0.4)
    assert metrics.recall_at_k == pytest.approx(0.666667, abs=1e-6)
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.ndcg_at_k == pytest.approx(0.498189, abs=1e-6)

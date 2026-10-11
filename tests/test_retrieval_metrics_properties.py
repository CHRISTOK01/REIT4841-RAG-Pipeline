"""Property-based tests (Hypothesis): statements that must hold for *every*
ranking and relevance set, checked on hundreds of generated cases each."""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k

IDS = [f"c{i}" for i in range(20)]
rankings = st.lists(st.sampled_from(IDS), unique=True, max_size=15)
relevant_sets = st.sets(st.sampled_from(IDS), min_size=1, max_size=10)
ks = st.integers(min_value=1, max_value=25)
ALL_FOUR = [precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k]
CASES = settings(max_examples=300, deadline=None)


@CASES
@given(rankings, relevant_sets, ks)
def test_every_metric_stays_between_0_and_1(ranked, relevant, k):
    for fn in ALL_FOUR:
        assert 0.0 <= fn(ranked, relevant, k) <= 1.0 + 1e-12


@CASES
@given(relevant_sets, ks)
def test_relevant_chunks_first_is_a_perfect_ranking(relevant, k):
    """All relevant chunks at the top, in any order, then everything else."""
    ranked = sorted(relevant) + [c for c in IDS if c not in relevant]
    assert reciprocal_rank(ranked, relevant, k) == 1.0
    assert math.isclose(ndcg_at_k(ranked, relevant, k), 1.0)
    assert math.isclose(recall_at_k(ranked, relevant, k), min(k, len(relevant)) / len(relevant))
    assert math.isclose(precision_at_k(ranked, relevant, k), min(k, len(relevant)) / k)


@CASES
@given(relevant_sets, ks)
def test_reversing_a_perfect_ranking_never_scores_higher(relevant, k):
    best = sorted(relevant) + [c for c in IDS if c not in relevant]
    worst = list(reversed(best))
    for fn in ALL_FOUR:
        assert fn(worst, relevant, k) <= fn(best, relevant, k) + 1e-12
    if len(relevant) < len(IDS) and k < len(IDS) - len(relevant) + 1:
        # the top k of the reversed list is all non-relevant
        assert ndcg_at_k(worst, relevant, k) < ndcg_at_k(best, relevant, k)


@CASES
@given(rankings, relevant_sets, ks, st.lists(st.sampled_from([f"junk{i}" for i in range(10)]), unique=True, max_size=10))
def test_appending_irrelevant_chunks_below_never_changes_any_metric_at_a_fixed_cutoff(ranked, relevant, k, junk):
    """Non-relevant chunks added *after* the existing results cannot lower MRR
    (or anything else) when the list already had at least k results."""
    if len(ranked) < k:
        ranked = ranked + [c for c in IDS if c not in ranked][: k - len(ranked)]
    for fn in ALL_FOUR:
        assert fn(ranked + junk, relevant, k) == fn(ranked, relevant, k)


@CASES
@given(rankings, relevant_sets, st.lists(st.sampled_from([f"junk{i}" for i in range(10)]), unique=True, max_size=10))
def test_appending_irrelevant_chunks_never_lowers_mrr_even_with_a_deeper_cutoff(ranked, relevant, junk):
    before = reciprocal_rank(ranked, relevant, k=max(len(ranked), 1))
    after = reciprocal_rank(ranked + junk, relevant, k=max(len(ranked) + len(junk), 1))
    assert after == before


@CASES
@given(rankings, relevant_sets, ks)
def test_recall_and_hits_never_decrease_as_k_grows(ranked, relevant, k):
    assert recall_at_k(ranked, relevant, k + 1) >= recall_at_k(ranked, relevant, k)
    assert reciprocal_rank(ranked, relevant, k + 1) >= reciprocal_rank(ranked, relevant, k)
    # hits = precision * k
    assert precision_at_k(ranked, relevant, k + 1) * (k + 1) >= precision_at_k(ranked, relevant, k) * k - 1e-9


@CASES
@given(rankings, relevant_sets, ks)
def test_moving_a_relevant_chunk_up_one_place_never_lowers_ndcg_or_mrr(ranked, relevant, k):
    for i in range(1, len(ranked)):
        if ranked[i] in relevant and ranked[i - 1] not in relevant:
            swapped = ranked[: i - 1] + [ranked[i], ranked[i - 1]] + ranked[i + 1 :]
            assert ndcg_at_k(swapped, relevant, k) >= ndcg_at_k(ranked, relevant, k) - 1e-12
            assert reciprocal_rank(swapped, relevant, k) >= reciprocal_rank(ranked, relevant, k)


@CASES
@given(rankings, relevant_sets, ks)
def test_metrics_are_zero_exactly_when_nothing_relevant_is_in_the_top_k(ranked, relevant, k):
    any_hit = any(c in relevant for c in ranked[:k])
    for fn in ALL_FOUR:
        assert (fn(ranked, relevant, k) > 0) == any_hit


@CASES
@given(rankings, ks)
def test_no_relevant_chunks_is_always_nan_never_a_number(ranked, k):
    assert precision_at_k(ranked, set(), k) == 0.0
    for fn in (recall_at_k, reciprocal_rank, ndcg_at_k):
        assert math.isnan(fn(ranked, set(), k))

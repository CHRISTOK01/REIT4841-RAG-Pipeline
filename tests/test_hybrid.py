"""Reciprocal rank fusion checked against fused scores worked out by hand.

    score(chunk) = sum over lists containing it of 1 / (k + rank in that list)

Useful constants for k = 60:  1/61 = 0.016393   1/62 = 0.016129   1/63 = 0.015873
"""

import pytest

from retrieval.base import RetrievedChunk
from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def rc(chunk_id: str, rank: int = 0, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=f"text-{chunk_id}", score=score, rank=rank)


def ranked(*chunk_ids: str) -> list[RetrievedChunk]:
    return [rc(cid, rank=i) for i, cid in enumerate(chunk_ids, start=1)]


class StubRetriever:
    def __init__(self, results: list[RetrievedChunk]):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.results[:k]


def scores(fused: list[RetrievedChunk]) -> dict[str, float]:
    return {item.chunk_id: item.score for item in fused}


def test_fused_scores_and_order_match_hand_calculation_k60():
    fused = reciprocal_rank_fusion([ranked("x", "y", "z"), ranked("y", "w")], k_constant=60)
    # x: rank 1 in A only            -> 1/61           = 0.016393
    # y: rank 2 in A, rank 1 in B    -> 1/62 + 1/61    = 0.032522
    # z: rank 3 in A only            -> 1/63           = 0.015873
    # w: rank 2 in B only            -> 1/62           = 0.016129
    assert scores(fused) == pytest.approx({"x": 0.016393, "y": 0.032522, "z": 0.015873, "w": 0.016129}, abs=1e-6)
    assert [item.chunk_id for item in fused] == ["y", "x", "w", "z"]
    assert [item.rank for item in fused] == [1, 2, 3, 4]


def test_the_k_constant_changes_scores_and_can_change_the_order():
    lists = [ranked("p", "m", "q"), ranked("n", "o", "q")]
    # k = 60: q = 1/63 + 1/63 = 0.031746 ; p = n = 1/61 = 0.016393 ; m = o = 1/62 = 0.016129
    at_60 = reciprocal_rank_fusion(lists, k_constant=60)
    assert scores(at_60) == pytest.approx({"q": 0.031746, "p": 0.016393, "n": 0.016393, "m": 0.016129, "o": 0.016129}, abs=1e-6)
    assert [item.chunk_id for item in at_60] == ["q", "n", "p", "m", "o"]
    # k = 0: q = 1/3 + 1/3 = 0.666667 ; p = n = 1/1 = 1 ; m = o = 1/2 -> q drops from 1st to 3rd
    at_0 = reciprocal_rank_fusion(lists, k_constant=0)
    assert scores(at_0) == pytest.approx({"q": 0.666667, "p": 1.0, "n": 1.0, "m": 0.5, "o": 0.5}, abs=1e-6)
    assert [item.chunk_id for item in at_0] == ["n", "p", "q", "m", "o"]


def test_default_constant_is_60():
    fused = reciprocal_rank_fusion([ranked("only")])
    assert fused[0].score == pytest.approx(1 / 61)


def test_ties_are_broken_by_chunk_id_regardless_of_list_order():
    a_first = reciprocal_rank_fusion([ranked("b"), ranked("a")])
    b_first = reciprocal_rank_fusion([ranked("a"), ranked("b")])
    assert [i.chunk_id for i in a_first] == [i.chunk_id for i in b_first] == ["a", "b"]
    assert a_first[0].score == a_first[1].score


def test_chunk_in_only_one_list_gets_exactly_one_term():
    fused = reciprocal_rank_fusion([ranked("solo", "both"), ranked("both")], k_constant=60)
    assert scores(fused)["solo"] == pytest.approx(1 / 61)  # no term from the list it is absent from
    assert scores(fused)["both"] == pytest.approx(1 / 62 + 1 / 61)


def test_fusion_uses_list_position_and_ignores_raw_scores_and_rank_fields():
    """BM25 scores (tens) and cosine scores (<1) are on different scales; RRF
    must not be swayed by either, nor by a stale `rank` field."""
    list_a = [rc("a", rank=9, score=1000.0), rc("b", rank=1, score=0.001)]
    list_b = [rc("b", rank=5, score=0.5), rc("a", rank=2, score=0.4)]
    fused = reciprocal_rank_fusion([list_a, list_b], k_constant=60)
    # by position: a = 1/61 + 1/62, b = 1/62 + 1/61 -> exact tie, broken by id
    assert fused[0].score == pytest.approx(fused[1].score)
    assert [i.chunk_id for i in fused] == ["a", "b"]


def test_duplicate_ids_within_one_list_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([ranked("a", "a")])


def test_empty_lists_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


def test_hybrid_retriever_returns_the_top_k_of_the_fused_ranking():
    lexical = StubRetriever(ranked("a", "b", "c"))
    dense = StubRetriever(ranked("b", "a", "d"))
    hybrid = HybridRetriever([lexical, dense], k_per_retriever=3)

    results = hybrid.retrieve("query", k=2)

    # a = 1/61 + 1/62 = b (tie -> a first by id); c = d = 1/63
    assert [r.chunk_id for r in results] == ["a", "b"]
    assert [r.rank for r in results] == [1, 2]


def test_hybrid_retriever_passes_its_rrf_constant_through():
    lists = (ranked("p", "m", "q"), ranked("n", "o", "q"))
    top_at = lambda rrf_k: HybridRetriever(  # noqa: E731
        [StubRetriever(lists[0]), StubRetriever(lists[1])], k_per_retriever=3, rrf_k=rrf_k
    ).retrieve("query", k=1)[0].chunk_id
    assert top_at(60) == "q"
    assert top_at(0) == "n"


def test_hybrid_retriever_asks_each_sub_retriever_for_k_per_retriever_candidates():
    lexical = StubRetriever(ranked(*[f"c{i}" for i in range(10)]))
    dense = StubRetriever(ranked(*[f"c{i}" for i in range(10)]))
    HybridRetriever([lexical, dense], k_per_retriever=5).retrieve("query", k=2)
    assert lexical.calls == [("query", 5)]
    assert dense.calls == [("query", 5)]

    HybridRetriever([lexical], k_per_retriever=5).retrieve("query", k=8)
    assert lexical.calls[-1] == ("query", 8)  # never fewer than k


def test_hybrid_retriever_requires_at_least_one_retriever():
    with pytest.raises(ValueError):
        HybridRetriever([])

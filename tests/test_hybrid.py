from retrieval.base import RetrievedChunk
from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def rc(chunk_id: str, rank: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=f"text-{chunk_id}", score=score, rank=rank)


class StubRetriever:
    def __init__(self, results: list[RetrievedChunk]):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.results[:k]


def test_reciprocal_rank_fusion_formula():
    list_a = [rc("x", rank=1), rc("y", rank=2)]
    list_b = [rc("y", rank=1), rc("x", rank=2)]
    fused = reciprocal_rank_fusion([list_a, list_b], k_constant=60)

    # both chunks get the same combined score (1/61 + 1/62) by symmetry -> tie
    scores = {item.chunk_id: item.score for item in fused}
    assert scores["x"] == scores["y"]
    assert abs(scores["x"] - (1 / 61 + 1 / 62)) < 1e-9
    assert [item.rank for item in fused] == [1, 2]


def test_reciprocal_rank_fusion_prefers_chunk_ranked_well_in_both_lists():
    list_a = [rc("a", rank=1), rc("b", rank=2), rc("c", rank=3)]
    list_b = [rc("a", rank=1), rc("c", rank=2), rc("b", rank=3)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0].chunk_id == "a"  # ranked #1 in both lists


def test_reciprocal_rank_fusion_includes_chunks_present_in_only_one_list():
    list_a = [rc("only_in_a", rank=1)]
    list_b = [rc("only_in_b", rank=1)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert {item.chunk_id for item in fused} == {"only_in_a", "only_in_b"}


def test_hybrid_retriever_fuses_and_truncates_to_k():
    lexical = StubRetriever([rc("a", 1), rc("b", 2), rc("c", 3)])
    dense = StubRetriever([rc("b", 1), rc("a", 2), rc("d", 3)])
    hybrid = HybridRetriever([lexical, dense], k_per_retriever=3)

    results = hybrid.retrieve("query", k=2)

    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]


def test_hybrid_retriever_requests_at_least_k_per_retriever_from_each_sub_retriever():
    lexical = StubRetriever([rc(f"c{i}", i + 1) for i in range(10)])
    dense = StubRetriever([rc(f"c{i}", i + 1) for i in range(10)])
    hybrid = HybridRetriever([lexical, dense], k_per_retriever=5)

    hybrid.retrieve("query", k=2)

    assert lexical.calls == [("query", 5)]
    assert dense.calls == [("query", 5)]


def test_hybrid_retriever_requires_at_least_one_retriever():
    try:
        HybridRetriever([])
        assert False, "expected ValueError"
    except ValueError:
        pass

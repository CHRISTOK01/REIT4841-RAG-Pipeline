"""Dense retriever on small hand-built vectors, where the cosine similarities
can be written down by hand. (The real embedded corpus is checked against
brute force in tests/test_retrievers_real.py.)"""

import numpy as np
import pytest

from data.schemas import Chunk
from retrieval.dense import DenseRetriever, l2_normalize
from retrieval.embedding_cache import EmbeddingCache


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        chunk_index=0,
        text=text,
        subset="Test",
        focus="Test Focus",
        question_type="information",
        source_question="irrelevant",
    )


def lookup_embedder(table: dict[str, list[float]], calls: list | None = None):
    def embed(texts):
        if calls is not None:
            calls.append(list(texts))
        return [np.array(table[t], dtype=np.float32) for t in texts]

    return embed


@pytest.fixture
def cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    yield cache
    cache.close()


def test_ranking_is_by_cosine_not_raw_inner_product(cache):
    """`long` has the larger inner product with the query (10 vs 2) but the
    smaller cosine (0.707 vs 1.0). A retriever that forgot to normalise would
    rank `long` first."""
    table = {"long": [10.0, 0.0], "aligned": [1.0, 1.0], "query": [1.0, 1.0]}
    chunks = [make_chunk("long#0", "long"), make_chunk("aligned#0", "aligned")]
    retriever = DenseRetriever.build(chunks, lookup_embedder(table), "toy", cache)

    results = retriever.retrieve("query", k=2)

    assert [r.chunk_id for r in results] == ["aligned#0", "long#0"]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)  # identical direction
    assert results[1].score == pytest.approx(0.707107, abs=1e-6)  # cos 45 degrees = 1/sqrt(2)


def test_query_vector_length_does_not_change_scores(cache):
    table = {"a": [3.0, 4.0], "b": [4.0, 3.0], "short query": [0.6, 0.8], "long query": [300.0, 400.0]}
    chunks = [make_chunk("a#0", "a"), make_chunk("b#0", "b")]
    retriever = DenseRetriever.build(chunks, lookup_embedder(table), "toy", cache)

    short = retriever.retrieve("short query", k=2)
    long = retriever.retrieve("long query", k=2)

    assert [(r.chunk_id, round(r.score, 6)) for r in short] == [(r.chunk_id, round(r.score, 6)) for r in long]
    assert short[0].score == pytest.approx(1.0, abs=1e-6)  # (3,4) vs (0.6,0.8): same direction
    assert short[1].score == pytest.approx(0.96, abs=1e-6)  # (4*0.6 + 3*0.8) / 5 = 4.8 / 5


def test_stored_index_vectors_are_unit_length(cache):
    table = {"a": [3.0, 4.0], "b": [0.0, 7.0], "c": [-2.0, 0.5]}
    chunks = [make_chunk(f"{t}#0", t) for t in table]
    retriever = DenseRetriever.build(chunks, lookup_embedder(table), "toy", cache)
    stored = retriever.index.reconstruct_n(0, retriever.index.ntotal)
    np.testing.assert_allclose(np.linalg.norm(stored, axis=1), 1.0, atol=1e-6)


def test_l2_normalize_leaves_a_zero_vector_alone_instead_of_dividing_by_zero():
    out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(out, [[0.0, 0.0], [0.6, 0.8]], atol=1e-6)


def test_identical_vectors_tie_and_are_returned_in_corpus_order(cache):
    table = {f"dup{i}": [1.0, 2.0] for i in range(6)} | {"far": [-1.0, 0.0], "query": [1.0, 2.0]}
    chunks = [make_chunk("far#0", "far")] + [make_chunk(f"dup{i}#0", f"dup{i}") for i in range(6)]
    retriever = DenseRetriever.build(chunks, lookup_embedder(table), "toy", cache)

    results = retriever.retrieve("query", k=3)

    assert [r.chunk_id for r in results] == ["dup0#0", "dup1#0", "dup2#0"]
    assert [r.rank for r in results] == [1, 2, 3]


def test_k_larger_than_corpus_returns_every_chunk_once(cache):
    table = {"a": [1.0, 0.0], "b": [0.0, 1.0], "query": [1.0, 0.2]}
    chunks = [make_chunk("a#0", "a"), make_chunk("b#0", "b")]
    retriever = DenseRetriever.build(chunks, lookup_embedder(table), "toy", cache)
    results = retriever.retrieve("query", k=100)
    assert [r.chunk_id for r in results] == ["a#0", "b#0"]


def test_chunk_and_query_embeddings_go_through_the_cache(cache):
    table = {"a": [1.0, 0.0], "b": [0.0, 1.0], "query": [1.0, 0.2]}
    chunks = [make_chunk("a#0", "a"), make_chunk("b#0", "b")]
    calls: list = []
    retriever = DenseRetriever.build(chunks, lookup_embedder(table, calls), "toy", cache)
    assert calls == [["a", "b"]]

    calls.clear()
    retriever.retrieve("query", k=1)
    retriever.retrieve("query", k=1)
    assert calls == [["query"]]  # second retrieval is served from the cache


def test_index_and_chunk_list_must_be_the_same_size(cache):
    table = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    chunks = [make_chunk("a#0", "a"), make_chunk("b#0", "b")]
    retriever = DenseRetriever.build(chunks, lookup_embedder(table), "toy", cache)
    with pytest.raises(ValueError, match="FAISS index holds 2 vectors"):
        DenseRetriever(chunks[:1], retriever.index, "toy", lookup_embedder(table), cache)


def test_retrieve_on_empty_corpus_returns_empty(cache):
    retriever = DenseRetriever.build([], lookup_embedder({}), "toy", cache)
    assert retriever.retrieve("anything", k=5) == []

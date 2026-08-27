import numpy as np

from data.schemas import Chunk
from retrieval.dense import DenseRetriever
from retrieval.embedding_cache import EmbeddingCache

# Toy deterministic "embedding": a bag-of-keywords count vector over a small
# fixed vocabulary. No network call, but shared vocabulary between query and
# chunk text still produces closer cosine similarity -- enough to exercise
# real FAISS index/search behavior meaningfully.
VOCAB = ["diabetes", "asthma", "exercise", "heart", "thirst", "fatigue", "inhaler", "cardio"]


def toy_embed_fn(texts):
    vectors = []
    for t in texts:
        tl = t.lower()
        vectors.append(np.array([tl.count(term) for term in VOCAB], dtype=np.float32))
    return vectors


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


CHUNKS = [
    make_chunk("d1#0", "Diabetes causes increased thirst and fatigue."),
    make_chunk("d2#0", "Asthma treatment often includes an inhaler."),
    make_chunk("d3#0", "Exercise improves heart health via cardio training."),
]


def test_retrieve_ranks_semantically_closest_chunk_first(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    retriever = DenseRetriever.build(CHUNKS, toy_embed_fn, "toy-model", cache)

    results = retriever.retrieve("what causes thirst and fatigue in diabetes", k=3)

    assert results[0].chunk_id == "d1#0"
    assert results[0].rank == 1
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    cache.close()


def test_retrieve_uses_embedding_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    calls = []

    def counting_embed_fn(texts):
        calls.append(list(texts))
        return toy_embed_fn(texts)

    retriever = DenseRetriever.build(CHUNKS, counting_embed_fn, "toy-model", cache)
    assert calls == [[c.text for c in CHUNKS]]  # one batch call to embed all chunks

    calls.clear()
    retriever.retrieve("asthma inhaler", k=1)
    assert calls == [["asthma inhaler"]]  # query text not cached yet -> one call

    calls.clear()
    retriever.retrieve("asthma inhaler", k=1)
    assert calls == []  # same query text now served from cache
    cache.close()


def test_retrieve_k_capped_at_corpus_size(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    retriever = DenseRetriever.build(CHUNKS, toy_embed_fn, "toy-model", cache)
    results = retriever.retrieve("diabetes", k=100)
    assert len(results) == len(CHUNKS)
    cache.close()


def test_retrieve_on_empty_corpus_returns_empty(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    retriever = DenseRetriever.build([], toy_embed_fn, "toy-model", cache)
    assert retriever.retrieve("anything", k=5) == []
    cache.close()

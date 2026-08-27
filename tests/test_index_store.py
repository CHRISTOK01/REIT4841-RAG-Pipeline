import numpy as np

from data.schemas import Chunk
from retrieval.dense import DenseRetriever
from retrieval.embedding_cache import EmbeddingCache
from retrieval.index_store import (
    build_or_load_variant_indexes,
    index_exists,
    load_variant_indexes,
    save_variant_indexes,
    variant_dir,
)
from retrieval.lexical import LexicalRetriever

VOCAB = ["diabetes", "asthma", "exercise", "thirst", "fatigue", "inhaler"]


def toy_embed_fn(texts):
    return [np.array([t.lower().count(term) for term in VOCAB], dtype=np.float32) for t in texts]


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
    make_chunk("d3#0", "Exercise improves overall health."),
]


def test_variant_dir_layout(tmp_path):
    d = variant_dir(tmp_path, "abc123", "curated")
    assert d == tmp_path / "abc123" / "curated"


def test_save_and_load_round_trips_identical_retrieval(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    lexical = LexicalRetriever.build(CHUNKS)
    dense = DenseRetriever.build(CHUNKS, toy_embed_fn, "toy-model", cache)

    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    save_variant_indexes(CHUNKS, lexical, dense, out_dir)
    assert index_exists(out_dir)

    loaded_lexical, loaded_dense = load_variant_indexes(out_dir, toy_embed_fn, cache)

    query = "what causes thirst and fatigue"
    orig_lex = lexical.retrieve(query, k=3)
    loaded_lex = loaded_lexical.retrieve(query, k=3)
    assert [r.chunk_id for r in orig_lex] == [r.chunk_id for r in loaded_lex]
    assert [r.score for r in orig_lex] == [r.score for r in loaded_lex]

    orig_dense = dense.retrieve(query, k=3)
    loaded_dense_results = loaded_dense.retrieve(query, k=3)
    assert [r.chunk_id for r in orig_dense] == [r.chunk_id for r in loaded_dense_results]
    for a, b in zip(orig_dense, loaded_dense_results):
        assert abs(a.score - b.score) < 1e-6
    cache.close()


def test_build_or_load_builds_when_missing_then_loads_when_present(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    calls = []

    def counting_embed_fn(texts):
        calls.append(list(texts))
        return toy_embed_fn(texts)

    assert not index_exists(out_dir)
    lexical1, dense1 = build_or_load_variant_indexes(CHUNKS, out_dir, counting_embed_fn, "toy-model", cache)
    assert index_exists(out_dir)
    assert len(calls) == 1  # chunks were embedded once during build

    calls.clear()
    lexical2, dense2 = build_or_load_variant_indexes(CHUNKS, out_dir, counting_embed_fn, "toy-model", cache)
    assert calls == []  # second call loaded from disk, no re-embedding

    query = "asthma inhaler"
    assert [r.chunk_id for r in dense1.retrieve(query, 2)] == [r.chunk_id for r in dense2.retrieve(query, 2)]
    cache.close()

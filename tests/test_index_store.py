import json

import numpy as np

from data.schemas import Chunk
from retrieval.dense import DenseRetriever
from retrieval.embedding_cache import EmbeddingCache
from retrieval.index_store import (
    build_or_load_variant_indexes,
    chunks_fingerprint,
    index_exists,
    index_is_current,
    load_variant_indexes,
    save_variant_indexes,
    variant_dir,
)
from retrieval.lexical import LexicalRetriever

VOCAB = ["diabetes", "asthma", "exercise", "thirst", "fatigue", "inhaler"]


def toy_embed_fn(texts):
    return [np.array([1.0] + [t.lower().count(term) for term in VOCAB], dtype=np.float32) for t in texts]


def counting(calls):
    def embed(texts):
        calls.append(list(texts))
        return toy_embed_fn(texts)

    return embed


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
    assert variant_dir(tmp_path, "abc123", "curated") == tmp_path / "abc123" / "curated"


def test_fingerprint_changes_with_any_chunk_id_text_or_order():
    base = chunks_fingerprint(CHUNKS)
    assert base == chunks_fingerprint(list(CHUNKS))
    assert chunks_fingerprint(CHUNKS[:2]) != base
    assert chunks_fingerprint(list(reversed(CHUNKS))) != base  # FAISS rows map to chunks by position
    assert chunks_fingerprint([make_chunk("d1#0", "Diabetes causes increased thirst and fatigue!"), *CHUNKS[1:]]) != base
    assert chunks_fingerprint([make_chunk("dX#0", CHUNKS[0].text), *CHUNKS[1:]]) != base


def test_save_and_load_round_trips_identical_retrieval(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    lexical = LexicalRetriever.build(CHUNKS)
    dense = DenseRetriever.build(CHUNKS, toy_embed_fn, "toy-model", cache)
    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    save_variant_indexes(CHUNKS, lexical, dense, out_dir)
    assert index_exists(out_dir) and index_is_current(out_dir, CHUNKS, "toy-model")

    loaded_lexical, loaded_dense = load_variant_indexes(CHUNKS, out_dir, toy_embed_fn, cache)
    query = "what causes thirst and fatigue"
    assert loaded_lexical.retrieve(query, k=3) == lexical.retrieve(query, k=3)
    assert loaded_dense.retrieve(query, k=3) == dense.retrieve(query, k=3)
    assert loaded_dense.retrieve(query, k=3)[0].chunk_id == "d1#0"
    cache.close()


def test_build_or_load_builds_once_then_reuses(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    calls: list = []
    build_or_load_variant_indexes(CHUNKS, out_dir, counting(calls), "toy-model", cache)
    assert calls == [[c.text for c in CHUNKS]]

    calls.clear()
    build_or_load_variant_indexes(CHUNKS, out_dir, counting(calls), "toy-model", cache)
    assert calls == []  # loaded from disk, nothing embedded
    cache.close()


def test_index_built_from_different_chunks_is_not_reused(tmp_path):
    """Regression: an index used to be loaded whenever its files existed, even
    if the corpus had since changed -- retrieval and relevance labels would then
    refer to different chunk lists."""
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    build_or_load_variant_indexes(CHUNKS, out_dir, toy_embed_fn, "toy-model", cache)

    new_chunks = [*CHUNKS[:2], make_chunk("d9#0", "Asthma inhaler technique matters."), make_chunk("d4#0", "Fatigue has many causes.")]
    assert not index_is_current(out_dir, new_chunks, "toy-model")
    lexical, dense = build_or_load_variant_indexes(new_chunks, out_dir, toy_embed_fn, "toy-model", cache)

    assert dense.index.ntotal == 4 and lexical.bm25.corpus_size == 4
    assert {r.chunk_id for r in dense.retrieve("asthma inhaler", k=4)} == {c.chunk_id for c in new_chunks}
    assert index_is_current(out_dir, new_chunks, "toy-model")  # the directory now describes the new corpus
    cache.close()


def test_same_ids_but_edited_text_is_not_reused(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    build_or_load_variant_indexes(CHUNKS, out_dir, toy_embed_fn, "toy-model", cache)
    edited = [make_chunk("d1#0", "Diabetes asthma asthma asthma."), *CHUNKS[1:]]
    _, dense = build_or_load_variant_indexes(edited, out_dir, toy_embed_fn, "toy-model", cache)
    assert dense.retrieve("asthma", k=1)[0].chunk_id == "d1#0"  # ranking reflects the edited text
    cache.close()


def test_index_built_with_a_different_embedding_model_is_not_reused(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    build_or_load_variant_indexes(CHUNKS, out_dir, toy_embed_fn, "model-a", cache)
    calls: list = []
    _, dense = build_or_load_variant_indexes(CHUNKS, out_dir, counting(calls), "model-b", cache)
    assert calls == [[c.text for c in CHUNKS]]  # re-embedded under the new model name
    assert dense.model == "model-b"
    assert json.loads((out_dir / "dense_meta.json").read_text())["model"] == "model-b"
    cache.close()


def test_half_written_index_directory_is_rebuilt(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    out_dir = tmp_path / "artifacts" / "hash1" / "curated"
    build_or_load_variant_indexes(CHUNKS, out_dir, toy_embed_fn, "toy-model", cache)
    (out_dir / "dense_meta.json").unlink()  # the file written last: its absence marks an incomplete save
    assert not index_is_current(out_dir, CHUNKS, "toy-model")
    _, dense = build_or_load_variant_indexes(CHUNKS, out_dir, toy_embed_fn, "toy-model", cache)
    assert dense.index.ntotal == 3 and index_is_current(out_dir, CHUNKS, "toy-model")

    (out_dir / "dense_meta.json").write_text("{ not json", encoding="utf-8")
    assert not index_is_current(out_dir, CHUNKS, "toy-model")
    cache.close()

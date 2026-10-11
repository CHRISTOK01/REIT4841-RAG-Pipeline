"""Embedding cache: hits, misses, and what happens when the cache is damaged."""

import sqlite3

import numpy as np
import pytest

from retrieval.embedding_cache import CorruptEmbeddingError, EmbeddingCache, get_or_compute_embeddings


def counting_embed_fn(call_log):
    def embed(texts):
        call_log.append(list(texts))
        return [np.array([float(len(t)), 1.0, 2.0, 3.0], dtype=np.float32) for t in texts]

    return embed


@pytest.fixture
def cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    yield cache
    cache.close()


def test_put_and_get_many_roundtrip(cache):
    vectors = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    cache.put_many("model-a", ["hello", "world"], vectors)

    found = cache.get_many("model-a", ["hello", "world", "missing"])
    assert len(found) == 2
    np.testing.assert_array_equal(found[EmbeddingCache.cache_key("model-a", "hello")], vectors[0])
    np.testing.assert_array_equal(found[EmbeddingCache.cache_key("model-a", "world")], vectors[1])


def test_same_text_is_a_hit_and_makes_no_embed_call(cache):
    calls: list = []
    first = get_or_compute_embeddings(cache, "model-a", ["alpha", "beta"], counting_embed_fn(calls))
    assert calls == [["alpha", "beta"]]

    calls.clear()
    second = get_or_compute_embeddings(cache, "model-a", ["alpha", "beta"], counting_embed_fn(calls))
    assert calls == []
    np.testing.assert_array_equal(first, second)


def test_changed_text_is_a_miss(cache):
    calls: list = []
    get_or_compute_embeddings(cache, "model-a", ["alpha"], counting_embed_fn(calls))
    calls.clear()
    get_or_compute_embeddings(cache, "model-a", ["alpha", "alpha "], counting_embed_fn(calls))
    assert calls == [["alpha "]]  # one trailing space is a different text


def test_changed_model_name_is_a_miss(cache):
    calls: list = []
    get_or_compute_embeddings(cache, "model-a", ["alpha"], counting_embed_fn(calls))
    calls.clear()
    get_or_compute_embeddings(cache, "model-b", ["alpha"], counting_embed_fn(calls))
    assert calls == [["alpha"]]


def test_result_rows_follow_input_order_including_repeats(cache):
    calls: list = []
    vectors = get_or_compute_embeddings(cache, "m", ["bb", "a", "bb", "cccc"], counting_embed_fn(calls))
    assert calls == [["bb", "a", "cccc"]]  # the repeated text is embedded once
    assert vectors[:, 0].tolist() == [2.0, 1.0, 2.0, 4.0]


def test_cache_persists_across_reopen(tmp_path):
    path = tmp_path / "cache.sqlite"
    first = EmbeddingCache(path)
    first.put_many("model-a", ["persisted"], [np.array([9.0, 9.0], dtype=np.float32)])
    first.close()

    second = EmbeddingCache(path)
    found = second.get_many("model-a", ["persisted"])
    second.close()
    np.testing.assert_array_equal(found[EmbeddingCache.cache_key("model-a", "persisted")], [9.0, 9.0])


def test_get_many_handles_batches_over_the_sqlite_variable_limit(cache):
    texts = [f"text-{i}" for i in range(1500)]
    cache.put_many("model-a", texts, [np.array([float(i)], dtype=np.float32) for i in range(1500)])
    found = cache.get_many("model-a", texts)
    assert len(found) == 1500
    assert found[EmbeddingCache.cache_key("model-a", "text-1234")][0] == 1234.0


# -- damaged cache contents must be loud, never quietly used ---------------------


def _overwrite_blob(cache: EmbeddingCache, model: str, text: str, blob: bytes) -> None:
    cache._conn.execute(
        "UPDATE embeddings SET vector = ? WHERE key = ?", (blob, EmbeddingCache.cache_key(model, text))
    )
    cache._conn.commit()


def test_truncated_vector_in_cache_raises(cache):
    calls: list = []
    get_or_compute_embeddings(cache, "m", ["alpha", "beta"], counting_embed_fn(calls))
    _overwrite_blob(cache, "m", "alpha", np.array([1.0, 2.0], dtype=np.float32).tobytes()[:-3])  # 5 bytes

    with pytest.raises(CorruptEmbeddingError, match="truncated"):
        get_or_compute_embeddings(cache, "m", ["alpha", "beta"], counting_embed_fn(calls))


def test_shortened_vector_with_whole_floats_is_caught_as_a_dimension_mismatch(cache):
    """A row cut at a 4-byte boundary still decodes; it must not be stacked
    into the index next to full-length vectors."""
    calls: list = []
    get_or_compute_embeddings(cache, "m", ["alpha", "beta"], counting_embed_fn(calls))
    _overwrite_blob(cache, "m", "alpha", np.array([1.0, 2.0], dtype=np.float32).tobytes())  # 2 dims, not 4

    with pytest.raises(CorruptEmbeddingError, match="mixed dimensions"):
        get_or_compute_embeddings(cache, "m", ["alpha", "beta"], counting_embed_fn(calls))


def test_nan_vector_in_cache_raises(cache):
    cache.put_many("m", ["alpha"], [np.array([1.0, 2.0], dtype=np.float32)])
    _overwrite_blob(cache, "m", "alpha", np.array([np.nan, 2.0], dtype=np.float32).tobytes())
    with pytest.raises(CorruptEmbeddingError, match="NaN"):
        cache.get_many("m", ["alpha"])


def test_empty_vector_in_cache_raises(cache):
    cache.put_many("m", ["alpha"], [np.array([1.0, 2.0], dtype=np.float32)])
    _overwrite_blob(cache, "m", "alpha", b"")
    with pytest.raises(CorruptEmbeddingError):
        cache.get_many("m", ["alpha"])


def test_garbage_cache_file_fails_to_open_or_read(tmp_path):
    path = tmp_path / "cache.sqlite"
    path.write_bytes(b"this is not a sqlite database, it is 60 bytes of plain text....")
    with pytest.raises(sqlite3.DatabaseError):
        EmbeddingCache(path).get_many("m", ["alpha"])


def test_cache_file_cut_off_mid_write_fails_loudly(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = EmbeddingCache(path)
    texts = [f"text-{i}" for i in range(400)]
    cache.put_many("m", texts, [np.arange(64, dtype=np.float32) + i for i in range(400)])
    cache.close()
    size = path.stat().st_size
    path.write_bytes(path.read_bytes()[: size // 2])  # simulate a partial file

    # Acceptable: a loud error, or the surviving rows returned intact (the lost
    # ones are then simply re-embedded). Not acceptable: a wrong vector.
    try:
        found = EmbeddingCache(path).get_many("m", texts)
    except (sqlite3.DatabaseError, CorruptEmbeddingError):
        return
    assert len(found) < len(texts)
    for i, text in enumerate(texts):
        key = EmbeddingCache.cache_key("m", text)
        if key in found:
            np.testing.assert_array_equal(found[key], np.arange(64, dtype=np.float32) + i)


def test_embedder_returning_too_few_vectors_raises_and_caches_nothing(cache):
    def short_embed(texts):
        return [np.array([1.0, 2.0], dtype=np.float32)]  # one vector, whatever was asked

    with pytest.raises(CorruptEmbeddingError, match="returned 1 vectors for 2 texts"):
        get_or_compute_embeddings(cache, "m", ["alpha", "beta"], short_embed)
    assert cache.get_many("m", ["alpha", "beta"]) == {}


def test_embedder_returning_nan_raises_and_caches_nothing(cache):
    def nan_embed(texts):
        return [np.array([np.nan, 1.0], dtype=np.float32) for _ in texts]

    with pytest.raises(CorruptEmbeddingError):
        get_or_compute_embeddings(cache, "m", ["alpha"], nan_embed)
    assert cache.get_many("m", ["alpha"]) == {}

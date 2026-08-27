import numpy as np

from retrieval.embedding_cache import EmbeddingCache, get_or_compute_embeddings


def fake_embed_fn(call_log):
    def embed(texts):
        call_log.append(list(texts))
        return [np.full(4, float(len(t)), dtype=np.float32) for t in texts]

    return embed


def test_put_and_get_many_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    vectors = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    cache.put_many("model-a", ["hello", "world"], vectors)

    found = cache.get_many("model-a", ["hello", "world", "missing"])
    assert len(found) == 2
    key_hello = EmbeddingCache.cache_key("model-a", "hello")
    np.testing.assert_array_equal(found[key_hello], vectors[0])
    cache.close()


def test_cache_key_depends_on_model_and_text():
    k1 = EmbeddingCache.cache_key("model-a", "hello")
    k2 = EmbeddingCache.cache_key("model-b", "hello")
    k3 = EmbeddingCache.cache_key("model-a", "goodbye")
    assert len({k1, k2, k3}) == 3


def test_get_or_compute_only_calls_embed_fn_for_missing_texts(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    calls = []
    embed_fn = fake_embed_fn(calls)

    vectors1 = get_or_compute_embeddings(cache, "model-a", ["alpha", "beta"], embed_fn)
    assert vectors1.shape == (2, 4)
    assert calls == [["alpha", "beta"]]

    calls.clear()
    vectors2 = get_or_compute_embeddings(cache, "model-a", ["alpha", "gamma"], embed_fn)
    assert calls == [["gamma"]]  # "alpha" was already cached
    np.testing.assert_array_equal(vectors1[0], vectors2[0])  # same cached "alpha" vector
    cache.close()


def test_cache_persists_across_reopen(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache1 = EmbeddingCache(path)
    cache1.put_many("model-a", ["persisted"], [np.array([9.0, 9.0], dtype=np.float32)])
    cache1.close()

    cache2 = EmbeddingCache(path)
    found = cache2.get_many("model-a", ["persisted"])
    assert len(found) == 1
    cache2.close()


def test_get_many_handles_large_batches_over_sqlite_variable_limit(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    texts = [f"text-{i}" for i in range(1500)]  # exceeds the 900-per-query batching threshold
    vectors = [np.array([float(i)], dtype=np.float32) for i in range(1500)]
    cache.put_many("model-a", texts, vectors)

    found = cache.get_many("model-a", texts)
    assert len(found) == 1500
    cache.close()

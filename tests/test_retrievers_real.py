"""Retriever checks on the real embedded MedQuAD KB variants and the real
150-query set. No API calls: every embedding comes from the cache.

- FAISS vs exact brute-force cosine similarity in NumPy (float64).
- Hybrid vs an RRF computed independently in this file from the lexical and
  dense result lists.
- BM25 sanity: a passage's own words retrieve that passage.
- The interface contract every retriever must satisfy.
"""

from collections import defaultdict

import numpy as np
import pytest

from experiments.runner import build_retriever, keyed_rng
from retrieval.base import RetrievedChunk
from retrieval.baselines import oracle_retrieve, random_retrieve
from retrieval.embedding_cache import get_or_compute_embeddings
from retrieval.lexical import tokenize
from tests.conftest import VARIANTS, cache_only_embed_fn

pytestmark = pytest.mark.realdata

SCORE_TOLERANCE = 1e-5  # float32 (FAISS) vs float64 (NumPy) rounding


@pytest.fixture(scope="module")
def chunk_vectors(sweep_config, variant_chunks, embedding_cache):
    """{variant: raw cached embedding matrix, float64, one row per chunk}."""
    model = sweep_config.generation.embedding_model
    return {
        v: get_or_compute_embeddings(embedding_cache, model, [c.text for c in variant_chunks[v]], cache_only_embed_fn).astype(np.float64)
        for v in VARIANTS
    }


@pytest.fixture(scope="module")
def query_vectors(sweep_config, queries, embedding_cache):
    model = sweep_config.generation.embedding_model
    return get_or_compute_embeddings(embedding_cache, model, [q.question for q in queries], cache_only_embed_fn).astype(np.float64)


def brute_force_cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity from its definition: dot / (|a| * |b|)."""
    return (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("k", [5, 50])
def test_faiss_top_k_matches_brute_force_cosine_for_every_query(variant, k, real_retrievers, chunk_vectors, query_vectors, queries, variant_chunks):
    _, dense = real_retrievers[variant]
    chunks = variant_chunks[variant]
    index_of = {c.chunk_id: i for i, c in enumerate(chunks)}
    for query, query_vector in zip(queries, query_vectors):
        cosine = brute_force_cosine(chunk_vectors[variant], query_vector)
        expected_scores = np.sort(cosine)[::-1][:k]
        results = dense.retrieve(query.question, k)

        assert len(results) == k
        # same scores at every rank...
        np.testing.assert_allclose([r.score for r in results], expected_scores, atol=SCORE_TOLERANCE, err_msg=query.query_id)
        # ...and each returned chunk really has (to rounding) the score claimed for it
        for r in results:
            assert abs(cosine[index_of[r.chunk_id]] - r.score) < SCORE_TOLERANCE, (query.query_id, r.chunk_id)
        # nothing with a clearly higher cosine was left out
        kth = expected_scores[-1]
        returned = {index_of[r.chunk_id] for r in results}
        clearly_better = set(np.flatnonzero(cosine > kth + SCORE_TOLERANCE).tolist())
        assert clearly_better <= returned, query.query_id


@pytest.mark.parametrize("variant", VARIANTS)
def test_vectors_stored_in_the_faiss_index_are_unit_length_and_match_the_cache(variant, real_retrievers, chunk_vectors):
    _, dense = real_retrievers[variant]
    stored = dense.index.reconstruct_n(0, dense.index.ntotal).astype(np.float64)
    assert stored.shape == chunk_vectors[variant].shape
    np.testing.assert_allclose(np.linalg.norm(stored, axis=1), 1.0, atol=1e-5)
    # row i of the index is chunk i's cached embedding, normalised (i.e. rows are not shuffled)
    raw = chunk_vectors[variant]
    np.testing.assert_allclose(stored, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-5)


def test_raw_openai_embeddings_are_already_unit_length(chunk_vectors):
    """Documents a fact about the data: text-embedding-3-small returns unit
    vectors, so on this corpus inner product and cosine happen to coincide.
    The retriever normalises anyway (tests/test_dense.py proves that on
    vectors of unequal length)."""
    norms = np.linalg.norm(chunk_vectors["curated"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_hybrid_equals_independently_computed_rrf_of_lexical_and_dense(variant, real_retrievers, queries, sweep_config):
    lexical, dense = real_retrievers[variant]
    hybrid = build_retriever("hybrid", lexical, dense, sweep_config.hybrid)
    depth, rrf_k, top_k = sweep_config.hybrid.k_per_retriever, sweep_config.hybrid.rrf_k, sweep_config.top_k
    assert (depth, rrf_k) == (50, 60)  # the values the thesis reports

    for query in queries:
        fused: dict[str, float] = defaultdict(float)
        for result_list in (lexical.retrieve(query.question, depth), dense.retrieve(query.question, depth)):
            for position, item in enumerate(result_list, start=1):
                fused[item.chunk_id] += 1.0 / (60 + position)
        expected = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]

        results = hybrid.retrieve(query.question, top_k)
        assert [r.chunk_id for r in results] == expected, query.query_id
        np.testing.assert_allclose([r.score for r in results], [fused[cid] for cid in expected], atol=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bm25_retrieves_a_passage_from_its_own_words(variant, real_retrievers, variant_chunks):
    """Query = a chunk's entire text. In at least 95% of cases the top hit must
    be that exact text. The measured exceptions (3-4%) are all one GARD
    boilerplate paragraph that recurs, cut at slightly different word offsets,
    in hundreds of documents; there BM25 may prefer a near-identical window,
    so the top hit must still share at least 80% of its vocabulary."""
    lexical, _ = real_retrievers[variant]
    chunks = variant_chunks[variant]
    sample = chunks[:: max(1, len(chunks) // 150)]
    exact = 0
    for chunk in sample:
        top = lexical.retrieve(chunk.text, k=1)
        assert top, chunk.chunk_id
        own, hit = set(tokenize(chunk.text)), set(tokenize(top[0].text))
        assert len(own & hit) / len(own | hit) >= 0.8, (chunk.chunk_id, top[0].chunk_id)
        exact += top[0].text == chunk.text
    assert exact / len(sample) >= 0.95, f"only {exact}/{len(sample)} chunks retrieved themselves"


def test_bm25_topic_keyword_query_returns_chunks_containing_the_keyword(real_retrievers, queries):
    """Every query names its topic. With the topic's rarest word as a one-word
    query, every result must contain that word."""
    lexical, _ = real_retrievers["curated"]
    checked = 0
    for query in queries:
        focus_tokens = [t for t in tokenize(query.focus) if len(t) > 4]
        if not focus_tokens:
            continue
        keyword = max(focus_tokens, key=lambda t: lexical.bm25.idf.get(t, -1.0))  # highest idf = rarest
        if keyword not in lexical.bm25.idf:
            continue
        results = lexical.retrieve(keyword, k=5)
        assert results, keyword
        assert all(keyword in tokenize(r.text) for r in results), keyword
        checked += 1
    assert checked >= 100


@pytest.mark.parametrize("variant", VARIANTS)
def test_bm25_never_returns_a_chunk_that_shares_no_term_with_the_query(variant, real_retrievers, queries):
    lexical, _ = real_retrievers[variant]
    assert lexical.retrieve("zzqxvjk qqwzzx", k=5) == []
    for query in queries:
        terms = set(tokenize(query.question))
        for r in lexical.retrieve(query.question, 50):
            assert terms & set(tokenize(r.text)), (query.query_id, r.chunk_id)


# -- interface contract ----------------------------------------------------------


def _check_contract(results, k, valid_ids, expect_exactly_k):
    assert isinstance(results, list)
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert len(results) <= k
    if expect_exactly_k:
        assert len(results) == k
    ids = [r.chunk_id for r in results]
    assert len(set(ids)) == len(ids), "duplicate chunk ids"
    assert set(ids) <= valid_ids, "chunk id not in this KB variant"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert all(isinstance(s, float) and np.isfinite(s) for s in scores)
    assert scores == sorted(scores, reverse=True), "not sorted by descending score"
    for r in results:
        assert r.text and r.doc_id == r.chunk_id.rsplit("#", 1)[0]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("method", ["lexical", "dense", "hybrid"])
@pytest.mark.parametrize("k", [1, 5, 20])
def test_every_retriever_honours_the_same_contract(variant, method, k, real_retrievers, queries, variant_chunks, sweep_config):
    lexical, dense = real_retrievers[variant]
    retriever = build_retriever(method, lexical, dense, sweep_config.hybrid)
    valid_ids = {c.chunk_id for c in variant_chunks[variant]}
    for query in queries:
        results = retriever.retrieve(query.question, k)
        # dense and hybrid always fill k from a corpus this size; BM25 may fall short only if < k chunks match
        _check_contract(results, k, valid_ids, expect_exactly_k=method != "lexical")
        assert results, (method, query.query_id)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("method", ["lexical", "dense", "hybrid"])
def test_retrieval_is_identical_when_repeated(variant, method, real_retrievers, queries, sweep_config):
    lexical, dense = real_retrievers[variant]
    retriever = build_retriever(method, lexical, dense, sweep_config.hybrid)
    for query in queries[:50]:
        first = retriever.retrieve(query.question, sweep_config.top_k)
        second = retriever.retrieve(query.question, sweep_config.top_k)
        assert first == second


def test_top_k_is_a_prefix_of_top_k_plus_more(real_retrievers, queries, sweep_config):
    """Asking for more results must not reorder the ones already returned."""
    lexical, dense = real_retrievers["curated"]
    for method in ("lexical", "dense"):
        retriever = build_retriever(method, lexical, dense, sweep_config.hybrid)
        for query in queries[:50]:
            five = [r.chunk_id for r in retriever.retrieve(query.question, 5)]
            twenty = [r.chunk_id for r in retriever.retrieve(query.question, 20)]
            assert twenty[:5] == five, (method, query.query_id)


@pytest.mark.parametrize("variant", VARIANTS)
def test_baseline_retrievers_honour_the_contract_and_their_definitions(variant, variant_chunks, queries, sweep_config):
    chunks = variant_chunks[variant]
    chunks_by_id = {c.chunk_id: c for c in chunks}
    valid_ids = set(chunks_by_id)
    k = sweep_config.top_k
    for query in queries:
        gold = {c.chunk_id for c in chunks if c.doc_id == query.query_id}

        oracle = oracle_retrieve(chunks_by_id, gold, k)
        _check_contract(oracle, k, valid_ids, expect_exactly_k=False)
        assert {r.chunk_id for r in oracle} <= gold  # only gold chunks, never padded
        assert len(oracle) == min(k, len(gold))

        rng = keyed_rng(sweep_config, "random", variant, query.query_id)
        randoms = random_retrieve(chunks, k, rng)
        _check_contract(randoms, k, valid_ids, expect_exactly_k=True)
        again = random_retrieve(chunks, k, keyed_rng(sweep_config, "random", variant, query.query_id))
        assert randoms == again  # same seed and key -> same draw

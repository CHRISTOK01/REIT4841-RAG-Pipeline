"""Shared fixtures and the switches that separate the test tiers.

- default `pytest`: fast deterministic tests + `realdata` tests (real MedQuAD
  corpus, real KB variants, real cached embeddings; never calls the API).
- `pytest --run-api`: additionally runs the `api` tests, which make real, paid
  OpenAI calls under a hard spend cap (see tests/test_api_*.py).

`realdata` tests need the processed datasets and indexes on disk
(`python -m data.ingest_medquad`, `data.kb_variants`, `data.query_set`,
`scripts.build_indexes`). If they are missing the tests are reported as
SKIPPED with that reason -- scripts/verify_all.py treats a skip as "not
verified", not as a pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.chunking import load_chunks_jsonl
from data.ingest_medquad import load_source_documents_jsonl
from data.query_set import load_query_set_jsonl
from experiments.config import load_sweep_config
from retrieval.embedding_cache import EmbeddingCache
from retrieval.index_store import index_is_current, load_variant_indexes, variant_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "datasets" / "processed"
VARIANTS = ("curated", "mixed", "reduced")


def pytest_addoption(parser):
    parser.addoption("--run-api", action="store_true", default=False, help="run tests that make real, paid OpenAI calls")
    parser.addoption("--api-max-usd", type=float, default=0.50, help="hard spend cap for one --run-api session")
    parser.addoption("--judge-model", default=None, help="judge model for the --run-api negative controls (default: the configured judge)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-api"):
        return
    skip_api = pytest.mark.skip(reason="real-API test: pass --run-api to run it")
    for item in items:
        if "api" in item.keywords:
            item.add_marker(skip_api)


@pytest.fixture(scope="session")
def sweep_config():
    return load_sweep_config(REPO_ROOT / "configs" / "full_sweep.yaml")


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"real data not on disk: {path.relative_to(REPO_ROOT)} (see tests/conftest.py)")
    return path


@pytest.fixture(scope="session")
def corpus_docs():
    return load_source_documents_jsonl(_require(PROCESSED / "medquad_documents.jsonl"))


@pytest.fixture(scope="session")
def variant_docs():
    return {v: load_source_documents_jsonl(_require(PROCESSED / f"kb_{v}" / "documents.jsonl")) for v in VARIANTS}


@pytest.fixture(scope="session")
def variant_chunks():
    return {v: load_chunks_jsonl(_require(PROCESSED / f"kb_{v}" / "chunks.jsonl")) for v in VARIANTS}


@pytest.fixture(scope="session")
def queries():
    return load_query_set_jsonl(_require(PROCESSED / "query_set.jsonl"))


def cache_only_embed_fn(texts):
    """Stands in for the OpenAI embedder in realdata tests: every text those
    tests embed must already be in the cache, so reaching this is a failure."""
    raise AssertionError(f"embedding cache miss for {len(texts)} text(s); realdata tests must not call the API")


@pytest.fixture(scope="session")
def embedding_cache(sweep_config):
    cache = EmbeddingCache(_require(REPO_ROOT / sweep_config.embedding_cache_path))
    yield cache
    cache.close()


@pytest.fixture(scope="session")
def real_retrievers(sweep_config, variant_chunks, embedding_cache):
    """{variant: (lexical, dense)} loaded from the indexes the sweep itself uses."""
    retrievers = {}
    for variant in VARIANTS:
        out_dir = variant_dir(REPO_ROOT / sweep_config.artifacts_root, sweep_config.kb_config_hash, variant)
        chunks = variant_chunks[variant]
        if not index_is_current(out_dir, chunks, sweep_config.generation.embedding_model):
            pytest.skip(f"no current index for {variant!r}: run python -m scripts.build_indexes")
        retrievers[variant] = load_variant_indexes(chunks, out_dir, cache_only_embed_fn, embedding_cache)
    return retrievers

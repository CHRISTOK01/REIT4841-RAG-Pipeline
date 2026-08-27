import random
from pathlib import Path

import pytest

from data.kb_variants import KBVariantConfig
from data.query_set import QueryItem
from data.schemas import Chunk, ChunkingConfig
from evaluation.generation_metrics import GenerationMetrics
from experiments.config import (
    GenerationMetricsConfig,
    GenerationSettings,
    HybridConfig,
    RobustnessConfig,
    SweepConfig,
)
from experiments.runner import (
    build_retriever,
    robustness_run_key,
    run_robustness_for_query,
    run_standard_query,
    standard_run_key,
)
from retrieval.base import RetrievedChunk
from retrieval.hybrid import HybridRetriever

RAW_RESPONSE = """## Summary
Short answer about the topic.

## Key Points
1. A grounded claim about the topic. [1]

## Disclaimer
Not medical advice.
"""


class FakeSweepClient:
    def __init__(self, chat_responses, embed_vectors=None):
        self.chat_responses = list(chat_responses)
        self.chat_calls = []
        self.embed_vectors = embed_vectors or {}
        self.embed_calls = []

    def chat_completion(self, messages, model, temperature, max_tokens):
        self.chat_calls.append(messages)
        return self.chat_responses.pop(0)

    def embed(self, texts, model):
        self.embed_calls.append(list(texts))
        return [self.embed_vectors.get(t, [0.0, 0.0]) for t in texts]

    def embed_fn(self, model):
        def _fn(texts):
            return self.embed(list(texts), model)

        return _fn


class StubRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, k):
        return self.results[:k]


def rc(chunk_id: str, rank: int, text: str = "context text") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], text=text, score=1.0, rank=rank)


def make_chunk(chunk_id: str, focus: str, doc_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id or chunk_id.split("#")[0],
        chunk_index=0,
        text=f"text-{chunk_id}",
        subset="Test",
        focus=focus,
        question_type="information",
        source_question="q",
    )


def make_config(generation_metrics_enabled: bool = True, top_k: int = 2) -> SweepConfig:
    return SweepConfig(
        retrieval_methods=["lexical", "dense", "hybrid"],
        kb_variant_names=["curated"],
        top_k=top_k,
        hybrid=HybridConfig(),
        generation_metrics=GenerationMetricsConfig(enabled=generation_metrics_enabled),
        robustness=RobustnessConfig(),
        kb_variant_config=KBVariantConfig(),
        chunking_config=ChunkingConfig(),
        generation=GenerationSettings(
            model="gpt-test",
            embedding_model="embed-test",
            prompt_template_id="wellness_summary_v1",
            temperature=0.2,
            max_tokens=500,
        ),
        query_set_path=Path("unused"),
        processed_data_dir=Path("unused"),
        artifacts_root=Path("unused"),
        embedding_cache_path=Path("unused"),
        results_path=Path("unused"),
    )


def make_query(query_id: str = "doc1", focus: str = "TopicA") -> QueryItem:
    return QueryItem(query_id=query_id, question="What about X?", subset="Test", focus=focus, question_type="information")


# -- key formatting ---------------------------------------------------------


def test_standard_run_key_format():
    assert standard_run_key("curated", "dense", "doc1") == "curated:dense:doc1"


def test_robustness_run_key_format():
    assert robustness_run_key("curated", "dense", "doc1", "inject_noise") == "curated:dense:doc1:robustness:inject_noise"


# -- build_retriever ----------------------------------------------------------


def test_build_retriever_selects_correct_implementation():
    lexical = StubRetriever([])
    dense = StubRetriever([])
    assert build_retriever("lexical", lexical, dense, HybridConfig()) is lexical
    assert build_retriever("dense", lexical, dense, HybridConfig()) is dense

    hybrid = build_retriever("hybrid", lexical, dense, HybridConfig(k_per_retriever=10))
    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.k_per_retriever == 10


def test_build_retriever_rejects_unknown_method():
    with pytest.raises(ValueError):
        build_retriever("unknown", StubRetriever([]), StubRetriever([]), HybridConfig())


# -- run_standard_query ----------------------------------------------------------


def test_run_standard_query_full_record_with_metrics_enabled():
    client = FakeSweepClient(
        chat_responses=[RAW_RESPONSE, "[true]", '["generated q"]', "[true, false]"],
        embed_vectors={"What about X?": [1.0, 0.0], "generated q": [1.0, 0.0]},
    )
    retriever = StubRetriever(
        [rc("doc1#0", 1, "context about topic"), rc("doc2#0", 2, "other context")]
    )
    config = make_config(generation_metrics_enabled=True, top_k=2)
    query = make_query(query_id="doc1")

    record = run_standard_query(client, retriever, query, {"doc1#0"}, config, "curated", "dense")

    assert record["record_type"] == "standard"
    assert record["run_key"] == "curated:dense:doc1"
    assert record["retrieval_metrics"]["precision_at_k"] == 0.5
    assert record["retrieval_metrics"]["recall_at_k"] == 1.0
    assert record["retrieval_metrics"]["mrr"] == 1.0
    assert record["generation"]["output"]["key_points"][0]["cited_chunk_ids"] == ["doc1#0"]
    assert record["generation_metrics"]["faithfulness"] == 1.0
    assert record["generation_metrics"]["answer_relevance"] == pytest.approx(1.0)
    assert record["generation_metrics"]["context_relevance"] == pytest.approx(0.5)


def test_run_standard_query_skips_generation_metrics_when_disabled():
    client = FakeSweepClient(chat_responses=[RAW_RESPONSE])
    retriever = StubRetriever([rc("doc1#0", 1, "context about topic")])
    config = make_config(generation_metrics_enabled=False, top_k=1)
    query = make_query(query_id="doc1")

    record = run_standard_query(client, retriever, query, {"doc1#0"}, config, "curated", "lexical")

    assert record["generation_metrics"] is None
    assert len(client.chat_calls) == 1


# -- run_robustness_for_query ----------------------------------------------------


def test_run_robustness_returns_empty_without_baseline_metrics():
    client = FakeSweepClient(chat_responses=[])
    chunks = [make_chunk("doc1#0", "TopicA", "doc1")]
    query = make_query(query_id="doc1", focus="TopicA")
    config = make_config()

    records = run_robustness_for_query(
        client,
        chunks,
        query,
        {"doc1#0"},
        config,
        "curated",
        "dense",
        baseline_retrieved=[rc("doc1#0", 1)],
        baseline_generation_metrics=None,
        rng=random.Random(1),
    )

    assert records == []
    assert client.chat_calls == []


def test_run_robustness_produces_drop_and_inject_records_with_deltas():
    chunks = [make_chunk("doc1#0", "TopicA", "doc1"), make_chunk("doc2#0", "TopicB", "doc2")]
    query = make_query(query_id="doc1", focus="TopicA")
    config = make_config(top_k=3)
    baseline_retrieved = [rc("doc1#0", 1, "context a"), rc("doc2#0", 2, "context b")]
    baseline_metrics = GenerationMetrics(faithfulness=1.0, answer_relevance=1.0, context_relevance=1.0)

    # drop_top_relevant leaves 1 context chunk (doc2#0); inject_noise leaves 3
    # (doc1#0, doc2#0, + the injected noise chunk, capped at top_k=3) -- context
    # relevance responses must match each perturbation's actual chunk count.
    drop_perturbation_responses = [RAW_RESPONSE, "[true]", '["gen q"]', "[true]"]
    inject_perturbation_responses = [RAW_RESPONSE, "[true]", '["gen q"]', "[true, false, true]"]
    client = FakeSweepClient(
        chat_responses=drop_perturbation_responses + inject_perturbation_responses,
        embed_vectors={"What about X?": [1.0, 0.0], "gen q": [1.0, 0.0]},
    )

    records = run_robustness_for_query(
        client,
        chunks,
        query,
        {"doc1#0"},
        config,
        "curated",
        "dense",
        baseline_retrieved,
        baseline_metrics,
        rng=random.Random(7),
    )

    assert {r["perturbation"] for r in records} == {"drop_top_relevant", "inject_noise"}
    for r in records:
        assert r["record_type"] == "robustness"
        assert r["baseline_generation_metrics"] == {
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "context_relevance": 1.0,
        }
        assert r["faithfulness_delta"] == 0.0  # perturbed also scored 1.0 via canned responses

    drop_record = next(r for r in records if r["perturbation"] == "drop_top_relevant")
    assert drop_record["noise_chunk_id"] is None
    assert all(c["chunk_id"] != "doc1#0" for c in drop_record["perturbed_retrieved"])

    inject_record = next(r for r in records if r["perturbation"] == "inject_noise")
    assert inject_record["noise_chunk_id"] == "doc2#0"  # only chunk with a different focus
    assert inject_record["perturbed_retrieved"][0]["chunk_id"] == "doc2#0"  # injected at rank 1


def test_run_robustness_skips_drop_when_nothing_relevant_was_retrieved():
    chunks = [make_chunk("doc1#0", "TopicA", "doc1"), make_chunk("doc2#0", "TopicB", "doc2")]
    query = make_query(query_id="doc1", focus="TopicA")
    config = make_config(top_k=3)
    baseline_retrieved = [rc("other#0", 1, "irrelevant")]  # nothing here is in relevant_chunk_ids
    baseline_metrics = GenerationMetrics(faithfulness=0.5, answer_relevance=0.5, context_relevance=0.5)

    # only inject_noise applies here (drop is a no-op, see assertion below); it leaves
    # 2 context chunks (the original "other#0" + the injected noise chunk).
    client = FakeSweepClient(
        chat_responses=[RAW_RESPONSE, "[true]", '["gen q"]', "[true, false]"],
        embed_vectors={"What about X?": [1.0, 0.0], "gen q": [1.0, 0.0]},
    )

    records = run_robustness_for_query(
        client,
        chunks,
        query,
        {"doc1#0"},  # relevant chunk id never appears in baseline_retrieved
        config,
        "curated",
        "dense",
        baseline_retrieved,
        baseline_metrics,
        rng=random.Random(1),
    )

    assert {r["perturbation"] for r in records} == {"inject_noise"}  # drop skipped: nothing to drop

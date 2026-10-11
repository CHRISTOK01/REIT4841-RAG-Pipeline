"""Unit tests for the runner's building blocks. The orchestration itself is
tested end to end in tests/test_run_sweep_integration.py."""

from dataclasses import replace

import pytest

from data.query_set import QueryItem
from data.schemas import Chunk
from experiments.config import HybridConfig
from experiments.runner import (
    Perturbed,
    build_perturbation,
    build_retriever,
    keyed_rng,
    record_seed,
    robustness_run_key,
    standard_run_key,
)
from retrieval.base import RetrievedChunk
from retrieval.hybrid import HybridRetriever
from tests.harness_fixtures import make_workspace


class StubRetriever:
    def retrieve(self, query, k):
        return []


def rc(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], text=f"text-{chunk_id}", score=1.0 / rank, rank=rank)


def make_chunk(chunk_id: str, focus: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        chunk_index=0,
        text=f"text-{chunk_id}",
        subset="Test",
        focus=focus,
        question_type="information",
        source_question="q",
    )


QUERY = QueryItem(query_id="doc1", question="What about X?", subset="Test", focus="TopicA", question_type="information")
KB = [make_chunk("doc1#0", "TopicA"), make_chunk("doc2#0", "TopicB"), make_chunk("doc3#0", "TopicC"), make_chunk("doc4#0", "TopicD")]
BASELINE = [rc("doc2#0", 1), rc("doc1#0", 2), rc("doc3#0", 3)]


@pytest.fixture
def config(tmp_path):
    return make_workspace(tmp_path)


def test_run_key_formats():
    assert standard_run_key("curated", "dense", "GARD:1:1-2") == "curated:dense:GARD:1:1-2"
    assert robustness_run_key("curated", "dense", "doc1", "inject_noise") == "curated:dense:doc1:robustness:inject_noise"


def test_build_retriever_selects_the_right_implementation_and_passes_hybrid_settings():
    lexical, dense = StubRetriever(), StubRetriever()
    assert build_retriever("lexical", lexical, dense, HybridConfig()) is lexical
    assert build_retriever("dense", lexical, dense, HybridConfig()) is dense
    hybrid = build_retriever("hybrid", lexical, dense, HybridConfig(k_per_retriever=10, rrf_k=7))
    assert isinstance(hybrid, HybridRetriever)
    assert (hybrid.k_per_retriever, hybrid.rrf_k, hybrid.retrievers) == (10, 7, [lexical, dense])
    with pytest.raises(ValueError):
        build_retriever("unknown", lexical, dense, HybridConfig())


def test_record_seed_depends_only_on_sweep_seed_repeat_and_key(config):
    seed = record_seed(config, "curated:dense:q1")
    assert seed == record_seed(config, "curated:dense:q1")
    assert 0 <= seed < 2**31
    assert seed != record_seed(config, "curated:dense:q2")
    assert seed != record_seed(replace(config, seed=124), "curated:dense:q1")
    assert seed != record_seed(replace(config, repeat=1), "curated:dense:q1")
    assert seed == record_seed(replace(config, top_k=9, max_workers=7), "curated:dense:q1")


def test_keyed_rng_is_reproducible_and_ignores_repeat(config):
    draw = lambda cfg, *parts: keyed_rng(cfg, *parts).random()  # noqa: E731
    assert draw(config, "noise", "curated", "q1") == draw(config, "noise", "curated", "q1")
    assert draw(config, "noise", "curated", "q1") != draw(config, "noise", "curated", "q2")
    assert draw(config, "noise", "curated", "q1") != draw(config, "random", "curated", "q1")
    assert draw(config, "noise", "curated", "q1") != draw(replace(config, seed=1), "noise", "curated", "q1")
    # repeat runs measure generation variance, so retrieval-side randomness must not move
    assert draw(config, "random", "curated", "q1") == draw(replace(config, repeat=3), "random", "curated", "q1")


def test_control_rerun_is_the_untouched_baseline(config):
    result = build_perturbation("control_rerun", config, "curated", QUERY, KB, BASELINE, {"doc1#0"}, {"doc1#0"})
    assert result == Perturbed(BASELINE, None, None)
    assert result.retrieved is not BASELINE


def test_drop_perturbation_names_what_it_dropped(config):
    result = build_perturbation("drop_top_relevant", config, "curated", QUERY, KB, BASELINE, {"doc1#0"}, {"doc1#0"})
    assert [c.chunk_id for c in result.retrieved] == ["doc2#0", "doc3#0"]
    assert (result.changed_chunk_id, result.noise_chunk_id) == ("doc1#0", None)


def test_drop_perturbation_explains_itself_when_nothing_relevant_was_retrieved(config):
    result = build_perturbation("drop_top_relevant", config, "curated", QUERY, KB, BASELINE, {"doc9#0"}, {"doc9#0"})
    assert result == "no relevant chunk in the baseline context to drop"


def test_inject_perturbation_protects_lenient_relevant_chunks_and_avoids_retrieved_ones(config):
    # doc3#0 is lenient-relevant, so the lowest-ranked replaceable chunk is doc2#0 (rank 1)
    result = build_perturbation("inject_noise", config, "curated", QUERY, KB, BASELINE, {"doc1#0"}, {"doc1#0", "doc3#0"})
    assert result.changed_chunk_id == "doc2#0"
    assert result.noise_chunk_id == "doc4#0"  # the only chunk that is off-topic, not relevant and not already retrieved
    assert [c.chunk_id for c in result.retrieved] == ["doc4#0", "doc1#0", "doc3#0"]


def test_inject_perturbation_explains_itself_when_it_cannot_be_applied(config):
    all_relevant = {"doc1#0", "doc2#0", "doc3#0"}
    assert build_perturbation("inject_noise", config, "curated", QUERY, KB, BASELINE, {"doc1#0"}, all_relevant) == (
        "every chunk in the baseline context is relevant; nothing to replace"
    )
    only_same_topic = [make_chunk("doc1#0", "TopicA"), make_chunk("doc5#0", "TopicA")]
    assert build_perturbation("inject_noise", config, "curated", QUERY, only_same_topic, BASELINE, {"doc1#0"}, {"doc1#0"}) == (
        "no off-topic chunk available to inject"
    )


def test_inject_perturbation_picks_the_same_noise_chunk_every_time_for_a_query(config):
    big_kb = KB + [make_chunk(f"extra{i}#0", f"Extra{i}") for i in range(40)]
    picks = {
        build_perturbation("inject_noise", config, "curated", QUERY, big_kb, BASELINE, {"doc1#0"}, {"doc1#0"}).noise_chunk_id
        for _ in range(5)
    }
    assert len(picks) == 1
    other_variant = build_perturbation("inject_noise", config, "mixed", QUERY, big_kb, BASELINE, {"doc1#0"}, {"doc1#0"})
    other_query = build_perturbation("inject_noise", config, "curated", replace(QUERY, query_id="doc7"), big_kb, BASELINE, {"doc1#0"}, {"doc1#0"})
    assert {other_variant.noise_chunk_id, other_query.noise_chunk_id} != picks  # keyed by variant and query


def test_unknown_perturbation_is_rejected(config):
    with pytest.raises(ValueError):
        build_perturbation("shuffle", config, "curated", QUERY, KB, BASELINE, set(), set())

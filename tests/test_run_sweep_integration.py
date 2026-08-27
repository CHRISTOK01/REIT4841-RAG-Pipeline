"""End-to-end wiring test for run_sweep: real index building (FAISS+BM25),
real per-query orchestration, real resumability, real robustness pass --
everything except the actual OpenAI network call, which SmartFakeClient
replaces by pattern-matching the judge prompts (they literally say "exactly
{n} boolean values" -- see evaluation/generation_metrics.py) so it can
respond correctly regardless of how many chunks/claims a given call involves.
"""

from __future__ import annotations

import re
from pathlib import Path

from data.chunking import save_chunks_jsonl
from data.kb_variants import KBVariantConfig
from data.query_set import QueryItem, save_query_set_jsonl
from data.schemas import Chunk, ChunkingConfig
from experiments.config import (
    GenerationMetricsConfig,
    GenerationSettings,
    HybridConfig,
    RobustnessConfig,
    SweepConfig,
)
from experiments.results_store import ResultsStore
from experiments.runner import run_sweep

RAW_RESPONSE = """## Summary
Short answer.

## Key Points
1. A grounded claim. [1]

## Disclaimer
Not medical advice.
"""

_BOOL_COUNT_RE = re.compile(r"exactly (\d+) boolean values")


class SmartFakeClient:
    """Responds correctly to any of generation_metrics.py's judge prompts by
    reading the required array length straight out of the system prompt,
    rather than needing a pre-scripted response queue."""

    def __init__(self):
        self.chat_calls = 0
        self.embed_calls = 0

    def chat_completion(self, messages, model, temperature, max_tokens):
        self.chat_calls += 1
        system = messages[0]["content"]
        match = _BOOL_COUNT_RE.search(system)
        if match:
            n = int(match.group(1))
            return "[" + ", ".join(["true"] * n) + "]"
        if "distinct questions" in system:
            return '["generated question"]'
        if "## Summary" in system:
            return RAW_RESPONSE
        raise AssertionError(f"Unrecognized judge/generation prompt: {system[:120]!r}")

    def embed(self, texts, model):
        self.embed_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def embed_fn(self, model):
        def _fn(texts):
            return self.embed(list(texts), model)

        return _fn


def make_chunk(chunk_id: str, doc_id: str, focus: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        subset="Test",
        focus=focus,
        question_type="information",
        source_question=text,
    )


def build_config(tmp_path: Path) -> SweepConfig:
    return SweepConfig(
        retrieval_methods=["lexical", "dense", "hybrid"],
        kb_variant_names=["curated"],
        top_k=2,
        hybrid=HybridConfig(k_per_retriever=5),
        generation_metrics=GenerationMetricsConfig(enabled=True),
        robustness=RobustnessConfig(enabled=True, seed=1, retrieval_method="dense"),
        kb_variant_config=KBVariantConfig(),
        chunking_config=ChunkingConfig(),
        generation=GenerationSettings(
            model="gpt-test",
            embedding_model="embed-test",
            prompt_template_id="wellness_summary_v1",
            temperature=0.2,
            max_tokens=500,
        ),
        query_set_path=tmp_path / "query_set.jsonl",
        processed_data_dir=tmp_path / "processed",
        artifacts_root=tmp_path / "artifacts",
        embedding_cache_path=tmp_path / "cache.sqlite",
        results_path=tmp_path / "results.jsonl",
    )


def setup_corpus(tmp_path: Path) -> None:
    chunks = [
        make_chunk("d1#0", "d1", "Diabetes", "Diabetes causes increased thirst and fatigue."),
        make_chunk("d2#0", "d2", "Asthma", "Asthma treatment often includes an inhaler."),
        make_chunk("d3#0", "d3", "Exercise", "Exercise improves cardiovascular health."),
    ]
    save_chunks_jsonl(chunks, tmp_path / "processed" / "kb_curated" / "chunks.jsonl")

    queries = [
        QueryItem(query_id="d1", question="What causes thirst and fatigue?", subset="Test", focus="Diabetes", question_type="information"),
        QueryItem(query_id="d2", question="How is asthma treated?", subset="Test", focus="Asthma", question_type="information"),
    ]
    save_query_set_jsonl(queries, tmp_path / "query_set.jsonl")


def test_run_sweep_produces_all_standard_and_robustness_records(tmp_path):
    setup_corpus(tmp_path)
    config = build_config(tmp_path)

    run_sweep(config, SmartFakeClient())

    store = ResultsStore(config.results_path)
    records = list(store.iter_records())
    keys = {r["run_key"] for r in records}

    expected_standard = {
        f"curated:{method}:{qid}" for method in ["lexical", "dense", "hybrid"] for qid in ["d1", "d2"]
    }
    assert expected_standard <= keys

    robustness_keys = {k for k in keys if ":robustness:" in k}
    assert robustness_keys  # at least one robustness record was produced
    assert all(k.startswith("curated:dense:") for k in robustness_keys)  # only for the configured method

    standard_records = [r for r in records if r["record_type"] == "standard"]
    assert all(r["generation_metrics"] is not None for r in standard_records)
    assert all(r["generation"]["output"]["summary"] for r in standard_records)


def test_run_sweep_is_resumable_and_makes_no_calls_on_full_rerun(tmp_path):
    setup_corpus(tmp_path)
    config = build_config(tmp_path)

    run_sweep(config, SmartFakeClient())
    keys_after_first_run = ResultsStore(config.results_path).load_completed_keys()

    second_client = SmartFakeClient()
    run_sweep(config, second_client)
    keys_after_second_run = ResultsStore(config.results_path).load_completed_keys()

    assert keys_after_second_run == keys_after_first_run  # nothing new logged
    assert second_client.chat_calls == 0
    assert second_client.embed_calls == 0

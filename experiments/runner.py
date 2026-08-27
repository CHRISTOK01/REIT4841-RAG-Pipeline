"""Sweep runner: for every (retrieval_method, kb_variant) combination in the
grid, retrieve + generate + evaluate for every query in the fixed query set,
logging one record per run to the results store. Resumable -- combinations
already present in the results file are skipped, and a query that fails
(after the client's own retry/backoff is exhausted) is logged as an error
record but NOT marked completed, so the next run retries it rather than
silently dropping it.

query set, LLM, prompt template, and top-k are all held fixed for the whole
sweep (config.top_k, config.generation.*) -- only retrieval_method and
kb_variant vary, so any metric difference between runs is attributable to
retrieval, not incidental generation-side changes.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Protocol

from data.chunking import load_chunks_jsonl
from data.query_set import QueryItem, load_query_set_jsonl
from data.schemas import Chunk
from evaluation.generation_metrics import GenerationMetrics, compute_generation_metrics
from evaluation.relevance_judgments import build_relevance_judgments
from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.robustness import RobustnessComparison, drop_top_relevant, inject_noise_chunk, select_noise_chunk
from experiments.config import HybridConfig, SweepConfig
from experiments.results_store import ResultsStore
from generation.generator import ChatClient, generate_summary
from retrieval.base import Retriever, RetrievedChunk
from retrieval.embedding_cache import EmbeddingCache
from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from retrieval.index_store import build_or_load_variant_indexes, variant_dir


class SweepClient(ChatClient, Protocol):
    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...

    def embed_fn(self, model: str): ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def standard_run_key(kb_variant: str, retrieval_method: str, query_id: str) -> str:
    return f"{kb_variant}:{retrieval_method}:{query_id}"


def robustness_run_key(kb_variant: str, retrieval_method: str, query_id: str, perturbation: str) -> str:
    return f"{kb_variant}:{retrieval_method}:{query_id}:robustness:{perturbation}"


def build_retriever(
    method: str, lexical: Retriever, dense: Retriever, hybrid_config: HybridConfig
) -> Retriever:
    if method == "lexical":
        return lexical
    if method == "dense":
        return dense
    if method == "hybrid":
        return HybridRetriever(
            [lexical, dense], k_per_retriever=hybrid_config.k_per_retriever, fusion_fn=reciprocal_rank_fusion
        )
    raise ValueError(f"Unknown retrieval method: {method}")


def run_standard_query(
    client: SweepClient,
    retriever: Retriever,
    query: QueryItem,
    relevant_chunk_ids: set[str],
    config: SweepConfig,
    kb_variant: str,
    retrieval_method: str,
) -> dict:
    """Retrieve, generate, and (optionally) score one query. Returns a fully
    JSON-serializable record. Raises on failure -- the caller decides how to
    handle/log that (see run_sweep)."""
    retrieved = retriever.retrieve(query.question, config.top_k)
    retrieval_metrics = compute_retrieval_metrics(retrieved, relevant_chunk_ids, config.top_k)

    generation_record = generate_summary(
        client,
        query.question,
        retrieved,
        model=config.generation.model,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    )

    generation_metrics: GenerationMetrics | None = None
    if config.generation_metrics.enabled:
        generation_metrics = compute_generation_metrics(
            client,
            query.question,
            generation_record.output.summary,
            [kp.text for kp in generation_record.output.key_points],
            [c.text for c in retrieved],
            chat_model=config.generation.model,
            embedding_model=config.generation.embedding_model,
        )

    return {
        "record_type": "standard",
        "run_key": standard_run_key(kb_variant, retrieval_method, query.query_id),
        "kb_variant": kb_variant,
        "retrieval_method": retrieval_method,
        "kb_config_hash": config.kb_config_hash,
        "top_k": config.top_k,
        "query_id": query.query_id,
        "question": query.question,
        "question_type": query.question_type,
        "retrieved": [asdict(c) for c in retrieved],
        "relevant_chunk_ids": sorted(relevant_chunk_ids),
        "retrieval_metrics": asdict(retrieval_metrics),
        "generation": asdict(generation_record),
        "generation_metrics": asdict(generation_metrics) if generation_metrics else None,
        "generated_at": _now_iso(),
    }


def run_robustness_for_query(
    client: SweepClient,
    chunks: list[Chunk],
    query: QueryItem,
    relevant_chunk_ids: set[str],
    config: SweepConfig,
    kb_variant: str,
    retrieval_method: str,
    baseline_retrieved: list[RetrievedChunk],
    baseline_generation_metrics: GenerationMetrics | None,
    rng: random.Random,
) -> list[dict]:
    """Re-generate on a perturbed version of baseline_retrieved (top relevant
    chunk dropped; an off-topic distractor injected) and record how much
    generation quality shifts. Skips a perturbation that has no effect
    (nothing relevant was in baseline_retrieved to drop) or isn't possible
    (no off-topic chunk exists to inject). Requires a baseline generation
    metrics result to diff against."""
    if baseline_generation_metrics is None:
        return []

    perturbations: list[tuple[str, list[RetrievedChunk], str | None]] = []

    dropped = drop_top_relevant(baseline_retrieved, relevant_chunk_ids)
    if [c.chunk_id for c in dropped] != [c.chunk_id for c in baseline_retrieved]:
        perturbations.append(("drop_top_relevant", dropped, None))

    noise_chunk = select_noise_chunk(chunks, exclude_focus=query.focus, rng=rng)
    if noise_chunk is not None:
        injected = inject_noise_chunk(baseline_retrieved, noise_chunk, position=1)[: config.top_k]
        perturbations.append(("inject_noise", injected, noise_chunk.chunk_id))

    records: list[dict] = []
    for perturbation_name, perturbed_retrieved, noise_chunk_id in perturbations:
        generation_record = generate_summary(
            client,
            query.question,
            perturbed_retrieved,
            model=config.generation.model,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
        )
        perturbed_metrics = compute_generation_metrics(
            client,
            query.question,
            generation_record.output.summary,
            [kp.text for kp in generation_record.output.key_points],
            [c.text for c in perturbed_retrieved],
            chat_model=config.generation.model,
            embedding_model=config.generation.embedding_model,
        )
        comparison = RobustnessComparison(
            baseline_faithfulness=baseline_generation_metrics.faithfulness,
            perturbed_faithfulness=perturbed_metrics.faithfulness,
            baseline_answer_relevance=baseline_generation_metrics.answer_relevance,
            perturbed_answer_relevance=perturbed_metrics.answer_relevance,
            baseline_context_relevance=baseline_generation_metrics.context_relevance,
            perturbed_context_relevance=perturbed_metrics.context_relevance,
        )
        records.append(
            {
                "record_type": "robustness",
                "run_key": robustness_run_key(kb_variant, retrieval_method, query.query_id, perturbation_name),
                "kb_variant": kb_variant,
                "retrieval_method": retrieval_method,
                "kb_config_hash": config.kb_config_hash,
                "query_id": query.query_id,
                "question": query.question,
                "perturbation": perturbation_name,
                "noise_chunk_id": noise_chunk_id,
                "perturbed_retrieved": [asdict(c) for c in perturbed_retrieved],
                "perturbed_generation": asdict(generation_record),
                "baseline_generation_metrics": asdict(baseline_generation_metrics),
                "perturbed_generation_metrics": asdict(perturbed_metrics),
                "faithfulness_delta": comparison.faithfulness_delta,
                "answer_relevance_delta": comparison.answer_relevance_delta,
                "context_relevance_delta": comparison.context_relevance_delta,
                "generated_at": _now_iso(),
            }
        )
    return records


def run_sweep(config: SweepConfig, client: SweepClient) -> None:
    store = ResultsStore(config.results_path)
    completed = store.load_completed_keys()

    queries = load_query_set_jsonl(config.query_set_path)
    embed_fn = client.embed_fn(config.generation.embedding_model)
    cache = EmbeddingCache(config.embedding_cache_path)
    rng = random.Random(config.robustness.seed)

    try:
        for kb_variant in config.kb_variant_names:
            chunks = load_chunks_jsonl(config.kb_variant_chunks_path(kb_variant))
            relevance = build_relevance_judgments(queries, chunks)

            out_dir = variant_dir(config.artifacts_root, config.kb_config_hash, kb_variant)
            lexical, dense = build_or_load_variant_indexes(
                chunks, out_dir, embed_fn, config.generation.embedding_model, cache
            )

            for retrieval_method in config.retrieval_methods:
                retriever = build_retriever(retrieval_method, lexical, dense, config.hybrid)
                run_robustness_here = (
                    config.robustness.enabled and retrieval_method == config.robustness.retrieval_method
                )

                for query in queries:
                    run_key = standard_run_key(kb_variant, retrieval_method, query.query_id)
                    if run_key in completed:
                        continue

                    try:
                        record = run_standard_query(
                            client,
                            retriever,
                            query,
                            relevance[query.query_id],
                            config,
                            kb_variant,
                            retrieval_method,
                        )
                    except Exception as e:  # noqa: BLE001 -- log and keep the sweep going
                        print(f"[error] {run_key}: {e}")
                        store.append(
                            {
                                "record_type": "error",
                                "run_key": f"{run_key}:error:{_now_iso()}",
                                "kb_variant": kb_variant,
                                "retrieval_method": retrieval_method,
                                "query_id": query.query_id,
                                "stage": "standard",
                                "error": str(e),
                                "generated_at": _now_iso(),
                            }
                        )
                        continue

                    store.append(record)
                    completed.add(run_key)
                    print(f"[ok] {run_key}")

                    if run_robustness_here:
                        baseline_retrieved = [
                            RetrievedChunk(**c) for c in record["retrieved"]
                        ]
                        baseline_metrics = (
                            GenerationMetrics(**record["generation_metrics"])
                            if record["generation_metrics"]
                            else None
                        )
                        try:
                            robustness_records = run_robustness_for_query(
                                client,
                                chunks,
                                query,
                                relevance[query.query_id],
                                config,
                                kb_variant,
                                retrieval_method,
                                baseline_retrieved,
                                baseline_metrics,
                                rng,
                            )
                        except Exception as e:  # noqa: BLE001
                            print(f"[error] robustness for {run_key}: {e}")
                            robustness_records = []
                        for r_record in robustness_records:
                            if r_record["run_key"] in completed:
                                continue
                            store.append(r_record)
                            completed.add(r_record["run_key"])
                            print(f"[ok] {r_record['run_key']}")
    finally:
        cache.close()

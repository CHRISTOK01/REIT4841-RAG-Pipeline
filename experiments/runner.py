"""Sweep runner: for every (kb_variant, retrieval method or baseline) in the
grid, retrieve + generate + evaluate every query in the fixed query set and
log one record per run to the results store.

Held fixed for the whole sweep: query set, generation model, judge model,
prompt template, temperature and top-k. Only the KB variant and the way the
context is obtained vary.

Resumability and failure handling:
- Every record has a `run_key`. A key already in the results file is skipped,
  so a rerun never repeats (or re-bills) finished work.
- Robustness records have their own keys and are rebuilt from the logged
  baseline record, so an interruption between a baseline and its perturbations
  loses nothing: the next run fills in exactly what is missing.
- A query that fails (API error after retries, unusable model output,
  malformed judge reply) is logged as a `record_type: "error"` record and is
  NOT marked completed, so the next run retries it. It is never written as a
  result row.
- A perturbation that cannot be applied (nothing relevant to drop, nothing
  non-relevant to replace) is logged as `record_type: "robustness_skipped"`
  with the reason, so the expected number of robustness rows is always
  accounted for.
- Rows from a different experimental setup (see `SweepConfig.run_fingerprint`)
  in the same results file make the run stop before doing anything.

All randomness is derived from `config.seed` and the record's own key, never
from a shared generator, so what a record contains does not depend on run
order, interruption, or the number of worker threads.
"""

from __future__ import annotations

import hashlib
import random
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from data.chunking import load_chunks_jsonl
from data.ingest_medquad import load_source_documents_jsonl
from data.query_set import QueryItem, load_query_set_jsonl
from data.schemas import Chunk
from evaluation.generation_metrics import compute_generation_metrics
from evaluation.relevance_judgments import build_relevance_judgments
from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.robustness import (
    drop_top_relevant,
    replace_lowest_nonrelevant_with_noise,
    select_noise_chunk,
)
from experiments.config import NO_KB, HybridConfig, SweepConfig
from experiments.results_store import ResultsStore
from generation.budget import BudgetExceeded
from generation.generator import ChatClient, generate_summary
from generation.prompts import CLOSED_BOOK_TEMPLATE_ID
from retrieval.base import Retriever, RetrievedChunk
from retrieval.baselines import oracle_retrieve, random_retrieve
from retrieval.embedding_cache import EmbeddingCache, get_or_compute_embeddings
from retrieval.hybrid import HybridRetriever
from retrieval.index_store import build_or_load_variant_indexes, variant_dir

DELTA_METRICS = ("faithfulness", "answer_relevance", "context_relevance", "reference_agreement")


class SweepClient(ChatClient, Protocol):
    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...

    def embed_fn(self, model: str): ...


class ResultsConfigMismatch(RuntimeError):
    """The results file already holds rows produced under different settings."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_version() -> str:
    """Git commit of the code that produced a row ("+dirty" if uncommitted changes)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return f"{sha}+dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


SOURCE_PACKAGES = ("data", "retrieval", "generation", "evaluation", "experiments", "analysis", "configs")


def source_tree_sha256(root: Path | None = None) -> str:
    """Hash of every source and config file's content (line endings
    normalised). Unlike the git commit, this pins the exact code behind a row
    even when it was run with uncommitted changes; `python -m scripts.verify_all`
    prints the current value so a commit can be matched to a results file."""
    root = root or Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for package in SOURCE_PACKAGES:
        for path in sorted((root / package).rglob("*")):
            if path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts:
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(b"\x00")
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
                digest.update(b"\x01")
    return digest.hexdigest()[:16]


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standard_run_key(kb_variant: str, retrieval_method: str, query_id: str) -> str:
    return f"{kb_variant}:{retrieval_method}:{query_id}"


def robustness_run_key(kb_variant: str, retrieval_method: str, query_id: str, perturbation: str) -> str:
    return f"{kb_variant}:{retrieval_method}:{query_id}:robustness:{perturbation}"


def record_seed(config: SweepConfig, run_key: str) -> int:
    """Deterministic per-record seed for the generation and judge calls."""
    digest = hashlib.sha256(f"{config.seed}|{config.repeat}|{run_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def keyed_rng(config: SweepConfig, *parts: str) -> random.Random:
    """A random generator that depends only on the sweep seed and `parts`."""
    return random.Random("|".join([str(config.seed), *parts]))


def build_retriever(
    method: str, lexical: Retriever, dense: Retriever, hybrid_config: HybridConfig
) -> Retriever:
    if method == "lexical":
        return lexical
    if method == "dense":
        return dense
    if method == "hybrid":
        return HybridRetriever(
            [lexical, dense], k_per_retriever=hybrid_config.k_per_retriever, rrf_k=hybrid_config.rrf_k
        )
    raise ValueError(f"Unknown retrieval method: {method}")


def _nan_safe_delta(perturbed: float | None, baseline: float | None) -> float | None:
    if perturbed is None or baseline is None:
        return None
    return perturbed - baseline


def generate_and_score(
    client: SweepClient,
    config: SweepConfig,
    query: QueryItem,
    retrieved: list[RetrievedChunk],
    reference_answer: str | None,
    seed: int,
    template_id: str | None = None,
) -> tuple[dict, dict | None]:
    """One generation call plus (if enabled) its judge calls. Raises on any
    failure; the caller logs that as an error record."""
    generation_record = generate_summary(
        client,
        query.question,
        retrieved,
        model=config.generation.model,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        template_id=template_id or config.generation.prompt_template_id,
        seed=seed,
        max_attempts=config.generation.max_attempts,
    )
    metrics = None
    if config.generation_metrics.enabled:
        metrics = asdict(
            compute_generation_metrics(
                client,
                query.question,
                generation_record.output,
                [c.text for c in retrieved],
                judge_model=config.generation.effective_judge_model,
                embedding_model=config.generation.embedding_model,
                num_questions=config.generation_metrics.answer_relevance_num_questions,
                reference_answer=reference_answer if config.generation_metrics.reference_agreement else None,
                seed=seed,
            )
        )
    return asdict(generation_record), metrics


_SOURCE_TREE_SHA256 = source_tree_sha256()  # the code as loaded by this process


def _common_fields(config: SweepConfig, fingerprint: str, version: str, seed: int) -> dict:
    return {
        "run_fingerprint": fingerprint,
        "settings": config.settings_for_log(),
        "code_version": version,
        "source_tree_sha256": _SOURCE_TREE_SHA256,
        "seed": seed,
    }


def run_standard_query(
    client: SweepClient,
    config: SweepConfig,
    query: QueryItem,
    retrieved: list[RetrievedChunk],
    strict_relevant: set[str],
    lenient_relevant: set[str],
    kb_variant: str,
    retrieval_method: str,
    reference_answer: str | None = None,
    fingerprint: str = "",
    version: str = "unknown",
) -> dict:
    """Generate and score one query on an already-retrieved context. Returns a
    fully JSON-serializable record."""
    run_key = standard_run_key(kb_variant, retrieval_method, query.query_id)
    seed = record_seed(config, run_key)
    closed_book = kb_variant == NO_KB
    generation, generation_metrics = generate_and_score(
        client,
        config,
        query,
        retrieved,
        reference_answer,
        seed,
        template_id=CLOSED_BOOK_TEMPLATE_ID if closed_book else None,
    )
    return {
        "record_type": "standard",
        "run_key": run_key,
        "kb_variant": kb_variant,
        "retrieval_method": retrieval_method,
        "kb_config_hash": config.kb_config_hash,
        "top_k": config.top_k,
        "query_id": query.query_id,
        "question": query.question,
        "question_type": query.question_type,
        "retrieved": [asdict(c) for c in retrieved],
        "relevant_chunk_ids": sorted(strict_relevant),
        "relevant_chunk_ids_lenient": sorted(lenient_relevant),
        "retrieval_metrics": None
        if closed_book
        else asdict(compute_retrieval_metrics(retrieved, strict_relevant, config.top_k)),
        "retrieval_metrics_lenient": None
        if closed_book
        else asdict(compute_retrieval_metrics(retrieved, lenient_relevant, config.top_k)),
        "generation": generation,
        "generation_metrics": generation_metrics,
        **_common_fields(config, fingerprint, version, seed),
        "generated_at": _now_iso(),
    }


@dataclass(frozen=True)
class Perturbed:
    retrieved: list[RetrievedChunk]
    changed_chunk_id: str | None  # the chunk that was dropped / replaced
    noise_chunk_id: str | None


def build_perturbation(
    perturbation: str,
    config: SweepConfig,
    kb_variant: str,
    query: QueryItem,
    chunks: list[Chunk],
    baseline_retrieved: list[RetrievedChunk],
    strict_relevant: set[str],
    lenient_relevant: set[str],
) -> Perturbed | str:
    """The perturbed context, or a string saying why this perturbation cannot
    be applied to this query."""
    if perturbation == "control_rerun":
        return Perturbed(list(baseline_retrieved), None, None)

    if perturbation == "drop_top_relevant":
        dropped, dropped_id = drop_top_relevant(baseline_retrieved, strict_relevant)
        if dropped_id is None:
            return "no relevant chunk in the baseline context to drop"
        return Perturbed(dropped, dropped_id, None)

    if perturbation == "inject_noise":
        protected = strict_relevant | lenient_relevant
        retrieved_ids = {c.chunk_id for c in baseline_retrieved}
        noise_chunk = select_noise_chunk(
            chunks,
            exclude_focus=query.focus,
            rng=keyed_rng(config, "noise", kb_variant, query.query_id),
            exclude_chunk_ids=protected | retrieved_ids,
            exclude_texts={c.text for c in baseline_retrieved},
        )
        if noise_chunk is None:
            return "no off-topic chunk available to inject"
        injected, replaced_id = replace_lowest_nonrelevant_with_noise(baseline_retrieved, noise_chunk, protected)
        if replaced_id is None:
            return "every chunk in the baseline context is relevant; nothing to replace"
        return Perturbed(injected, replaced_id, noise_chunk.chunk_id)

    raise ValueError(f"Unknown perturbation: {perturbation}")


def run_robustness_perturbation(
    client: SweepClient,
    config: SweepConfig,
    query: QueryItem,
    perturbation: str,
    perturbed: Perturbed,
    baseline_record: dict,
    strict_relevant: set[str],
    reference_answer: str | None = None,
    fingerprint: str = "",
    version: str = "unknown",
) -> dict:
    """Re-generate on a perturbed context and record the shift against the
    logged baseline record for the same query."""
    kb_variant = baseline_record["kb_variant"]
    retrieval_method = baseline_record["retrieval_method"]
    run_key = robustness_run_key(kb_variant, retrieval_method, query.query_id, perturbation)
    seed = record_seed(config, run_key)
    generation, perturbed_metrics = generate_and_score(
        client, config, query, perturbed.retrieved, reference_answer, seed
    )
    baseline_metrics = baseline_record["generation_metrics"]
    record = {
        "record_type": "robustness",
        "run_key": run_key,
        "baseline_run_key": baseline_record["run_key"],
        "kb_variant": kb_variant,
        "retrieval_method": retrieval_method,
        "kb_config_hash": config.kb_config_hash,
        "top_k": config.top_k,
        "query_id": query.query_id,
        "question": query.question,
        "perturbation": perturbation,
        "changed_chunk_id": perturbed.changed_chunk_id,
        "noise_chunk_id": perturbed.noise_chunk_id,
        "relevant_chunk_ids": sorted(strict_relevant),
        "baseline_retrieved_chunk_ids": [c["chunk_id"] for c in baseline_record["retrieved"]],
        "perturbed_retrieved": [asdict(c) for c in perturbed.retrieved],
        "baseline_retrieval_metrics": baseline_record["retrieval_metrics"],
        "perturbed_retrieval_metrics": asdict(
            compute_retrieval_metrics(perturbed.retrieved, strict_relevant, config.top_k)
        ),
        "perturbed_generation": generation,
        "baseline_generation_metrics": baseline_metrics,
        "perturbed_generation_metrics": perturbed_metrics,
        **_common_fields(config, fingerprint, version, seed),
        "generated_at": _now_iso(),
    }
    for metric in DELTA_METRICS:
        record[f"{metric}_delta"] = _nan_safe_delta(perturbed_metrics.get(metric), baseline_metrics.get(metric))
    return record


@dataclass
class _Job:
    run_key: str
    stage: str
    kb_variant: str
    retrieval_method: str
    query_id: str
    fn: Callable[[], dict]


@dataclass
class SweepSummary:
    written: int = 0
    skipped_existing: int = 0
    errors: int = 0

    def __str__(self) -> str:
        return f"{self.written} record(s) written, {self.skipped_existing} already done, {self.errors} error(s)"


def _execute_jobs(jobs: list[_Job], store: ResultsStore, completed: set[str], max_workers: int, summary: SweepSummary) -> None:
    """Run jobs (inline when max_workers == 1, else on a thread pool), appending
    each result as it finishes. A job that raises becomes an error record.
    BudgetExceeded stops everything and is re-raised."""
    stop = threading.Event()

    def guarded(job: _Job) -> dict:
        if stop.is_set():
            raise BudgetExceeded("stopped: spend cap reached earlier in this run")
        return job.fn()

    def finish(job: _Job, outcome: Callable[[], dict]) -> None:
        try:
            record = outcome()
        except BudgetExceeded:
            stop.set()
            raise
        except Exception as e:  # noqa: BLE001 -- logged as an error record; never as a result
            print(f"[error] {job.run_key}: {type(e).__name__}: {e}")
            store.append(
                {
                    "record_type": "error",
                    "run_key": f"{job.run_key}:error:{_now_iso()}",
                    "failed_run_key": job.run_key,
                    "kb_variant": job.kb_variant,
                    "retrieval_method": job.retrieval_method,
                    "query_id": job.query_id,
                    "stage": job.stage,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "generated_at": _now_iso(),
                }
            )
            summary.errors += 1
            return
        store.append(record)
        completed.add(record["run_key"])
        summary.written += 1
        print(f"[ok] {record['run_key']}")

    if max_workers == 1:
        for job in jobs:
            finish(job, lambda job=job: guarded(job))
        return

    budget_error: BudgetExceeded | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(guarded, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                finish(futures[future], future.result)
            except BudgetExceeded as e:
                budget_error = e
    if budget_error is not None:
        raise budget_error


def run_sweep(config: SweepConfig, client: SweepClient) -> SweepSummary:
    store = ResultsStore(config.results_path)
    store.discard_torn_tail()
    fingerprint = config.run_fingerprint(file_sha256(config.query_set_path))
    version = code_version()

    completed: set[str] = set()
    baseline_records: dict[str, dict] = {}  # standard records of the robustness method, by run_key
    for record in store.iter_records():
        if record.get("record_type") == "error":
            continue
        if record.get("run_fingerprint") != fingerprint:
            raise ResultsConfigMismatch(
                f"{config.results_path} contains rows with run_fingerprint="
                f"{record.get('run_fingerprint')!r}, but the current settings give {fingerprint!r}. "
                "Use a new results file (or move the old one) instead of mixing setups."
            )
        completed.add(record["run_key"])
        if record["record_type"] == "standard" and record["retrieval_method"] == config.robustness.retrieval_method:
            baseline_records[record["run_key"]] = record

    summary = SweepSummary()
    queries = load_query_set_jsonl(config.query_set_path)
    if len({q.query_id for q in queries}) != len(queries):
        raise ValueError(f"{config.query_set_path} contains duplicate query ids")

    curated_dir = config.processed_data_dir / "kb_curated"
    reference_chunks = load_chunks_jsonl(curated_dir / "chunks.jsonl")
    needs_reference = config.generation_metrics.enabled and config.generation_metrics.reference_agreement
    reference_answers: dict[str, str] = {}
    if needs_reference:
        reference_answers = {
            d.doc_id: d.answer for d in load_source_documents_jsonl(curated_dir / "documents.jsonl")
        }
        missing = [q.query_id for q in queries if q.query_id not in reference_answers]
        if missing:
            raise ValueError(f"no reference answer for {len(missing)} query id(s), e.g. {missing[:3]}")

    def reference_for(query: QueryItem) -> str | None:
        return reference_answers.get(query.query_id)

    embed_fn = client.embed_fn(config.generation.embedding_model)
    cache = EmbeddingCache(config.embedding_cache_path)
    context_methods = list(config.retrieval_methods) + [b for b in config.baselines if b != "none"]

    try:
        # One batched call for every query embedding the dense retriever will need.
        if {"dense", "hybrid"} & set(config.retrieval_methods):
            get_or_compute_embeddings(
                cache, config.generation.embedding_model, [q.question for q in queries], embed_fn
            )

        for kb_variant in config.kb_variant_names:
            chunks = load_chunks_jsonl(config.kb_variant_chunks_path(kb_variant))
            if len({c.chunk_id for c in chunks}) != len(chunks):
                raise ValueError(f"KB variant {kb_variant!r} contains duplicate chunk ids")
            chunks_by_id = {c.chunk_id: c for c in chunks}
            relevance = build_relevance_judgments(queries, chunks, reference_chunks)

            out_dir = variant_dir(config.artifacts_root, config.kb_config_hash, kb_variant)
            lexical, dense = build_or_load_variant_indexes(
                chunks, out_dir, embed_fn, config.generation.embedding_model, cache
            )

            for method in context_methods:
                retriever = (
                    build_retriever(method, lexical, dense, config.hybrid)
                    if method in config.retrieval_methods
                    else None
                )
                jobs: list[_Job] = []
                for query in queries:
                    run_key = standard_run_key(kb_variant, method, query.query_id)
                    if run_key in completed:
                        summary.skipped_existing += 1
                        continue
                    strict = relevance.strict[query.query_id]
                    lenient = relevance.lenient[query.query_id]
                    # Retrieval happens here, on the main thread (FAISS + the SQLite
                    # cache are not shared across threads); only API work is deferred.
                    if method == "oracle":
                        retrieved = oracle_retrieve(chunks_by_id, strict, config.top_k)
                    elif method == "random":
                        retrieved = random_retrieve(
                            chunks, config.top_k, keyed_rng(config, "random", kb_variant, query.query_id)
                        )
                    else:
                        retrieved = retriever.retrieve(query.question, config.top_k)
                    jobs.append(
                        _Job(
                            run_key,
                            "standard",
                            kb_variant,
                            method,
                            query.query_id,
                            lambda query=query, retrieved=retrieved, strict=strict, lenient=lenient, method=method: run_standard_query(
                                client,
                                config,
                                query,
                                retrieved,
                                strict,
                                lenient,
                                kb_variant,
                                method,
                                reference_for(query),
                                fingerprint,
                                version,
                            ),
                        )
                    )
                _execute_jobs(jobs, store, completed, config.max_workers, summary)

                if not (config.robustness.enabled and method == config.robustness.retrieval_method):
                    continue
                if not config.generation_metrics.enabled:
                    continue  # nothing to compare without generation metrics

                # Standard records written just now are needed as baselines.
                for record in store.iter_records():
                    if (
                        record.get("record_type") == "standard"
                        and record["kb_variant"] == kb_variant
                        and record["retrieval_method"] == method
                    ):
                        baseline_records[record["run_key"]] = record

                jobs = []
                for query in queries:
                    baseline = baseline_records.get(standard_run_key(kb_variant, method, query.query_id))
                    if baseline is None:
                        continue  # its standard run failed; retried (with robustness) next run
                    baseline_retrieved = [RetrievedChunk(**c) for c in baseline["retrieved"]]
                    strict = relevance.strict[query.query_id]
                    for perturbation in config.robustness.perturbations:
                        run_key = robustness_run_key(kb_variant, method, query.query_id, perturbation)
                        if run_key in completed:
                            summary.skipped_existing += 1
                            continue
                        perturbed = build_perturbation(
                            perturbation,
                            config,
                            kb_variant,
                            query,
                            chunks,
                            baseline_retrieved,
                            strict,
                            relevance.lenient[query.query_id],
                        )
                        if isinstance(perturbed, str):
                            store.append(
                                {
                                    "record_type": "robustness_skipped",
                                    "run_key": run_key,
                                    "run_fingerprint": fingerprint,
                                    "kb_variant": kb_variant,
                                    "retrieval_method": method,
                                    "query_id": query.query_id,
                                    "perturbation": perturbation,
                                    "reason": perturbed,
                                    "generated_at": _now_iso(),
                                }
                            )
                            completed.add(run_key)
                            summary.written += 1
                            continue
                        jobs.append(
                            _Job(
                                run_key,
                                "robustness",
                                kb_variant,
                                method,
                                query.query_id,
                                lambda query=query, perturbation=perturbation, perturbed=perturbed, baseline=baseline, strict=strict: run_robustness_perturbation(
                                    client,
                                    config,
                                    query,
                                    perturbation,
                                    perturbed,
                                    baseline,
                                    strict,
                                    reference_for(query),
                                    fingerprint,
                                    version,
                                ),
                            )
                        )
                _execute_jobs(jobs, store, completed, config.max_workers, summary)

        if "none" in config.baselines:
            jobs = []
            for query in queries:
                run_key = standard_run_key(NO_KB, "none", query.query_id)
                if run_key in completed:
                    summary.skipped_existing += 1
                    continue
                jobs.append(
                    _Job(
                        run_key,
                        "standard",
                        NO_KB,
                        "none",
                        query.query_id,
                        lambda query=query: run_standard_query(
                            client,
                            config,
                            query,
                            [],
                            set(),
                            set(),
                            NO_KB,
                            "none",
                            reference_for(query),
                            fingerprint,
                            version,
                        ),
                    )
                )
            _execute_jobs(jobs, store, completed, config.max_workers, summary)
    finally:
        cache.close()
        budget = getattr(client, "budget", None)
        if budget is not None:
            budget.save()

    print(f"Sweep finished: {summary}")
    return summary

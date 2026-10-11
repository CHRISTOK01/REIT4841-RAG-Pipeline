"""Estimate what a sweep will cost before any paid call is made.

`estimate_sweep` counts the records a run would still have to produce (it
reads the results file, so finished work costs nothing) and prices each one
from token counts: prompt sizes are measured from the real chunks, reference
answers and prompt templates; reply sizes are the fixed allowances below,
which were set from the token usage logged by real runs of this pipeline.

This is an estimate for planning. The enforced limit is the spend cap in
generation/budget.py, which uses the token counts the API actually reports.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from data.chunking import load_chunks_jsonl
from data.ingest_medquad import load_source_documents_jsonl
from data.query_set import load_query_set_jsonl
from evaluation.generation_metrics import (
    ABSTENTION_SYSTEM_PROMPT,
    ANSWER_RELEVANCE_SYSTEM_PROMPT,
    CLAIM_SUPPORT_SYSTEM_PROMPT,
    CONTEXT_RELEVANCE_SYSTEM_PROMPT,
)
from experiments.config import NO_KB, SweepConfig
from experiments.results_store import ResultsStore
from experiments.runner import robustness_run_key, standard_run_key
from generation.budget import chat_cost_usd, embedding_cost_usd, load_pricing
from generation.prompts import SYSTEM_PROMPT_V1

# Typical reply sizes, in tokens.
ANSWER_TOKENS = 300  # one generated summary (Summary + Key Points + Disclaimer)
CLAIMS_PER_ANSWER = 7
TOKENS_PER_CLAIM_LABEL = 6
TOKENS_PER_GENERATED_QUESTION = 22
MESSAGE_OVERHEAD_TOKENS = 12  # chat formatting per call
PASSAGE_HEADER_TOKENS = 22  # "[n] (chunk_id=...)" line per passage in the generation prompt


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:  # noqa: BLE001 -- tiktoken missing or no encoding file: fall back to a rule of thumb
        return round(len(text.split()) * 1.4)


@dataclass(frozen=True)
class SweepEstimate:
    pending_records: int
    total_records: int
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    chat_usd: float
    embedding_usd: float

    @property
    def total_usd(self) -> float:
        return self.chat_usd + self.embedding_usd

    def __str__(self) -> str:
        return (
            f"{self.pending_records} of {self.total_records} records still to run: "
            f"~{self.prompt_tokens / 1e6:.2f}M prompt + ~{self.completion_tokens / 1e6:.2f}M completion tokens, "
            f"~{self.embedding_tokens / 1e6:.2f}M embedding tokens => about ${self.total_usd:.2f} "
            f"(chat ${self.chat_usd:.2f}, embeddings ${self.embedding_usd:.2f})"
        )


def estimate_sweep(config: SweepConfig, pricing: dict | None = None) -> SweepEstimate:
    pricing = pricing if pricing is not None else load_pricing()
    queries = load_query_set_jsonl(config.query_set_path)
    completed = ResultsStore(config.results_path).load_completed_keys()

    curated_dir = config.processed_data_dir / "kb_curated"
    answers = {d.doc_id: d.answer for d in load_source_documents_jsonl(curated_dir / "documents.jsonl")}
    reference_tokens = statistics.mean(count_tokens(answers[q.query_id]) for q in queries)
    question_tokens = statistics.mean(count_tokens(q.question) for q in queries)

    system_tokens = count_tokens(SYSTEM_PROMPT_V1)
    claim_tokens = CLAIMS_PER_ANSWER * 30
    n_questions = config.generation_metrics.answer_relevance_num_questions

    def record_cost(n_passages: int, passage_tokens: float) -> tuple[float, float, float]:
        """(prompt, completion, embedding) tokens for one generated-and-scored record."""
        context = n_passages * passage_tokens
        prompt = system_tokens + question_tokens + context + n_passages * PASSAGE_HEADER_TOKENS + MESSAGE_OVERHEAD_TOKENS
        completion = ANSWER_TOKENS
        embedding = 0.0
        if config.generation_metrics.enabled:
            if n_passages:
                prompt += count_tokens(CLAIM_SUPPORT_SYSTEM_PROMPT) + question_tokens + context + claim_tokens
                completion += CLAIMS_PER_ANSWER * TOKENS_PER_CLAIM_LABEL
                prompt += count_tokens(CONTEXT_RELEVANCE_SYSTEM_PROMPT) + question_tokens + context
                completion += 12 + 2 * n_passages
            prompt += count_tokens(ABSTENTION_SYSTEM_PROMPT) + ANSWER_TOKENS
            completion += 8
            prompt += count_tokens(ANSWER_RELEVANCE_SYSTEM_PROMPT) + ANSWER_TOKENS
            completion += n_questions * TOKENS_PER_GENERATED_QUESTION
            embedding += question_tokens + n_questions * TOKENS_PER_GENERATED_QUESTION
            if config.generation_metrics.reference_agreement:
                prompt += count_tokens(CLAIM_SUPPORT_SYSTEM_PROMPT) + question_tokens + reference_tokens + claim_tokens
                completion += CLAIMS_PER_ANSWER * TOKENS_PER_CLAIM_LABEL
            prompt += 4 * MESSAGE_OVERHEAD_TOKENS
        return prompt, completion, embedding

    prompt_total = completion_total = embedding_total = 0.0
    pending = total = 0

    def add(run_key: str, n_passages: int, passage_tokens: float) -> None:
        nonlocal prompt_total, completion_total, embedding_total, pending, total
        total += 1
        if run_key in completed:
            return
        pending += 1
        p, c, e = record_cost(n_passages, passage_tokens)
        prompt_total += p
        completion_total += c
        embedding_total += e

    context_methods = list(config.retrieval_methods) + [b for b in config.baselines if b != "none"]
    for kb_variant in config.kb_variant_names:
        chunks = load_chunks_jsonl(config.kb_variant_chunks_path(kb_variant))
        sample = chunks[:: max(1, len(chunks) // 500)]
        passage_tokens = statistics.mean(count_tokens(c.text) for c in sample)
        for method in context_methods:
            for q in queries:
                add(standard_run_key(kb_variant, method, q.query_id), config.top_k, passage_tokens)
            if config.robustness.enabled and method == config.robustness.retrieval_method:
                for q in queries:
                    for perturbation in config.robustness.perturbations:
                        add(
                            robustness_run_key(kb_variant, method, q.query_id, perturbation),
                            config.top_k,
                            passage_tokens,
                        )
    if "none" in config.baselines:
        for q in queries:
            add(standard_run_key(NO_KB, "none", q.query_id), 0, 0.0)

    # Generation and judging may use different models; price each at its own rate.
    gen_model, judge_model = config.generation.model, config.generation.effective_judge_model
    if gen_model == judge_model:
        chat_usd = chat_cost_usd(pricing, gen_model, round(prompt_total), round(completion_total))
    else:
        # Rough split: the generation call carries ~1/3 of the prompt tokens and the answer itself.
        gen_prompt = prompt_total / 3
        gen_completion = ANSWER_TOKENS * pending
        chat_usd = chat_cost_usd(pricing, gen_model, round(gen_prompt), round(gen_completion)) + chat_cost_usd(
            pricing, judge_model, round(prompt_total - gen_prompt), round(completion_total - gen_completion)
        )
    return SweepEstimate(
        pending_records=pending,
        total_records=total,
        prompt_tokens=round(prompt_total),
        completion_tokens=round(completion_total),
        embedding_tokens=round(embedding_total),
        chat_usd=chat_usd,
        embedding_usd=embedding_cost_usd(pricing, config.generation.embedding_model, round(embedding_total)),
    )

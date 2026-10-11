"""Run exactly one question through the pipeline with timing printed at each
stage, so you can see that "asking the AI" is a real, separately-timed network
call -- not something that already happened. Reuses the already-built curated
KB index (no re-embedding), so this itself only makes 4 new API calls: 1
generation call + 3 grading calls (faithfulness, answer relevance, context
relevance -- the last of which also does an embedding call).

    python scripts/live_single_query.py --question "What are the symptoms of Ascher Syndrome ?"
    python scripts/live_single_query.py --method lexical
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from data.chunking import load_chunks_jsonl
from evaluation.generation_metrics import compute_generation_metrics
from experiments.config import HybridConfig, load_sweep_config
from experiments.runner import build_retriever
from generation.client import OpenAIClient
from generation.generator import generate_summary
from retrieval.embedding_cache import EmbeddingCache
from retrieval.index_store import build_or_load_variant_indexes, variant_dir


def _timed(label: str, fn):
    print(f"-> {label} ...", flush=True)
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"   done in {elapsed:.2f}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one query through the pipeline with visible timing.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo_sweep.yaml"))
    parser.add_argument("--question", default="What is the outlook for Adult Acute Myeloid Leukemia ?")
    parser.add_argument("--method", default="dense", choices=["lexical", "dense", "hybrid"])
    parser.add_argument("--kb-variant", default="curated")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    client = OpenAIClient()

    total_start = time.perf_counter()

    chunks = _timed(
        f"loading '{args.kb_variant}' chunk set from disk",
        lambda: load_chunks_jsonl(config.kb_variant_chunks_path(args.kb_variant)),
    )
    print(f"   ({len(chunks)} chunks)")

    embed_fn = client.embed_fn(config.generation.embedding_model)
    cache = EmbeddingCache(config.embedding_cache_path)
    out_dir = variant_dir(config.artifacts_root, config.kb_config_hash, args.kb_variant)
    lexical, dense = _timed(
        "loading FAISS + BM25 indexes (cached on disk, no re-embedding)",
        lambda: build_or_load_variant_indexes(
            chunks, out_dir, embed_fn, config.generation.embedding_model, cache
        ),
    )
    retriever = build_retriever(args.method, lexical, dense, HybridConfig())

    print(f"\nQuestion: {args.question}\n")

    retrieved = _timed(
        f"retrieving top-{config.top_k} chunks ({args.method})",
        lambda: retriever.retrieve(args.question, config.top_k),
    )
    for c in retrieved:
        print(f"     {c.chunk_id} (score {c.score:.3f}): {c.text[:90].strip()}...")

    generation_record = _timed(
        f"calling {config.generation.model} to generate the summary (real API round-trip)",
        lambda: generate_summary(
            client,
            args.question,
            retrieved,
            model=config.generation.model,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
        ),
    )
    print(f"\nGenerated summary:\n  {generation_record.output.summary}\n")

    metrics = _timed(
        "grading the answer: faithfulness + answer relevance + context relevance (3 more API calls)",
        lambda: compute_generation_metrics(
            client,
            args.question,
            generation_record.output.summary,
            [kp.text for kp in generation_record.output.key_points],
            [c.text for c in retrieved],
            chat_model=config.generation.model,
            embedding_model=config.generation.embedding_model,
        ),
    )
    print(f"\nMetrics: {metrics}")

    cache.close()
    print(f"\nTotal wall-clock time: {time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()

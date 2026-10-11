"""Build (or refresh) the BM25 + FAISS indexes for every KB variant in a sweep
config, embedding any chunk and query text not already in the cache.

    python -m scripts.build_indexes --config configs/full_sweep.yaml

Embedding is the only paid step here (a few cents for the whole corpus); it
goes through the same spend cap as the sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.chunking import load_chunks_jsonl
from data.query_set import load_query_set_jsonl
from experiments.config import load_sweep_config
from generation.budget import SpendTracker
from generation.client import OpenAIClient
from retrieval.embedding_cache import EmbeddingCache, get_or_compute_embeddings
from retrieval.index_store import build_or_load_variant_indexes, variant_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indexes for every KB variant in a sweep config.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    budget = SpendTracker(config.budget.max_usd, config.budget.ledger_path)
    client = OpenAIClient(budget=budget)
    embed_fn = client.embed_fn(config.generation.embedding_model)
    cache = EmbeddingCache(config.embedding_cache_path)
    try:
        for kb_variant in config.kb_variant_names:
            chunks = load_chunks_jsonl(config.kb_variant_chunks_path(kb_variant))
            out_dir = variant_dir(config.artifacts_root, config.kb_config_hash, kb_variant)
            build_or_load_variant_indexes(chunks, out_dir, embed_fn, config.generation.embedding_model, cache)
            print(f"{kb_variant}: {len(chunks)} chunks indexed at {out_dir}")
        queries = load_query_set_jsonl(config.query_set_path)
        get_or_compute_embeddings(cache, config.generation.embedding_model, [q.question for q in queries], embed_fn)
        print(f"{len(queries)} query embeddings cached")
    finally:
        cache.close()
        budget.save()
    print(f"Spend ledger: ${budget.spent_usd:.4f} of ${config.budget.max_usd:.2f}")


if __name__ == "__main__":
    main()

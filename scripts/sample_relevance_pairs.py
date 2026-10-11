"""Draw the sample of query/chunk pairs used to check the relevance labels by hand.

    python -m scripts.sample_relevance_pairs

Writes reports/validation/relevance_label_sample.json with two seeded samples
from the curated KB:

- 30 "labelled relevant" pairs: a query and one chunk the strict rule marks
  relevant. Checks for false positives (a chunk labelled relevant that does
  not actually help answer the question).
- 30 "retrieved but not labelled" pairs: a query and a chunk that the dense or
  BM25 retriever ranks in its top 5 for it but that neither label set marks
  relevant. Checks for false negatives (a helpful chunk the labels call wrong).

The judgements themselves are made by a person reading the pairs; see
reports/validation/relevance_label_sample.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.chunking import load_chunks_jsonl
from data.query_set import load_query_set_jsonl
from evaluation.relevance_judgments import build_relevance_judgments
from experiments.config import load_sweep_config
from experiments.runner import build_retriever
from retrieval.embedding_cache import EmbeddingCache
from retrieval.index_store import load_variant_indexes, variant_dir


def _no_api(texts):
    raise RuntimeError("embedding cache miss: run python -m scripts.build_indexes first")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample query/chunk pairs for manual relevance labelling.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/validation/relevance_label_sample.json"))
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    rng = random.Random(args.seed)
    queries = load_query_set_jsonl(config.query_set_path)
    chunks = load_chunks_jsonl(config.kb_variant_chunks_path("curated"))
    chunk_by_id = {c.chunk_id: c for c in chunks}
    judgments = build_relevance_judgments(queries, chunks, chunks)

    relevant_pairs = [(q, cid) for q in queries for cid in sorted(judgments.strict[q.query_id])]
    labelled = rng.sample(relevant_pairs, args.n)

    cache = EmbeddingCache(config.embedding_cache_path)
    lexical, dense = load_variant_indexes(
        chunks, variant_dir(config.artifacts_root, config.kb_config_hash, "curated"), _no_api, cache
    )
    unlabelled_pairs = []
    for q in queries:
        for method in ("dense", "lexical"):
            retriever = build_retriever(method, lexical, dense, config.hybrid)
            for item in retriever.retrieve(q.question, config.top_k):
                if item.chunk_id not in judgments.lenient[q.query_id]:
                    unlabelled_pairs.append((q, item.chunk_id, method, item.rank))
    cache.close()
    seen: set[tuple[str, str]] = set()
    unique_unlabelled = []
    for q, cid, method, rank in unlabelled_pairs:
        if (q.query_id, cid) not in seen:
            seen.add((q.query_id, cid))
            unique_unlabelled.append((q, cid, method, rank))
    unlabelled = rng.sample(unique_unlabelled, args.n)

    def row(q, cid, **extra):
        c = chunk_by_id[cid]
        return {
            "query_id": q.query_id,
            "question": q.question,
            "chunk_id": cid,
            "chunk_topic": c.focus,
            "chunk_question_type": c.question_type,
            "chunk_text": c.text,
            **extra,
        }

    out = {
        "seed": args.seed,
        "population": {"labelled_relevant_pairs": len(relevant_pairs), "retrieved_but_unlabelled_pairs": len(unique_unlabelled)},
        "labelled_relevant": [row(q, cid, n_chunks_in_answer=len(judgments.strict[q.query_id])) for q, cid in labelled],
        "retrieved_but_unlabelled": [row(q, cid, retrieved_by=method, rank=rank) for q, cid, method, rank in unlabelled],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}: {args.n} of {len(relevant_pairs)} labelled-relevant pairs, {args.n} of {len(unique_unlabelled)} retrieved-but-unlabelled pairs")


if __name__ == "__main__":
    main()

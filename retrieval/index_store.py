"""Persist and reload a KB variant's FAISS + BM25 indexes and chunk metadata.

Layout: artifacts/<config_hash>/<variant>/
    chunks.jsonl       -- shared chunk metadata (used by both retrievers)
    dense.faiss         -- FAISS IndexFlatIP over L2-normalized embeddings
    dense_meta.json      -- embedding model name
    bm25.pkl             -- pickled rank_bm25.BM25Okapi

Rebuilding an index from a config the `config_hash` has already built is
unnecessary -- callers should check `variant_dir.exists()` first (or use
`build_or_load_variant_indexes`) so a resumed/rerun sweep doesn't re-embed and
re-pay for chunks it already indexed.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss

from data.chunking import load_chunks_jsonl, save_chunks_jsonl
from data.schemas import Chunk
from retrieval.dense import DenseRetriever
from retrieval.embedding_cache import EmbedFn, EmbeddingCache
from retrieval.lexical import LexicalRetriever


def variant_dir(artifacts_root: Path, config_hash: str, variant: str) -> Path:
    return artifacts_root / config_hash / variant


def index_exists(out_dir: Path) -> bool:
    return all(
        (out_dir / name).exists()
        for name in ("chunks.jsonl", "dense.faiss", "dense_meta.json", "bm25.pkl")
    )


def save_variant_indexes(
    chunks: list[Chunk],
    lexical: LexicalRetriever,
    dense: DenseRetriever,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_chunks_jsonl(chunks, out_dir / "chunks.jsonl")
    faiss.write_index(dense.index, str(out_dir / "dense.faiss"))
    (out_dir / "dense_meta.json").write_text(
        json.dumps({"model": dense.model}, indent=2), encoding="utf-8"
    )
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(lexical.bm25, f)


def load_variant_indexes(
    out_dir: Path, embed_fn: EmbedFn, cache: EmbeddingCache
) -> tuple[LexicalRetriever, DenseRetriever]:
    chunks = load_chunks_jsonl(out_dir / "chunks.jsonl")

    with (out_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)  # noqa: S301 -- artifact we generated ourselves, not untrusted input
    lexical = LexicalRetriever(chunks, bm25=bm25)

    dense_index = faiss.read_index(str(out_dir / "dense.faiss"))
    dense_meta = json.loads((out_dir / "dense_meta.json").read_text(encoding="utf-8"))
    dense = DenseRetriever(chunks, dense_index, dense_meta["model"], embed_fn, cache)

    return lexical, dense


def build_or_load_variant_indexes(
    chunks: list[Chunk],
    out_dir: Path,
    embed_fn: EmbedFn,
    dense_model: str,
    cache: EmbeddingCache,
) -> tuple[LexicalRetriever, DenseRetriever]:
    if index_exists(out_dir):
        return load_variant_indexes(out_dir, embed_fn, cache)

    lexical = LexicalRetriever.build(chunks)
    dense = DenseRetriever.build(chunks, embed_fn, dense_model, cache)
    save_variant_indexes(chunks, lexical, dense, out_dir)
    return lexical, dense

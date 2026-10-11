"""Persist and reload a KB variant's FAISS + BM25 indexes and chunk metadata.

Layout: artifacts/<config_hash>/<variant>/
    chunks.jsonl       -- shared chunk metadata (used by both retrievers)
    dense.faiss         -- FAISS IndexFlatIP over L2-normalized embeddings
    dense_meta.json      -- embedding model name + fingerprint of the chunk set
    bm25.pkl             -- pickled rank_bm25.BM25Okapi

A saved index is only reused if it was built from exactly the chunks and the
embedding model being asked for now (`index_is_current`). Anything else --
re-ingested corpus, different embedding model, half-written directory -- is
rebuilt, which is cheap because embeddings come from the cache.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import faiss

from data.chunking import save_chunks_jsonl
from data.schemas import Chunk
from retrieval.dense import DenseRetriever
from retrieval.embedding_cache import EmbedFn, EmbeddingCache
from retrieval.lexical import LexicalRetriever

_INDEX_FILES = ("chunks.jsonl", "dense.faiss", "dense_meta.json", "bm25.pkl")


def variant_dir(artifacts_root: Path, config_hash: str, variant: str) -> Path:
    return artifacts_root / config_hash / variant


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    """Hash of every chunk id + text, in order: identifies the exact corpus an
    index was built over (order matters -- FAISS rows map to chunks by position)."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


def index_exists(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in _INDEX_FILES)


def index_is_current(out_dir: Path, chunks: list[Chunk], dense_model: str) -> bool:
    if not index_exists(out_dir):
        return False
    try:
        meta = json.loads((out_dir / "dense_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("model") == dense_model
        and meta.get("num_chunks") == len(chunks)
        and meta.get("chunks_fingerprint") == chunks_fingerprint(chunks)
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
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(lexical.bm25, f)
    # Written last: its presence + matching fingerprint marks the directory complete.
    (out_dir / "dense_meta.json").write_text(
        json.dumps(
            {
                "model": dense.model,
                "num_chunks": len(chunks),
                "chunks_fingerprint": chunks_fingerprint(chunks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_variant_indexes(
    chunks: list[Chunk], out_dir: Path, embed_fn: EmbedFn, cache: EmbeddingCache
) -> tuple[LexicalRetriever, DenseRetriever]:
    """Loads the saved indexes and binds them to `chunks` -- the caller's own
    chunk list, already checked against the saved fingerprint -- so retrieval
    and relevance judgments are guaranteed to be talking about the same chunks."""
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
    if index_is_current(out_dir, chunks, dense_model):
        return load_variant_indexes(chunks, out_dir, embed_fn, cache)
    if index_exists(out_dir):
        print(f"[index] {out_dir} is stale (different chunks or embedding model) -- rebuilding")

    lexical = LexicalRetriever.build(chunks)
    dense = DenseRetriever.build(chunks, embed_fn, dense_model, cache)
    save_variant_indexes(chunks, lexical, dense, out_dir)
    return lexical, dense

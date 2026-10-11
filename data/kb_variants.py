"""Build three knowledge-base variants from one base MedQuAD corpus:

- `curated`: a quality-filtered, stratified sample of the corpus.
- `mixed`:   curated + a configurable ratio of "distractor" documents drawn from
             the rest of the corpus, simulating a diluted/noisy knowledge base.
             Distractors are restricted to topics (focuses) that do not occur in
             curated at all, and to answers that do not duplicate a curated
             answer, so a distractor cannot be a second correct source for a
             query (see `eligible_noise_docs`).
- `reduced`: curated with a configurable fraction of topics (focuses) removed
             entirely, simulating deliberate coverage gaps.

All sampling is seeded and deterministic so a variant can be exactly
reproduced from its config (see `config_hash`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from data.chunking import chunk_document, chunk_documents, save_chunks_jsonl
from data.ingest_medquad import load_source_documents_jsonl, save_source_documents_jsonl
from data.schemas import Chunk, ChunkingConfig, SourceDocument


# Bumped whenever the construction rules change, so indexes/artifacts built under
# older rules can never be picked up by `config_hash`.
#   1: mixed noise drawn from all leftover documents (could share a curated topic)
#   2: mixed noise restricted to topics absent from curated, with no chunk that
#      repeats a curated chunk verbatim and no duplicate answers
KB_BUILD_VERSION = 2


def normalize_topic(focus: str) -> str:
    return " ".join(focus.lower().split())


@dataclass(frozen=True)
class KBVariantConfig:
    seed: int = 42
    base_corpus_size: int = 1500
    min_answer_words: int = 15
    mixed_noise_ratio: float = 0.3
    reduced_drop_topic_fraction: float = 0.35

    def __post_init__(self) -> None:
        if self.base_corpus_size <= 0:
            raise ValueError("base_corpus_size must be positive")
        if self.min_answer_words < 0:
            raise ValueError("min_answer_words cannot be negative")
        if self.mixed_noise_ratio < 0:
            raise ValueError("mixed_noise_ratio cannot be negative")
        if not 0 <= self.reduced_drop_topic_fraction < 1:
            raise ValueError("reduced_drop_topic_fraction must be in [0, 1)")


def config_hash(kb_config: KBVariantConfig, chunking_config: ChunkingConfig) -> str:
    payload = {
        "kb_variant": asdict(kb_config),
        "chunking": asdict(chunking_config),
        "build_version": KB_BUILD_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _quality_filter(docs: list[SourceDocument], min_answer_words: int) -> list[SourceDocument]:
    return [d for d in docs if len(d.answer.split()) >= min_answer_words]


def _stratified_sample(
    docs: list[SourceDocument], n: int, rng: random.Random
) -> tuple[list[SourceDocument], list[SourceDocument]]:
    """Proportionally sample n docs across `subset` groups (largest-remainder
    rounding so group quotas sum exactly to n). Returns (sampled, remainder)."""
    by_subset: dict[str, list[SourceDocument]] = defaultdict(list)
    for d in docs:
        by_subset[d.subset].append(d)
    for group in by_subset.values():
        rng.shuffle(group)

    n = min(n, len(docs))
    total = len(docs)
    quotas = {s: (len(g) / total) * n for s, g in by_subset.items()}
    floor_quotas = {s: int(q) for s, q in quotas.items()}
    remainder = n - sum(floor_quotas.values())
    by_fractional_part_desc = sorted(quotas, key=lambda s: quotas[s] - floor_quotas[s], reverse=True)
    for s in by_fractional_part_desc[:remainder]:
        floor_quotas[s] += 1

    sampled: list[SourceDocument] = []
    sampled_ids: set[str] = set()
    for s, group in by_subset.items():
        take = min(floor_quotas.get(s, 0), len(group))
        sampled.extend(group[:take])
        sampled_ids.update(d.doc_id for d in group[:take])

    remainder_docs = [d for d in docs if d.doc_id not in sampled_ids]
    return sampled, remainder_docs


def sample_base_pool(
    all_docs: list[SourceDocument], config: KBVariantConfig
) -> tuple[list[SourceDocument], list[SourceDocument]]:
    """Filter to quality docs, then stratified-sample the curated base corpus.

    Returns (curated_docs, distractor_pool): distractor_pool is the unsampled
    leftover quality docs, available for `mixed` noise injection.
    """
    rng = random.Random(config.seed)
    quality_pool = _quality_filter(all_docs, config.min_answer_words)
    quality_pool.sort(key=lambda d: d.doc_id)  # deterministic order before shuffling
    return _stratified_sample(quality_pool, config.base_corpus_size, rng)


def eligible_noise_docs(
    curated_docs: list[SourceDocument],
    distractor_pool: list[SourceDocument],
    chunking_config: ChunkingConfig,
) -> list[SourceDocument]:
    """Leftover documents that are safe to inject as noise:
    - their topic does not occur anywhere in curated;
    - none of their chunks repeats a curated chunk word for word (MedQuAD
      reuses whole answers and boilerplate passages across documents, and a
      verbatim copy of a relevant passage is not noise);
    - no two of them share an answer."""
    curated_topics = {normalize_topic(d.focus) for d in curated_docs}
    curated_chunk_texts = {c.text for c in chunk_documents(curated_docs, chunking_config)}
    seen_answers: set[str] = set()
    eligible: list[SourceDocument] = []
    for doc in sorted(distractor_pool, key=lambda d: d.doc_id):
        if normalize_topic(doc.focus) in curated_topics or doc.answer in seen_answers:
            continue
        if any(c.text in curated_chunk_texts for c in chunk_document(doc, chunking_config)):
            continue
        seen_answers.add(doc.answer)
        eligible.append(doc)
    return eligible


def build_mixed(
    curated_docs: list[SourceDocument],
    distractor_pool: list[SourceDocument],
    config: KBVariantConfig,
    chunking_config: ChunkingConfig,
) -> list[SourceDocument]:
    rng = random.Random(config.seed + 1)
    n_noise = round(len(curated_docs) * config.mixed_noise_ratio)
    pool = eligible_noise_docs(curated_docs, distractor_pool, chunking_config)
    if n_noise > len(pool):
        # Never silently build a `mixed` KB with a lower ratio than configured.
        raise ValueError(
            f"mixed_noise_ratio={config.mixed_noise_ratio} needs {n_noise} off-topic noise "
            f"documents but only {len(pool)} are eligible"
        )
    rng.shuffle(pool)
    return curated_docs + pool[:n_noise]


def build_reduced(
    curated_docs: list[SourceDocument], config: KBVariantConfig
) -> tuple[list[SourceDocument], list[str]]:
    """Drop a fraction of topics (focuses) entirely from the curated set.
    Returns (reduced_docs, dropped_topics)."""
    rng = random.Random(config.seed + 2)
    topics = sorted({normalize_topic(d.focus) for d in curated_docs})
    n_drop = round(len(topics) * config.reduced_drop_topic_fraction)
    rng.shuffle(topics)
    dropped = set(topics[:n_drop])
    reduced = [d for d in curated_docs if normalize_topic(d.focus) not in dropped]
    return reduced, sorted(dropped)


@dataclass(frozen=True)
class KBVariant:
    name: str
    documents: list[SourceDocument]
    chunks: list[Chunk]
    dropped_topics: list[str] | None = None


def build_all_variants(
    all_docs: list[SourceDocument],
    kb_config: KBVariantConfig,
    chunking_config: ChunkingConfig,
) -> dict[str, KBVariant]:
    curated_docs, distractor_pool = sample_base_pool(all_docs, kb_config)
    mixed_docs = build_mixed(curated_docs, distractor_pool, kb_config, chunking_config)
    reduced_docs, dropped_topics = build_reduced(curated_docs, kb_config)

    return {
        "curated": KBVariant(
            "curated", curated_docs, chunk_documents(curated_docs, chunking_config)
        ),
        "mixed": KBVariant("mixed", mixed_docs, chunk_documents(mixed_docs, chunking_config)),
        "reduced": KBVariant(
            "reduced",
            reduced_docs,
            chunk_documents(reduced_docs, chunking_config),
            dropped_topics,
        ),
    }


def save_variant(variant: KBVariant, out_dir: Path, manifest_extra: dict) -> None:
    variant_dir = out_dir / f"kb_{variant.name}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_source_documents_jsonl(variant.documents, variant_dir / "documents.jsonl")
    save_chunks_jsonl(variant.chunks, variant_dir / "chunks.jsonl")

    manifest = {
        "variant": variant.name,
        "num_documents": len(variant.documents),
        "num_chunks": len(variant.chunks),
        "chunks_sha256": hashlib.sha256((variant_dir / "chunks.jsonl").read_bytes()).hexdigest(),
        "dropped_topics": variant.dropped_topics,
        **manifest_extra,
    }
    (variant_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated/mixed/reduced KB variants from MedQuAD.")
    parser.add_argument("--config", type=Path, default=Path("configs/kb_variants.yaml"))
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/medquad_documents.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/processed"))
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    kb_config = KBVariantConfig(**raw["kb_variant"])
    chunking_config = ChunkingConfig(**raw["chunking"])

    all_docs = load_source_documents_jsonl(args.corpus)
    variants = build_all_variants(all_docs, kb_config, chunking_config)
    chash = config_hash(kb_config, chunking_config)

    manifest_extra = {
        "config_hash": chash,
        "build_version": KB_BUILD_VERSION,
        "kb_variant_config": asdict(kb_config),
        "chunking_config": asdict(chunking_config),
        "source_corpus": args.corpus.as_posix(),
        "source_corpus_size": len(all_docs),
    }

    for variant in variants.values():
        save_variant(variant, args.out_dir, manifest_extra)
        print(
            f"{variant.name}: {len(variant.documents)} docs -> {len(variant.chunks)} chunks "
            f"(hash={chash})"
        )


if __name__ == "__main__":
    main()

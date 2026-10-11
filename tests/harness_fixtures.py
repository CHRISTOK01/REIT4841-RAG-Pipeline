"""A small on-disk workspace for harness tests, built with the project's own
data layer (real chunker, real variant builder, real query sampler) from a
hand-written mini corpus, so the sweep runner is exercised exactly as in a
real run -- only the OpenAI API is replaced (tests/fakes.py)."""

from __future__ import annotations

from pathlib import Path

from data.chunking import save_chunks_jsonl
from data.ingest_medquad import save_source_documents_jsonl
from data.kb_variants import KBVariantConfig, build_all_variants
from data.query_set import QuerySetConfig, sample_query_set, save_query_set_jsonl
from data.schemas import ChunkingConfig, SourceDocument
from experiments.config import (
    BudgetConfig,
    GenerationMetricsConfig,
    GenerationSettings,
    HybridConfig,
    RobustnessConfig,
    SweepConfig,
)

TOPICS = {
    "Diabetes": "diabetes thirst insulin blood sugar glucose",
    "Asthma": "asthma inhaler wheezing airways breathing",
    "Migraine": "migraine headache aura nausea light",
    "Kidney Stones": "kidney stones pain urine minerals",
    "Heart Failure": "heart failure fatigue swelling fluid",
    "Influenza": "influenza vaccine fever cough virus",
    "Anemia": "anemia iron fatigue pale blood",
    "Gout": "gout joint uric acid swelling",
    "Eczema": "eczema skin itching rash dry",
    "Insomnia": "insomnia sleep tired night rest",
    "Glaucoma": "glaucoma eye pressure vision nerve",
    "Scurvy": "scurvy vitamin gums citrus bruising",
}
QUESTION_TYPES = ("symptoms", "treatment")
EXTRA_TOPICS = [
    "Rabies", "Tetanus", "Malaria", "Measles", "Mumps", "Rickets",
    "Shingles", "Tinnitus", "Vertigo", "Psoriasis", "Sciatica", "Bursitis",
]  # fmt: skip


def mini_corpus() -> list[SourceDocument]:
    docs = []
    for t_idx, (topic, vocabulary) in enumerate(TOPICS.items()):
        for q_idx, question_type in enumerate(QUESTION_TYPES):
            words = vocabulary.split()
            # ~40 words: three chunks at chunk size 20 / overlap 5
            answer = " ".join(f"{words[i % len(words)]} {question_type} detail{i}" for i in range(14))
            docs.append(
                SourceDocument(
                    doc_id=f"Mini:{t_idx:02d}:{t_idx:02d}-{q_idx}",
                    subset="Mini",
                    document_id=f"{t_idx:02d}",
                    focus=topic,
                    question_id=f"{t_idx:02d}-{q_idx}",
                    question_type=question_type,
                    question=f"What is the {question_type} of {topic} ?",
                    answer=answer,
                    url=None,
                    umls_cui=None,
                )
            )
    # Single-document topics, so that some leftover documents are on topics
    # absent from curated (the only documents eligible as `mixed` noise).
    for e_idx, topic in enumerate(EXTRA_TOPICS):
        token = topic.lower()
        docs.append(
            SourceDocument(
                doc_id=f"Mini:x{e_idx:02d}:x{e_idx:02d}-0",
                subset="Mini",
                document_id=f"x{e_idx:02d}",
                focus=topic,
                question_id=f"x{e_idx:02d}-0",
                question_type="information",
                question=f"What is {topic} ?",
                answer=" ".join(f"{token} information fact{i}" for i in range(14)),
                url=None,
                umls_cui=None,
            )
        )
    return docs


def make_workspace(tmp_path: Path, **overrides) -> SweepConfig:
    """Writes KB variants + a query set under tmp_path and returns a SweepConfig
    pointing at them. Keyword overrides replace SweepConfig fields."""
    kb_config = KBVariantConfig(seed=7, base_corpus_size=16, min_answer_words=5, mixed_noise_ratio=0.25, reduced_drop_topic_fraction=0.4)
    chunking_config = ChunkingConfig(chunk_size_words=20, chunk_overlap_words=5)
    variants = build_all_variants(mini_corpus(), kb_config, chunking_config)

    processed = tmp_path / "processed"
    for name, variant in variants.items():
        save_source_documents_jsonl(variant.documents, processed / f"kb_{name}" / "documents.jsonl")
        save_chunks_jsonl(variant.chunks, processed / f"kb_{name}" / "chunks.jsonl")
    queries = sample_query_set(variants["curated"].documents, QuerySetConfig(seed=3, num_queries=8))
    save_query_set_jsonl(queries, tmp_path / "query_set.jsonl")

    settings = dict(
        retrieval_methods=["lexical", "dense", "hybrid"],
        kb_variant_names=["curated", "mixed", "reduced"],
        baselines=["oracle", "random", "none"],
        top_k=3,
        seed=123,
        hybrid=HybridConfig(k_per_retriever=10, rrf_k=60),
        generation_metrics=GenerationMetricsConfig(enabled=True, answer_relevance_num_questions=2, reference_agreement=True),
        robustness=RobustnessConfig(enabled=True, retrieval_method="dense"),
        kb_variant_config=kb_config,
        chunking_config=chunking_config,
        generation=GenerationSettings(
            model="gen-model",
            judge_model="judge-model",
            embedding_model="embed-model",
            prompt_template_id="wellness_summary_v1",
            temperature=0.2,
            max_tokens=500,
            max_attempts=2,
        ),
        query_set_path=tmp_path / "query_set.jsonl",
        processed_data_dir=processed,
        artifacts_root=tmp_path / "artifacts",
        embedding_cache_path=tmp_path / "cache.sqlite",
        results_path=tmp_path / "results.jsonl",
        budget=BudgetConfig(max_usd=1.0, ledger_path=tmp_path / "ledger.json"),
    )
    settings.update(overrides)
    return SweepConfig(**settings)


def expected_standard_keys(config: SweepConfig, query_ids: list[str]) -> set[str]:
    methods = config.retrieval_methods + [b for b in config.baselines if b != "none"]
    keys = {f"{v}:{m}:{q}" for v in config.kb_variant_names for m in methods for q in query_ids}
    if "none" in config.baselines:
        keys |= {f"none:none:{q}" for q in query_ids}
    return keys


def expected_robustness_keys(config: SweepConfig, query_ids: list[str]) -> set[str]:
    if not config.robustness.enabled:
        return set()
    return {
        f"{v}:{config.robustness.retrieval_method}:{q}:robustness:{p}"
        for v in config.kb_variant_names
        for q in query_ids
        for p in config.robustness.perturbations
    }

from data.kb_variants import (
    KBVariantConfig,
    build_all_variants,
    build_mixed,
    build_reduced,
    config_hash,
    sample_base_pool,
)
from data.schemas import ChunkingConfig, SourceDocument


def make_docs(subset: str, focus_prefix: str, count: int, words: int) -> list[SourceDocument]:
    answer = " ".join(f"word{i}" for i in range(words))
    return [
        SourceDocument(
            doc_id=f"{subset}:doc{i}:q{i}",
            subset=subset,
            document_id=f"doc{i}",
            focus=f"{focus_prefix}{i}",
            question_id=f"q{i}",
            question_type="information",
            question=f"What is {focus_prefix}{i}?",
            answer=answer,
            url=None,
            umls_cui=None,
        )
        for i in range(count)
    ]


def test_quality_filter_drops_short_answers():
    docs = make_docs("A", "TopicA", 5, words=5) + make_docs("A", "TopicB", 5, words=50)
    config = KBVariantConfig(seed=1, base_corpus_size=100, min_answer_words=15)
    curated, distractors = sample_base_pool(docs, config)
    assert all(len(d.answer.split()) >= 15 for d in curated)
    assert all(len(d.answer.split()) >= 15 for d in distractors)
    assert len(curated) + len(distractors) == 5  # the 5 short ones were dropped entirely


def test_stratified_sample_proportional_and_exact():
    docs = make_docs("A", "TopicA", 100, words=50) + make_docs("B", "TopicB", 50, words=50)
    config = KBVariantConfig(seed=1, base_corpus_size=30, min_answer_words=15)
    curated, distractors = sample_base_pool(docs, config)

    assert len(curated) == 30
    from_a = sum(1 for d in curated if d.subset == "A")
    from_b = sum(1 for d in curated if d.subset == "B")
    assert (from_a, from_b) == (20, 10)  # exact proportional split, no rounding ambiguity
    assert len(distractors) == 120


def test_sample_base_pool_is_deterministic():
    docs = make_docs("A", "Topic", 40, words=50)
    config = KBVariantConfig(seed=7, base_corpus_size=10, min_answer_words=15)
    curated1, _ = sample_base_pool(docs, config)
    curated2, _ = sample_base_pool(docs, config)
    assert [d.doc_id for d in curated1] == [d.doc_id for d in curated2]


def test_base_corpus_size_capped_when_larger_than_pool():
    docs = make_docs("A", "Topic", 10, words=50)
    config = KBVariantConfig(seed=1, base_corpus_size=1000, min_answer_words=15)
    curated, distractors = sample_base_pool(docs, config)
    assert len(curated) == 10
    assert distractors == []


def test_build_mixed_adds_distractors_not_duplicating_curated():
    docs = make_docs("A", "Topic", 100, words=50)
    config = KBVariantConfig(seed=1, base_corpus_size=20, min_answer_words=15, mixed_noise_ratio=0.5)
    curated, distractors = sample_base_pool(docs, config)
    mixed = build_mixed(curated, distractors, config)

    assert len(mixed) == len(curated) + 10  # round(20 * 0.5)
    curated_ids = {d.doc_id for d in curated}
    noise_docs = [d for d in mixed if d.doc_id not in curated_ids]
    assert len(noise_docs) == 10
    assert all(d.doc_id in {dd.doc_id for dd in distractors} for d in noise_docs)


def test_build_mixed_caps_noise_at_distractor_pool_size():
    docs = make_docs("A", "Topic", 12, words=50)
    config = KBVariantConfig(seed=1, base_corpus_size=10, min_answer_words=15, mixed_noise_ratio=5.0)
    curated, distractors = sample_base_pool(docs, config)
    assert len(distractors) == 2
    mixed = build_mixed(curated, distractors, config)
    assert len(mixed) == len(curated) + 2  # capped, not 50


def test_build_reduced_drops_entire_topics():
    docs = make_docs("A", "Topic", 20, words=50)  # 20 unique focuses, one doc each
    config = KBVariantConfig(seed=3, base_corpus_size=20, min_answer_words=15, reduced_drop_topic_fraction=0.4)
    curated, _ = sample_base_pool(docs, config)
    reduced, dropped_topics = build_reduced(curated, config)

    assert len(dropped_topics) == 8  # round(20 * 0.4)
    assert len(reduced) == len(curated) - 8
    remaining_focuses = {d.focus for d in reduced}
    assert remaining_focuses.isdisjoint(set(dropped_topics))


def test_build_all_variants_end_to_end():
    docs = make_docs("A", "TopicA", 60, words=200) + make_docs("B", "TopicB", 40, words=200)
    kb_config = KBVariantConfig(
        seed=42, base_corpus_size=50, min_answer_words=15, mixed_noise_ratio=0.3, reduced_drop_topic_fraction=0.3
    )
    chunking_config = ChunkingConfig(chunk_size_words=50, chunk_overlap_words=10)

    variants = build_all_variants(docs, kb_config, chunking_config)

    assert set(variants.keys()) == {"curated", "mixed", "reduced"}
    assert len(variants["curated"].documents) == 50
    assert len(variants["mixed"].documents) > len(variants["curated"].documents)
    assert len(variants["reduced"].documents) < len(variants["curated"].documents)
    for variant in variants.values():
        assert len(variant.chunks) > 0
        assert all(c.doc_id in {d.doc_id for d in variant.documents} for c in variant.chunks)


def test_config_hash_deterministic_and_sensitive_to_changes():
    kb_config = KBVariantConfig()
    chunking_config = ChunkingConfig()
    h1 = config_hash(kb_config, chunking_config)
    h2 = config_hash(kb_config, chunking_config)
    assert h1 == h2

    changed = KBVariantConfig(seed=99)
    h3 = config_hash(changed, chunking_config)
    assert h3 != h1

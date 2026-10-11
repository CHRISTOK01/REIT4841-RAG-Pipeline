"""KB variant construction rules on small hand-built corpora, where the right
answer can be read off. (The real MedQuAD build is checked in tests/test_data_real.py.)"""

import pytest

from data.kb_variants import (
    KBVariantConfig,
    build_all_variants,
    build_mixed,
    build_reduced,
    config_hash,
    eligible_noise_docs,
    normalize_topic,
    sample_base_pool,
)
from data.schemas import ChunkingConfig, SourceDocument


def make_doc(doc_id: str, focus: str, subset: str = "A", words: int = 50, answer: str | None = None) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        subset=subset,
        document_id=doc_id,
        focus=focus,
        question_id=doc_id,
        question_type="information",
        question=f"What is {focus}?",
        answer=answer if answer is not None else " ".join(f"{doc_id}w{i}" for i in range(words)),
        url=None,
        umls_cui=None,
    )


CHUNKING = ChunkingConfig(chunk_size_words=120, chunk_overlap_words=20)


def make_docs(subset: str, focus_prefix: str, count: int, words: int = 50) -> list[SourceDocument]:
    return [make_doc(f"{subset}:doc{i}:q{i}", f"{focus_prefix}{i}", subset, words) for i in range(count)]


def test_quality_filter_drops_short_answers():
    docs = make_docs("A", "Short", 5, words=5) + make_docs("B", "Long", 5, words=50)
    config = KBVariantConfig(seed=1, base_corpus_size=100, min_answer_words=15)
    curated, distractors = sample_base_pool(docs, config)
    assert {d.subset for d in curated} == {"B"} and len(curated) == 5
    assert distractors == []


def test_answer_exactly_at_the_minimum_length_is_kept():
    docs = [make_doc("a", "T1", words=14), make_doc("b", "T2", words=15), make_doc("c", "T3", words=16)]
    curated, _ = sample_base_pool(docs, KBVariantConfig(seed=1, base_corpus_size=10, min_answer_words=15))
    assert sorted(d.doc_id for d in curated) == ["b", "c"]


def test_stratified_sample_proportional_and_exact():
    docs = make_docs("A", "TopicA", 100) + make_docs("B", "TopicB", 50)
    curated, distractors = sample_base_pool(docs, KBVariantConfig(seed=1, base_corpus_size=30))
    assert len(curated) == 30
    # 100:50 split of 30 -> 20 and 10
    assert (sum(d.subset == "A" for d in curated), sum(d.subset == "B" for d in curated)) == (20, 10)
    assert len(distractors) == 120
    assert {d.doc_id for d in curated}.isdisjoint(d.doc_id for d in distractors)


def test_stratified_sample_largest_remainder_rounding_sums_exactly():
    docs = make_docs("A", "TA", 10) + make_docs("B", "TB", 10) + make_docs("C", "TC", 10)
    curated, _ = sample_base_pool(docs, KBVariantConfig(seed=1, base_corpus_size=10))
    # quotas 3.33 each -> 3 + 3 + 3 and one extra: total must still be exactly 10
    assert len(curated) == 10
    assert sorted(sum(d.subset == s for d in curated) for s in "ABC") == [3, 3, 4]


def test_sampling_is_deterministic_and_independent_of_input_order():
    docs = make_docs("A", "Topic", 40)
    config = KBVariantConfig(seed=7, base_corpus_size=10)
    first, _ = sample_base_pool(docs, config)
    second, _ = sample_base_pool(list(reversed(docs)), config)
    assert [d.doc_id for d in first] == [d.doc_id for d in second]


def test_a_different_seed_gives_a_different_sample():
    docs = make_docs("A", "Topic", 200)
    a, _ = sample_base_pool(docs, KBVariantConfig(seed=1, base_corpus_size=20))
    b, _ = sample_base_pool(docs, KBVariantConfig(seed=2, base_corpus_size=20))
    assert {d.doc_id for d in a} != {d.doc_id for d in b}


def test_base_corpus_size_capped_when_larger_than_pool():
    curated, distractors = sample_base_pool(make_docs("A", "Topic", 10), KBVariantConfig(seed=1, base_corpus_size=1000))
    assert len(curated) == 10 and distractors == []


# -- mixed -----------------------------------------------------------------------


def test_noise_excludes_documents_on_a_curated_topic_even_with_different_spelling_case():
    curated = [make_doc("c1", "Heart Failure"), make_doc("c2", "Asthma")]
    pool = [
        make_doc("p1", "heart   failure"),  # same topic, different case/spacing -> not noise
        make_doc("p2", "Migraine"),
        make_doc("p3", "ASTHMA"),  # same topic -> not noise
        make_doc("p4", "Gout"),
    ]
    assert [d.doc_id for d in eligible_noise_docs(curated, pool, CHUNKING)] == ["p2", "p4"]


def test_noise_excludes_duplicate_answers():
    curated = [make_doc("c1", "Asthma", answer="shared boilerplate answer text")]
    pool = [
        make_doc("p1", "Gout", answer="shared boilerplate answer text"),  # copy of a curated answer
        make_doc("p2", "Migraine", answer="unique text one"),
        make_doc("p3", "Lupus", answer="unique text one"),  # copy of another noise answer
    ]
    assert [d.doc_id for d in eligible_noise_docs(curated, pool, CHUNKING)] == ["p2"]


def test_noise_excludes_documents_that_share_a_single_chunk_with_curated():
    """Regression: the answers differ as a whole, but one chunk of the leftover
    document is a word-for-word copy of a curated chunk (in MedQuAD: boilerplate
    such as "This condition is inherited in an autosomal recessive pattern ...")."""
    chunking = ChunkingConfig(chunk_size_words=4, chunk_overlap_words=0)
    curated = [make_doc("c1", "Asthma", answer="unique asthma opening words shared boilerplate passage here")]
    pool = [
        make_doc("p1", "Gout", answer="different gout opening words shared boilerplate passage here"),
        make_doc("p2", "Lupus", answer="different lupus opening words another closing passage entirely"),
    ]
    # c1 chunks: [unique asthma opening words] [shared boilerplate passage here]; p1's 2nd chunk is identical
    assert [d.doc_id for d in eligible_noise_docs(curated, pool, chunking)] == ["p2"]


def test_build_mixed_adds_exactly_the_configured_ratio_of_off_topic_documents():
    docs = make_docs("A", "Topic", 100)
    config = KBVariantConfig(seed=1, base_corpus_size=20, mixed_noise_ratio=0.5)
    curated, distractors = sample_base_pool(docs, config)
    mixed = build_mixed(curated, distractors, config, CHUNKING)

    assert mixed[: len(curated)] == curated  # curated is kept whole, in order
    noise = mixed[len(curated) :]
    assert len(noise) == 10  # 0.5 * 20
    assert len({d.doc_id for d in mixed}) == len(mixed)
    assert {normalize_topic(d.focus) for d in noise}.isdisjoint(normalize_topic(d.focus) for d in curated)


def test_build_mixed_refuses_to_build_with_less_noise_than_configured():
    curated = [make_doc(f"c{i}", f"Topic{i}") for i in range(10)]
    # 6 leftover docs, but 3 share a curated topic -> only 3 eligible; ratio 0.5 needs 5
    pool = [make_doc(f"p{i}", f"Topic{i}") for i in range(3)] + [make_doc(f"n{i}", f"New{i}") for i in range(3)]
    with pytest.raises(ValueError, match="needs 5 off-topic noise documents but only 3"):
        build_mixed(curated, pool, KBVariantConfig(seed=1, base_corpus_size=10, mixed_noise_ratio=0.5), CHUNKING)


def test_zero_noise_ratio_leaves_curated_unchanged():
    curated = [make_doc(f"c{i}", f"Topic{i}") for i in range(4)]
    pool = [make_doc("n1", "New")]
    assert build_mixed(curated, pool, KBVariantConfig(mixed_noise_ratio=0.0), CHUNKING) == curated


# -- reduced -----------------------------------------------------------------------


def test_build_reduced_drops_whole_topics_and_every_document_on_them():
    # 10 topics, 3 documents each
    curated = [make_doc(f"t{t}d{d}", f"Topic{t}") for t in range(10) for d in range(3)]
    config = KBVariantConfig(seed=3, base_corpus_size=30, reduced_drop_topic_fraction=0.4)
    reduced, dropped = build_reduced(curated, config)

    assert len(dropped) == 4  # 0.4 * 10 topics
    assert len(reduced) == 18  # (10 - 4) topics * 3 documents
    assert {normalize_topic(d.focus) for d in reduced}.isdisjoint(dropped)
    assert {d.doc_id for d in reduced} < {d.doc_id for d in curated}  # strict subset
    kept_topics = {normalize_topic(d.focus) for d in reduced}
    assert all(sum(normalize_topic(d.focus) == t for d in reduced) == 3 for t in kept_topics)  # kept topics are whole


def test_build_reduced_treats_case_variants_as_one_topic():
    curated = [make_doc("a", "Asthma"), make_doc("b", "ASTHMA"), make_doc("c", "Gout"), make_doc("d", "gout")]
    reduced, dropped = build_reduced(curated, KBVariantConfig(seed=1, reduced_drop_topic_fraction=0.5))
    assert len(dropped) == 1  # 2 topics, half dropped
    assert len(reduced) == 2 and len({normalize_topic(d.focus) for d in reduced}) == 1


def test_zero_drop_fraction_keeps_everything():
    curated = [make_doc(f"c{i}", f"Topic{i}") for i in range(5)]
    reduced, dropped = build_reduced(curated, KBVariantConfig(reduced_drop_topic_fraction=0.0))
    assert reduced == curated and dropped == []


# -- end to end ----------------------------------------------------------------------


def test_build_all_variants_set_relations():
    docs = make_docs("A", "TopicA", 60, words=200) + make_docs("B", "TopicB", 40, words=200)
    kb_config = KBVariantConfig(seed=42, base_corpus_size=50, mixed_noise_ratio=0.3, reduced_drop_topic_fraction=0.3)
    variants = build_all_variants(docs, kb_config, ChunkingConfig(chunk_size_words=50, chunk_overlap_words=10))

    ids = {name: {d.doc_id for d in v.documents} for name, v in variants.items()}
    assert len(ids["curated"]) == 50
    assert ids["reduced"] < ids["curated"] < ids["mixed"]
    assert len(ids["mixed"]) == 65  # 50 + 0.3 * 50
    assert len(ids["reduced"]) == 35  # every topic has one doc: 50 - 0.3 * 50
    for variant in variants.values():
        assert {c.doc_id for c in variant.chunks} == {d.doc_id for d in variant.documents}
        # 200 words, size 50, step 40 -> windows at 0, 40, 80, 120, 160 -> 5 chunks per document
        assert len(variant.chunks) == 5 * len(variant.documents)


def test_config_hash_changes_with_any_setting():
    base = config_hash(KBVariantConfig(), ChunkingConfig())
    assert base == config_hash(KBVariantConfig(), ChunkingConfig())
    assert len(base) == 12
    changed = [
        config_hash(KBVariantConfig(seed=99), ChunkingConfig()),
        config_hash(KBVariantConfig(base_corpus_size=10), ChunkingConfig()),
        config_hash(KBVariantConfig(min_answer_words=1), ChunkingConfig()),
        config_hash(KBVariantConfig(mixed_noise_ratio=0.1), ChunkingConfig()),
        config_hash(KBVariantConfig(reduced_drop_topic_fraction=0.1), ChunkingConfig()),
        config_hash(KBVariantConfig(), ChunkingConfig(chunk_size_words=99)),
        config_hash(KBVariantConfig(), ChunkingConfig(chunk_overlap_words=5)),
    ]
    assert base not in changed and len(set(changed)) == len(changed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_corpus_size": 0},
        {"min_answer_words": -1},
        {"mixed_noise_ratio": -0.1},
        {"reduced_drop_topic_fraction": 1.0},
        {"reduced_drop_topic_fraction": -0.1},
    ],
)
def test_invalid_kb_config_rejected(kwargs):
    with pytest.raises(ValueError):
        KBVariantConfig(**kwargs)

"""Invariants of the REAL data on disk: the ingested MedQuAD corpus, the three
KB variants the sweep uses, the 150-query set and the relevance judgments
derived from them. No API calls."""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter

import pytest
import yaml

from data.kb_variants import KBVariantConfig, build_all_variants, config_hash, normalize_topic
from data.query_set import QuerySetConfig, sample_query_set
from data.schemas import ChunkingConfig
from evaluation.relevance_judgments import build_relevance_judgments
from retrieval.lexical import tokenize
from tests.conftest import PROCESSED, REPO_ROOT, VARIANTS

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def kb_yaml():
    return yaml.safe_load((REPO_ROOT / "configs" / "kb_variants.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kb_config(kb_yaml):
    return KBVariantConfig(**kb_yaml["kb_variant"])


@pytest.fixture(scope="module")
def chunking_config(kb_yaml):
    return ChunkingConfig(**kb_yaml["chunking"])


@pytest.fixture(scope="module")
def manifests():
    return {v: json.loads((PROCESSED / f"kb_{v}" / "manifest.json").read_text(encoding="utf-8")) for v in VARIANTS}


@pytest.fixture(scope="module")
def judgments(queries, variant_chunks):
    return {v: build_relevance_judgments(queries, variant_chunks[v], variant_chunks["curated"]) for v in VARIANTS}


# -- corpus ------------------------------------------------------------------------


def test_corpus_is_medquad_from_the_nine_usable_subsets_with_unique_ids(corpus_docs):
    assert len(corpus_docs) > 16000
    assert {d.subset for d in corpus_docs} == {
        "CancerGov", "GARD", "GHR", "MPlusHealthTopics", "NIDDK", "NINDS", "SeniorHealth", "NHLBI", "CDC",
    }  # fmt: skip
    ids = [d.doc_id for d in corpus_docs]
    assert len(set(ids)) == len(ids)  # regression: MedQuAD reuses <Document id> + qid across some CancerGov files
    assert all(d.question.strip() and d.answer.strip() for d in corpus_docs)


def test_documents_that_shared_an_id_in_medquad_are_now_distinct(corpus_docs):
    by_id = {d.doc_id: d for d in corpus_docs}
    a = by_id["CancerGov:0000013_2:0000013_2-1"]
    b = by_id["CancerGov:0000013_2_1:0000013_2-1"]
    assert a.focus == "Chronic Myeloproliferative Neoplasms" and b.focus == "Polycythemia Vera"
    assert a.answer != b.answer


# -- variant construction ---------------------------------------------------------------


def test_manifests_match_the_current_config(manifests, kb_config, chunking_config, kb_yaml):
    for variant in VARIANTS:
        m = manifests[variant]
        assert m["config_hash"] == config_hash(kb_config, chunking_config)
        assert m["kb_variant_config"] == kb_yaml["kb_variant"]
        assert m["chunking_config"] == kb_yaml["chunking"]


def test_curated_has_the_configured_size_and_passes_the_quality_filter(variant_docs, kb_config):
    curated = variant_docs["curated"]
    assert len(curated) == kb_config.base_corpus_size == 1500
    assert all(len(d.answer.split()) >= kb_config.min_answer_words for d in curated)
    assert len({d.doc_id for d in curated}) == len(curated)


def test_curated_is_stratified_in_proportion_to_the_quality_filtered_corpus(corpus_docs, variant_docs, kb_config):
    pool = Counter(d.subset for d in corpus_docs if len(d.answer.split()) >= kb_config.min_answer_words)
    sample = Counter(d.subset for d in variant_docs["curated"])
    total = sum(pool.values())
    for subset, count in pool.items():
        exact_quota = count / total * 1500
        assert abs(sample[subset] - exact_quota) < 1.0, subset  # largest-remainder rounding: off by < 1


def test_reduced_is_a_strict_subset_of_curated(variant_docs, variant_chunks):
    curated, reduced = variant_docs["curated"], variant_docs["reduced"]
    assert {d.doc_id for d in reduced} < {d.doc_id for d in curated}
    assert set(reduced) < set(curated)  # the documents themselves are unchanged
    assert set(variant_chunks["reduced"]) < set(variant_chunks["curated"])  # and so is every chunk


def test_reduced_drops_the_configured_share_of_topics_whole(variant_docs, manifests, kb_config):
    curated, reduced = variant_docs["curated"], variant_docs["reduced"]
    topics = {normalize_topic(d.focus) for d in curated}
    dropped = set(manifests["reduced"]["dropped_topics"])
    assert dropped <= topics
    assert len(dropped) == round(len(topics) * kb_config.reduced_drop_topic_fraction)
    assert kb_config.reduced_drop_topic_fraction == 0.35
    # every document on a dropped topic is gone, every other document is kept
    assert {d.doc_id for d in reduced} == {d.doc_id for d in curated if normalize_topic(d.focus) not in dropped}


def test_mixed_is_curated_plus_exactly_the_configured_ratio_of_noise(variant_docs, variant_chunks, kb_config):
    curated, mixed = variant_docs["curated"], variant_docs["mixed"]
    assert mixed[: len(curated)] == curated
    noise = mixed[len(curated) :]
    assert kb_config.mixed_noise_ratio == 0.3
    assert len(noise) == round(len(curated) * kb_config.mixed_noise_ratio) == 450
    assert len(noise) / len(curated) == pytest.approx(0.30)
    assert len({d.doc_id for d in mixed}) == len(mixed)
    assert set(variant_chunks["curated"]) < set(variant_chunks["mixed"])


def test_noise_documents_are_off_topic_and_duplicate_nothing(variant_docs):
    curated, mixed = variant_docs["curated"], variant_docs["mixed"]
    noise = mixed[len(curated) :]
    curated_topics = {normalize_topic(d.focus) for d in curated}
    assert not [d.doc_id for d in noise if normalize_topic(d.focus) in curated_topics]
    curated_answers = {d.answer for d in curated}
    assert not [d.doc_id for d in noise if d.answer in curated_answers]
    assert len({d.answer for d in noise}) == len(noise)


def test_noise_is_not_secretly_relevant_to_any_query(variant_docs, variant_chunks, queries, judgments):
    curated_ids = {d.doc_id for d in variant_docs["curated"]}
    noise_chunks = [c for c in variant_chunks["mixed"] if c.doc_id not in curated_ids]
    noise_chunk_ids = {c.chunk_id for c in noise_chunks}
    assert len({c.doc_id for c in noise_chunks}) == 450

    query_topics = {normalize_topic(q.focus) for q in queries}
    assert not [c.chunk_id for c in noise_chunks if normalize_topic(c.focus) in query_topics]
    # not relevant under either label definition
    for q in queries:
        assert not judgments["mixed"].strict[q.query_id] & noise_chunk_ids
        assert not judgments["mixed"].lenient[q.query_id] & noise_chunk_ids
    # so adding the noise changes no query's relevant set
    assert judgments["mixed"].strict == judgments["curated"].strict
    assert judgments["mixed"].lenient == judgments["curated"].lenient
    # and no noise chunk repeats, word for word, a chunk that answers a query
    answer_texts = {c.text for c in variant_chunks["curated"] if c.doc_id in {q.query_id for q in queries}}
    assert not [c.chunk_id for c in noise_chunks if c.text in answer_texts]


def test_rebuilding_from_the_corpus_reproduces_the_files_on_disk_exactly(corpus_docs, kb_config, chunking_config, variant_docs, variant_chunks):
    rebuilt = build_all_variants(corpus_docs, kb_config, chunking_config)
    for variant in VARIANTS:
        assert rebuilt[variant].documents == variant_docs[variant]
        assert rebuilt[variant].chunks == variant_chunks[variant]


def test_chunk_file_hashes_match_the_manifests(manifests):
    for variant in VARIANTS:
        digest = hashlib.sha256((PROCESSED / f"kb_{variant}" / "chunks.jsonl").read_bytes()).hexdigest()
        assert digest == manifests[variant]["chunks_sha256"]


_REBUILD_AND_HASH = """
import hashlib, json, sys, yaml
from pathlib import Path
from data.ingest_medquad import load_source_documents_jsonl
from data.kb_variants import KBVariantConfig, build_all_variants
from data.schemas import ChunkingConfig
raw = yaml.safe_load(Path("configs/kb_variants.yaml").read_text(encoding="utf-8"))
docs = load_source_documents_jsonl(Path("datasets/processed/medquad_documents.jsonl"))
variants = build_all_variants(docs, KBVariantConfig(**raw["kb_variant"]), ChunkingConfig(**raw["chunking"]))
out = {}
for name, variant in variants.items():
    blob = "".join(json.dumps(c.__dict__, ensure_ascii=False) + "\\n" for c in variant.chunks).encode("utf-8")
    out[name] = hashlib.sha256(blob).hexdigest()
print(json.dumps(out))
"""


@pytest.mark.parametrize("hash_seed", ["0", "1", "4242"])
def test_build_is_identical_in_a_fresh_process_with_a_different_hash_seed(hash_seed, manifests):
    """Python randomises set/dict-of-str ordering per process (PYTHONHASHSEED).
    A build that leaned on set order would differ between processes; each fresh
    process must reproduce the hash recorded when the KB was built."""
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _REBUILD_AND_HASH], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True
    )
    hashes = json.loads(result.stdout.strip().splitlines()[-1])
    assert hashes == {v: manifests[v]["chunks_sha256"] for v in VARIANTS}


# -- chunking, on every real document ------------------------------------------------------


def test_real_chunks_reconstruct_every_document_with_exact_overlap(variant_docs, variant_chunks, chunking_config):
    size, overlap = chunking_config.chunk_size_words, chunking_config.chunk_overlap_words
    assert (size, overlap) == (120, 20)
    by_doc: dict[str, list] = {}
    for chunk in variant_chunks["mixed"]:  # mixed contains every curated and noise document
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    assert set(by_doc) == {d.doc_id for d in variant_docs["mixed"]}
    for doc in variant_docs["mixed"]:
        chunks = by_doc[doc.doc_id]
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert [c.chunk_id for c in chunks] == [f"{doc.doc_id}#{i}" for i in range(len(chunks))]
        words: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_words = chunk.text.split(" ")
            assert 0 < len(chunk_words) <= size
            if i < len(chunks) - 1:
                assert len(chunk_words) == size
            if i > 0:
                assert chunk_words[:overlap] == chunks[i - 1].text.split(" ")[size - overlap :]
            words.extend(chunk_words if i == 0 else chunk_words[overlap:])
        assert " ".join(words) == doc.answer, doc.doc_id  # nothing lost, added or reordered


def test_real_chunk_ids_are_unique_and_no_chunk_is_empty(variant_chunks):
    for variant in VARIANTS:
        chunks = variant_chunks[variant]
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(c.text.strip() for c in chunks)


def test_real_corpus_exercises_the_edge_cases(variant_docs, variant_chunks):
    """The properties above are only meaningful if the real data contains the hard cases."""
    lengths = [len(d.answer.split()) for d in variant_docs["mixed"]]
    assert min(lengths) < 20  # very short documents (single chunk)
    assert max(lengths) > 3000  # very long documents (dozens of chunks)
    assert any(any(ord(ch) > 127 for ch in d.answer) for d in variant_docs["mixed"])  # non-ASCII text
    per_doc = Counter(c.doc_id for c in variant_chunks["mixed"])
    assert max(per_doc.values()) > 25 and min(per_doc.values()) == 1


# -- query set and relevance judgments -------------------------------------------------------


def test_query_set_is_150_distinct_curated_questions(queries, variant_docs):
    curated = {d.doc_id: d for d in variant_docs["curated"]}
    assert len(queries) == 150
    assert len({q.query_id for q in queries}) == 150
    for q in queries:
        doc = curated[q.query_id]  # KeyError here = a query that is not a curated document
        assert (q.question, q.focus, q.question_type, q.subset) == (doc.question, doc.focus, doc.question_type, doc.subset)


def test_query_set_on_disk_is_what_the_config_reproduces(queries, variant_docs):
    raw = yaml.safe_load((REPO_ROOT / "configs" / "query_set.yaml").read_text(encoding="utf-8"))["query_set"]
    assert raw == {"seed": 42, "num_queries": 150}
    assert sample_query_set(variant_docs["curated"], QuerySetConfig(**raw)) == queries


def test_query_set_is_stratified_by_question_type(queries, variant_docs):
    pool = Counter(d.question_type for d in variant_docs["curated"])
    sample = Counter(q.question_type for q in queries)
    for question_type, count in pool.items():
        assert abs(sample[question_type] - count / 1500 * 150) < 1.0, question_type


def test_every_query_has_relevant_chunks_in_curated_and_mixed(queries, judgments):
    for variant in ("curated", "mixed"):
        assert all(judgments[variant].strict[q.query_id] for q in queries)


def test_strict_relevant_chunks_are_exactly_the_chunks_of_the_query_own_answer(queries, variant_docs, variant_chunks, judgments):
    """Independent check of the label: stitch each query's relevant chunks back
    together and compare with the MedQuAD answer to that very question."""
    answers = {d.doc_id: d.answer for d in variant_docs["curated"]}
    text_of = {c.chunk_id: c for c in variant_chunks["curated"]}
    for q in queries:
        ordered = sorted((text_of[cid] for cid in judgments["curated"].strict[q.query_id]), key=lambda c: c.chunk_index)
        words: list[str] = []
        for i, chunk in enumerate(ordered):
            words.extend(chunk.text.split(" ") if i == 0 else chunk.text.split(" ")[20:])
            assert chunk.source_question == q.question
        assert " ".join(words) == answers[q.query_id], q.query_id


def test_reduced_really_removes_the_answers_to_some_queries_but_not_all(queries, variant_docs, judgments):
    reduced_ids = {d.doc_id for d in variant_docs["reduced"]}
    unanswerable = [q for q in queries if not judgments["reduced"].strict[q.query_id]]
    assert {q.query_id for q in unanswerable} == {q.query_id for q in queries if q.query_id not in reduced_ids}
    # "limited coverage" is only being tested if a real share of queries lose their answer, and a real share keep it
    assert 0.15 <= len(unanswerable) / len(queries) <= 0.55, len(unanswerable)
    # a query's answer survives in `reduced` exactly as it was in `curated`
    for q in queries:
        if q.query_id in reduced_ids:
            assert judgments["reduced"].strict[q.query_id] == judgments["curated"].strict[q.query_id]


def test_lenient_labels_follow_their_definition_and_contain_strict(queries, variant_chunks, judgments):
    for variant in VARIANTS:
        chunks = {c.chunk_id: c for c in variant_chunks[variant]}
        own_texts = {q.query_id: {c.text for c in variant_chunks["curated"] if c.doc_id == q.query_id} for q in queries}
        for q in queries:
            strict, lenient = judgments[variant].strict[q.query_id], judgments[variant].lenient[q.query_id]
            assert strict <= lenient
            for chunk_id in lenient - strict:
                chunk = chunks[chunk_id]
                same_text = chunk.text in own_texts[q.query_id]
                same_question = normalize_topic(chunk.focus) == normalize_topic(q.focus) and chunk.question_type == q.question_type
                assert same_text or same_question, (q.query_id, chunk_id)


# -- leakage --------------------------------------------------------------------------------


def test_no_query_contains_its_reference_answer(queries, variant_docs):
    answers = {d.doc_id: d.answer for d in variant_docs["curated"]}
    for q in queries:
        assert answers[q.query_id] not in q.question
        assert len(q.question.split()) < 40  # a question, not a pasted passage


def test_the_question_is_not_part_of_the_indexed_chunk_text(queries, variant_chunks, real_retrievers):
    """Chunks carry their source question as metadata. If it were indexed, every
    retriever would find the gold chunk by matching the query against itself."""
    lexical, _ = real_retrievers["curated"]
    chunks = variant_chunks["curated"]
    assert lexical.bm25.corpus_size == len(chunks)
    for i in range(0, len(chunks), 37):
        assert lexical.bm25.doc_len[i] == len(tokenize(chunks[i].text))  # text only, no question appended

    # A few MedQuAD answers repeat their own question; that is a property of the
    # source data, bounded here so it cannot quietly grow.
    own_chunks = {q.query_id: [c for c in chunks if c.doc_id == q.query_id] for q in queries}
    restated = [q.query_id for q in queries if any(q.question.rstrip(" ?").lower() in c.text.lower() for c in own_chunks[q.query_id])]
    assert len(restated) <= 0.05 * len(queries), restated

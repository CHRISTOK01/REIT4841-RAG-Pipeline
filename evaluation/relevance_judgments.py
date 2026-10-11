"""Derive retrieval relevance judgments from MedQuAD's native question -> source-document link.

Two label sets are built for every query, and every retrieval metric is
reported under both:

- **strict**: a chunk is relevant iff it was cut from the query's own QA pair's
  answer text (query.query_id == chunk.doc_id, see data/query_set.py).
- **lenient**: strict, plus any chunk that
    (a) has text identical to one of the query's own answer chunks (MedQuAD
        repeats whole answers and boilerplate passages across documents), or
    (b) comes from a document with the same topic (focus) AND the same
        question type as the query -- i.e. another source answering the same
        question about the same condition.

Strict under-counts (a retriever returning an identical twin of the answer is
marked wrong); lenient over-counts a little. The truth lies between them, so a
finding that only holds under one of the two is not robust.

Both are document-level, not manually verified passage-level, relevance. A
query's relevant set can legitimately be empty for a given KB variant (its
topic was dropped in `reduced`); see evaluation/retrieval_metrics.py for how
that is scored.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from data.kb_variants import normalize_topic
from data.query_set import QueryItem
from data.schemas import Chunk


@dataclass(frozen=True)
class RelevanceJudgments:
    strict: dict[str, set[str]]  # query_id -> relevant chunk_ids
    lenient: dict[str, set[str]]


def build_doc_to_chunk_ids(chunks: list[Chunk]) -> dict[str, set[str]]:
    doc_to_chunk_ids: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        doc_to_chunk_ids[chunk.doc_id].add(chunk.chunk_id)
    return doc_to_chunk_ids


def build_strict_judgments(queries: list[QueryItem], chunks: list[Chunk]) -> dict[str, set[str]]:
    """{query_id: chunk_ids cut from the query's own source document}. Empty
    if that document isn't present in this KB variant."""
    doc_to_chunk_ids = build_doc_to_chunk_ids(chunks)
    return {q.query_id: set(doc_to_chunk_ids.get(q.query_id, set())) for q in queries}


def build_lenient_judgments(
    queries: list[QueryItem], chunks: list[Chunk], reference_chunks: list[Chunk]
) -> dict[str, set[str]]:
    """`reference_chunks` must contain every query's own source chunks (the
    curated KB does); it supplies the answer text to match against even when
    the KB variant being judged (`chunks`) has dropped the query's document."""
    strict = build_strict_judgments(queries, chunks)

    own_texts: dict[str, set[str]] = defaultdict(set)
    for chunk in reference_chunks:
        own_texts[chunk.doc_id].add(chunk.text)

    by_text: dict[str, set[str]] = defaultdict(set)
    by_topic_and_type: dict[tuple[str, str], set[str]] = defaultdict(set)
    for chunk in chunks:
        by_text[chunk.text].add(chunk.chunk_id)
        by_topic_and_type[(normalize_topic(chunk.focus), chunk.question_type)].add(chunk.chunk_id)

    lenient: dict[str, set[str]] = {}
    for q in queries:
        relevant = set(strict[q.query_id])
        for text in own_texts.get(q.query_id, set()):
            relevant |= by_text.get(text, set())
        relevant |= by_topic_and_type.get((normalize_topic(q.focus), q.question_type), set())
        lenient[q.query_id] = relevant
    return lenient


def build_relevance_judgments(
    queries: list[QueryItem], chunks: list[Chunk], reference_chunks: list[Chunk]
) -> RelevanceJudgments:
    return RelevanceJudgments(
        strict=build_strict_judgments(queries, chunks),
        lenient=build_lenient_judgments(queries, chunks, reference_chunks),
    )

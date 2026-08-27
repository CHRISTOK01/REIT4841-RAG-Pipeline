"""Derive retrieval relevance judgments from MedQuAD's native question -> source-document link.

A chunk is relevant to a query if it was cut from that query's own QA pair's
answer text (query.query_id == chunk.doc_id, see data/query_set.py). This is
document-level relevance, not manually verified passage-level relevance -- a
simplifying assumption worth stating explicitly in the methodology chapter.

A query's relevant set can legitimately be empty for a given KB variant (e.g.
its source document's whole topic was dropped in `reduced`) -- callers must
treat that as "undefined ground truth for this query in this KB", not as
"zero relevant chunks retrieved". See evaluation/retrieval_metrics.py, which
returns NaN rather than 0 for metrics that are undefined when there is no
ground truth to measure against.
"""

from __future__ import annotations

from collections import defaultdict

from data.query_set import QueryItem
from data.schemas import Chunk


def build_doc_to_chunk_ids(chunks: list[Chunk]) -> dict[str, set[str]]:
    doc_to_chunk_ids: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        doc_to_chunk_ids[chunk.doc_id].add(chunk.chunk_id)
    return doc_to_chunk_ids


def relevant_chunk_ids_for_query(query: QueryItem, doc_to_chunk_ids: dict[str, set[str]]) -> set[str]:
    return doc_to_chunk_ids.get(query.query_id, set())


def build_relevance_judgments(
    queries: list[QueryItem], chunks: list[Chunk]
) -> dict[str, set[str]]:
    """Returns {query_id: {relevant chunk_id, ...}} for every query, using the
    given KB variant's chunk list. The set is empty if the query's own source
    document isn't present in this KB variant at all."""
    doc_to_chunk_ids = build_doc_to_chunk_ids(chunks)
    return {query.query_id: relevant_chunk_ids_for_query(query, doc_to_chunk_ids) for query in queries}

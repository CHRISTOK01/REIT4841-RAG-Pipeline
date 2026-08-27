from data.query_set import QueryItem
from data.schemas import Chunk
from evaluation.relevance_judgments import build_doc_to_chunk_ids, build_relevance_judgments


def make_chunk(chunk_id: str, doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        text="text",
        subset="Test",
        focus="Focus",
        question_type="information",
        source_question="q",
    )


def make_query(query_id: str) -> QueryItem:
    return QueryItem(query_id=query_id, question="q", subset="Test", focus="Focus", question_type="information")


def test_build_doc_to_chunk_ids_groups_by_doc():
    chunks = [make_chunk("d1#0", "d1"), make_chunk("d1#1", "d1"), make_chunk("d2#0", "d2")]
    mapping = build_doc_to_chunk_ids(chunks)
    assert mapping["d1"] == {"d1#0", "d1#1"}
    assert mapping["d2"] == {"d2#0"}


def test_relevance_judgments_finds_all_chunks_of_source_doc():
    chunks = [make_chunk("d1#0", "d1"), make_chunk("d1#1", "d1"), make_chunk("d2#0", "d2")]
    queries = [make_query("d1")]
    judgments = build_relevance_judgments(queries, chunks)
    assert judgments["d1"] == {"d1#0", "d1#1"}


def test_relevance_judgments_empty_when_source_doc_absent_from_variant():
    chunks = [make_chunk("d2#0", "d2")]  # d1 (the query's own source doc) got dropped, e.g. `reduced`
    queries = [make_query("d1")]
    judgments = build_relevance_judgments(queries, chunks)
    assert judgments["d1"] == set()


def test_relevance_judgments_covers_multiple_queries():
    chunks = [make_chunk("d1#0", "d1"), make_chunk("d2#0", "d2")]
    queries = [make_query("d1"), make_query("d2"), make_query("d3")]
    judgments = build_relevance_judgments(queries, chunks)
    assert judgments == {"d1": {"d1#0"}, "d2": {"d2#0"}, "d3": set()}

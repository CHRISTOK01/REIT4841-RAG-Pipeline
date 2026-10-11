from data.query_set import QueryItem
from data.schemas import Chunk
from evaluation.relevance_judgments import (
    build_doc_to_chunk_ids,
    build_lenient_judgments,
    build_relevance_judgments,
    build_strict_judgments,
)


def make_chunk(chunk_id: str, text: str = "text", focus: str = "Focus", question_type: str = "information") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        chunk_index=int(chunk_id.split("#")[1]),
        text=text,
        subset="Test",
        focus=focus,
        question_type=question_type,
        source_question="q",
    )


def make_query(query_id: str, focus: str = "Focus", question_type: str = "information") -> QueryItem:
    return QueryItem(query_id=query_id, question="q", subset="Test", focus=focus, question_type=question_type)


def test_build_doc_to_chunk_ids_groups_by_doc():
    chunks = [make_chunk("d1#0"), make_chunk("d1#1"), make_chunk("d2#0")]
    assert build_doc_to_chunk_ids(chunks) == {"d1": {"d1#0", "d1#1"}, "d2": {"d2#0"}}


def test_strict_is_every_chunk_of_the_query_source_document_and_nothing_else():
    chunks = [make_chunk("d1#0", "a"), make_chunk("d1#1", "b"), make_chunk("d2#0", "c", focus="Other")]
    assert build_strict_judgments([make_query("d1")], chunks) == {"d1": {"d1#0", "d1#1"}}


def test_strict_is_empty_when_the_source_document_is_absent_from_the_variant():
    chunks = [make_chunk("d2#0", focus="Other")]  # d1 was dropped, as in `reduced`
    assert build_strict_judgments([make_query("d1")], chunks) == {"d1": set()}


def test_strict_covers_every_query_even_unknown_ones():
    chunks = [make_chunk("d1#0", focus="A"), make_chunk("d2#0", focus="B")]
    judgments = build_strict_judgments([make_query("d1"), make_query("d2"), make_query("d3")], chunks)
    assert judgments == {"d1": {"d1#0"}, "d2": {"d2#0"}, "d3": set()}


def test_lenient_adds_chunks_with_text_identical_to_the_query_own_answer():
    chunks = [
        make_chunk("d1#0", "boilerplate passage", focus="Asthma"),
        make_chunk("d1#1", "asthma specific text", focus="Asthma"),
        make_chunk("d2#0", "boilerplate passage", focus="Gout"),  # verbatim twin of d1#0
        make_chunk("d3#0", "something else", focus="Lupus"),
    ]
    lenient = build_lenient_judgments([make_query("d1", focus="Asthma")], chunks, reference_chunks=chunks)
    assert lenient == {"d1": {"d1#0", "d1#1", "d2#0"}}


def test_lenient_adds_chunks_on_the_same_topic_and_question_type_only():
    chunks = [
        make_chunk("d1#0", "a", focus="Asthma", question_type="treatment"),
        make_chunk("d2#0", "b", focus="asthma ", question_type="treatment"),  # same topic (case/space), same type
        make_chunk("d3#0", "c", focus="Asthma", question_type="symptoms"),  # same topic, other question
        make_chunk("d4#0", "d", focus="Gout", question_type="treatment"),  # other topic, same question
    ]
    query = make_query("d1", focus="Asthma", question_type="treatment")
    assert build_lenient_judgments([query], chunks, reference_chunks=chunks) == {"d1": {"d1#0", "d2#0"}}


def test_lenient_can_be_non_empty_when_strict_is_empty():
    """The query's own document was dropped from this variant, but a verbatim
    copy of its answer survives in another document."""
    curated = [make_chunk("d1#0", "shared answer", focus="Asthma"), make_chunk("d2#0", "shared answer", focus="Gout")]
    reduced = [curated[1]]
    judgments = build_relevance_judgments([make_query("d1", focus="Asthma")], reduced, reference_chunks=curated)
    assert judgments.strict == {"d1": set()}
    assert judgments.lenient == {"d1": {"d2#0"}}


def test_strict_is_always_a_subset_of_lenient():
    chunks = [make_chunk(f"d{i}#0", f"t{i % 3}", focus=f"F{i % 2}") for i in range(8)]
    queries = [make_query(f"d{i}", focus=f"F{i % 2}") for i in range(8)]
    judgments = build_relevance_judgments(queries, chunks, reference_chunks=chunks)
    for q in queries:
        assert judgments.strict[q.query_id] <= judgments.lenient[q.query_id]


def test_returned_sets_are_independent_copies():
    chunks = [make_chunk("d1#0")]
    judgments = build_relevance_judgments([make_query("d1")], chunks, reference_chunks=chunks)
    judgments.strict["d1"].add("mutated")
    assert build_strict_judgments([make_query("d1")], chunks) == {"d1": {"d1#0"}}

import pytest

from data.chunking import chunk_document, chunk_documents
from data.schemas import ChunkingConfig, SourceDocument


def make_doc(answer: str, doc_id: str = "TestSubset:doc1:q1") -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        subset="TestSubset",
        document_id="doc1",
        focus="Test Focus",
        question_id="q1",
        question_type="information",
        question="What is X?",
        answer=answer,
        url="https://example.com",
        umls_cui="C000000",
    )


def test_short_answer_produces_single_chunk():
    doc = make_doc("one two three four five")
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_words=120, chunk_overlap_words=20))
    assert len(chunks) == 1
    assert chunks[0].text == "one two three four five"
    assert chunks[0].chunk_id == "TestSubset:doc1:q1#0"


def test_long_answer_splits_with_overlap():
    words = [f"w{i}" for i in range(25)]
    doc = make_doc(" ".join(words))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_words=10, chunk_overlap_words=3))
    # step = 7 words/chunk; starts at 0, 7, 14, 21 -> 4 chunks
    assert len(chunks) == 4
    assert chunks[0].text == " ".join(words[0:10])
    assert chunks[1].text == " ".join(words[7:17])
    # last chunk must reach the end of the document exactly once, not run past it
    assert chunks[-1].text == " ".join(words[21:25])
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"TestSubset:doc1:q1#{i}"


def test_empty_answer_produces_no_chunks():
    doc = make_doc("")
    assert chunk_document(doc, ChunkingConfig()) == []


def test_chunk_metadata_propagates_from_document():
    doc = make_doc("some medical answer text here")
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_words=50, chunk_overlap_words=5))
    assert chunks[0].subset == doc.subset
    assert chunks[0].focus == doc.focus
    assert chunks[0].question_type == doc.question_type
    assert chunks[0].source_question == doc.question
    assert chunks[0].doc_id == doc.doc_id


def test_chunk_documents_flattens_across_multiple_docs():
    docs = [make_doc("a b c", doc_id="S:d1:q1"), make_doc("d e f", doc_id="S:d2:q2")]
    chunks = chunk_documents(docs, ChunkingConfig(chunk_size_words=10, chunk_overlap_words=2))
    assert len(chunks) == 2
    assert {c.doc_id for c in chunks} == {"S:d1:q1", "S:d2:q2"}


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (10, 10), (10, 15), (-5, 0)],
)
def test_invalid_chunking_config_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size_words=chunk_size, chunk_overlap_words=overlap)

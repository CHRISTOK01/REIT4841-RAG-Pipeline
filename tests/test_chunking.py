import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from data.chunking import chunk_document, chunk_documents, load_chunks_jsonl, save_chunks_jsonl
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


def reconstruct(chunks, overlap: int) -> list[str]:
    """Undo the chunking: the first chunk whole, then each later chunk minus
    the `overlap` words it repeats from the one before."""
    words: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_words = chunk.text.split(" ")
        words.extend(chunk_words if i == 0 else chunk_words[overlap:])
    return words


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
    # step = 7 words/chunk; windows start at 0, 7, 14 and 21 -> 4 chunks
    assert [c.text for c in chunks] == [
        " ".join(words[0:10]),
        " ".join(words[7:17]),
        " ".join(words[14:24]),
        " ".join(words[21:25]),  # last chunk stops at the end of the document
    ]
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"TestSubset:doc1:q1#{i}"


@pytest.mark.parametrize(
    "n_words,expected_chunks",
    [(1, 1), (9, 1), (10, 1), (11, 2), (17, 2), (18, 3), (24, 3), (25, 4)],
)
def test_chunk_count_at_window_boundaries(n_words, expected_chunks):
    """size 10, overlap 3 (step 7): one chunk up to 10 words, then one more per
    extra 7 words (or part of 7). Worked by hand: 11..17 -> 2, 18..24 -> 3, 25 -> 4."""
    doc = make_doc(" ".join(f"w{i}" for i in range(n_words)))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_words=10, chunk_overlap_words=3))
    assert len(chunks) == expected_chunks


def test_document_exactly_one_window_long_does_not_get_a_trailing_overlap_only_chunk():
    doc = make_doc(" ".join(f"w{i}" for i in range(10)))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_words=10, chunk_overlap_words=3))
    assert len(chunks) == 1


def test_empty_and_whitespace_only_answers_produce_no_chunks():
    assert chunk_document(make_doc(""), ChunkingConfig()) == []
    assert chunk_document(make_doc("   \n\t  "), ChunkingConfig()) == []


def test_zero_overlap_partitions_the_document():
    words = [f"w{i}" for i in range(23)]
    chunks = chunk_document(make_doc(" ".join(words)), ChunkingConfig(chunk_size_words=10, chunk_overlap_words=0))
    assert [len(c.text.split()) for c in chunks] == [10, 10, 3]
    assert reconstruct(chunks, 0) == words


def test_unicode_text_is_preserved_exactly():
    answer = "Ménière’s disease — naïve café ≥ 5 µg/dL β-blockers 治疗 Sjögren’s"
    chunks = chunk_document(make_doc(answer), ChunkingConfig(chunk_size_words=4, chunk_overlap_words=1))
    assert reconstruct(chunks, 1) == answer.split()
    assert "治疗" in chunks[-1].text or any("治疗" in c.text for c in chunks)


def test_very_long_document():
    words = [f"w{i}" for i in range(5000)]
    config = ChunkingConfig(chunk_size_words=120, chunk_overlap_words=20)
    chunks = chunk_document(make_doc(" ".join(words)), config)
    # windows start every 100 words: 0, 100, ..., 4900 -> 50 chunks; the last holds words 4900..4999
    assert len(chunks) == 50
    assert chunks[-1].text.split() == words[4900:5000]
    assert reconstruct(chunks, 20) == words


word = st.text(alphabet=st.characters(blacklist_categories=("Z", "C")), min_size=1, max_size=8)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(word, max_size=400),
    st.integers(min_value=1, max_value=40).flatmap(
        lambda size: st.tuples(st.just(size), st.integers(min_value=0, max_value=size - 1))
    ),
)
def test_chunking_invariants_hold_for_any_document_and_config(words, size_and_overlap):
    size, overlap = size_and_overlap
    doc = make_doc(" ".join(words))
    words = doc.answer.split()  # some unicode "letters" are whitespace to str.split
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_words=size, chunk_overlap_words=overlap))

    assert reconstruct(chunks, overlap) == words or (not words and chunks == [])
    assert all(c.text.strip() for c in chunks)  # no empty chunks
    assert all(1 <= len(c.text.split(" ")) <= size for c in chunks)
    assert all(len(c.text.split(" ")) == size for c in chunks[:-1])  # only the last may be short
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for previous, current in zip(chunks, chunks[1:]):
        # the overlap is exactly `overlap` words, and it is the same words
        assert previous.text.split(" ")[size - overlap :] == current.text.split(" ")[:overlap]
    if len(chunks) > 1:
        assert len(chunks[-1].text.split(" ")) > overlap  # last chunk adds at least one new word


def test_chunk_ids_are_stable_across_runs():
    doc = make_doc(" ".join(f"w{i}" for i in range(300)))
    config = ChunkingConfig(chunk_size_words=50, chunk_overlap_words=10)
    assert chunk_document(doc, config) == chunk_document(doc, config)


def test_chunk_metadata_propagates_from_document():
    doc = make_doc("some medical answer text here")
    chunk = chunk_document(doc, ChunkingConfig(chunk_size_words=50, chunk_overlap_words=5))[0]
    assert (chunk.subset, chunk.focus, chunk.question_type, chunk.source_question, chunk.doc_id) == (
        doc.subset,
        doc.focus,
        doc.question_type,
        doc.question,
        doc.doc_id,
    )


def test_chunk_documents_flattens_across_multiple_docs_in_order():
    docs = [make_doc("a b c", doc_id="S:d1:q1"), make_doc("d e f", doc_id="S:d2:q2")]
    chunks = chunk_documents(docs, ChunkingConfig(chunk_size_words=10, chunk_overlap_words=2))
    assert [(c.doc_id, c.text) for c in chunks] == [("S:d1:q1", "a b c"), ("S:d2:q2", "d e f")]


def test_chunks_jsonl_roundtrip_is_exact_and_uses_unix_newlines(tmp_path):
    chunks = chunk_document(make_doc("Ménière’s disease naïve café 治疗 " * 20), ChunkingConfig(chunk_size_words=7, chunk_overlap_words=2))
    path = tmp_path / "chunks.jsonl"
    save_chunks_jsonl(chunks, path)
    assert load_chunks_jsonl(path) == chunks
    assert b"\r\n" not in path.read_bytes()  # byte-identical on Windows and Linux, so file hashes compare


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (10, 10), (10, 15), (-5, 0), (10, -1)])
def test_invalid_chunking_config_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size_words=chunk_size, chunk_overlap_words=overlap)

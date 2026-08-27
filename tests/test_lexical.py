from data.schemas import Chunk
from retrieval.lexical import LexicalRetriever, tokenize


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        chunk_index=0,
        text=text,
        subset="Test",
        focus="Test Focus",
        question_type="information",
        source_question="irrelevant",
    )


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Diabetes Type-2 Symptoms!") == ["diabetes", "type", "2", "symptoms"]


def test_retrieve_ranks_relevant_chunk_first():
    chunks = [
        make_chunk("d1#0", "The symptoms of diabetes include increased thirst and fatigue."),
        make_chunk("d2#0", "Treatment for asthma often includes inhaled corticosteroids."),
        make_chunk("d3#0", "Regular exercise improves cardiovascular health."),
    ]
    retriever = LexicalRetriever.build(chunks)
    results = retriever.retrieve("what are the symptoms of diabetes", k=3)

    assert results[0].chunk_id == "d1#0"
    assert results[0].rank == 1
    # scores should be non-increasing
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_k():
    chunks = [make_chunk(f"d{i}#0", f"document number {i} about health topic {i}") for i in range(10)]
    retriever = LexicalRetriever.build(chunks)
    results = retriever.retrieve("health topic", k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]


def test_retrieve_on_empty_corpus_returns_empty():
    retriever = LexicalRetriever.build([])
    assert retriever.retrieve("anything", k=5) == []

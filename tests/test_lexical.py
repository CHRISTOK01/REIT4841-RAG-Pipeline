"""BM25 checked against scores worked out by hand from the Okapi BM25 formula
(rank_bm25's defaults: k1 = 1.5, b = 0.75):

    idf(t)   = ln((N - n_t + 0.5) / (n_t + 0.5))
    score(d) = sum over query terms of
               idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
"""

import pytest

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


# Five chunks of exactly 4 tokens each, so len(d) / avg_len = 1 and the length
# term is k1 * (1 - b + b) = 1.5 for every chunk.
EQUAL_LENGTH = [
    make_chunk("d0#0", "aspirin aspirin relieves pain"),
    make_chunk("d1#0", "aspirin reduces fever quickly"),
    make_chunk("d2#0", "insulin lowers blood sugar"),
    make_chunk("d3#0", "vaccines prevent viral infection"),
    make_chunk("d4#0", "exercise strengthens heart muscle"),
]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Diabetes Type-2 Symptoms!") == ["diabetes", "type", "2", "symptoms"]


def test_term_frequency_ranking_matches_hand_computed_scores():
    results = LexicalRetriever.build(EQUAL_LENGTH).retrieve("aspirin", k=5)
    # "aspirin" is in 2 of 5 chunks: idf = ln(3.5 / 2.5) = 0.336472
    # d0: tf = 2 -> 2 * 2.5 / (2 + 1.5) = 1.428571 -> 0.336472 * 1.428571 = 0.480675
    # d1: tf = 1 -> 1 * 2.5 / (1 + 1.5) = 1        -> 0.336472
    assert [r.chunk_id for r in results] == ["d0#0", "d1#0"]
    assert results[0].score == pytest.approx(0.480675, abs=1e-6)
    assert results[1].score == pytest.approx(0.336472, abs=1e-6)
    assert [r.rank for r in results] == [1, 2]


def test_two_term_query_adds_the_per_term_scores():
    results = LexicalRetriever.build(EQUAL_LENGTH).retrieve("insulin sugar", k=5)
    # each term is in 1 of 5 chunks: idf = ln(4.5 / 1.5) = 1.098612; tf = 1 -> factor 1
    # d2 contains both: 1.098612 + 1.098612 = 2.197225
    assert [r.chunk_id for r in results] == ["d2#0"]
    assert results[0].score == pytest.approx(2.197225, abs=1e-6)


def test_length_normalisation_matches_hand_computed_scores():
    chunks = [
        make_chunk("e0#0", "migraine"),  # 1 token
        make_chunk("e1#0", "migraine causes severe throbbing headache pain today"),  # 7 tokens
        make_chunk("e2#0", "sleep helps recovery"),  # 3
        make_chunk("e3#0", "water"),  # 1
        make_chunk("e4#0", "rest often helps"),  # 3
        make_chunk("e5#0", "drink more fluids"),  # 3
    ]  # 18 tokens over 6 chunks: avg_len = 3
    results = LexicalRetriever.build(chunks).retrieve("migraine", k=6)
    # "migraine" is in 2 of 6 chunks: idf = ln(4.5 / 2.5) = 0.587787; tf = 1 in both
    # e0: 2.5 / (1 + 1.5 * (0.25 + 0.75 * 1/3)) = 2.5 / 1.75 = 1.428571 -> 0.839695
    # e1: 2.5 / (1 + 1.5 * (0.25 + 0.75 * 7/3)) = 2.5 / 4.0  = 0.625    -> 0.367367
    assert [r.chunk_id for r in results] == ["e0#0", "e1#0"]
    assert results[0].score == pytest.approx(0.839695, abs=1e-6)
    assert results[1].score == pytest.approx(0.367367, abs=1e-6)


def test_chunks_sharing_no_term_with_the_query_are_never_returned():
    retriever = LexicalRetriever.build(EQUAL_LENGTH)
    assert retriever.retrieve("zzqxv", k=5) == []  # no chunk contains this token
    assert len(retriever.retrieve("aspirin", k=5)) == 2  # 2 matches, not padded to 5


def test_k_truncates_the_ranked_list():
    results = LexicalRetriever.build(EQUAL_LENGTH).retrieve("aspirin", k=1)
    assert [r.chunk_id for r in results] == ["d0#0"]


def test_tied_scores_keep_corpus_order():
    chunks = [make_chunk(f"t{i}#0", "fever rash") for i in range(4)] + [make_chunk("other#0", "unrelated words here")] * 1
    chunks += [make_chunk(f"pad{i}#0", f"padding token{i}") for i in range(6)]
    results = LexicalRetriever.build(chunks).retrieve("fever", k=3)
    assert len({r.score for r in results}) == 1  # genuinely tied
    assert [r.chunk_id for r in results] == ["t0#0", "t1#0", "t2#0"]


def test_retrieve_on_empty_corpus_returns_empty():
    assert LexicalRetriever.build([]).retrieve("anything", k=5) == []

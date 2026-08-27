import random

from data.schemas import Chunk
from evaluation.robustness import drop_top_relevant, inject_noise_chunk, select_noise_chunk
from retrieval.base import RetrievedChunk


def rc(chunk_id: str, rank: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=f"text-{chunk_id}", score=score, rank=rank)


def make_chunk(chunk_id: str, focus: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        chunk_index=0,
        text=f"text-{chunk_id}",
        subset="Test",
        focus=focus,
        question_type="information",
        source_question="q",
    )


RETRIEVED = [rc("a", 1), rc("b", 2), rc("c", 3)]


def test_drop_top_relevant_removes_first_relevant_hit_and_renumbers():
    result = drop_top_relevant(RETRIEVED, relevant_chunk_ids={"b"})
    assert [c.chunk_id for c in result] == ["a", "c"]
    assert [c.rank for c in result] == [1, 2]


def test_drop_top_relevant_removes_earliest_when_multiple_relevant():
    result = drop_top_relevant(RETRIEVED, relevant_chunk_ids={"b", "c"})
    assert [c.chunk_id for c in result] == ["a", "c"]  # b (rank 2) dropped, not c (rank 3)


def test_drop_top_relevant_no_op_when_nothing_relevant_retrieved():
    result = drop_top_relevant(RETRIEVED, relevant_chunk_ids={"z"})
    assert [c.chunk_id for c in result] == ["a", "b", "c"]


def test_inject_noise_chunk_at_top_pushes_others_down():
    noise = make_chunk("noise1", focus="Unrelated")
    result = inject_noise_chunk(RETRIEVED, noise, position=1)
    assert [c.chunk_id for c in result] == ["noise1", "a", "b", "c"]
    assert [c.rank for c in result] == [1, 2, 3, 4]


def test_inject_noise_chunk_at_arbitrary_position():
    noise = make_chunk("noise1", focus="Unrelated")
    result = inject_noise_chunk(RETRIEVED, noise, position=2)
    assert [c.chunk_id for c in result] == ["a", "noise1", "b", "c"]


def test_inject_noise_chunk_position_clamped_to_valid_range():
    noise = make_chunk("noise1", focus="Unrelated")
    result = inject_noise_chunk(RETRIEVED, noise, position=100)
    assert [c.chunk_id for c in result] == ["a", "b", "c", "noise1"]


def test_inject_noise_chunk_into_empty_list():
    noise = make_chunk("noise1", focus="Unrelated")
    result = inject_noise_chunk([], noise, position=1)
    assert [c.chunk_id for c in result] == ["noise1"]
    assert result[0].score == 0.0


def test_select_noise_chunk_excludes_query_focus():
    chunks = [make_chunk("a", "TopicA"), make_chunk("b", "TopicA"), make_chunk("c", "TopicB")]
    rng = random.Random(1)
    for _ in range(10):
        chosen = select_noise_chunk(chunks, exclude_focus="TopicA", rng=rng)
        assert chosen.focus != "TopicA"


def test_select_noise_chunk_deterministic_with_seed():
    chunks = [make_chunk(f"c{i}", f"Topic{i}") for i in range(20)]
    a = select_noise_chunk(chunks, exclude_focus="TopicNone", rng=random.Random(42))
    b = select_noise_chunk(chunks, exclude_focus="TopicNone", rng=random.Random(42))
    assert a.chunk_id == b.chunk_id


def test_select_noise_chunk_returns_none_when_no_candidates():
    chunks = [make_chunk("a", "OnlyTopic")]
    assert select_noise_chunk(chunks, exclude_focus="OnlyTopic", rng=random.Random(1)) is None

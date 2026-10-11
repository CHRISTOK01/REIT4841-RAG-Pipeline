"""The perturbation functions do exactly what they claim, and nothing else."""

import random

import pytest

from data.schemas import Chunk
from evaluation.robustness import (
    PERTURBATIONS,
    drop_top_relevant,
    replace_lowest_nonrelevant_with_noise,
    select_noise_chunk,
)
from retrieval.base import RetrievedChunk


def rc(chunk_id: str, rank: int, score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=chunk_id, text=f"text-{chunk_id}", score=score if score is not None else 1.0 / rank, rank=rank
    )


def make_chunk(chunk_id: str, focus: str, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        chunk_index=0,
        text=text or f"text-{chunk_id}",
        subset="Test",
        focus=focus,
        question_type="information",
        source_question="q",
    )


RETRIEVED = [rc("a", 1), rc("b", 2), rc("c", 3), rc("d", 4), rc("e", 5)]
NOISE = make_chunk("noise1", focus="Unrelated")


def ids(chunks):
    return [c.chunk_id for c in chunks]


def test_the_three_conditions():
    assert PERTURBATIONS == ("drop_top_relevant", "inject_noise", "control_rerun")


# -- drop_top_relevant ------------------------------------------------------------------


def test_drop_removes_exactly_the_highest_ranked_relevant_chunk():
    result, dropped = drop_top_relevant(RETRIEVED, relevant_chunk_ids={"b", "d"})
    assert dropped == "b"  # rank 2 beats rank 4
    assert ids(result) == ["a", "c", "d", "e"]  # everything else, in the same order
    assert [c.rank for c in result] == [1, 2, 3, 4]


def test_drop_leaves_every_other_chunk_untouched():
    result, _ = drop_top_relevant(RETRIEVED, relevant_chunk_ids={"c"})
    original = {c.chunk_id: c for c in RETRIEVED}
    for chunk in result:
        assert (chunk.text, chunk.score, chunk.doc_id) == (
            original[chunk.chunk_id].text,
            original[chunk.chunk_id].score,
            original[chunk.chunk_id].doc_id,
        )


def test_drop_when_the_relevant_chunk_is_first_or_last():
    assert ids(drop_top_relevant(RETRIEVED, {"a"})[0]) == ["b", "c", "d", "e"]
    assert ids(drop_top_relevant(RETRIEVED, {"e"})[0]) == ["a", "b", "c", "d"]


def test_drop_reports_when_there_was_nothing_to_drop():
    result, dropped = drop_top_relevant(RETRIEVED, relevant_chunk_ids={"z"})
    assert dropped is None
    assert result == RETRIEVED and result is not RETRIEVED
    assert drop_top_relevant([], {"a"}) == ([], None)


def test_drop_does_not_modify_its_input():
    before = list(RETRIEVED)
    drop_top_relevant(RETRIEVED, {"a"})
    assert RETRIEVED == before


# -- inject noise --------------------------------------------------------------------------


def test_noise_replaces_the_lowest_ranked_non_relevant_chunk_in_place():
    result, replaced = replace_lowest_nonrelevant_with_noise(RETRIEVED, NOISE, protected_chunk_ids={"b", "e"})
    assert replaced == "d"  # e is protected, so d is the lowest-ranked non-relevant chunk
    assert ids(result) == ["a", "b", "c", "noise1", "e"]
    assert len(result) == len(RETRIEVED)  # context size unchanged
    assert [c.rank for c in result] == [1, 2, 3, 4, 5]


def test_noise_injection_keeps_every_relevant_chunk_and_touches_one_slot_only():
    protected = {"b", "e"}
    result, replaced = replace_lowest_nonrelevant_with_noise(RETRIEVED, NOISE, protected)
    assert protected <= set(ids(result))
    changed = [i for i, (old, new) in enumerate(zip(RETRIEVED, result)) if old != new]
    assert changed == [3]  # only the replaced slot differs
    assert result[3].text == NOISE.text and result[3].doc_id == NOISE.doc_id
    assert set(ids(RETRIEVED)) - set(ids(result)) == {replaced}
    assert set(ids(result)) - set(ids(RETRIEVED)) == {"noise1"}


def test_noise_injection_when_only_the_top_chunk_is_non_relevant():
    result, replaced = replace_lowest_nonrelevant_with_noise(RETRIEVED, NOISE, protected_chunk_ids={"b", "c", "d", "e"})
    assert replaced == "a"
    assert ids(result) == ["noise1", "b", "c", "d", "e"]


def test_noise_injection_is_refused_when_everything_is_relevant():
    result, replaced = replace_lowest_nonrelevant_with_noise(RETRIEVED, NOISE, protected_chunk_ids=set("abcde"))
    assert replaced is None
    assert result == RETRIEVED


def test_noise_injection_rejects_a_noise_chunk_that_is_already_retrieved():
    with pytest.raises(ValueError, match="already in the retrieved list"):
        replace_lowest_nonrelevant_with_noise(RETRIEVED, make_chunk("c", "Unrelated"), protected_chunk_ids=set())


def test_noise_injection_does_not_modify_its_input():
    before = list(RETRIEVED)
    replace_lowest_nonrelevant_with_noise(RETRIEVED, NOISE, {"b"})
    assert RETRIEVED == before


# -- choosing the noise chunk -----------------------------------------------------------------

KB = [
    make_chunk("a", "TopicA"),
    make_chunk("b", "topica "),  # same topic, different case/spacing
    make_chunk("c", "TopicB"),
    make_chunk("d", "TopicC"),
    make_chunk("e", "TopicD", text="duplicate of a relevant passage"),
]


def test_noise_chunk_never_shares_the_query_topic():
    rng = random.Random(1)
    chosen = {select_noise_chunk(KB, exclude_focus="TopicA", rng=rng).chunk_id for _ in range(200)}
    assert chosen == {"c", "d", "e"}  # a and b (TopicA in any spelling) never; the rest all reachable


def test_noise_chunk_is_never_an_excluded_id_or_a_copy_of_an_excluded_text():
    rng = random.Random(1)
    chosen = {
        select_noise_chunk(
            KB, exclude_focus="TopicA", rng=rng, exclude_chunk_ids={"c"}, exclude_texts={"duplicate of a relevant passage"}
        ).chunk_id
        for _ in range(100)
    }
    assert chosen == {"d"}


def test_noise_choice_is_reproducible_from_the_seed():
    kb = [make_chunk(f"c{i}", f"Topic{i}") for i in range(50)]
    first = select_noise_chunk(kb, "none", random.Random("123|noise|curated|q1"))
    second = select_noise_chunk(kb, "none", random.Random("123|noise|curated|q1"))
    other = [select_noise_chunk(kb, "none", random.Random(f"123|noise|curated|q{i}")).chunk_id for i in range(20)]
    assert first == second
    assert len(set(other)) > 5  # different queries get different noise chunks


def test_no_candidate_returns_none():
    assert select_noise_chunk([make_chunk("a", "OnlyTopic")], exclude_focus="OnlyTopic", rng=random.Random(1)) is None
    assert select_noise_chunk([], exclude_focus="x", rng=random.Random(1)) is None

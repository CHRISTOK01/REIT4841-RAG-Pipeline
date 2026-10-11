"""Prompt templates. The snapshot tests pin the exact text: if a prompt is
edited, even by one character, they fail until the new hash is written here
on purpose (and the thesis methodology is updated to match)."""

import pytest

from generation.prompts import (
    CLOSED_BOOK_TEMPLATE_ID,
    PROMPT_TEMPLATE_ID,
    SYSTEM_PROMPT_V1,
    USER_PROMPT_TEMPLATE_V1,
    format_context_block,
    prompt_hash,
    render_prompt,
)
from retrieval.base import RetrievedChunk

# SHA-256 of "<system prompt>\x00<user template>" for each template.
WELLNESS_SUMMARY_V1_SHA256 = "d207ca59ae076d4a4f34fb56a9ad01e2a641297359059322fadacd8c375cdb4c"
CLOSED_BOOK_V1_SHA256 = "ec600688d61f7884a0e15ae7ad6ad551256146cbde186c875de1412281f17d38"


def rc(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=text, score=1.0, rank=1)


def test_generation_prompt_snapshot():
    assert PROMPT_TEMPLATE_ID == "wellness_summary_v1"
    assert prompt_hash(PROMPT_TEMPLATE_ID) == WELLNESS_SUMMARY_V1_SHA256


def test_closed_book_prompt_snapshot():
    assert prompt_hash(CLOSED_BOOK_TEMPLATE_ID) == CLOSED_BOOK_V1_SHA256


def test_prompt_hash_is_sensitive_to_a_one_character_edit(monkeypatch):
    import generation.prompts as prompts

    edited = (SYSTEM_PROMPT_V1.replace("ONLY", "only"), USER_PROMPT_TEMPLATE_V1)
    monkeypatch.setitem(prompts.PROMPT_TEMPLATES, PROMPT_TEMPLATE_ID, edited)
    assert prompt_hash(PROMPT_TEMPLATE_ID) != WELLNESS_SUMMARY_V1_SHA256


def test_generation_prompt_carries_the_rules_the_metrics_rely_on():
    """The parser needs these three headings and numbered, cited key points."""
    for required in ("## Summary", "## Key Points", "## Disclaimer", "ONLY on the numbered context passages", "say so explicitly"):
        assert required in SYSTEM_PROMPT_V1


def test_rendered_prompt_is_exactly_system_plus_question_plus_numbered_passages():
    chunks = [rc("d1#0", "Diabetes causes thirst."), rc("d2#0", "Asthma needs an inhaler.")]
    rendered = render_prompt("What causes thirst?", chunks)

    assert rendered.messages == [
        {"role": "system", "content": SYSTEM_PROMPT_V1},
        {
            "role": "user",
            "content": (
                "Question: What causes thirst?\n\nContext passages:\n"
                "[1] (chunk_id=d1#0)\nDiabetes causes thirst.\n\n"
                "[2] (chunk_id=d2#0)\nAsthma needs an inhaler.\n"
            ),
        },
    ]
    assert rendered.passage_map == {1: "d1#0", 2: "d2#0"}
    assert rendered.template_id == "wellness_summary_v1"


def test_passages_are_numbered_in_the_order_given():
    chunks = [rc("d3#0", "c"), rc("d1#0", "a"), rc("d2#0", "b")]
    assert render_prompt("q", chunks).passage_map == {1: "d3#0", 2: "d1#0", 3: "d2#0"}
    assert format_context_block(chunks).index("[1] (chunk_id=d3#0)") < format_context_block(chunks).index("[2] (chunk_id=d1#0)")


def test_render_prompt_with_no_chunks_says_so():
    rendered = render_prompt("question", [])
    assert rendered.passage_map == {}
    assert rendered.messages[1]["content"] == "Question: question\n\nContext passages:\n(no context passages retrieved)\n"


def test_closed_book_prompt_has_no_context_and_refuses_chunks():
    rendered = render_prompt("What is asthma?", [], CLOSED_BOOK_TEMPLATE_ID)
    assert rendered.messages[1]["content"] == "Question: What is asthma?\n"
    assert "No context passages are provided" in rendered.messages[0]["content"]
    assert rendered.template_id == "closed_book_v1"
    with pytest.raises(ValueError, match="takes no context"):
        render_prompt("q", [rc("d1#0", "text")], CLOSED_BOOK_TEMPLATE_ID)


def test_unknown_template_id_is_rejected():
    with pytest.raises(ValueError, match="Unknown prompt template"):
        render_prompt("q", [], "wellness_summary_v2")


def test_question_text_with_braces_is_not_interpreted_as_a_template_field():
    rendered = render_prompt("What about {context_block} and {question}?", [rc("d1#0", "text with {braces}")])
    assert "What about {context_block} and {question}?" in rendered.messages[1]["content"]
    assert "text with {braces}" in rendered.messages[1]["content"]

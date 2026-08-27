from generation.prompts import format_context_block, render_prompt
from retrieval.base import RetrievedChunk


def rc(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=text, score=1.0, rank=1)


def test_format_context_block_numbers_passages_and_shows_chunk_id():
    chunks = [rc("d1#0", "Diabetes causes thirst."), rc("d2#0", "Asthma needs an inhaler.")]
    block = format_context_block(chunks)
    assert "[1] (chunk_id=d1#0)\nDiabetes causes thirst." in block
    assert "[2] (chunk_id=d2#0)\nAsthma needs an inhaler." in block


def test_format_context_block_empty_list():
    assert format_context_block([]) == ""


def test_render_prompt_builds_system_and_user_messages():
    chunks = [rc("d1#0", "Diabetes causes thirst.")]
    rendered = render_prompt("What causes thirst?", chunks)

    assert len(rendered.messages) == 2
    assert rendered.messages[0]["role"] == "system"
    assert rendered.messages[1]["role"] == "user"
    assert "What causes thirst?" in rendered.messages[1]["content"]
    assert "Diabetes causes thirst." in rendered.messages[1]["content"]


def test_render_prompt_passage_map_matches_chunk_order():
    chunks = [rc("d1#0", "text a"), rc("d2#0", "text b"), rc("d3#0", "text c")]
    rendered = render_prompt("question", chunks)
    assert rendered.passage_map == {1: "d1#0", 2: "d2#0", 3: "d3#0"}


def test_render_prompt_with_no_chunks_uses_placeholder():
    rendered = render_prompt("question", [])
    assert rendered.passage_map == {}
    assert "no context passages retrieved" in rendered.messages[1]["content"]

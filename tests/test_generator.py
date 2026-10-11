import pytest

from generation.generator import MalformedGenerationOutput, generate_summary
from generation.prompts import CLOSED_BOOK_TEMPLATE_ID, SYSTEM_PROMPT_V1, prompt_hash
from retrieval.base import RetrievedChunk
from tests.fakes import GOOD_ANSWER, ScriptedClient, chat_result


def rc(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=text, score=1.0, rank=1)


def test_generate_summary_builds_a_complete_record():
    client = ScriptedClient([chat_result(GOOD_ANSWER, model="gpt-4o-mini-2024-07-18")])
    record = generate_summary(
        client, "What is X?", [rc("d1#0", "Some context text.")], model="gpt-4o-mini", temperature=0.1, max_tokens=500, seed=11
    )

    assert record.question == "What is X?"
    assert record.model == "gpt-4o-mini"  # what was asked for
    assert record.returned_model == "gpt-4o-mini-2024-07-18"  # what the API said it used
    assert record.system_fingerprint == "fp_fake"
    assert record.finish_reason == "stop"
    assert record.prompt_template_id == "wellness_summary_v1"
    assert record.prompt_hash == prompt_hash("wellness_summary_v1")
    assert (record.temperature, record.max_tokens, record.seed, record.attempts) == (0.1, 500, 11, 1)
    assert (record.prompt_tokens, record.completion_tokens) == (100, 20)
    assert record.retrieved_chunk_ids == ["d1#0"]
    assert record.raw_text == GOOD_ANSWER
    assert record.output.summary == "Short answer about the topic. It has two sentences."
    assert record.output.key_points[0].cited_chunk_ids == ["d1#0"]
    assert record.generated_at.endswith("+00:00")


def test_generate_summary_sends_the_fixed_system_prompt_and_the_given_parameters():
    client = ScriptedClient([GOOD_ANSWER])
    generate_summary(client, "question text", [rc("d1#0", "context")], model="m", temperature=0.3, max_tokens=200, seed=5)

    call = client.chat_calls[0]
    assert (call["model"], call["temperature"], call["max_tokens"], call["seed"], call["json_mode"]) == ("m", 0.3, 200, 5, False)
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert call["messages"][1]["content"].startswith("Question: question text\n\nContext passages:\n[1] (chunk_id=d1#0)\ncontext")


def test_closed_book_generation_uses_the_closed_book_template():
    client = ScriptedClient([GOOD_ANSWER])
    record = generate_summary(client, "q", [], model="m", template_id=CLOSED_BOOK_TEMPLATE_ID)
    assert record.prompt_template_id == "closed_book_v1"
    assert record.prompt_hash == prompt_hash(CLOSED_BOOK_TEMPLATE_ID)
    assert "Context passages" not in client.chat_calls[0]["messages"][1]["content"]


TRUNCATED = chat_result("## Summary\nAn answer that was cut", finish_reason="length")
NO_KEY_POINTS = "## Summary\nA summary.\n\n## Disclaimer\nd\n"
NO_SUMMARY = "## Key Points\n1. a point [1]\n\n## Disclaimer\nd\n"
FREE_TEXT = "I'm sorry, but I can't help with that."


@pytest.mark.parametrize(
    "bad_reply,reason",
    [
        (TRUNCATED, "finish_reason='length'"),
        (chat_result(GOOD_ANSWER, finish_reason="content_filter"), "finish_reason='content_filter'"),
        (NO_KEY_POINTS, "no key points"),
        (NO_SUMMARY, "no '## Summary'"),
        (FREE_TEXT, "no '## Summary'"),
    ],
)
def test_unusable_output_is_an_error_never_a_record(bad_reply, reason):
    client = ScriptedClient([bad_reply] * 3)
    with pytest.raises(MalformedGenerationOutput, match=reason):
        generate_summary(client, "q", [rc("d1#0", "context")], model="m", max_attempts=3)
    assert len(client.chat_calls) == 3  # it tried the allowed number of times, no more


def test_unusable_output_is_regenerated_with_a_new_seed_and_the_attempt_count_is_logged():
    client = ScriptedClient([FREE_TEXT, TRUNCATED, GOOD_ANSWER])
    record = generate_summary(client, "q", [rc("d1#0", "context")], model="m", seed=100, max_attempts=3)
    assert record.attempts == 3
    assert [c["seed"] for c in client.chat_calls] == [100, 101, 102]
    assert record.seed == 102  # the seed of the attempt that was kept
    assert record.raw_text == GOOD_ANSWER


def test_api_errors_propagate_untouched():
    client = ScriptedClient([TimeoutError("simulated timeout")])
    with pytest.raises(TimeoutError):
        generate_summary(client, "q", [], model="m")

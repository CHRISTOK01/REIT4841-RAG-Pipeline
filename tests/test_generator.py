from generation.generator import generate_summary
from retrieval.base import RetrievedChunk

RAW_RESPONSE = """## Summary
Short answer.

## Key Points
1. A grounded claim. [1]

## Disclaimer
Not medical advice.
"""


class FakeChatClient:
    def __init__(self, response_text: str = RAW_RESPONSE):
        self.response_text = response_text
        self.calls: list[dict] = []

    def chat_completion(self, messages, model, temperature, max_tokens):
        self.calls.append(
            {"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.response_text


def rc(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=text, score=1.0, rank=1)


def test_generate_summary_builds_a_complete_record():
    client = FakeChatClient()
    chunks = [rc("d1#0", "Some context text.")]

    record = generate_summary(client, "What is X?", chunks, model="gpt-4o-mini", temperature=0.1, max_tokens=500)

    assert record.question == "What is X?"
    assert record.model == "gpt-4o-mini"
    assert record.prompt_template_id == "wellness_summary_v1"
    assert record.retrieved_chunk_ids == ["d1#0"]
    assert record.raw_text == RAW_RESPONSE
    assert record.output.summary == "Short answer."
    assert record.output.key_points[0].cited_chunk_ids == ["d1#0"]
    assert record.generated_at  # non-empty ISO timestamp


def test_generate_summary_calls_client_with_rendered_prompt_and_params():
    client = FakeChatClient()
    chunks = [rc("d1#0", "context")]

    generate_summary(client, "question text", chunks, model="gpt-4o-mini", temperature=0.3, max_tokens=200)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 200
    assert call["messages"][1]["content"].startswith("Question: question text")


def test_generate_summary_with_no_retrieved_chunks():
    client = FakeChatClient()
    record = generate_summary(client, "question", [], model="gpt-4o-mini")
    assert record.retrieved_chunk_ids == []

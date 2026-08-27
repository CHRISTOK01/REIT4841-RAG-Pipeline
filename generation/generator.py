"""Generate one structured wellness-summary response for a query + retrieved chunks.

`client` only needs to provide a `chat_completion(messages, model, temperature, max_tokens)`
method -- duck-typed against generation.client.OpenAIClient so tests can inject a fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from generation.parsing import GenerationOutput, parse_generation_output
from generation.prompts import PROMPT_TEMPLATE_ID, render_prompt
from retrieval.base import RetrievedChunk


class ChatClient(Protocol):
    def chat_completion(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int | None
    ) -> str: ...


@dataclass(frozen=True)
class GenerationRecord:
    question: str
    model: str
    prompt_template_id: str
    retrieved_chunk_ids: list[str]
    raw_text: str
    output: GenerationOutput
    generated_at: str  # ISO 8601 UTC


def generate_summary(
    client: ChatClient,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str,
    temperature: float = 0.2,
    max_tokens: int | None = 800,
) -> GenerationRecord:
    rendered = render_prompt(question, retrieved_chunks)
    raw_text = client.chat_completion(
        rendered.messages, model=model, temperature=temperature, max_tokens=max_tokens
    )
    output = parse_generation_output(raw_text, rendered.passage_map)
    return GenerationRecord(
        question=question,
        model=model,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
        raw_text=raw_text,
        output=output,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

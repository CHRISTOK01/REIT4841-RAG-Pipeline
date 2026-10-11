"""Generate one structured wellness-summary response for a query + retrieved chunks.

`client` only needs to provide a `chat_completion(...) -> ChatResult` method --
duck-typed against generation.client.OpenAIClient so tests can inject a fake.

An output that cannot be used -- cut off at max_tokens, or missing the Summary
or Key Points section the prompt requires -- is regenerated (with a different
seed) up to `max_attempts` times and then raised as
`MalformedGenerationOutput`. It is never passed on to be scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from generation.client import ChatResult
from generation.parsing import GenerationOutput, parse_generation_output
from generation.prompts import PROMPT_TEMPLATE_ID, prompt_hash, render_prompt
from retrieval.base import RetrievedChunk


class MalformedGenerationOutput(RuntimeError):
    """The model's output could not be used as an answer (see module docstring)."""


class ChatClient(Protocol):
    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int | None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class GenerationRecord:
    question: str
    model: str  # model requested
    returned_model: str  # model the API reports having used
    system_fingerprint: str | None
    finish_reason: str
    prompt_template_id: str
    prompt_hash: str
    temperature: float
    max_tokens: int | None
    seed: int | None
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    retrieved_chunk_ids: list[str]
    raw_text: str
    output: GenerationOutput
    generated_at: str  # ISO 8601 UTC


def _problem_with(result: ChatResult, output: GenerationOutput) -> str | None:
    if result.finish_reason != "stop":
        return f"finish_reason={result.finish_reason!r} (output cut off or filtered)"
    if not output.summary.strip():
        return "no '## Summary' section found"
    if not output.key_points:
        return "no key points found under '## Key Points'"
    return None


def generate_summary(
    client: ChatClient,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str,
    temperature: float = 0.2,
    max_tokens: int | None = 800,
    template_id: str = PROMPT_TEMPLATE_ID,
    seed: int | None = None,
    max_attempts: int = 3,
) -> GenerationRecord:
    rendered = render_prompt(question, retrieved_chunks, template_id)
    problem = "no attempt made"
    for attempt in range(1, max_attempts + 1):
        attempt_seed = None if seed is None else seed + attempt - 1
        result = client.chat_completion(
            rendered.messages, model=model, temperature=temperature, max_tokens=max_tokens, seed=attempt_seed
        )
        output = parse_generation_output(result.text, rendered.passage_map)
        problem = _problem_with(result, output)
        if problem is None:
            return GenerationRecord(
                question=question,
                model=model,
                returned_model=result.model,
                system_fingerprint=result.system_fingerprint,
                finish_reason=result.finish_reason,
                prompt_template_id=template_id,
                prompt_hash=prompt_hash(template_id),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=attempt_seed,
                attempts=attempt,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
                raw_text=result.text,
                output=output,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
    raise MalformedGenerationOutput(f"unusable generation after {max_attempts} attempt(s): {problem}")

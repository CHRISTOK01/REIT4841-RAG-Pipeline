"""Prompt templates for wellness-summary generation.

Templates are versioned by id so an experiment config can pin an exact
prompt and the methodology chapter can quote it verbatim. Output is
structured (Summary / numbered, passage-cited Key Points / Disclaimer) so
the evaluation layer can parse individual claims and check their cited
passages against what was actually retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.base import RetrievedChunk

PROMPT_TEMPLATE_ID = "wellness_summary_v1"

SYSTEM_PROMPT_V1 = """You are a medical-information assistant that writes short, structured wellness \
summaries in response to a user's health question.

Rules:
- Base every factual statement ONLY on the numbered context passages provided below. Do not use \
outside/prior medical knowledge.
- If the provided context does not fully answer the question, say so explicitly rather than \
filling the gap from general knowledge.
- After every claim in "Key Points", cite the context passage number(s) it is based on, like [1] \
or [2][3].
- Do not invent citations to passage numbers that were not provided.
- This is not a substitute for professional medical advice; end with a brief disclaimer saying so.

Output format (Markdown, use exactly these three headings):
## Summary
<one short paragraph directly answering the question>

## Key Points
1. <claim> [<passage number(s)>]
2. <claim> [<passage number(s)>]

## Disclaimer
<one sentence>
"""

USER_PROMPT_TEMPLATE_V1 = """Question: {question}

Context passages:
{context_block}
"""


@dataclass(frozen=True)
class RenderedPrompt:
    messages: list[dict]
    passage_map: dict[int, str]  # passage number (as shown to the model) -> chunk_id


def format_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] (chunk_id={c.chunk_id})\n{c.text}" for i, c in enumerate(chunks, start=1))


def render_prompt(question: str, chunks: list[RetrievedChunk]) -> RenderedPrompt:
    context_block = format_context_block(chunks) if chunks else "(no context passages retrieved)"
    user_prompt = USER_PROMPT_TEMPLATE_V1.format(question=question, context_block=context_block)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V1},
        {"role": "user", "content": user_prompt},
    ]
    passage_map = {i: c.chunk_id for i, c in enumerate(chunks, start=1)}
    return RenderedPrompt(messages=messages, passage_map=passage_map)

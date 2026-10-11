"""Prompt templates for wellness-summary generation.

Templates are versioned by id so an experiment config can pin an exact
prompt and the methodology chapter can quote it verbatim. Output is
structured (Summary / numbered, passage-cited Key Points / Disclaimer) so
the evaluation layer can parse individual claims and check their cited
passages against what was actually retrieved.
"""

from __future__ import annotations

import hashlib
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


# No-retrieval baseline ("LLM only"): same task and output format, but the model is
# told to answer from its own knowledge. Used only for the `none` baseline, never for
# any of the retrieval configurations.
CLOSED_BOOK_TEMPLATE_ID = "closed_book_v1"

SYSTEM_PROMPT_CLOSED_BOOK_V1 = """You are a medical-information assistant that writes short, structured wellness \
summaries in response to a user's health question.

Rules:
- No context passages are provided. Answer from your own general medical knowledge.
- If you do not know the answer, say so explicitly rather than guessing.
- This is not a substitute for professional medical advice; end with a brief disclaimer saying so.

Output format (Markdown, use exactly these three headings):
## Summary
<one short paragraph directly answering the question>

## Key Points
1. <claim>
2. <claim>

## Disclaimer
<one sentence>
"""

USER_PROMPT_TEMPLATE_CLOSED_BOOK_V1 = """Question: {question}
"""

# template id -> (system prompt, user prompt template)
PROMPT_TEMPLATES: dict[str, tuple[str, str]] = {
    PROMPT_TEMPLATE_ID: (SYSTEM_PROMPT_V1, USER_PROMPT_TEMPLATE_V1),
    CLOSED_BOOK_TEMPLATE_ID: (SYSTEM_PROMPT_CLOSED_BOOK_V1, USER_PROMPT_TEMPLATE_CLOSED_BOOK_V1),
}


def prompt_hash(template_id: str = PROMPT_TEMPLATE_ID) -> str:
    """SHA-256 over the exact system + user template text. Logged with every
    result row, so any edit to a prompt -- even whitespace -- is visible in the log."""
    system_prompt, user_template = PROMPT_TEMPLATES[template_id]
    return hashlib.sha256(f"{system_prompt}\x00{user_template}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RenderedPrompt:
    messages: list[dict]
    passage_map: dict[int, str]  # passage number (as shown to the model) -> chunk_id
    template_id: str = PROMPT_TEMPLATE_ID


def format_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] (chunk_id={c.chunk_id})\n{c.text}" for i, c in enumerate(chunks, start=1))


def render_prompt(
    question: str, chunks: list[RetrievedChunk], template_id: str = PROMPT_TEMPLATE_ID
) -> RenderedPrompt:
    if template_id not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template id: {template_id!r}")
    system_prompt, user_template = PROMPT_TEMPLATES[template_id]
    if template_id == CLOSED_BOOK_TEMPLATE_ID:
        if chunks:
            raise ValueError("the closed-book template takes no context passages")
        user_prompt = user_template.format(question=question)
    else:
        context_block = format_context_block(chunks) if chunks else "(no context passages retrieved)"
        user_prompt = user_template.format(question=question, context_block=context_block)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    passage_map = {i: c.chunk_id for i, c in enumerate(chunks, start=1)}
    return RenderedPrompt(messages=messages, passage_map=passage_map, template_id=template_id)

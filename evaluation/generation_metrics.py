"""LLM-as-judge generation quality metrics: faithfulness, answer relevance,
context relevance.

These replicate Ragas' scoring logic directly against our own OpenAIClient
rather than depending on the `ragas` package. Decision, not a shortcut: as of
this writing, `ragas` 0.4.3 (latest) fails to import at all under a clean
`pip install ragas` -- its `ragas.llms.base` unconditionally imports
`langchain_community.chat_models.vertexai.ChatVertexAI`, which no longer
exists at that path in current `langchain-community`, and pinning
`langchain-community` back down to a version that still has it breaks
`langchain_openai`'s required `langchain-core` version instead (verified: the
resulting cross-package pin cascade has no consistent resolution against the
rest of that dependency tree at the time of writing). The build spec
explicitly pre-authorized this fallback ("...or replicate its scoring logic
if Ragas integration proves awkward"). A self-built implementation is also
arguably better for a reproducible thesis pipeline: no fragile, fast-moving
external dependency chain to still be resolvable when the thesis is
defended, and every scoring prompt is inline here to quote verbatim in the
methodology chapter.

Each metric is computed via a single structured chat completion (`client`
duck-types generation.client.OpenAIClient: `chat_completion(...)` and, for
answer relevance, `embed(...)`), so this module has no network dependency of
its own -- tests inject a fake client with canned JSON responses.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

_JSON_BLOCK_RE = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)


class EvalClient(Protocol):
    def chat_completion(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int | None
    ) -> str: ...

    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...


class MalformedJudgeResponse(ValueError):
    """The judge model's response couldn't be parsed into the expected shape."""


def _extract_json(text: str):
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise MalformedJudgeResponse(f"No JSON found in model response: {text!r}")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        raise MalformedJudgeResponse(f"Invalid JSON in model response: {text!r}") from e


def _parse_bool_array(text: str, expected_length: int) -> list[bool]:
    data = _extract_json(text)
    if not isinstance(data, list) or len(data) != expected_length:
        raise MalformedJudgeResponse(
            f"Expected a JSON array of {expected_length} booleans, got: {text!r}"
        )
    return [bool(v) for v in data]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -- Faithfulness ------------------------------------------------------------

FAITHFULNESS_SYSTEM_PROMPT = """You are a strict fact-checking judge. You are given a question, a set \
of numbered context passages, and a numbered list of claims made in an answer to that question.

For each claim, decide whether it is directly supported by the context passages: a careful reader \
could verify it is true using ONLY the given context, with no outside knowledge. If a claim adds \
any information, number, or specific detail beyond what the context states, it is NOT supported.

Respond with ONLY a JSON array of exactly {n} boolean values (true/false), one per claim, in the \
same order as the claims. No other text."""


def compute_faithfulness(
    client: EvalClient,
    question: str,
    claim_texts: list[str],
    context_texts: list[str],
    model: str,
) -> float:
    """Fraction of claim_texts (e.g. a GenerationOutput's Key Point sentences)
    that the judge model marks as directly supported by context_texts.
    NaN if there are no claims to check (see module docstring re: undefined
    metrics vs. 0.0)."""
    if not claim_texts:
        return math.nan

    numbered_claims = "\n".join(f"{i}. {c}" for i, c in enumerate(claim_texts, start=1))
    context_block = (
        "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context_texts, start=1))
        or "(no context passages provided)"
    )
    messages = [
        {"role": "system", "content": FAITHFULNESS_SYSTEM_PROMPT.format(n=len(claim_texts))},
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext passages:\n{context_block}\n\nClaims:\n{numbered_claims}",
        },
    ]
    response = client.chat_completion(messages, model=model, temperature=0.0, max_tokens=500)
    verdicts = _parse_bool_array(response, expected_length=len(claim_texts))
    return sum(verdicts) / len(verdicts)


# -- Context relevance ---------------------------------------------------------

CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You are given a question and a numbered list of context passages \
that were retrieved for it. For each passage, judge whether it is relevant and useful for answering \
the question -- i.e. it contains information that helps answer it, even partially.

Respond with ONLY a JSON array of exactly {n} boolean values (true/false), one per passage, in the \
same order as the passages. No other text."""


def compute_context_relevance(
    client: EvalClient,
    question: str,
    context_texts: list[str],
    model: str,
) -> float:
    """Fraction of context_texts the judge model marks as relevant to the
    question. NaN if no context was retrieved at all (undefined, distinct
    from "context was retrieved but judged irrelevant")."""
    if not context_texts:
        return math.nan

    numbered_passages = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context_texts, start=1))
    messages = [
        {"role": "system", "content": CONTEXT_RELEVANCE_SYSTEM_PROMPT.format(n=len(context_texts))},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{numbered_passages}"},
    ]
    response = client.chat_completion(messages, model=model, temperature=0.0, max_tokens=500)
    verdicts = _parse_bool_array(response, expected_length=len(context_texts))
    return sum(verdicts) / len(verdicts)


# -- Answer relevance ------------------------------------------------------------

ANSWER_RELEVANCE_SYSTEM_PROMPT = """Given an answer, generate {n} distinct questions that this answer \
would be an appropriate, specific response to. If the answer does not commit to any specific \
information (e.g. it says the information is unavailable, or is empty/near-empty), respond with an \
empty array instead.

Respond with ONLY a JSON array of question strings. No other text."""


def compute_answer_relevance(
    client: EvalClient,
    question: str,
    answer_text: str,
    chat_model: str,
    embedding_model: str,
    num_questions: int = 3,
) -> float:
    """Mean cosine similarity between the original question and `num_questions`
    LLM-generated questions that the answer would appropriately respond to
    (Ragas' actual approach). 0.0 if the answer is empty or the judge model
    marks it as non-committal (generates no questions)."""
    if not answer_text.strip():
        return 0.0

    messages = [
        {"role": "system", "content": ANSWER_RELEVANCE_SYSTEM_PROMPT.format(n=num_questions)},
        {"role": "user", "content": f"Answer:\n{answer_text}"},
    ]
    response = client.chat_completion(messages, model=chat_model, temperature=0.7, max_tokens=300)
    generated = _extract_json(response)
    if not isinstance(generated, list):
        raise MalformedJudgeResponse(f"Expected a JSON array of questions, got: {response!r}")
    generated_questions = [q for q in generated if isinstance(q, str) and q.strip()]

    if not generated_questions:
        return 0.0

    vectors = client.embed([question, *generated_questions], model=embedding_model)
    original_vec = np.asarray(vectors[0], dtype=np.float32)
    similarities = [_cosine_similarity(original_vec, np.asarray(v, dtype=np.float32)) for v in vectors[1:]]
    return float(np.mean(similarities))


# -- Combined ------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationMetrics:
    faithfulness: float
    answer_relevance: float
    context_relevance: float


def compute_generation_metrics(
    client: EvalClient,
    question: str,
    answer_text: str,
    claim_texts: list[str],
    context_texts: list[str],
    chat_model: str,
    embedding_model: str,
) -> GenerationMetrics:
    return GenerationMetrics(
        faithfulness=compute_faithfulness(client, question, claim_texts, context_texts, chat_model),
        answer_relevance=compute_answer_relevance(
            client, question, answer_text, chat_model, embedding_model
        ),
        context_relevance=compute_context_relevance(client, question, context_texts, chat_model),
    )

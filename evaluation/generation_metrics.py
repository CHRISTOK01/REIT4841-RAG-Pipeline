"""LLM-as-judge generation quality metrics (definitions version `METRICS_VERSION`).

These follow Ragas' scoring ideas but are implemented directly against our own
OpenAIClient: `ragas` 0.4.3 would not import under a clean install at the time
of writing (unresolvable langchain pins), and a self-contained implementation
keeps every judge prompt inline and quotable in the methodology chapter.

What is measured, per generated answer:

- **claims**: every sentence of the Summary plus every Key Point (citation
  markers stripped). Extraction is a deterministic rule, not an LLM call.
- **faithfulness**: each claim is labelled by the judge as `supported` (fully
  verifiable from the retrieved passages), `unsupported` (asserts something the
  passages do not state), or `no_information` (asserts nothing about the topic,
  e.g. "the context does not say"). faithfulness = supported / (supported +
  unsupported). Statements of "no information" are in neither the numerator
  nor the denominator. Undefined (NaN) when there is no factual claim or no
  context.
- **abstention**: the judge labels the whole answer `answered`, `partial` or
  `abstained` (the answer declines / says the information is unavailable).
  Reported as its own rate; it is not a score of 0 on anything.
- **answer relevance**: mean cosine similarity between the user's question and
  questions the judge writes back from the answer. Computed only for answers
  that are not `abstained`; NaN otherwise.
- **context relevance**: fraction of retrieved passages the judge marks as
  helping to answer the question. NaN with no context.
- **reference agreement** (optional): the same claim labelling, but against
  the MedQuAD reference answer instead of the retrieved passages. The
  reference answer is never shown to the generator.

Every judge call runs at temperature 0 in JSON mode. A judge reply that is not
exactly the expected shape (wrong length, a string where a boolean belongs, an
unknown label) is retried and then raised as `MalformedJudgeResponse`; it is
never coerced into a score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypeVar

import numpy as np

from generation.client import ChatResult
from generation.parsing import GenerationOutput

METRICS_VERSION = "v2"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_ATTEMPTS = 3

CLAIM_LABELS = ("supported", "unsupported", "no_information")
ABSTENTION_LABELS = ("answered", "partial", "abstained")

_CITATION_RE = re.compile(r"\s*\[\d+\]")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_MIN_CLAIM_WORDS = 3

T = TypeVar("T")


class EvalClient(Protocol):
    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int | None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult: ...

    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...


class MalformedJudgeResponse(ValueError):
    """The judge model's response couldn't be parsed into the expected shape."""


# -- claim extraction (deterministic) ----------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace and a capital
    letter, digit, bracket or quote. A fragment of fewer than three words
    ("Yes.", "No.") is not a checkable claim on its own, so it is joined to the
    sentence that follows it. Deliberately simple and reproducible; an
    abbreviation such as "e.g. The" would be split in two."""
    parts = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text.strip()) if s.strip()]
    sentences: list[str] = []
    carry = ""
    for part in parts:
        part = f"{carry} {part}".strip()
        if len(part.split()) < _MIN_CLAIM_WORDS:
            carry = part
        else:
            sentences.append(part)
            carry = ""
    if carry:
        if sentences:
            sentences[-1] = f"{sentences[-1]} {carry}"
        else:
            sentences.append(carry)
    return sentences


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def extract_claims(output: GenerationOutput) -> list[str]:
    """Summary sentences followed by Key Points, citation markers removed."""
    claims = [strip_citations(s) for s in split_sentences(output.summary)]
    claims += [strip_citations(kp.text) for kp in output.key_points]
    return [c for c in claims if c]


def answer_text_for_judging(output: GenerationOutput) -> str:
    """Summary + Key Points, without the boilerplate disclaimer."""
    key_points = "\n".join(f"- {strip_citations(kp.text)}" for kp in output.key_points)
    return f"{output.summary.strip()}\n{key_points}".strip()


# -- judge plumbing -----------------------------------------------------------


def _ask_judge(
    client: EvalClient,
    messages: list[dict],
    model: str,
    max_tokens: int,
    seed: int | None,
    parse: Callable[[str], T],
) -> T:
    last_error: MalformedJudgeResponse | None = None
    for attempt in range(JUDGE_MAX_ATTEMPTS):
        attempt_seed = None if seed is None else seed + attempt
        result = client.chat_completion(
            messages,
            model=model,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=max_tokens,
            seed=attempt_seed,
            json_mode=True,
        )
        try:
            if result.finish_reason != "stop":
                raise MalformedJudgeResponse(f"judge reply cut off (finish_reason={result.finish_reason!r})")
            return parse(result.text)
        except MalformedJudgeResponse as e:
            last_error = e
    assert last_error is not None
    raise last_error


def _load_json_object(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise MalformedJudgeResponse(f"judge reply is not valid JSON: {text!r}") from e
    if not isinstance(data, dict):
        raise MalformedJudgeResponse(f"judge reply is not a JSON object: {text!r}")
    return data


def _parse_list(text: str, key: str, expected_length: int) -> list:
    data = _load_json_object(text)
    values = data.get(key)
    if not isinstance(values, list) or len(values) != expected_length:
        raise MalformedJudgeResponse(f"expected {key!r} to be a list of {expected_length} items, got: {text!r}")
    return values


def parse_label_list(text: str, key: str, expected_length: int, allowed: tuple[str, ...]) -> list[str]:
    values = _parse_list(text, key, expected_length)
    if not all(isinstance(v, str) and v in allowed for v in values):
        raise MalformedJudgeResponse(f"expected every item of {key!r} to be one of {allowed}, got: {text!r}")
    return values


def parse_bool_list(text: str, key: str, expected_length: int) -> list[bool]:
    values = _parse_list(text, key, expected_length)
    if not all(isinstance(v, bool) for v in values):  # "false" (a string) is rejected, not truthy
        raise MalformedJudgeResponse(f"expected every item of {key!r} to be true/false, got: {text!r}")
    return values


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def _passage_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        raise MalformedJudgeResponse("zero-length embedding in answer relevance")
    return float(np.dot(a, b) / denom)


# -- claim support (faithfulness / reference agreement) -----------------------

CLAIM_SUPPORT_SYSTEM_PROMPT = """You are a strict fact-checking judge. You are given a question, a set of \
numbered source passages, and a numbered list of claims taken from an answer to that question.

Label each claim with exactly one of:
- "supported": the claim asserts a fact, and a careful reader could verify every part of it using \
ONLY the source passages, with no outside knowledge.
- "unsupported": the claim asserts a fact, and any part of it (a fact, number, name, cause, or \
specific detail) is not stated in the source passages or contradicts them. This includes a claim \
that is true in general but not stated in the passages, and a claim about a different subject from \
the one the passages discuss.
- "no_information": the claim asserts no fact about the world at all. Its only content is that the \
passages or context do not contain, mention, or provide something, or that the question cannot be \
answered from them.

Use "no_information" only for such statements about what the passages lack. Every claim that \
asserts a fact, on any subject, must be labelled "supported" or "unsupported".

A claim that something does not exist or is not available (for example "there is no vaccine" or \
"there is no cure") asserts a fact. It is "supported" only if the passages explicitly say so; if the \
passages are silent on the matter it is "unsupported".

Respond with ONLY a JSON object of the form {{"labels": [...]}} containing exactly {n} labels, one per \
claim, in the same order as the claims."""


@dataclass(frozen=True)
class ClaimSupport:
    score: float  # supported / (supported + unsupported); NaN if no factual claims
    labels: list[str] = field(default_factory=list)

    @property
    def n_supported(self) -> int:
        return self.labels.count("supported")

    @property
    def n_unsupported(self) -> int:
        return self.labels.count("unsupported")

    @property
    def n_no_information(self) -> int:
        return self.labels.count("no_information")


def support_score(labels: list[str]) -> float:
    factual = labels.count("supported") + labels.count("unsupported")
    if factual == 0:
        return math.nan
    return labels.count("supported") / factual


def compute_claim_support(
    client: EvalClient,
    question: str,
    claims: list[str],
    passages: list[str],
    model: str,
    seed: int | None = None,
) -> ClaimSupport:
    """Labels every claim against `passages`. No judge call (score NaN, no
    labels) when there are no claims or no passages to check against."""
    if not claims or not passages:
        return ClaimSupport(score=math.nan, labels=[])
    messages = [
        {"role": "system", "content": CLAIM_SUPPORT_SYSTEM_PROMPT.format(n=len(claims))},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nSource passages:\n{_passage_block(passages)}\n\n"
                f"Claims:\n{_numbered(claims)}"
            ),
        },
    ]
    labels = _ask_judge(
        client,
        messages,
        model,
        max_tokens=40 + 12 * len(claims),
        seed=seed,
        parse=lambda text: parse_label_list(text, "labels", len(claims), CLAIM_LABELS),
    )
    return ClaimSupport(score=support_score(labels), labels=labels)


# -- abstention ----------------------------------------------------------------

ABSTENTION_SYSTEM_PROMPT = """You are given a text that an assistant wrote in reply to a question. You \
are not shown the question. Decide whether the text gives information or declines to give it.

Label the text with exactly one of:
- "answered": it states substantive information and nowhere says that information is missing, not \
provided, or unavailable.
- "partial": it states some substantive information, and also says that some information is missing, \
not provided, not available, or cannot be answered.
- "abstained": it states no substantive information. It only says that the information is not \
available, not provided, or not in the context.

Respond with ONLY a JSON object of the form {"label": "..."}."""


def compute_abstention_label(client: EvalClient, answer_text: str, model: str, seed: int | None = None) -> str:
    """Whether the answer commits to content or declines. The judge sees the
    answer only, not the question, so that an answer about the wrong subject
    is not mistaken for an abstention (relevance is measured separately)."""
    def parse(text: str) -> str:
        label = _load_json_object(text).get("label")
        if label not in ABSTENTION_LABELS:
            raise MalformedJudgeResponse(f"expected 'label' to be one of {ABSTENTION_LABELS}, got: {text!r}")
        return label

    messages = [
        {"role": "system", "content": ABSTENTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Text:\n{answer_text}"},
    ]
    return _ask_judge(client, messages, model, max_tokens=20, seed=seed, parse=parse)


# -- context relevance ----------------------------------------------------------

CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You are given a question and a numbered list of context passages \
that were retrieved for it. For each passage, judge whether it is relevant and useful for answering \
the question -- i.e. it contains information that helps answer it, even partially.

Respond with ONLY a JSON object of the form {{"relevant": [...]}} containing exactly {n} boolean values \
(true/false), one per passage, in the same order as the passages."""


@dataclass(frozen=True)
class ContextRelevance:
    score: float  # fraction of passages judged relevant; NaN with no passages
    labels: list[bool] = field(default_factory=list)


def compute_context_relevance(
    client: EvalClient, question: str, context_texts: list[str], model: str, seed: int | None = None
) -> ContextRelevance:
    if not context_texts:
        return ContextRelevance(score=math.nan, labels=[])
    messages = [
        {"role": "system", "content": CONTEXT_RELEVANCE_SYSTEM_PROMPT.format(n=len(context_texts))},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{_passage_block(context_texts)}"},
    ]
    labels = _ask_judge(
        client,
        messages,
        model,
        max_tokens=40 + 6 * len(context_texts),
        seed=seed,
        parse=lambda text: parse_bool_list(text, "relevant", len(context_texts)),
    )
    return ContextRelevance(score=sum(labels) / len(labels), labels=labels)


# -- answer relevance -----------------------------------------------------------

ANSWER_RELEVANCE_SYSTEM_PROMPT = """Given an answer, write {n} distinct questions that this answer would \
be an appropriate, specific response to.

Respond with ONLY a JSON object of the form {{"questions": [...]}} containing exactly {n} question \
strings."""


def compute_answer_relevance(
    client: EvalClient,
    question: str,
    answer_text: str,
    chat_model: str,
    embedding_model: str,
    num_questions: int = 3,
    seed: int | None = None,
) -> float:
    """Mean cosine similarity between `question` and `num_questions` questions
    the judge writes back from the answer. Only call this for an answer that
    was not labelled `abstained`."""
    if not answer_text.strip():
        raise MalformedJudgeResponse("answer relevance requested for an empty answer")

    def parse(text: str) -> list[str]:
        questions = _parse_list(text, "questions", num_questions)
        if not all(isinstance(q, str) and q.strip() for q in questions):
            raise MalformedJudgeResponse(f"expected {num_questions} non-empty question strings, got: {text!r}")
        return questions

    messages = [
        {"role": "system", "content": ANSWER_RELEVANCE_SYSTEM_PROMPT.format(n=num_questions)},
        {"role": "user", "content": f"Answer:\n{answer_text}"},
    ]
    generated = _ask_judge(client, messages, chat_model, max_tokens=60 + 60 * num_questions, seed=seed, parse=parse)

    vectors = client.embed([question, *generated], model=embedding_model)
    if len(vectors) != len(generated) + 1:
        raise MalformedJudgeResponse("embedding call returned the wrong number of vectors")
    original_vec = np.asarray(vectors[0], dtype=np.float32)
    similarities = [_cosine_similarity(original_vec, np.asarray(v, dtype=np.float32)) for v in vectors[1:]]
    return float(np.mean(similarities))


# -- prompt fingerprint ------------------------------------------------------------


def judge_prompts_hash() -> str:
    """SHA-256 over the four judge prompts, logged with every result row: an
    edit to any of them changes how answers are scored."""
    blob = "\x00".join(
        [
            CLAIM_SUPPORT_SYSTEM_PROMPT,
            ABSTENTION_SYSTEM_PROMPT,
            CONTEXT_RELEVANCE_SYSTEM_PROMPT,
            ANSWER_RELEVANCE_SYSTEM_PROMPT,
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -- combined ---------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationMetrics:
    faithfulness: float
    n_claims: int
    n_supported: int
    n_unsupported: int
    n_no_information: int
    claim_labels: list[str]
    abstention_label: str
    abstained: bool
    answer_relevance: float
    context_relevance: float
    context_labels: list[bool]
    reference_agreement: float | None  # None when the reference check is switched off
    reference_labels: list[str] | None
    judge_model: str
    metrics_version: str = METRICS_VERSION


def compute_generation_metrics(
    client: EvalClient,
    question: str,
    output: GenerationOutput,
    context_texts: list[str],
    judge_model: str,
    embedding_model: str,
    num_questions: int = 3,
    reference_answer: str | None = None,
    seed: int | None = None,
) -> GenerationMetrics:
    claims = extract_claims(output)
    if not claims:
        raise MalformedJudgeResponse("no claims could be extracted from the generated answer")
    answer_text = answer_text_for_judging(output)

    support = compute_claim_support(client, question, claims, context_texts, judge_model, seed)
    abstention_label = compute_abstention_label(client, answer_text, judge_model, seed)
    abstained = abstention_label == "abstained"
    answer_relevance = (
        math.nan
        if abstained
        else compute_answer_relevance(
            client, question, answer_text, judge_model, embedding_model, num_questions, seed
        )
    )
    context = compute_context_relevance(client, question, context_texts, judge_model, seed)
    reference = (
        compute_claim_support(client, question, claims, [reference_answer], judge_model, seed)
        if reference_answer is not None
        else None
    )

    return GenerationMetrics(
        faithfulness=support.score,
        n_claims=len(claims),
        n_supported=support.n_supported,
        n_unsupported=support.n_unsupported,
        n_no_information=support.n_no_information,
        claim_labels=support.labels,
        abstention_label=abstention_label,
        abstained=abstained,
        answer_relevance=answer_relevance,
        context_relevance=context.score,
        context_labels=context.labels,
        reference_agreement=reference.score if reference is not None else None,
        reference_labels=reference.labels if reference is not None else None,
        judge_model=judge_model,
    )

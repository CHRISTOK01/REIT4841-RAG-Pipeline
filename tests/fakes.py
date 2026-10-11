"""Stand-ins for the OpenAI API (the one external service), used by the fast
tests. They replace the network call only; the code under test is real.

`ScriptedClient` replays a fixed list of replies (for testing how a specific
reply is handled). `RuleBasedClient` answers any generation/judge prompt from
this project with a well-formed reply, and can be told to fail in specific
ways at specific call numbers (for crash, timeout and malformed-reply tests).
"""

from __future__ import annotations

import json
import re

from generation.client import ChatResult

GOOD_ANSWER = """## Summary
Short answer about the topic. It has two sentences.

## Key Points
1. A grounded claim about the topic. [1]
2. A second grounded claim. [1]

## Disclaimer
Not medical advice.
"""


def chat_result(text: str, finish_reason: str = "stop", model: str = "fake-model-2024-01-01") -> ChatResult:
    return ChatResult(
        text=text,
        model=model,
        finish_reason=finish_reason,
        prompt_tokens=100,
        completion_tokens=20,
        system_fingerprint="fp_fake",
    )


class ScriptedClient:
    """Returns `replies` in order. A reply may be a string, a ChatResult, or an
    exception instance (which is raised)."""

    def __init__(self, replies, embed_vectors=None):
        self.replies = list(replies)
        self.chat_calls: list[dict] = []
        self.embed_vectors = embed_vectors or {}
        self.embed_calls: list[list[str]] = []

    def chat_completion(self, messages, model, temperature, max_tokens, seed=None, json_mode=False):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "json_mode": json_mode,
            }
        )
        if not self.replies:
            raise AssertionError("ScriptedClient ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, ChatResult) else chat_result(reply)

    def embed(self, texts, model):
        self.embed_calls.append(list(texts))
        return [self.embed_vectors[t] for t in texts]


_N_LABELS_RE = re.compile(r"exactly (\d+) labels")
_N_BOOLS_RE = re.compile(r"exactly (\d+) boolean values")
_N_QUESTIONS_RE = re.compile(r"exactly (\d+) question")
_VOCAB = ["diabetes", "asthma", "exercise", "heart", "thirst", "fatigue", "inhaler", "kidney", "migraine", "vaccine"]


def keyword_vector(text: str) -> list[float]:
    """A deterministic toy embedding: counts of a small vocabulary plus a
    constant, so texts differ and no vector is all-zero."""
    lowered = text.lower()
    return [1.0] + [float(lowered.count(term)) for term in _VOCAB]


class RuleBasedClient:
    """Answers every prompt this project sends with a well-formed reply.

    `fail_on`: {chat call number (1-based): exception instance or str}. An
    exception is raised on that call; a str is returned as the reply text
    (use it to inject malformed output).
    """

    def __init__(self, fail_on: dict | None = None, answer: str = GOOD_ANSWER):
        self.fail_on = dict(fail_on or {})
        self.answer = answer
        self.chat_call_count = 0
        self.embed_call_count = 0
        self.generation_calls: list[dict] = []
        self.budget = None

    def chat_completion(self, messages, model, temperature, max_tokens, seed=None, json_mode=False):
        self.chat_call_count += 1
        injected = self.fail_on.get(self.chat_call_count)
        if isinstance(injected, BaseException):
            raise injected
        if isinstance(injected, str):
            return chat_result(injected)

        system = messages[0]["content"]
        if "## Summary" in system:
            self.generation_calls.append({"messages": messages, "model": model, "temperature": temperature, "seed": seed})
            return chat_result(self.answer)
        if match := _N_LABELS_RE.search(system):
            return chat_result(json.dumps({"labels": ["supported"] * int(match.group(1))}))
        if match := _N_BOOLS_RE.search(system):
            return chat_result(json.dumps({"relevant": [True] * int(match.group(1))}))
        if match := _N_QUESTIONS_RE.search(system):
            return chat_result(json.dumps({"questions": [f"generated question {i}" for i in range(int(match.group(1)))]}))
        if '{"label": "..."}' in system:
            return chat_result(json.dumps({"label": "answered"}))
        raise AssertionError(f"RuleBasedClient does not recognise this prompt: {system[:120]!r}")

    def embed(self, texts, model):
        self.embed_call_count += 1
        return [keyword_vector(t) for t in texts]

    def embed_fn(self, model):
        def _fn(texts):
            return self.embed(list(texts), model)

        return _fn

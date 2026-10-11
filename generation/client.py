"""Thin wrapper around the OpenAI SDK: chat completions + embeddings, both
with explicit retry/backoff on transient errors.

Never hardcode the API key -- it's always read from the OPENAI_API_KEY
environment variable (a `.env` file in the repo root is loaded if present).
Retry policy is owned entirely by tenacity here (the SDK's own retries are
disabled via max_retries=0) so it's one place to tune/cite in the methodology
chapter.

A response that carries no usable content (no choices, empty text, wrong
number of embeddings) raises `MalformedAPIResponse`; it is retried like a
transient error and, if it persists, propagates. It is never returned as an
empty string for a caller to mistake for an answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tenacity.wait import wait_base

from generation.budget import SpendTracker

load_dotenv()

DEFAULT_EMBED_BATCH_SIZE = 100
DEFAULT_TIMEOUT_SECONDS = 90.0


class MalformedAPIResponse(RuntimeError):
    """The API answered, but without the content the call needs."""


RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    MalformedAPIResponse,
)


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str  # the model name the API reports having used (a dated snapshot)
    finish_reason: str  # "stop", "length" (hit max_tokens), "content_filter", ...
    prompt_tokens: int
    completion_tokens: int
    system_fingerprint: str | None


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        max_retry_attempts: int = 5,
        wait_strategy: wait_base | None = None,
        budget: SpendTracker | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Put it in a .env file in the repo root.")
        self._client = OpenAI(api_key=api_key, max_retries=0, timeout=timeout)
        self.budget = budget
        self._retry = retry(
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            wait=wait_strategy or wait_random_exponential(min=1, max=30),
            stop=stop_after_attempt(max_retry_attempts),
            reraise=True,
        )

    def embed(
        self, texts: list[str], model: str, batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    ) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            @self._retry
            def _call(batch: list[str] = batch) -> list[list[float]]:
                if self.budget is not None:
                    self.budget.check(model, "embedding")
                response = self._client.embeddings.create(model=model, input=batch)
                items = list(response.data)
                if len(items) != len(batch):
                    raise MalformedAPIResponse(
                        f"embeddings API returned {len(items)} vectors for {len(batch)} inputs"
                    )
                # The API tags each vector with the index of its input; order by it
                # rather than trusting response order.
                items.sort(key=lambda item: getattr(item, "index", 0))
                if self.budget is not None:
                    usage = getattr(response, "usage", None)
                    self.budget.record_embedding(model, getattr(usage, "total_tokens", 0) or 0)
                return [item.embedding for item in items]

            results.extend(_call())
        return results

    def embed_fn(self, model: str, batch_size: int = DEFAULT_EMBED_BATCH_SIZE):
        """A callable matching retrieval.embedding_cache.EmbedFn, for wiring
        this client into DenseRetriever.build / EmbeddingCache."""

        def _fn(texts):
            return self.embed(list(texts), model, batch_size=batch_size)

        return _fn

    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        @self._retry
        def _call() -> ChatResult:
            if self.budget is not None:
                self.budget.check(model, "chat")
            kwargs: dict = {}
            if seed is not None:
                kwargs["seed"] = seed
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            if self.budget is not None:
                self.budget.record_chat(model, prompt_tokens, completion_tokens)

            if not response.choices:
                raise MalformedAPIResponse("chat completion returned no choices")
            choice = response.choices[0]
            text = choice.message.content
            if text is None or not text.strip():
                raise MalformedAPIResponse(
                    f"chat completion returned empty content (finish_reason={choice.finish_reason!r})"
                )
            return ChatResult(
                text=text,
                model=getattr(response, "model", None) or model,
                finish_reason=choice.finish_reason or "unknown",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                system_fingerprint=getattr(response, "system_fingerprint", None),
            )

        return _call()


def main() -> None:
    """Manual smoke test: `python -m generation.client`. Makes one real
    embedding call and one real chat completion call to confirm OPENAI_API_KEY
    is set and working. Costs a trivial amount of API credit."""
    client = OpenAIClient()

    vectors = client.embed(["hello world"], model="text-embedding-3-small")
    print(f"Embedding OK: dim={len(vectors[0])}")

    reply = client.chat_completion(
        [{"role": "user", "content": "Reply with exactly the word: OK"}],
        model="gpt-4o-mini-2024-07-18",
        max_tokens=5,
    )
    print(f"Chat completion OK: {reply.text!r} (model={reply.model})")


if __name__ == "__main__":
    main()

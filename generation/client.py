"""Thin wrapper around the OpenAI SDK: chat completions + embeddings, both
with explicit retry/backoff on transient errors.

Never hardcode the API key -- it's always read from the OPENAI_API_KEY
environment variable (copy .env.example to .env and fill it in). Retry
policy is owned entirely by tenacity here (the SDK's own retries are
disabled via max_retries=0) so it's one place to tune/cite in the
methodology chapter.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tenacity.wait import wait_base

load_dotenv()

RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

DEFAULT_EMBED_BATCH_SIZE = 100


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        max_retry_attempts: int = 5,
        wait_strategy: wait_base | None = None,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key."
            )
        self._client = OpenAI(api_key=api_key, max_retries=0)
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
                response = self._client.embeddings.create(model=model, input=batch)
                return [item.embedding for item in response.data]

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
    ) -> str:
        @self._retry
        def _call() -> str:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

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
        model="gpt-4o-mini",
        max_tokens=5,
    )
    print(f"Chat completion OK: {reply!r}")


if __name__ == "__main__":
    main()

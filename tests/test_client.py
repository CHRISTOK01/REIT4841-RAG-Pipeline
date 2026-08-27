import httpx2
import pytest
from openai import RateLimitError
from tenacity import wait_none

from generation.client import OpenAIClient


def make_rate_limit_error() -> RateLimitError:
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient(api_key=None)


def test_explicit_api_key_used_even_without_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIClient(api_key="sk-explicit-test-key")
    assert client is not None


class FakeEmbeddingsEndpoint:
    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model, input):
        self.calls.append(list(input))

        class Item:
            def __init__(self, vector):
                self.embedding = vector

        class Response:
            def __init__(self, items):
                self.data = items

        return Response([Item([float(len(t))]) for t in input])


def test_embed_batches_requests(monkeypatch):
    client = OpenAIClient(api_key="sk-test", wait_strategy=wait_none())
    fake_embeddings = FakeEmbeddingsEndpoint()
    client._client.embeddings = fake_embeddings

    texts = [f"text-{i}" for i in range(5)]
    vectors = client.embed(texts, model="text-embedding-3-small", batch_size=2)

    assert len(vectors) == 5
    assert len(fake_embeddings.calls) == 3  # batches of 2, 2, 1
    assert fake_embeddings.calls == [["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]]


def test_embed_empty_list_makes_no_call():
    client = OpenAIClient(api_key="sk-test", wait_strategy=wait_none())
    fake_embeddings = FakeEmbeddingsEndpoint()
    client._client.embeddings = fake_embeddings
    assert client.embed([], model="text-embedding-3-small") == []
    assert fake_embeddings.calls == []


def test_embed_fn_wraps_embed_for_retrieval_module():
    client = OpenAIClient(api_key="sk-test", wait_strategy=wait_none())
    fake_embeddings = FakeEmbeddingsEndpoint()
    client._client.embeddings = fake_embeddings

    embed_fn = client.embed_fn(model="text-embedding-3-small")
    vectors = embed_fn(["abc", "de"])
    assert [v[0] for v in vectors] == [3.0, 2.0]


class FlakyChatCompletions:
    """Fails with a retryable error `fail_times`, then succeeds."""

    def __init__(self, fail_times: int, reply: str = "ok"):
        self.fail_times = fail_times
        self.reply = reply
        self.call_count = 0

    def create(self, model, messages, temperature, max_tokens):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise make_rate_limit_error()

        class Message:
            content = self.reply

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


def test_chat_completion_retries_on_rate_limit_then_succeeds():
    client = OpenAIClient(api_key="sk-test", max_retry_attempts=5, wait_strategy=wait_none())
    flaky = FlakyChatCompletions(fail_times=2, reply="finally worked")
    client._client.chat = FakeChat(flaky)

    result = client.chat_completion([{"role": "user", "content": "hi"}], model="gpt-4o-mini")

    assert result == "finally worked"
    assert flaky.call_count == 3


def test_chat_completion_gives_up_after_max_attempts():
    client = OpenAIClient(api_key="sk-test", max_retry_attempts=2, wait_strategy=wait_none())
    always_fails = FlakyChatCompletions(fail_times=999)
    client._client.chat = FakeChat(always_fails)

    with pytest.raises(RateLimitError):
        client.chat_completion([{"role": "user", "content": "hi"}], model="gpt-4o-mini")

    assert always_fails.call_count == 2

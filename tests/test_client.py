"""OpenAIClient with the SDK's network endpoints replaced by fakes: retry
behaviour, malformed responses, and spend accounting. The wrapper itself is
real; only the HTTP call is simulated."""

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError
from tenacity import wait_none

from generation.budget import BudgetExceeded, SpendTracker, UnknownModelPrice
from generation.client import MalformedAPIResponse, OpenAIClient

PRICING = {"chat": {"gpt-test": {"input": 1.0, "output": 2.0}}, "embedding": {"embed-test": {"input": 0.5}}}
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def rate_limit_error():
    return RateLimitError("rate limited", response=httpx.Response(429, request=REQUEST), body=None)


def server_error():
    return InternalServerError("server error", response=httpx.Response(500, request=REQUEST), body=None)


def bad_request_error():
    return BadRequestError("context too long", response=httpx.Response(400, request=REQUEST), body=None)


class Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = prompt_tokens, completion_tokens, total_tokens


class ChatResponse:
    def __init__(self, content, finish_reason="stop", choices=True):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
        self.choices = [choice] if choices else []
        self.model = "gpt-test-2024-01-01"
        self.system_fingerprint = "fp_abc"
        self.usage = Usage(prompt_tokens=1000, completion_tokens=100)


class ScriptedCompletions:
    """Each call pops the next outcome: an exception to raise or a response to return."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_client(outcomes, budget=None, max_retry_attempts=5):
    client = OpenAIClient(api_key="sk-test", max_retry_attempts=max_retry_attempts, wait_strategy=wait_none(), budget=budget)
    completions = ScriptedCompletions(outcomes)
    client._client.chat = type("Chat", (), {"completions": completions})()
    return client, completions


def chat(client, **kwargs):
    return client.chat_completion([{"role": "user", "content": "hi"}], model="gpt-test", **kwargs)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient(api_key=None)


def test_successful_call_returns_text_and_the_metadata_the_log_needs():
    client, completions = make_client([ChatResponse("the answer")])
    result = chat(client, temperature=0.2, max_tokens=50, seed=7, json_mode=True)

    assert result.text == "the answer"
    assert result.model == "gpt-test-2024-01-01"  # the model the API reports, not the one requested
    assert (result.finish_reason, result.system_fingerprint) == ("stop", "fp_abc")
    assert (result.prompt_tokens, result.completion_tokens) == (1000, 100)
    sent = completions.calls[0]
    assert (sent["model"], sent["temperature"], sent["max_tokens"], sent["seed"]) == ("gpt-test", 0.2, 50, 7)
    assert sent["response_format"] == {"type": "json_object"}


def test_seed_and_json_mode_are_not_sent_unless_asked_for():
    client, completions = make_client([ChatResponse("ok")])
    chat(client)
    assert "seed" not in completions.calls[0] and "response_format" not in completions.calls[0]


@pytest.mark.parametrize(
    "error_factory",
    [rate_limit_error, server_error, lambda: APITimeoutError(request=REQUEST), lambda: APIConnectionError(request=REQUEST)],
    ids=["rate_limit", "server_error", "timeout", "connection_error"],
)
def test_transient_errors_are_retried_and_one_result_comes_back(error_factory):
    client, completions = make_client([error_factory(), error_factory(), ChatResponse("finally worked")])
    assert chat(client).text == "finally worked"
    assert len(completions.calls) == 3


def test_gives_up_after_max_attempts_and_raises_the_original_error():
    client, completions = make_client([rate_limit_error()] * 10, max_retry_attempts=2)
    with pytest.raises(RateLimitError):
        chat(client)
    assert len(completions.calls) == 2


def test_non_transient_errors_are_not_retried():
    client, completions = make_client([bad_request_error(), ChatResponse("never reached")])
    with pytest.raises(BadRequestError):
        chat(client)
    assert len(completions.calls) == 1


@pytest.mark.parametrize(
    "bad_response",
    [ChatResponse(None), ChatResponse(""), ChatResponse("   \n"), ChatResponse("x", choices=False), ChatResponse(None, finish_reason="content_filter")],
    ids=["none_content", "empty_content", "whitespace_content", "no_choices", "filtered"],
)
def test_empty_or_malformed_response_is_retried_then_raised_never_returned_as_an_empty_answer(bad_response):
    client, completions = make_client([bad_response] * 3, max_retry_attempts=3)
    with pytest.raises(MalformedAPIResponse):
        chat(client)
    assert len(completions.calls) == 3


def test_malformed_response_followed_by_a_good_one_recovers():
    client, _ = make_client([ChatResponse(None), ChatResponse("good")])
    assert chat(client).text == "good"


def test_a_cut_off_answer_is_returned_with_its_finish_reason_for_the_caller_to_judge():
    client, _ = make_client([ChatResponse("half an ans", finish_reason="length")])
    assert chat(client).finish_reason == "length"


# -- embeddings ------------------------------------------------------------------------


class Item:
    def __init__(self, vector, index):
        self.embedding, self.index = vector, index


class FakeEmbeddings:
    def __init__(self, shuffle=False, drop_last=False, outcomes=None):
        self.calls: list[list[str]] = []
        self.shuffle, self.drop_last, self.outcomes = shuffle, drop_last, list(outcomes or [])

    def create(self, model, input):
        self.calls.append(list(input))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        items = [Item([float(len(t))], i) for i, t in enumerate(input)]
        if self.drop_last:
            items = items[:-1]
        if self.shuffle:
            items = list(reversed(items))
        return type("Response", (), {"data": items, "usage": Usage(total_tokens=10 * len(input))})()


def make_embed_client(fake, budget=None):
    client = OpenAIClient(api_key="sk-test", wait_strategy=wait_none(), budget=budget, max_retry_attempts=3)
    client._client.embeddings = fake
    return client


def test_embed_batches_requests_and_keeps_order():
    fake = FakeEmbeddings()
    vectors = make_embed_client(fake).embed(["a", "bb", "ccc", "dddd", "eeeee"], model="embed-test", batch_size=2)
    assert fake.calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_embed_reorders_a_response_that_arrives_out_of_order():
    """The API labels each vector with the index of its input. A response in a
    different order must not pair texts with the wrong vectors."""
    vectors = make_embed_client(FakeEmbeddings(shuffle=True)).embed(["a", "bb", "ccc"], model="embed-test")
    assert vectors == [[1.0], [2.0], [3.0]]


def test_embed_with_too_few_vectors_raises():
    with pytest.raises(MalformedAPIResponse, match="returned 2 vectors for 3 inputs"):
        make_embed_client(FakeEmbeddings(drop_last=True)).embed(["a", "bb", "ccc"], model="embed-test")


def test_embed_retries_a_transient_error():
    fake = FakeEmbeddings(outcomes=[rate_limit_error()])
    assert make_embed_client(fake).embed(["a"], model="embed-test") == [[1.0]]
    assert len(fake.calls) == 2


def test_embed_empty_list_makes_no_call():
    fake = FakeEmbeddings()
    assert make_embed_client(fake).embed([], model="embed-test") == []
    assert fake.calls == []


def test_embed_fn_wraps_embed_for_the_retrieval_module():
    embed_fn = make_embed_client(FakeEmbeddings()).embed_fn(model="embed-test")
    assert embed_fn(("abc", "de")) == [[3.0], [2.0]]


# -- spend cap ---------------------------------------------------------------------------


def test_every_attempt_is_billed_but_only_one_result_is_returned(tmp_path):
    """A retried call costs money each time it reaches the API; the ledger must
    count every attempt even though the caller gets a single result."""
    budget = SpendTracker(max_usd=100.0, ledger_path=tmp_path / "ledger.json", pricing=PRICING)
    client, _ = make_client([ChatResponse(None), ChatResponse("good")], budget=budget)
    assert chat(client).text == "good"
    # two billed attempts of 1000 prompt + 100 completion tokens: 2 * (1000 * $1 + 100 * $2) / 1e6
    assert budget.spent_usd == pytest.approx(0.0024)


def test_call_is_refused_once_the_cap_is_reached_and_is_not_retried(tmp_path):
    budget = SpendTracker(max_usd=0.001, ledger_path=tmp_path / "ledger.json", pricing=PRICING)
    client, completions = make_client([ChatResponse("first"), ChatResponse("second")], budget=budget)
    assert chat(client).text == "first"  # costs $0.0012, taking the ledger past the $0.001 cap
    with pytest.raises(BudgetExceeded):
        chat(client)
    assert len(completions.calls) == 1  # the refused call never reached the API


def test_unpriced_model_is_refused_before_any_call(tmp_path):
    budget = SpendTracker(max_usd=1.0, ledger_path=tmp_path / "ledger.json", pricing=PRICING)
    client, completions = make_client([ChatResponse("x")], budget=budget)
    with pytest.raises(UnknownModelPrice):
        client.chat_completion([{"role": "user", "content": "hi"}], model="some-other-model")
    assert completions.calls == []


def test_embedding_spend_is_recorded(tmp_path):
    budget = SpendTracker(max_usd=1.0, ledger_path=tmp_path / "ledger.json", pricing=PRICING)
    make_embed_client(FakeEmbeddings(), budget=budget).embed(["a", "bb", "ccc"], model="embed-test")
    assert budget.spent_usd == pytest.approx(30 * 0.5 / 1e6)  # 3 inputs * 10 tokens at $0.5 per million

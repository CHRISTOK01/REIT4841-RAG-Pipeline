import math

import pytest

from evaluation.generation_metrics import (
    MalformedJudgeResponse,
    compute_answer_relevance,
    compute_context_relevance,
    compute_faithfulness,
    compute_generation_metrics,
)


class FakeEvalClient:
    def __init__(self, chat_responses=None, embed_vectors=None):
        self.chat_responses = list(chat_responses or [])
        self.chat_calls = []
        self.embed_vectors = embed_vectors or {}
        self.embed_calls = []

    def chat_completion(self, messages, model, temperature, max_tokens):
        self.chat_calls.append(
            {"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.chat_responses.pop(0)

    def embed(self, texts, model):
        self.embed_calls.append(list(texts))
        return [self.embed_vectors[t] for t in texts]


# -- faithfulness --------------------------------------------------------------


def test_faithfulness_scores_fraction_of_supported_claims():
    client = FakeEvalClient(chat_responses=["[true, false]"])
    score = compute_faithfulness(client, "q", ["claim 1", "claim 2"], ["context"], model="m")
    assert score == 0.5
    assert len(client.chat_calls) == 1


def test_faithfulness_nan_with_no_claims_and_no_api_call():
    client = FakeEvalClient(chat_responses=[])
    score = compute_faithfulness(client, "q", [], ["context"], model="m")
    assert math.isnan(score)
    assert client.chat_calls == []


def test_faithfulness_raises_on_wrong_length_response():
    client = FakeEvalClient(chat_responses=["[true]"])  # 2 claims expected, only 1 returned
    with pytest.raises(MalformedJudgeResponse):
        compute_faithfulness(client, "q", ["c1", "c2"], ["context"], model="m")


def test_faithfulness_raises_on_non_json_response():
    client = FakeEvalClient(chat_responses=["sorry, I cannot help with that"])
    with pytest.raises(MalformedJudgeResponse):
        compute_faithfulness(client, "q", ["c1"], ["context"], model="m")


def test_faithfulness_tolerates_surrounding_prose_around_json():
    client = FakeEvalClient(chat_responses=["Here is my answer:\n[true, true]\nHope that helps."])
    score = compute_faithfulness(client, "q", ["c1", "c2"], ["context"], model="m")
    assert score == 1.0


# -- context relevance -----------------------------------------------------------


def test_context_relevance_scores_fraction_of_relevant_passages():
    client = FakeEvalClient(chat_responses=["[true, true, false]"])
    score = compute_context_relevance(client, "q", ["p1", "p2", "p3"], model="m")
    assert score == pytest.approx(2 / 3)


def test_context_relevance_nan_with_no_context_and_no_api_call():
    client = FakeEvalClient(chat_responses=[])
    score = compute_context_relevance(client, "q", [], model="m")
    assert math.isnan(score)
    assert client.chat_calls == []


# -- answer relevance -------------------------------------------------------------


def test_answer_relevance_empty_answer_scores_zero_no_api_call():
    client = FakeEvalClient(chat_responses=[])
    score = compute_answer_relevance(client, "q", "   ", chat_model="m", embedding_model="e")
    assert score == 0.0
    assert client.chat_calls == []


def test_answer_relevance_non_committal_answer_scores_zero():
    client = FakeEvalClient(chat_responses=["[]"])
    score = compute_answer_relevance(client, "q", "I cannot answer that.", chat_model="m", embedding_model="e")
    assert score == 0.0
    assert client.embed_calls == []


def test_answer_relevance_averages_cosine_similarity_to_generated_questions():
    vectors = {
        "original question": [1.0, 0.0],
        "generated q1": [1.0, 0.0],  # identical direction -> cosine similarity 1.0
        "generated q2": [0.0, 1.0],  # orthogonal -> cosine similarity 0.0
    }
    client = FakeEvalClient(
        chat_responses=['["generated q1", "generated q2"]'], embed_vectors=vectors
    )
    score = compute_answer_relevance(
        client, "original question", "some answer", chat_model="m", embedding_model="e"
    )
    assert score == pytest.approx(0.5)
    assert client.embed_calls == [["original question", "generated q1", "generated q2"]]


def test_answer_relevance_raises_when_response_is_not_a_list():
    client = FakeEvalClient(chat_responses=['{"not": "a list"}'])
    with pytest.raises(MalformedJudgeResponse):
        compute_answer_relevance(client, "q", "answer", chat_model="m", embedding_model="e")


# -- combined ----------------------------------------------------------------------


def test_compute_generation_metrics_assembles_all_three():
    vectors = {"q": [1.0, 0.0], "gen q": [1.0, 0.0]}
    client = FakeEvalClient(
        chat_responses=[
            "[true]",  # faithfulness: 1 claim, supported
            '["gen q"]',  # answer relevance: 1 generated question
            "[true, false]",  # context relevance: 2 passages, 1 relevant
        ],
        embed_vectors=vectors,
    )
    metrics = compute_generation_metrics(
        client,
        question="q",
        answer_text="answer",
        claim_texts=["claim"],
        context_texts=["p1", "p2"],
        chat_model="m",
        embedding_model="e",
    )
    assert metrics.faithfulness == 1.0
    assert metrics.answer_relevance == pytest.approx(1.0)
    assert metrics.context_relevance == pytest.approx(0.5)

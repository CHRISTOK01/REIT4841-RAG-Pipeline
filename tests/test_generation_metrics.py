"""Generation-metric plumbing with the judge model replaced by scripted replies.

These tests can only prove what happens *given* a judge reply: the arithmetic,
which claims are sent, what is treated as malformed, and when a judge call is
skipped. Whether the real judge can tell a faithful answer from a hallucinated
one is a separate question, answered with the real model in
tests/test_api_negative_controls.py.
"""

import json
import math

import pytest

from evaluation.generation_metrics import (
    JUDGE_MAX_ATTEMPTS,
    MalformedJudgeResponse,
    compute_abstention_label,
    compute_answer_relevance,
    compute_claim_support,
    compute_context_relevance,
    compute_generation_metrics,
    extract_claims,
    split_sentences,
    strip_citations,
    support_score,
)
from generation.parsing import parse_generation_output
from tests.fakes import ScriptedClient, chat_result

ANSWER = """## Summary
Asthma is a chronic lung disease. It inflames and narrows the airways [1].

## Key Points
1. Asthma causes wheezing and coughing. [1]
2. Inhaled corticosteroids are a common treatment. [2][3]

## Disclaimer
This is not medical advice.
"""
OUTPUT = parse_generation_output(ANSWER, {1: "a#0", 2: "b#0", 3: "c#0"})
FOUR_CLAIMS = [
    "Asthma is a chronic lung disease.",
    "It inflames and narrows the airways.",
    "Asthma causes wheezing and coughing.",
    "Inhaled corticosteroids are a common treatment.",
]


def labels(*values: str) -> str:
    return json.dumps({"labels": list(values)})


# -- claim extraction ---------------------------------------------------------------


def test_claims_are_the_summary_sentences_then_the_key_points_without_citations_or_disclaimer():
    assert extract_claims(OUTPUT) == FOUR_CLAIMS


def test_split_sentences():
    assert split_sentences("It is common. Does it spread? It does not! The end is near") == [
        "It is common.",
        "Does it spread?",
        "It does not!",
        "The end is near",
    ]
    # a decimal point is not a sentence boundary
    assert split_sentences("A dose of 2.5 mg is used. Then stop taking it.") == ["A dose of 2.5 mg is used.", "Then stop taking it."]
    assert split_sentences("  ") == []


def test_fragments_too_short_to_be_a_claim_are_joined_to_their_neighbour():
    """Found by the real-judge negative controls: "Yes." on its own was sent to
    the judge as a claim and came back labelled as saying nothing."""
    assert split_sentences("Yes. Histidinemia is inherited in a recessive pattern.") == [
        "Yes. Histidinemia is inherited in a recessive pattern."
    ]
    assert split_sentences("The condition is inherited. Always.") == ["The condition is inherited. Always."]
    assert split_sentences("No.") == ["No."]
    assert split_sentences("Yes. No. It varies by person.") == ["Yes. No. It varies by person."]


def test_strip_citations():
    assert strip_citations("A claim [1][2]. Another [3]") == "A claim. Another"


# -- faithfulness arithmetic -----------------------------------------------------------


@pytest.mark.parametrize(
    "judge_labels,expected",
    [
        (["supported"] * 4, 1.0),
        (["supported", "unsupported", "supported", "unsupported"], 0.5),  # 2 of 4
        (["unsupported"] * 4, 0.0),
        # "no information" statements are in neither numerator nor denominator: 1 of (1 + 2), not 1 of 4
        (["supported", "unsupported", "unsupported", "no_information"], 1 / 3),
        (["supported", "no_information", "no_information", "no_information"], 1.0),  # 1 of 1
    ],
)
def test_faithfulness_is_supported_over_supported_plus_unsupported(judge_labels, expected):
    client = ScriptedClient([labels(*judge_labels)])
    support = compute_claim_support(client, "q", FOUR_CLAIMS, ["passage"], model="judge")
    assert support.score == pytest.approx(expected)
    assert support.labels == judge_labels
    assert support.n_supported + support.n_unsupported + support.n_no_information == 4


def test_an_answer_that_only_says_no_information_has_undefined_faithfulness_not_one():
    """Regression for the audit finding: "the context does not provide this"
    used to be scored as a fully faithful answer (1.0)."""
    client = ScriptedClient([labels("no_information", "no_information")])
    support = compute_claim_support(client, "q", ["The context does not say.", "No details are given."], ["p"], model="judge")
    assert math.isnan(support.score)
    assert support.n_no_information == 2


def test_support_score_function_directly():
    assert support_score(["supported", "supported", "unsupported"]) == pytest.approx(2 / 3)
    assert math.isnan(support_score([]))
    assert math.isnan(support_score(["no_information"]))


def test_no_claims_or_no_passages_means_undefined_and_no_judge_call():
    client = ScriptedClient([])
    assert math.isnan(compute_claim_support(client, "q", [], ["p"], model="judge").score)
    assert math.isnan(compute_claim_support(client, "q", ["a claim"], [], model="judge").score)
    assert client.chat_calls == []


def test_judge_is_called_at_temperature_zero_in_json_mode_with_every_claim_and_passage():
    client = ScriptedClient([labels(*["supported"] * 4)])
    compute_claim_support(client, "the question", FOUR_CLAIMS, ["passage one", "passage two"], model="judge-model", seed=9)
    call = client.chat_calls[0]
    assert (call["model"], call["temperature"], call["json_mode"], call["seed"]) == ("judge-model", 0.0, True, 9)
    user = call["messages"][1]["content"]
    assert "Question: the question" in user and "[1] passage one" in user and "[2] passage two" in user
    for i, claim in enumerate(FOUR_CLAIMS, start=1):
        assert f"{i}. {claim}" in user
    assert "exactly 4 labels" in call["messages"][0]["content"]


# -- malformed judge replies are errors, never scores ---------------------------------------


@pytest.mark.parametrize(
    "bad_reply",
    [
        "sorry, I cannot help with that",  # not JSON
        '["supported", "supported"]',  # JSON, but not an object
        '{"labels": ["supported"]}',  # too few
        '{"labels": ["supported", "supported", "supported"]}',  # too many
        '{"labels": ["supported", "maybe"]}',  # unknown label
        '{"labels": ["supported", true]}',  # wrong type
        '{"verdicts": ["supported", "supported"]}',  # wrong key
        '{"labels": "supported, supported"}',  # not a list
    ],
)
def test_malformed_claim_labels_raise_after_retrying(bad_reply):
    client = ScriptedClient([bad_reply] * JUDGE_MAX_ATTEMPTS)
    with pytest.raises(MalformedJudgeResponse):
        compute_claim_support(client, "q", ["c1", "c2"], ["p"], model="judge")
    assert len(client.chat_calls) == JUDGE_MAX_ATTEMPTS


def test_quoted_booleans_are_rejected_not_read_as_true():
    """Regression: bool("false") is True in Python, so a judge replying with
    strings used to have every passage counted as relevant."""
    client = ScriptedClient(['{"relevant": ["false", "false"]}'] * JUDGE_MAX_ATTEMPTS)
    with pytest.raises(MalformedJudgeResponse):
        compute_context_relevance(client, "q", ["p1", "p2"], model="judge")


def test_a_cut_off_judge_reply_is_malformed_even_if_it_parses():
    client = ScriptedClient([chat_result(labels("supported"), finish_reason="length")] * JUDGE_MAX_ATTEMPTS)
    with pytest.raises(MalformedJudgeResponse, match="cut off"):
        compute_claim_support(client, "q", ["c1"], ["p"], model="judge")


def test_a_malformed_reply_followed_by_a_good_one_is_retried_with_a_new_seed():
    client = ScriptedClient(["garbage", labels("supported", "unsupported")])
    support = compute_claim_support(client, "q", ["c1", "c2"], ["p"], model="judge", seed=40)
    assert support.score == 0.5
    assert [c["seed"] for c in client.chat_calls] == [40, 41]


# -- context relevance ------------------------------------------------------------------------


def test_context_relevance_is_the_fraction_of_passages_judged_relevant():
    client = ScriptedClient(['{"relevant": [true, true, false]}'])
    result = compute_context_relevance(client, "q", ["p1", "p2", "p3"], model="judge")
    assert result.score == pytest.approx(2 / 3)
    assert result.labels == [True, True, False]


def test_context_relevance_undefined_with_no_context_and_no_judge_call():
    client = ScriptedClient([])
    assert math.isnan(compute_context_relevance(client, "q", [], model="judge").score)
    assert client.chat_calls == []


# -- abstention ----------------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["answered", "partial", "abstained"])
def test_abstention_label_is_passed_through(label):
    client = ScriptedClient([json.dumps({"label": label})])
    assert compute_abstention_label(client, "an answer", model="judge") == label


def test_unknown_abstention_label_is_malformed():
    client = ScriptedClient(['{"label": "refused"}'] * JUDGE_MAX_ATTEMPTS)
    with pytest.raises(MalformedJudgeResponse):
        compute_abstention_label(client, "an answer", model="judge")


# -- answer relevance ----------------------------------------------------------------------------


def test_answer_relevance_is_the_mean_cosine_to_the_generated_questions():
    vectors = {
        "original question": [1.0, 0.0],
        "generated q1": [2.0, 0.0],  # same direction -> cosine 1.0
        "generated q2": [0.0, 3.0],  # orthogonal     -> cosine 0.0
        "generated q3": [1.0, 1.0],  # 45 degrees     -> cosine 0.707107
    }
    client = ScriptedClient(['{"questions": ["generated q1", "generated q2", "generated q3"]}'], embed_vectors=vectors)
    score = compute_answer_relevance(client, "original question", "some answer", "judge", "embed", num_questions=3)
    assert score == pytest.approx((1.0 + 0.0 + 0.707107) / 3, abs=1e-6)
    assert client.embed_calls == [["original question", "generated q1", "generated q2", "generated q3"]]
    assert client.chat_calls[0]["temperature"] == 0.0


def test_answer_relevance_asks_for_the_configured_number_of_questions():
    vectors = {"q": [1.0, 0.0], "g1": [1.0, 0.0], "g2": [1.0, 0.0]}
    client = ScriptedClient(['{"questions": ["g1", "g2"]}'], embed_vectors=vectors)
    compute_answer_relevance(client, "q", "answer", "judge", "embed", num_questions=2)
    assert "write 2 distinct questions" in client.chat_calls[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    "bad_reply",
    ['{"questions": []}', '{"questions": ["only one"]}', '{"questions": ["a", ""]}', '{"questions": ["a", 5]}', '{"not": "a list"}'],
)
def test_answer_relevance_never_turns_a_bad_judge_reply_into_zero(bad_reply):
    client = ScriptedClient([bad_reply] * JUDGE_MAX_ATTEMPTS, embed_vectors={})
    with pytest.raises(MalformedJudgeResponse):
        compute_answer_relevance(client, "q", "answer", "judge", "embed", num_questions=2)
    assert client.embed_calls == []


def test_answer_relevance_refuses_an_empty_answer():
    with pytest.raises(MalformedJudgeResponse):
        compute_answer_relevance(ScriptedClient([]), "q", "   ", "judge", "embed")


# -- combined --------------------------------------------------------------------------------------

EMBED = {"q": [1.0, 0.0], "g1": [1.0, 0.0], "g2": [0.0, 1.0], "g3": [1.0, 0.0]}


def test_compute_generation_metrics_assembles_everything_for_an_answered_question():
    client = ScriptedClient(
        [
            labels("supported", "supported", "unsupported", "supported"),  # claims vs retrieved context
            '{"label": "answered"}',
            '{"questions": ["g1", "g2", "g3"]}',
            '{"relevant": [true, false]}',
            labels("supported", "unsupported", "unsupported", "supported"),  # claims vs reference answer
        ],
        embed_vectors=EMBED,
    )
    metrics = compute_generation_metrics(
        client, "q", OUTPUT, ["p1", "p2"], judge_model="judge", embedding_model="embed", reference_answer="the reference"
    )
    assert metrics.faithfulness == pytest.approx(0.75)
    assert (metrics.n_claims, metrics.n_supported, metrics.n_unsupported, metrics.n_no_information) == (4, 3, 1, 0)
    assert (metrics.abstention_label, metrics.abstained) == ("answered", False)
    assert metrics.answer_relevance == pytest.approx(2 / 3)
    assert metrics.context_relevance == pytest.approx(0.5)
    assert metrics.reference_agreement == pytest.approx(0.5)
    assert (metrics.judge_model, metrics.metrics_version) == ("judge", "v2")
    # the reference answer goes only into the last (reference) call, never into the faithfulness call
    assert "the reference" not in client.chat_calls[0]["messages"][1]["content"]
    assert "[1] the reference" in client.chat_calls[4]["messages"][1]["content"]


def test_abstained_answer_gets_no_answer_relevance_call_and_no_zero():
    client = ScriptedClient(
        [labels("no_information", "no_information", "no_information", "no_information"), '{"label": "abstained"}', '{"relevant": [false, false]}'],
        embed_vectors=EMBED,
    )
    metrics = compute_generation_metrics(client, "q", OUTPUT, ["p1", "p2"], judge_model="judge", embedding_model="embed")
    assert metrics.abstained is True
    assert math.isnan(metrics.answer_relevance)  # undefined, not 0.0
    assert math.isnan(metrics.faithfulness)  # nothing factual was claimed
    assert metrics.context_relevance == 0.0
    assert metrics.reference_agreement is None  # reference check switched off
    assert client.embed_calls == [] and len(client.chat_calls) == 3


def test_partial_answer_still_gets_answer_relevance():
    client = ScriptedClient(
        [labels("supported", "supported", "no_information", "supported"), '{"label": "partial"}', '{"questions": ["g1", "g1", "g1"]}', '{"relevant": [true, true]}'],
        embed_vectors=EMBED,
    )
    metrics = compute_generation_metrics(client, "q", OUTPUT, ["p1", "p2"], judge_model="judge", embedding_model="embed")
    assert (metrics.abstention_label, metrics.abstained) == ("partial", False)
    assert metrics.answer_relevance == pytest.approx(1.0)
    assert metrics.faithfulness == pytest.approx(1.0)  # 3 of 3 factual claims


def test_no_context_baseline_has_undefined_faithfulness_and_context_relevance():
    client = ScriptedClient(['{"label": "answered"}', '{"questions": ["g1", "g1", "g1"]}', labels(*["supported"] * 4)], embed_vectors=EMBED)
    metrics = compute_generation_metrics(
        client, "q", OUTPUT, [], judge_model="judge", embedding_model="embed", reference_answer="the reference"
    )
    assert math.isnan(metrics.faithfulness) and math.isnan(metrics.context_relevance)
    assert metrics.reference_agreement == 1.0
    assert metrics.answer_relevance == pytest.approx(1.0)


def test_an_answer_with_no_extractable_claims_is_an_error():
    empty = parse_generation_output("## Summary\n\n## Key Points\n\n## Disclaimer\nd\n", {})
    with pytest.raises(MalformedJudgeResponse, match="no claims"):
        compute_generation_metrics(ScriptedClient([]), "q", empty, ["p"], judge_model="judge", embedding_model="embed")


def test_judge_prompts_snapshot():
    """The judge prompts define the generation metrics. If one is edited this
    fails until the new hash is written here on purpose; the same hash is
    logged with every result row and is part of the run fingerprint."""
    from evaluation.generation_metrics import judge_prompts_hash

    assert judge_prompts_hash() == "3e0d109812af789a6760820aed10e751412c0d77aaaee0f58353ee9ffafba838"

"""Negative controls for the generation metrics, run against the REAL judge
model (pytest --run-api). If these fail, the generation metrics cannot tell a
good answer from a bad one and every generation result in the thesis is void.

Each case pairs a real MedQuAD question from the query set with a short
passage stating the corpus answer to it, and three hand-written answers:
  (i)   grounded     -- every claim is stated in the passage;
  (ii)  hallucinated -- half the claims are from the passage, half are invented
                        specifics (numbers, drugs, causes) that are not in it;
  (iii) off-topic    -- fluent, true-sounding statements about a different subject.
plus an abstaining answer ("the context does not say").

Expected orderings:
  faithfulness          grounded > hallucinated > off-topic, grounded near 1, off-topic near 0
  answer relevance      grounded and hallucinated (both on-topic) > off-topic
  context relevance     the real passage > an unrelated passage
  abstention            only the abstaining answer is labelled `abstained`
  reference agreement   grounded > hallucinated

Cost: about 60 judge calls, under US$0.02 at gpt-4o-mini prices. The session
is capped by --api-max-usd (default $0.50).
"""

from __future__ import annotations

import math
import statistics

import pytest

from evaluation.generation_metrics import compute_generation_metrics
from generation.budget import SpendTracker
from generation.client import OpenAIClient
from generation.parsing import parse_generation_output
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.api


def answer(summary: str, *key_points: str) -> str:
    points = "\n".join(f"{i}. {p} [1]" for i, p in enumerate(key_points, start=1))
    return f"## Summary\n{summary}\n\n## Key Points\n{points}\n\n## Disclaimer\nThis is not a substitute for professional medical advice.\n"


CASES = [
    {
        "name": "la_crosse_prevention",
        "question": "How to prevent La Crosse Encephalitis ?",
        "passage": (
            "There is no vaccine against La Crosse encephalitis virus (LACV). Reducing exposure to mosquito bites is the "
            "best defense against getting infected with LACV or other mosquito-borne viruses. Use insect repellent "
            "containing DEET, picaridin, IR3535 or oil of lemon eucalyptus on exposed skin and/or clothing. Wear long "
            "sleeves, pants and socks when weather permits. Have secure, intact screens on windows and doors to keep "
            "mosquitoes out. Eliminate mosquito breeding sites by emptying standing water from flower pots, buckets, "
            "barrels, and tires. Fill tree holes in or around your yard with soil."
        ),
        "grounded": answer(
            "La Crosse encephalitis is prevented by reducing exposure to mosquito bites, because there is no vaccine against the virus.",
            "There is no vaccine against La Crosse encephalitis virus.",
            "Insect repellent containing DEET, picaridin, IR3535 or oil of lemon eucalyptus can be used on exposed skin and clothing.",
            "Wearing long sleeves, pants and socks when weather permits reduces exposure to bites.",
            "Emptying standing water from flower pots, buckets, barrels and tires eliminates mosquito breeding sites.",
        ),
        "hallucinated": answer(
            "La Crosse encephalitis is prevented by reducing exposure to mosquito bites and by a two-dose vaccine given to children under five.",
            "Insect repellent containing DEET or picaridin can be used on exposed skin and clothing.",
            "A two-dose vaccine licensed in 2014 gives about 92 percent protection against La Crosse encephalitis virus.",
            "Emptying standing water from flower pots, buckets and tires eliminates mosquito breeding sites.",
            "Taking 500 mg of vitamin B1 daily makes people unattractive to mosquitoes and is recommended by the CDC.",
        ),
    },
    {
        "name": "histidinemia_inheritance",
        "question": "Is histidinemia inherited ?",
        "passage": (
            "This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each "
            "cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy "
            "of the mutated gene, but they typically do not show signs and symptoms of the condition."
        ),
        "grounded": answer(
            "Yes. Histidinemia is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations.",
            "The condition is inherited in an autosomal recessive pattern.",
            "Both copies of the gene in each cell have mutations in an affected person.",
            "Each parent of an affected individual carries one copy of the mutated gene.",
            "Parents who carry one copy typically do not show signs and symptoms of the condition.",
        ),
        "hallucinated": answer(
            "Yes. Histidinemia is inherited in an autosomal recessive pattern and affects roughly 1 in 800 newborns in Japan.",
            "The condition is inherited in an autosomal recessive pattern.",
            "It is caused by mutations in the HAL gene located on chromosome 17.",
            "Each parent of an affected individual carries one copy of the mutated gene.",
            "Carriers have a 40 percent lifetime risk of developing seizures by the age of thirty.",
        ),
    },
    {
        "name": "kidney_stone_symptoms",
        "question": "What are the symptoms of Kidney Stones in Adults ?",
        "passage": (
            "People with kidney stones may have pain while urinating, see blood in the urine, or feel a sharp pain in the "
            "back or lower abdomen. The pain may last for a short or long time. People may experience nausea and vomiting "
            "with the pain. However, people who have small stones that pass easily through the urinary tract may not have "
            "symptoms at all."
        ),
        "grounded": answer(
            "Kidney stones can cause pain while urinating, blood in the urine, and sharp pain in the back or lower abdomen, although small stones may cause no symptoms.",
            "People with kidney stones may have pain while urinating.",
            "They may see blood in the urine.",
            "They may feel a sharp pain in the back or lower abdomen, sometimes with nausea and vomiting.",
            "People with small stones that pass easily may not have symptoms at all.",
        ),
        "hallucinated": answer(
            "Kidney stones can cause pain while urinating and blood in the urine, and in most adults they also cause a fever above 39 degrees Celsius.",
            "People with kidney stones may have pain while urinating.",
            "About 70 percent of adults with kidney stones develop a high fever and yellowing of the skin.",
            "They may feel a sharp pain in the back or lower abdomen.",
            "Symptoms usually begin exactly two weeks after the stone forms and are worse in the left kidney.",
        ),
    },
]

OFF_TOPIC = answer(
    "The Great Barrier Reef is the largest coral reef system in the world and lies off the coast of Queensland, Australia.",
    "The reef stretches for more than 2,300 kilometres.",
    "It is made up of nearly 3,000 individual reefs and about 900 islands.",
    "Coral bleaching occurs when sea temperatures stay too high for too long.",
    "The reef was listed as a World Heritage Site in 1981.",
)
ABSTAIN = answer(
    "The provided context does not contain information that answers this question.",
    "The context passages do not mention this topic.",
    "No information about this question is provided in the passages.",
)
UNRELATED_PASSAGE = (
    "Lease agreements for commercial property usually run for three to ten years. The tenant pays a fixed monthly "
    "rent plus a share of the building's maintenance costs, and may negotiate an option to renew at the end of the term."
)


@pytest.fixture(scope="module")
def judge(request, sweep_config):
    # Spend goes to the project ledger, so the overall cap sees it; this session
    # may add at most --api-max-usd on top of what is already there.
    ledger = REPO_ROOT / sweep_config.budget.ledger_path
    already_spent = SpendTracker(max_usd=float("inf"), ledger_path=ledger).spent_usd
    session_cap = request.config.getoption("--api-max-usd")
    budget = SpendTracker(max_usd=min(already_spent + session_cap, sweep_config.budget.max_usd), ledger_path=ledger)
    client = OpenAIClient(budget=budget)
    settings = sweep_config.generation
    judge_model = request.config.getoption("--judge-model") or settings.effective_judge_model

    def score(question: str, answer_text: str, passages: list[str], reference: str | None = None):
        output = parse_generation_output(answer_text, {1: "p#0"})
        return compute_generation_metrics(
            client,
            question,
            output,
            passages,
            judge_model=judge_model,
            embedding_model=settings.embedding_model,
            num_questions=sweep_config.generation_metrics.answer_relevance_num_questions,
            reference_answer=reference,
            seed=1,
        )

    yield score
    budget.save()
    print(f"\n[negative controls] judge={judge_model} spent ${budget.spent_usd - already_spent:.4f}")


@pytest.fixture(scope="module")
def scores(judge):
    table = {}
    for case in CASES:
        table[case["name"]] = {
            "grounded": judge(case["question"], case["grounded"], [case["passage"]], case["passage"]),
            "hallucinated": judge(case["question"], case["hallucinated"], [case["passage"]], case["passage"]),
            "off_topic": judge(case["question"], OFF_TOPIC, [case["passage"]], case["passage"]),
            "abstain": judge(case["question"], ABSTAIN, [case["passage"]], case["passage"]),
            "unrelated_context": judge(case["question"], case["grounded"], [UNRELATED_PASSAGE]),
        }
    print("\ncase                       answer          faith  labels(S/U/N)  abstention  ans_rel  ctx_rel  ref_agree")
    for name, row in table.items():
        for kind, m in row.items():
            print(
                f"{name:26s} {kind:15s} {m.faithfulness:5.2f}  {m.n_supported}/{m.n_unsupported}/{m.n_no_information}"
                f"          {m.abstention_label:10s} {m.answer_relevance:7.3f}  {m.context_relevance:7.2f}  "
                f"{(m.reference_agreement if m.reference_agreement is not None else float('nan')):7.2f}"
            )
    return table


@pytest.mark.parametrize("case", [c["name"] for c in CASES])
def test_faithfulness_ranks_grounded_above_hallucinated_above_off_topic(scores, case):
    row = scores[case]
    assert row["grounded"].faithfulness >= 0.85  # 5 or 6 claims, all in the passage
    assert row["grounded"].faithfulness > row["hallucinated"].faithfulness + 0.2
    assert 0.2 <= row["hallucinated"].faithfulness <= 0.75  # about half the claims are invented
    assert row["off_topic"].faithfulness <= 0.15  # true statements about a coral reef are not supported by a medical passage
    assert row["hallucinated"].faithfulness > row["off_topic"].faithfulness


@pytest.mark.parametrize("case", [c["name"] for c in CASES])
def test_the_invented_claims_are_the_ones_labelled_unsupported(scores, case):
    """Claim order in the hallucinated answer: [summary (mixed), true, INVENTED, true, INVENTED]."""
    labels = scores[case]["hallucinated"].claim_labels
    assert len(labels) == 5
    assert labels[2] == "unsupported" and labels[4] == "unsupported"
    assert labels[1] == "supported" and labels[3] == "supported"


@pytest.mark.parametrize("case", [c["name"] for c in CASES])
def test_answer_relevance_separates_on_topic_from_off_topic(scores, case):
    row = scores[case]
    assert row["grounded"].answer_relevance > row["off_topic"].answer_relevance + 0.25
    assert row["hallucinated"].answer_relevance > row["off_topic"].answer_relevance + 0.25
    assert row["grounded"].answer_relevance > 0.5
    assert row["off_topic"].answer_relevance < 0.35


@pytest.mark.parametrize("case", [c["name"] for c in CASES])
def test_abstention_is_detected_and_is_not_scored_as_faithful_or_as_zero(scores, case):
    row = scores[case]
    assert row["abstain"].abstention_label == "abstained"
    assert math.isnan(row["abstain"].answer_relevance)  # undefined, not 0
    assert math.isnan(row["abstain"].faithfulness)  # "the context does not say" asserts no facts: not 1.0
    assert row["abstain"].n_no_information == row["abstain"].n_claims
    assert row["grounded"].abstention_label == "answered"
    assert row["hallucinated"].abstention_label in ("answered", "partial")
    assert row["grounded"].n_no_information == 0


@pytest.mark.parametrize("case", [c["name"] for c in CASES])
def test_context_relevance_separates_the_real_passage_from_an_unrelated_one(scores, case):
    row = scores[case]
    assert row["grounded"].context_relevance == 1.0
    assert row["unrelated_context"].context_relevance == 0.0


def test_a_correct_answer_is_unfaithful_to_a_context_that_does_not_contain_it(scores):
    """Faithfulness is about the context, not about truth: the same correct
    answers, shown next to a lease-agreement passage, must score near 0.

    Tolerance, stated plainly: none of the 15 claims is supported by that
    passage, so the right count of `supported` labels is 0. The gpt-4o-mini
    judge was observed to label one of them -- "There is no vaccine against La
    Crosse encephalitis virus" -- as supported in roughly half of otherwise
    identical runs (temperature 0, same seed). This test therefore allows at
    most 1 wrong label in 15 (and none in any case but that one); the same
    weakness on "there is no X" claims is listed in TESTING.md."""
    rows = {c["name"]: scores[c["name"]]["unrelated_context"] for c in CASES}
    wrongly_supported = sum(row.n_supported for row in rows.values())
    assert sum(row.n_claims for row in rows.values()) == 15
    assert wrongly_supported <= 1, {name: row.claim_labels for name, row in rows.items()}
    assert rows["histidinemia_inheritance"].faithfulness == 0.0
    assert rows["kidney_stone_symptoms"].faithfulness == 0.0
    assert all(row.n_no_information == 0 for row in rows.values())  # factual claims, not statements about the context


@pytest.mark.parametrize("case", [c["name"] for c in CASES])
def test_reference_agreement_ranks_grounded_above_hallucinated(scores, case):
    row = scores[case]
    assert row["grounded"].reference_agreement >= 0.85
    assert row["grounded"].reference_agreement > row["hallucinated"].reference_agreement + 0.2


def test_mean_separation_across_cases_is_large(scores):
    """A summary number for the thesis: how far apart the three answer types sit on average."""
    mean = lambda kind, field: statistics.mean(getattr(scores[c["name"]][kind], field) for c in CASES)  # noqa: E731
    assert mean("grounded", "faithfulness") - mean("hallucinated", "faithfulness") > 0.35
    assert mean("hallucinated", "faithfulness") - mean("off_topic", "faithfulness") > 0.25

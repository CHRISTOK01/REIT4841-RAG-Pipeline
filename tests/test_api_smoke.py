"""Real-API smoke test (pytest --run-api): three real queries through the real
pipeline -- real embeddings, real FAISS/BM25 retrieval, real generation, real
judge -- on the curated KB, plus the three baselines.

Cost: 3 queries x (3 retrievers + oracle + random) + 3 no-retrieval rows = 18
records, about US$0.02. The run is refused before any call if the estimate
would exceed --api-max-usd, and the spend tracker enforces the same cap while
it runs.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from analysis.aggregate import load_standard_results
from data.query_set import save_query_set_jsonl
from experiments.config import BudgetConfig, RobustnessConfig
from experiments.cost_estimate import estimate_sweep
from experiments.results_store import ResultsStore
from experiments.runner import run_sweep
from generation.budget import SpendTracker
from generation.client import OpenAIClient
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.api

N_QUERIES = 3


@pytest.fixture(scope="module")
def smoke(request, sweep_config, queries, real_retrievers, tmp_path_factory):
    out = tmp_path_factory.mktemp("api_smoke")
    save_query_set_jsonl(queries[:N_QUERIES], out / "query_set.jsonl")
    ledger = REPO_ROOT / sweep_config.budget.ledger_path
    already_spent = SpendTracker(max_usd=float("inf"), ledger_path=ledger).spent_usd
    session_cap = request.config.getoption("--api-max-usd")
    cap = min(already_spent + session_cap, sweep_config.budget.max_usd)
    config = replace(
        sweep_config,
        kb_variant_names=["curated"],
        robustness=RobustnessConfig(enabled=False),
        query_set_path=out / "query_set.jsonl",
        results_path=out / "results.jsonl",
        processed_data_dir=REPO_ROOT / sweep_config.processed_data_dir,
        artifacts_root=REPO_ROOT / sweep_config.artifacts_root,
        embedding_cache_path=REPO_ROOT / sweep_config.embedding_cache_path,
        max_workers=4,
        budget=BudgetConfig(max_usd=cap, ledger_path=ledger),
    )
    estimate = estimate_sweep(config)
    print(f"\n[api smoke] estimate: {estimate}")
    assert estimate.pending_records == 18
    assert estimate.total_usd < session_cap, "estimated cost exceeds --api-max-usd; nothing was called"

    budget = SpendTracker(max_usd=cap, ledger_path=ledger)
    summary = run_sweep(config, OpenAIClient(budget=budget))
    spent = budget.spent_usd - already_spent
    print(f"[api smoke] actual spend: ${spent:.4f}")
    return config, summary, estimate, spent


def test_dry_run_estimate_makes_no_call_and_is_within_a_factor_of_two_of_the_real_cost(smoke):
    _, _, estimate, spent = smoke
    assert 0 < spent < 0.25
    assert estimate.total_usd / 2 <= spent <= estimate.total_usd * 2


def test_real_run_completes_every_row_without_errors(smoke):
    config, summary, _, _ = smoke
    df = load_standard_results(config.results_path)
    assert summary.errors == 0
    assert len(df) == 18
    assert set(df.groupby("retrieval_method").size()) == {N_QUERIES}


def test_the_api_reports_the_pinned_model_snapshot(smoke):
    config, _, _, _ = smoke
    for record in ResultsStore(config.results_path).iter_records():
        generation = record["generation"]
        assert generation["model"] == "gpt-4o-mini-2024-07-18"
        assert generation["returned_model"] == "gpt-4o-mini-2024-07-18"  # the API did not substitute another snapshot
        assert generation["finish_reason"] == "stop"
        assert generation["prompt_tokens"] > 100 and generation["completion_tokens"] > 20


def test_real_answers_have_the_required_structure_and_cite_only_passages_they_were_given(smoke):
    config, _, _, _ = smoke
    for record in ResultsStore(config.results_path).iter_records():
        output = record["generation"]["output"]
        assert output["summary"] and output["key_points"] and output["disclaimer"]
        n_passages = len(record["retrieved"])
        for point in output["key_points"]:
            assert all(1 <= n <= n_passages for n in point["cited_passage_numbers"]), record["run_key"]


def test_real_metric_values_are_in_range_and_undefined_where_they_should_be(smoke):
    config, _, _, _ = smoke
    for record in ResultsStore(config.results_path).iter_records():
        metrics = record["generation_metrics"]
        for name in ("faithfulness", "answer_relevance", "context_relevance", "reference_agreement"):
            value = metrics[name]
            assert math.isnan(value) or 0.0 <= value <= 1.0, (record["run_key"], name, value)
        assert metrics["abstention_label"] in ("answered", "partial", "abstained")
        assert math.isnan(metrics["answer_relevance"]) == (metrics["abstention_label"] == "abstained")
        if record["retrieval_method"] == "none":
            assert math.isnan(metrics["faithfulness"]) and math.isnan(metrics["context_relevance"])
        else:
            assert not math.isnan(metrics["context_relevance"])
            assert len(metrics["context_labels"]) == len(record["retrieved"])
            assert len(metrics["claim_labels"]) == metrics["n_claims"]


def test_oracle_context_is_judged_more_relevant_than_random_context(smoke):
    config, _, _, _ = smoke
    df = load_standard_results(config.results_path)
    means = df.groupby("retrieval_method")[["context_relevance", "abstained", "reference_agreement"]].mean()
    assert means.loc["oracle", "context_relevance"] > means.loc["random", "context_relevance"] + 0.5
    assert means.loc["random", "abstained"] > means.loc["oracle", "abstained"]  # nothing useful in random chunks

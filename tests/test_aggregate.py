import math

import pytest

from analysis.aggregate import (
    aggregate_robustness_deltas,
    aggregate_standard_metrics,
    load_robustness_results,
    load_standard_results,
)
from experiments.results_store import ResultsStore


def standard_record(kb_variant, retrieval_method, query_id, precision, recall, mrr, ndcg, faithfulness):
    return {
        "record_type": "standard",
        "run_key": f"{kb_variant}:{retrieval_method}:{query_id}",
        "kb_variant": kb_variant,
        "retrieval_method": retrieval_method,
        "query_id": query_id,
        "question_type": "information",
        "top_k": 5,
        "retrieval_metrics": {"precision_at_k": precision, "recall_at_k": recall, "mrr": mrr, "ndcg_at_k": ndcg},
        "generation_metrics": {"faithfulness": faithfulness, "answer_relevance": 0.8, "context_relevance": 0.9},
    }


def robustness_record(kb_variant, retrieval_method, query_id, perturbation, faithfulness_delta):
    return {
        "record_type": "robustness",
        "run_key": f"{kb_variant}:{retrieval_method}:{query_id}:robustness:{perturbation}",
        "kb_variant": kb_variant,
        "retrieval_method": retrieval_method,
        "query_id": query_id,
        "perturbation": perturbation,
        "noise_chunk_id": None,
        "baseline_generation_metrics": {"faithfulness": 1.0, "answer_relevance": 1.0, "context_relevance": 1.0},
        "perturbed_generation_metrics": {
            "faithfulness": 1.0 + faithfulness_delta,
            "answer_relevance": 1.0,
            "context_relevance": 1.0,
        },
        "faithfulness_delta": faithfulness_delta,
        "answer_relevance_delta": 0.0,
        "context_relevance_delta": 0.0,
    }


def test_load_standard_results_flattens_nested_metrics(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append(standard_record("curated", "dense", "q1", 0.5, 1.0, 1.0, 1.0, 0.9))
    store.append({"record_type": "error", "run_key": "x", "kb_variant": "curated"})  # must be ignored

    df = load_standard_results(tmp_path / "results.jsonl")

    assert len(df) == 1
    assert df.iloc[0]["kb_variant"] == "curated"
    assert df.iloc[0]["precision_at_k"] == 0.5
    assert df.iloc[0]["faithfulness"] == 0.9


def test_load_standard_results_empty_file_has_expected_columns(tmp_path):
    df = load_standard_results(tmp_path / "does_not_exist.jsonl")
    assert list(df.columns) == [
        "kb_variant",
        "retrieval_method",
        "query_id",
        "question_type",
        "top_k",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "faithfulness",
        "answer_relevance",
        "context_relevance",
    ]
    assert df.empty


def test_load_standard_results_preserves_nan_for_undefined_metrics(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append(standard_record("reduced", "dense", "q1", 0.0, math.nan, math.nan, math.nan, 0.0))
    df = load_standard_results(tmp_path / "results.jsonl")
    assert math.isnan(df.iloc[0]["recall_at_k"])


def test_aggregate_standard_metrics_means_per_group_and_skips_nan(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append(standard_record("curated", "dense", "q1", 1.0, 1.0, 1.0, 1.0, 1.0))
    store.append(standard_record("curated", "dense", "q2", 0.0, 0.0, 0.0, 0.0, 0.0))
    store.append(standard_record("reduced", "dense", "q3", 0.0, math.nan, math.nan, math.nan, 0.0))

    df = load_standard_results(tmp_path / "results.jsonl")
    agg = aggregate_standard_metrics(df)

    curated_row = agg[agg["kb_variant"] == "curated"].iloc[0]
    assert curated_row["precision_at_k"] == 0.5
    assert curated_row["n"] == 2

    reduced_row = agg[agg["kb_variant"] == "reduced"].iloc[0]
    assert reduced_row["precision_at_k"] == 0.0
    assert math.isnan(reduced_row["recall_at_k"])  # only NaN input -> mean is NaN, not 0
    assert reduced_row["n"] == 1


def test_aggregate_standard_metrics_per_metric_n_differs_from_total_n_when_some_rows_are_nan(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append(standard_record("reduced", "dense", "q1", 1.0, 1.0, 1.0, 1.0, 1.0))
    store.append(standard_record("reduced", "dense", "q2", 0.0, math.nan, math.nan, math.nan, 0.0))

    df = load_standard_results(tmp_path / "results.jsonl")
    agg = aggregate_standard_metrics(df)
    row = agg.iloc[0]

    assert row["n"] == 2  # total rows in the group
    assert row["precision_at_k_n"] == 2  # precision is never NaN
    assert row["recall_at_k_n"] == 1  # one of the two rows was NaN for recall
    assert row["recall_at_k"] == 1.0  # mean over the 1 non-NaN value, not diluted by the NaN


def test_aggregate_standard_metrics_empty_input(tmp_path):
    df = load_standard_results(tmp_path / "does_not_exist.jsonl")
    agg = aggregate_standard_metrics(df)
    assert agg.empty
    assert "kb_variant" in agg.columns


def test_load_robustness_results_flattens_deltas(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append(robustness_record("curated", "dense", "q1", "drop_top_relevant", -0.2))
    store.append(standard_record("curated", "dense", "q1", 1.0, 1.0, 1.0, 1.0, 1.0))  # must be ignored

    df = load_robustness_results(tmp_path / "results.jsonl")
    assert len(df) == 1
    assert df.iloc[0]["perturbation"] == "drop_top_relevant"
    assert df.iloc[0]["faithfulness_delta"] == -0.2


def test_aggregate_robustness_deltas_means_per_group(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append(robustness_record("curated", "dense", "q1", "drop_top_relevant", -0.2))
    store.append(robustness_record("curated", "dense", "q2", "drop_top_relevant", -0.4))
    store.append(robustness_record("mixed", "dense", "q3", "inject_noise", -0.1))

    df = load_robustness_results(tmp_path / "results.jsonl")
    agg = aggregate_robustness_deltas(df)

    curated_row = agg[(agg["kb_variant"] == "curated") & (agg["perturbation"] == "drop_top_relevant")].iloc[0]
    assert curated_row["faithfulness_delta"] == pytest.approx(-0.3)
    assert curated_row["n"] == 2


def test_aggregate_robustness_deltas_empty_input(tmp_path):
    df = load_robustness_results(tmp_path / "does_not_exist.jsonl")
    agg = aggregate_robustness_deltas(df)
    assert agg.empty
    assert "kb_variant" in agg.columns

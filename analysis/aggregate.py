"""Load experiment results (results/*.jsonl, from experiments.run_sweep) into
pandas DataFrames and aggregate them into comparison tables.

Uses pandas' default NaN-skipping `.mean()`, which matters here: some
per-query metrics are legitimately NaN (see evaluation/retrieval_metrics.py)
rather than 0.0, and skipping them from the mean is the correct behavior --
it excludes "there was nothing to measure" rows instead of dragging the
average down as if retrieval had failed on them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.results_store import ResultsStore

STANDARD_METRIC_COLUMNS = [
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
]

ROBUSTNESS_DELTA_COLUMNS = ["faithfulness_delta", "answer_relevance_delta", "context_relevance_delta"]


def load_standard_results(results_path: Path) -> pd.DataFrame:
    """One row per (kb_variant, retrieval_method, query) standard run."""
    rows = []
    for record in ResultsStore(results_path).iter_records():
        if record.get("record_type") != "standard":
            continue
        retrieval_metrics = record.get("retrieval_metrics") or {}
        generation_metrics = record.get("generation_metrics") or {}
        rows.append(
            {
                "kb_variant": record["kb_variant"],
                "retrieval_method": record["retrieval_method"],
                "query_id": record["query_id"],
                "question_type": record.get("question_type"),
                "top_k": record.get("top_k"),
                "precision_at_k": retrieval_metrics.get("precision_at_k"),
                "recall_at_k": retrieval_metrics.get("recall_at_k"),
                "mrr": retrieval_metrics.get("mrr"),
                "ndcg_at_k": retrieval_metrics.get("ndcg_at_k"),
                "faithfulness": generation_metrics.get("faithfulness"),
                "answer_relevance": generation_metrics.get("answer_relevance"),
                "context_relevance": generation_metrics.get("context_relevance"),
            }
        )
    columns = ["kb_variant", "retrieval_method", "query_id", "question_type", "top_k", *STANDARD_METRIC_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def load_robustness_results(results_path: Path) -> pd.DataFrame:
    """One row per (kb_variant, retrieval_method, query, perturbation) robustness run."""
    rows = []
    for record in ResultsStore(results_path).iter_records():
        if record.get("record_type") != "robustness":
            continue
        baseline = record["baseline_generation_metrics"]
        perturbed = record["perturbed_generation_metrics"]
        rows.append(
            {
                "kb_variant": record["kb_variant"],
                "retrieval_method": record["retrieval_method"],
                "query_id": record["query_id"],
                "perturbation": record["perturbation"],
                "noise_chunk_id": record.get("noise_chunk_id"),
                "baseline_faithfulness": baseline["faithfulness"],
                "perturbed_faithfulness": perturbed["faithfulness"],
                "faithfulness_delta": record["faithfulness_delta"],
                "baseline_answer_relevance": baseline["answer_relevance"],
                "perturbed_answer_relevance": perturbed["answer_relevance"],
                "answer_relevance_delta": record["answer_relevance_delta"],
                "baseline_context_relevance": baseline["context_relevance"],
                "perturbed_context_relevance": perturbed["context_relevance"],
                "context_relevance_delta": record["context_relevance_delta"],
            }
        )
    columns = [
        "kb_variant",
        "retrieval_method",
        "query_id",
        "perturbation",
        "noise_chunk_id",
        "baseline_faithfulness",
        "perturbed_faithfulness",
        "faithfulness_delta",
        "baseline_answer_relevance",
        "perturbed_answer_relevance",
        "answer_relevance_delta",
        "baseline_context_relevance",
        "perturbed_context_relevance",
        "context_relevance_delta",
    ]
    return pd.DataFrame(rows, columns=columns)


def _aggregate_with_per_metric_counts(
    df: pd.DataFrame, group_by: list[str], metric_columns: list[str]
) -> pd.DataFrame:
    """Mean (NaN-skipping) per metric, PLUS a `<metric>_n` column giving that
    metric's own non-NaN sample size -- not just the group's total row count.
    These can differ: e.g. precision_at_k is always defined, but recall_at_k /
    mrr / ndcg_at_k are NaN for a query with zero relevant chunks (see
    evaluation/retrieval_metrics.py), so their effective N within the same
    group can be smaller than the group's row count. A single shared `n`
    would silently overstate the sample size behind those specific means.
    """
    if df.empty:
        columns = [*group_by]
        for metric in metric_columns:
            columns += [metric, f"{metric}_n"]
        columns.append("n")
        return pd.DataFrame(columns=columns)

    grouped_mean = df.groupby(group_by)[metric_columns].mean()
    grouped_count = df.groupby(group_by)[metric_columns].count()
    grouped_count.columns = [f"{c}_n" for c in grouped_count.columns]
    result = pd.concat([grouped_mean, grouped_count], axis=1)
    result["n"] = df.groupby(group_by).size()
    return result.reset_index()


def aggregate_standard_metrics(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """Mean (NaN-skipping) + per-metric non-NaN count, default kb_variant x retrieval_method."""
    group_by = group_by or ["kb_variant", "retrieval_method"]
    return _aggregate_with_per_metric_counts(df, group_by, STANDARD_METRIC_COLUMNS)


def aggregate_robustness_deltas(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """Mean (NaN-skipping) delta + per-metric non-NaN count, default kb_variant x perturbation."""
    group_by = group_by or ["kb_variant", "perturbation"]
    return _aggregate_with_per_metric_counts(df, group_by, ROBUSTNESS_DELTA_COLUMNS)

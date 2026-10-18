"""Load experiment results (results/*.jsonl, from experiments.run_sweep) into
pandas DataFrames and aggregate them into comparison tables. Everything here
is computed from the results log alone.

Retrieval metric columns, for each of P@k, R@k, MRR, NDCG@k and each label set
(strict: no suffix; lenient: `_lenient`):

- `<metric>_all`  -- every query, with a query that has no relevant chunk in
                     the KB variant counted as 0. The headline number.
- `<metric>`      -- answerable queries only (NaN for the others), isolating
                     ranking quality from KB coverage.
- `coverage`      -- share of queries with at least one relevant chunk.

Generation columns: `abstained` / `partial` (0/1; their means are rates);
`faithfulness`, `answer_relevance` and `reference_agreement` (all NaN for
abstained answers; faithfulness also NaN when no factual claim was made);
`context_relevance`. Means skip NaN, and every mean is reported with the
number of queries behind it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from analysis.stats import bootstrap_mean_ci
from experiments.results_store import ResultsStore

RETRIEVAL_METRICS = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]
RETRIEVAL_ALL_COLUMNS = [f"{m}_all" for m in RETRIEVAL_METRICS]
RETRIEVAL_LENIENT_COLUMNS = [f"{m}_lenient" for m in RETRIEVAL_METRICS]
RETRIEVAL_LENIENT_ALL_COLUMNS = [f"{m}_lenient_all" for m in RETRIEVAL_METRICS]
GENERATION_METRIC_COLUMNS = [
    "faithfulness",
    "abstained",
    "partial",
    "answer_relevance",
    "context_relevance",
    "reference_agreement",
]

# Canonical column order of the aggregated comparison table.
STANDARD_METRIC_COLUMNS = [
    "coverage",
    *RETRIEVAL_ALL_COLUMNS,
    *RETRIEVAL_METRICS,
    "coverage_lenient",
    *RETRIEVAL_LENIENT_ALL_COLUMNS,
    *RETRIEVAL_LENIENT_COLUMNS,
    *GENERATION_METRIC_COLUMNS,
]

DELTA_BASES = ["faithfulness", "answer_relevance", "context_relevance", "reference_agreement", "abstained", "recall_at_k"]
ROBUSTNESS_DELTA_COLUMNS = [f"{m}_delta" for m in DELTA_BASES]

# Relative degradation (thesis Eq. 3.13): (baseline - perturbed) / baseline, for
# the generation metrics where higher is better. Positive = the perturbation made
# the score worse. A ratio over a baseline near zero is meaningless, so it is NaN
# when the baseline is below REL_DEGRADATION_MIN_BASELINE; those queries still
# appear in the absolute change, which is -<metric>_delta.
REL_DEGRADATION_BASES = ["faithfulness", "answer_relevance", "context_relevance", "reference_agreement"]
REL_DEGRADATION_MIN_BASELINE = 0.1
ROBUSTNESS_REL_DEGRADATION_COLUMNS = [f"{m}_rel_degradation" for m in REL_DEGRADATION_BASES]


class ResultsIntegrityError(RuntimeError):
    """The results log cannot be trusted as-is (unreadable lines or duplicate rows)."""


def read_records(results_path: Path) -> list[dict]:
    """Every record in the log. Raises if any line is unreadable or any
    run_key appears twice: both would silently distort every table."""
    store = ResultsStore(results_path)
    records = list(store.iter_records())
    if store.malformed_line_count:
        raise ResultsIntegrityError(f"{results_path}: {store.malformed_line_count} unreadable line(s)")
    keys = [r["run_key"] for r in records if "run_key" in r]
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise ResultsIntegrityError(f"{results_path}: duplicate run_key(s), e.g. {duplicates[:3]}")
    return records


def _num(value) -> float:
    return math.nan if value is None else float(value)


def _retrieval_columns(metrics: dict | None, relevant_ids: list, suffix: str) -> dict:
    """Both views of one label set's metrics for one row (see module docstring)."""
    columns: dict[str, float] = {}
    if metrics is None:  # no-retrieval baseline: retrieval metrics do not apply
        for m in RETRIEVAL_METRICS:
            columns[f"{m}{suffix}"] = math.nan
            columns[f"{m}{suffix}_all"] = math.nan
        columns[f"coverage{suffix}"] = math.nan
        return columns
    answerable = len(relevant_ids) > 0
    for m in RETRIEVAL_METRICS:
        value = _num(metrics[m])
        columns[f"{m}{suffix}"] = value if answerable else math.nan
        columns[f"{m}{suffix}_all"] = value if answerable else 0.0
    columns[f"coverage{suffix}"] = 1.0 if answerable else 0.0
    return columns


def _generation_columns(metrics: dict | None) -> dict:
    if not metrics:
        return {m: math.nan for m in GENERATION_METRIC_COLUMNS}
    abstained = metrics["abstention_label"] == "abstained"
    return {
        "faithfulness": math.nan if abstained else _num(metrics["faithfulness"]),
        "abstained": 1.0 if abstained else 0.0,
        "partial": 1.0 if metrics["abstention_label"] == "partial" else 0.0,
        "answer_relevance": _num(metrics["answer_relevance"]),
        "context_relevance": _num(metrics["context_relevance"]),
        # Faithfulness, answer relevance and reference agreement are properties of an
        # answer that was given. An abstention asserts nothing; the judge nevertheless
        # sometimes labels its "the context does not say" sentences as unsupported, which
        # would score a correct refusal as an unfaithful or wrong answer. The raw labels
        # stay in the log; only the reported scores are restricted to given answers.
        "reference_agreement": math.nan if abstained else _num(metrics.get("reference_agreement")),
    }


def load_standard_results(results_path: Path) -> pd.DataFrame:
    """One row per (kb_variant, retrieval_method, query) standard run."""
    rows = []
    for record in read_records(results_path):
        if record.get("record_type") != "standard":
            continue
        generation_metrics = record.get("generation_metrics") or {}
        rows.append(
            {
                "kb_variant": record["kb_variant"],
                "retrieval_method": record["retrieval_method"],
                "query_id": record["query_id"],
                "question_type": record.get("question_type"),
                "top_k": record.get("top_k"),
                "n_retrieved": len(record.get("retrieved") or []),
                **_retrieval_columns(record.get("retrieval_metrics"), record.get("relevant_chunk_ids") or [], ""),
                **_retrieval_columns(
                    record.get("retrieval_metrics_lenient"),
                    record.get("relevant_chunk_ids_lenient") or [],
                    "_lenient",
                ),
                **_generation_columns(generation_metrics),
                "n_claims": _num(generation_metrics.get("n_claims")),
                "n_unsupported": _num(generation_metrics.get("n_unsupported")),
                "n_no_information": _num(generation_metrics.get("n_no_information")),
            }
        )
    columns = [
        "kb_variant",
        "retrieval_method",
        "query_id",
        "question_type",
        "top_k",
        "n_retrieved",
        *STANDARD_METRIC_COLUMNS,
        "n_claims",
        "n_unsupported",
        "n_no_information",
    ]
    return pd.DataFrame(rows, columns=columns)


def relative_degradation(baseline: float, perturbed: float) -> float:
    """(baseline - perturbed) / baseline; NaN if either is undefined or the
    baseline is below REL_DEGRADATION_MIN_BASELINE."""
    if math.isnan(baseline) or math.isnan(perturbed) or baseline < REL_DEGRADATION_MIN_BASELINE:
        return math.nan
    return (baseline - perturbed) / baseline


def load_robustness_results(results_path: Path) -> pd.DataFrame:
    """One row per (kb_variant, retrieval_method, query, perturbation) robustness run."""
    rows = []
    for record in read_records(results_path):
        if record.get("record_type") != "robustness":
            continue
        baseline = _generation_columns(record["baseline_generation_metrics"])
        perturbed = _generation_columns(record["perturbed_generation_metrics"])
        relevant = record.get("relevant_chunk_ids") or []
        baseline["recall_at_k"] = _num((record.get("baseline_retrieval_metrics") or {}).get("recall_at_k")) if relevant else math.nan
        perturbed["recall_at_k"] = _num((record.get("perturbed_retrieval_metrics") or {}).get("recall_at_k")) if relevant else math.nan
        row = {
            "kb_variant": record["kb_variant"],
            "retrieval_method": record["retrieval_method"],
            "query_id": record["query_id"],
            "perturbation": record["perturbation"],
            "changed_chunk_id": record.get("changed_chunk_id"),
            "noise_chunk_id": record.get("noise_chunk_id"),
        }
        for metric in DELTA_BASES:
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"perturbed_{metric}"] = perturbed[metric]
            row[f"{metric}_delta"] = perturbed[metric] - baseline[metric]  # NaN if either side is undefined
        for metric in REL_DEGRADATION_BASES:
            row[f"{metric}_rel_degradation"] = relative_degradation(baseline[metric], perturbed[metric])
        rows.append(row)
    columns = ["kb_variant", "retrieval_method", "query_id", "perturbation", "changed_chunk_id", "noise_chunk_id"]
    for metric in DELTA_BASES:
        columns += [f"baseline_{metric}", f"perturbed_{metric}", f"{metric}_delta"]
    columns += ROBUSTNESS_REL_DEGRADATION_COLUMNS
    return pd.DataFrame(rows, columns=columns)


def _aggregate_with_per_metric_counts(
    df: pd.DataFrame, group_by: list[str], metric_columns: list[str]
) -> pd.DataFrame:
    """Mean (NaN-skipping) per metric, PLUS a `<metric>_n` column giving that
    metric's own non-NaN sample size -- not just the group's total row count."""
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
    """Mean (NaN-skipping) delta and relative degradation + per-metric non-NaN
    count, default kb_variant x perturbation."""
    group_by = group_by or ["kb_variant", "perturbation"]
    return _aggregate_with_per_metric_counts(df, group_by, ROBUSTNESS_DELTA_COLUMNS + ROBUSTNESS_REL_DEGRADATION_COLUMNS)


def summarize_with_uncertainty(
    df: pd.DataFrame, group_by: list[str], metric_columns: list[str], seed: int = 0
) -> pd.DataFrame:
    """Long table: one row per (group, metric) with mean, standard deviation,
    n (non-NaN queries) and a 95% bootstrap confidence interval for the mean."""
    rows = []
    if not df.empty:
        for group_key, group in df.groupby(group_by):
            group_key = group_key if isinstance(group_key, tuple) else (group_key,)
            for metric in metric_columns:
                values = group[metric].dropna()
                ci_low, ci_high = bootstrap_mean_ci(values.to_numpy(), seed=seed)
                rows.append(
                    {
                        **dict(zip(group_by, group_key)),
                        "metric": metric,
                        "mean": values.mean() if len(values) else math.nan,
                        "sd": values.std(ddof=1) if len(values) > 1 else math.nan,
                        "n": len(values),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    }
                )
    return pd.DataFrame(rows, columns=[*group_by, "metric", "mean", "sd", "n", "ci95_low", "ci95_high"])

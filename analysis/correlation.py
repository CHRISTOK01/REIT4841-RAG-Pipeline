"""RQ4: how well do retrieval metrics predict generation metrics, query by query?

For each (retrieval metric, generation metric) pair this reports Spearman's rho
and Kendall's tau-b (the measure used by Salemi & Zamani, 2024) across
individual (configuration, query) rows, with two-sided p-values.

Two scopes are produced:

- `pooled`: every row of the real retrieval methods on every KB variant
  together (the RQ4 headline). The same query appears once per configuration,
  so the rows are not independent; the 95% interval therefore comes from a
  cluster bootstrap that resamples whole queries, and the p-values (which
  assume independent rows) are only indicative.
- one scope per configuration (`<kb_variant>/<retrieval_method>`): each query
  appears once. p-values only, no interval.

Baselines (oracle, random, no retrieval) are left out: they sit at the extremes
of every retrieval metric by construction and would manufacture a correlation
that says nothing about real retrievers.

A pair where either variable is constant (e.g. recall on a configuration where
every query scores 1) has no defined correlation and is reported as NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

REAL_METHODS = ["lexical", "dense", "hybrid"]
CORRELATION_RETRIEVAL_METRICS = ["precision_at_k_all", "recall_at_k_all", "mrr_all", "ndcg_at_k_all"]
CORRELATION_GENERATION_METRICS = [
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "reference_agreement",
    "abstained",
]
CORRELATION_COLUMNS = [
    "scope",
    "retrieval_metric",
    "generation_metric",
    "n",
    "spearman_rho",
    "spearman_p",
    "spearman_ci95_low",
    "spearman_ci95_high",
    "kendall_tau",
    "kendall_p",
    "kendall_ci95_low",
    "kendall_ci95_high",
]
DEFAULT_N_BOOT = 2_000


@dataclass(frozen=True)
class Correlation:
    n: int
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float


def _is_constant(values: np.ndarray) -> bool:
    return values.size == 0 or bool(np.all(values == values[0]))


def rank_correlation(x, y) -> Correlation:
    """Spearman rho and Kendall tau-b of x against y, pairs with a NaN on either
    side dropped. NaN when fewer than 3 pairs remain or either side is constant."""
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("rank correlation needs two equally long, aligned sequences")
    keep = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr, y_arr = x_arr[keep], y_arr[keep]
    n = int(x_arr.size)
    if n < 3 or _is_constant(x_arr) or _is_constant(y_arr):
        return Correlation(n, math.nan, math.nan, math.nan, math.nan)
    spearman = scipy_stats.spearmanr(x_arr, y_arr)
    kendall = scipy_stats.kendalltau(x_arr, y_arr, variant="b")
    return Correlation(n, float(spearman.statistic), float(spearman.pvalue), float(kendall.statistic), float(kendall.pvalue))


def cluster_bootstrap_ci(
    df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str = "query_id", n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> tuple[float, float, float, float]:
    """95% percentile intervals for (Spearman, Kendall) when rows come in
    clusters (one query, several configurations): resample whole clusters with
    replacement. Resamples where a correlation is undefined are skipped.
    Returns (rho_low, rho_high, tau_low, tau_high); NaN if fewer than 2 clusters."""
    data = df[[cluster_col, x_col, y_col]].dropna()
    groups = [g[[x_col, y_col]].to_numpy(dtype=float) for _, g in data.groupby(cluster_col)]
    if len(groups) < 2:
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    rhos, taus = [], []
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[i] for i in picks])
        result = rank_correlation(sample[:, 0], sample[:, 1])
        if not math.isnan(result.spearman_rho):
            rhos.append(result.spearman_rho)
            taus.append(result.kendall_tau)
    if not rhos:
        return math.nan, math.nan, math.nan, math.nan
    rho_low, rho_high = np.percentile(rhos, [2.5, 97.5])
    tau_low, tau_high = np.percentile(taus, [2.5, 97.5])
    return float(rho_low), float(rho_high), float(tau_low), float(tau_high)


def _rows_for_scope(
    scope: str, data: pd.DataFrame, retrieval_metrics, generation_metrics, n_boot: int, seed: int
) -> list[dict]:
    rows = []
    for r_metric in retrieval_metrics:
        for g_metric in generation_metrics:
            result = rank_correlation(data[r_metric].to_numpy(), data[g_metric].to_numpy())
            ci = (math.nan,) * 4
            if n_boot and not math.isnan(result.spearman_rho):
                ci = cluster_bootstrap_ci(data, r_metric, g_metric, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "scope": scope,
                    "retrieval_metric": r_metric,
                    "generation_metric": g_metric,
                    "n": result.n,
                    "spearman_rho": result.spearman_rho,
                    "spearman_p": result.spearman_p,
                    "spearman_ci95_low": ci[0],
                    "spearman_ci95_high": ci[1],
                    "kendall_tau": result.kendall_tau,
                    "kendall_p": result.kendall_p,
                    "kendall_ci95_low": ci[2],
                    "kendall_ci95_high": ci[3],
                }
            )
    return rows


def retrieval_generation_correlations(
    df: pd.DataFrame,
    retrieval_metrics: list[str] | None = None,
    generation_metrics: list[str] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """The RQ4 table described in the module docstring. `df` is the per-query
    table from analysis.aggregate.load_standard_results."""
    retrieval_metrics = retrieval_metrics or CORRELATION_RETRIEVAL_METRICS
    generation_metrics = generation_metrics or CORRELATION_GENERATION_METRICS
    real = df[df["retrieval_method"].isin(REAL_METHODS)] if not df.empty else df
    if real.empty:
        return pd.DataFrame(columns=CORRELATION_COLUMNS)
    rows = _rows_for_scope("pooled", real, retrieval_metrics, generation_metrics, n_boot, seed)
    for (kb, method), group in real.groupby(["kb_variant", "retrieval_method"]):
        rows += _rows_for_scope(f"{kb}/{method}", group, retrieval_metrics, generation_metrics, 0, seed)
    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)

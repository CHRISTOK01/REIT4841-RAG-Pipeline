"""Paired comparisons between configurations, on the same queries.

Two families of comparison are produced for each metric:

- between retrieval methods, within one KB variant (e.g. hybrid vs lexical on curated);
- between KB variants, within one retrieval method (e.g. dense on mixed vs dense on curated).

Each row is a paired test over queries (analysis/stats.py). p-values are
Holm-adjusted within each (family, metric) so that looking at several pairs at
once does not manufacture a "significant" one.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from analysis.stats import holm_adjust, paired_comparison

COMPARISON_COLUMNS = [
    "family",
    "held_fixed",
    "metric",
    "a",
    "b",
    "n_pairs",
    "mean_a",
    "mean_b",
    "mean_diff",
    "ci95_low",
    "ci95_high",
    "wilcoxon_p",
    "p_holm",
    "sd_diff",
    "min_detectable_diff",
    "verdict",
]


def _verdict(ci_low: float, ci_high: float, p_holm: float) -> str:
    if pd.isna(ci_low) or pd.isna(p_holm):
        return "not enough data"
    if (ci_low > 0 or ci_high < 0) and p_holm < 0.05:
        return "difference"
    return "could be noise"


def _pairwise(
    df: pd.DataFrame, vary: str, fixed: str, metrics: list[str], family: str, seed: int
) -> list[dict]:
    rows: list[dict] = []
    for fixed_value, group in df.groupby(fixed):
        wide = {
            metric: group.pivot(index="query_id", columns=vary, values=metric) for metric in metrics
        }
        levels = sorted(group[vary].unique())
        for metric in metrics:
            for a, b in combinations(levels, 2):
                table = wide[metric]
                result = paired_comparison(table[a].to_numpy(), table[b].to_numpy(), seed=seed)
                rows.append(
                    {
                        "family": family,
                        "held_fixed": fixed_value,
                        "metric": metric,
                        "a": a,
                        "b": b,
                        "n_pairs": result.n_pairs,
                        "mean_a": result.mean_a,
                        "mean_b": result.mean_b,
                        "mean_diff": result.mean_diff,
                        "ci95_low": result.ci_low,
                        "ci95_high": result.ci_high,
                        "wilcoxon_p": result.wilcoxon_p,
                        "sd_diff": result.sd_diff,
                        "min_detectable_diff": result.min_detectable_diff,
                    }
                )
    return rows


def paired_comparisons(df: pd.DataFrame, metrics: list[str], seed: int = 0) -> pd.DataFrame:
    """All pairwise comparisons described in the module docstring. `df` is the
    per-query table from analysis.aggregate.load_standard_results."""
    if df.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    rows = _pairwise(df, "retrieval_method", "kb_variant", metrics, "methods_within_kb", seed)
    rows += _pairwise(df, "kb_variant", "retrieval_method", metrics, "kbs_within_method", seed)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    result["p_holm"] = result.groupby(["family", "metric"])["wilcoxon_p"].transform(
        lambda p: holm_adjust(p.tolist())
    )
    result["verdict"] = [
        _verdict(low, high, p) for low, high, p in zip(result["ci95_low"], result["ci95_high"], result["p_holm"])
    ]
    return result[COMPARISON_COLUMNS]

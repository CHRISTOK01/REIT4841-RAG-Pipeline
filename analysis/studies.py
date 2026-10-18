"""The planned analyses of the two studies (thesis Section 4.7), from the
per-query table of analysis.aggregate.load_standard_results.

- Study 1: the three retrieval methods compared on the curated KB.
- Study 2: the three KB variants compared with ONE retrieval method, chosen by
  a rule fixed in advance (`select_study2_method`): the method with the
  highest mean NDCG@k (strict labels, unanswerable queries counted as 0) on
  the curated KB; an exact tie is broken by the higher mean faithfulness, then
  by name so the rule always returns one method.
- Interaction (secondary): does the gap between two methods change when the
  KB is degraded? For each query, (a - b on the degraded KB) - (a - b on
  curated), tested with the same paired bootstrap + Wilcoxon as everything
  else.

Holm adjustment is applied within each planned family and metric (three pairs
for each study), as the analysis plan states. analysis.compare's
`paired_comparisons` remains the exhaustive, more conservative table.
"""

from __future__ import annotations

import math
from itertools import combinations

import pandas as pd

from analysis.compare import COMPARISON_COLUMNS, _holm_and_verdict, _pairwise, _verdict
from analysis.stats import holm_adjust, paired_comparison

REAL_METHODS = ["lexical", "dense", "hybrid"]
KB_VARIANTS = ["curated", "mixed", "reduced"]
REFERENCE_KB = "curated"
SELECTION_METRIC = "ndcg_at_k_all"
TIE_BREAK_METRIC = "faithfulness"
LOWER_IS_BETTER = {"abstained"}

INTERACTION_COLUMNS = [
    "metric",
    "kb_variant",
    "a",
    "b",
    "n_pairs",
    "gap_on_curated",
    "gap_on_kb",
    "change_in_gap",
    "ci95_low",
    "ci95_high",
    "wilcoxon_p",
    "p_holm",
    "verdict",
]


def _real(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["retrieval_method"].isin(REAL_METHODS)]


def select_study2_method(df: pd.DataFrame) -> str:
    """The Study 2 retrieval method under the rule in the module docstring."""
    curated = _real(df)[lambda d: d["kb_variant"] == REFERENCE_KB]
    if curated.empty:
        raise ValueError("no real-method rows on the curated KB to choose a Study 2 method from")
    means = curated.groupby("retrieval_method")[[SELECTION_METRIC, TIE_BREAK_METRIC]].mean()
    ranked = sorted(
        means.index,
        key=lambda m: (
            -means.loc[m, SELECTION_METRIC],
            -(means.loc[m, TIE_BREAK_METRIC] if not math.isnan(means.loc[m, TIE_BREAK_METRIC]) else -math.inf),
            m,
        ),
    )
    return ranked[0]


def planned_comparisons(df: pd.DataFrame, metrics: list[str], study2_method: str | None = None, seed: int = 0) -> pd.DataFrame:
    """Study 1 (methods within curated) and Study 2 (KB variants within the
    chosen method) pairwise tests, Holm-adjusted within (study, metric)."""
    real = _real(df) if not df.empty else df
    if real.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    study2_method = study2_method or select_study2_method(real)
    study1 = real[real["kb_variant"] == REFERENCE_KB]
    study2 = real[(real["retrieval_method"] == study2_method) & real["kb_variant"].isin(KB_VARIANTS)]
    rows = _pairwise(study1, "retrieval_method", "kb_variant", metrics, "study1_methods_on_curated", seed)
    rows += _pairwise(study2, "kb_variant", "retrieval_method", metrics, "study2_kbs_with_selected_method", seed)
    return _holm_and_verdict(rows)


def method_ranks_by_kb(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean of each real method per KB variant and its rank (1 = best; for
    `abstained` lower is better), to see whether the ordering of the methods
    changes as the KB is degraded."""
    real = _real(df)
    rows = []
    for metric in metrics:
        means = real.groupby(["kb_variant", "retrieval_method"])[metric].mean()
        for kb in [k for k in KB_VARIANTS if k in means.index.get_level_values(0)]:
            kb_means = means.loc[kb].dropna()
            ascending = metric in LOWER_IS_BETTER
            ranks = kb_means.rank(ascending=ascending, method="min")
            for method in kb_means.index:
                rows.append(
                    {"metric": metric, "kb_variant": kb, "retrieval_method": method, "mean": float(kb_means[method]), "rank": int(ranks[method])}
                )
    return pd.DataFrame(rows, columns=["metric", "kb_variant", "retrieval_method", "mean", "rank"])


def method_by_kb_interaction(df: pd.DataFrame, metrics: list[str], seed: int = 0) -> pd.DataFrame:
    """Change in the per-query gap between two methods from curated to each
    degraded KB variant (see module docstring). Holm-adjusted within metric."""
    real = _real(df)
    rows = []
    for metric in metrics:
        wide = real.pivot_table(index="query_id", columns=["kb_variant", "retrieval_method"], values=metric, aggfunc="first", dropna=False)
        for kb in [k for k in KB_VARIANTS if k != REFERENCE_KB]:
            for a, b in combinations(REAL_METHODS, 2):
                needed = [(REFERENCE_KB, a), (REFERENCE_KB, b), (kb, a), (kb, b)]
                if any(col not in wide.columns for col in needed):
                    continue
                gap_curated = (wide[(REFERENCE_KB, a)] - wide[(REFERENCE_KB, b)]).to_numpy()
                gap_kb = (wide[(kb, a)] - wide[(kb, b)]).to_numpy()
                result = paired_comparison(gap_kb, gap_curated, seed=seed)
                rows.append(
                    {
                        "metric": metric,
                        "kb_variant": kb,
                        "a": a,
                        "b": b,
                        "n_pairs": result.n_pairs,
                        "gap_on_curated": result.mean_b,
                        "gap_on_kb": result.mean_a,
                        "change_in_gap": result.mean_diff,
                        "ci95_low": result.ci_low,
                        "ci95_high": result.ci_high,
                        "wilcoxon_p": result.wilcoxon_p,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    result["p_holm"] = result.groupby("metric")["wilcoxon_p"].transform(lambda p: holm_adjust(p.tolist()))
    result["verdict"] = [_verdict(lo, hi, p) for lo, hi, p in zip(result["ci95_low"], result["ci95_high"], result["p_holm"])]
    return result[INTERACTION_COLUMNS]

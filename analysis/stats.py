"""Uncertainty and significance for per-query metric values.

Everything here works on per-query numbers for one configuration (or on the
per-query difference between two configurations run on the same queries).
NaN means "undefined for this query" and is dropped, pairwise where two
configurations are compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats

DEFAULT_N_BOOT = 10_000
Z_ALPHA_TWO_SIDED_05 = 1.959964
Z_POWER_80 = 0.841621


def _clean(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = DEFAULT_N_BOOT, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean: resample the
    queries with replacement `n_boot` times and take the middle `confidence`
    share of the resampled means. (NaN, NaN) with fewer than 2 values."""
    array = _clean(values)
    if array.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    resampled_means = array[rng.integers(0, array.size, size=(n_boot, array.size))].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(resampled_means, [tail, 100 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class PairedComparison:
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float  # mean of (a - b) over queries where both are defined
    ci_low: float  # 95% paired-bootstrap CI of mean_diff
    ci_high: float
    wilcoxon_p: float  # two-sided Wilcoxon signed-rank test on the paired differences
    sd_diff: float
    min_detectable_diff: float  # smallest true mean difference this n would detect with 80% power


def paired_comparison(
    a: Sequence[float], b: Sequence[float], n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> PairedComparison:
    """Compare two configurations query by query. `a[i]` and `b[i]` must be
    the same query. Pairs where either value is NaN are dropped."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired comparison needs two equally long, aligned sequences")
    keep = ~(np.isnan(a_arr) | np.isnan(b_arr))
    a_arr, b_arr = a_arr[keep], b_arr[keep]
    n = int(a_arr.size)
    if n < 2:
        return PairedComparison(n, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    diff = a_arr - b_arr
    ci_low, ci_high = bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed)
    if np.all(diff == 0):
        p_value = 1.0  # identical on every query; scipy cannot rank all-zero differences
    else:
        p_value = float(scipy_stats.wilcoxon(a_arr, b_arr, zero_method="wilcox").pvalue)
    sd_diff = float(diff.std(ddof=1))
    return PairedComparison(
        n_pairs=n,
        mean_a=float(a_arr.mean()),
        mean_b=float(b_arr.mean()),
        mean_diff=float(diff.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        wilcoxon_p=p_value,
        sd_diff=sd_diff,
        min_detectable_diff=min_detectable_difference(sd_diff, n),
    )


def min_detectable_difference(sd_diff: float, n_pairs: int) -> float:
    """Smallest true mean paired difference detectable with 80% power at a
    two-sided 5% level (normal approximation): (1.96 + 0.84) * sd / sqrt(n)."""
    if n_pairs < 2:
        return math.nan
    return (Z_ALPHA_TWO_SIDED_05 + Z_POWER_80) * sd_diff / math.sqrt(n_pairs)


def required_pairs(sd_diff: float, target_diff: float) -> float:
    """Number of paired queries needed to detect `target_diff` with 80% power."""
    if target_diff <= 0:
        return math.nan
    return ((Z_ALPHA_TWO_SIDED_05 + Z_POWER_80) * sd_diff / target_diff) ** 2


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate
    across several comparisons). NaN inputs stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    m = len(order)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running_max
    return adjusted.tolist()

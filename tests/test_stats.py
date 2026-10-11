"""Uncertainty and significance helpers, checked against hand calculations,
scipy's own bootstrap, and simulated data with a known answer."""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from analysis.stats import (
    bootstrap_mean_ci,
    holm_adjust,
    min_detectable_difference,
    paired_comparison,
    required_pairs,
)


def test_bootstrap_ci_of_a_constant_is_that_constant():
    assert bootstrap_mean_ci([0.4] * 30) == pytest.approx((0.4, 0.4))


def test_bootstrap_ci_brackets_the_sample_mean_and_ignores_nan():
    values = [0.0, 1.0, 1.0, 0.5, 0.25, math.nan, 0.75, 1.0, 0.0, 0.5, math.nan]
    low, high = bootstrap_mean_ci(values, seed=1)
    mean = np.nanmean(values)
    assert low < mean < high
    assert (low, high) == bootstrap_mean_ci([v for v in values if not math.isnan(v)], seed=1)


def test_bootstrap_ci_is_reproducible_for_a_seed():
    values = np.random.default_rng(0).random(80)
    assert bootstrap_mean_ci(values, seed=5) == bootstrap_mean_ci(values, seed=5)


def test_bootstrap_ci_needs_at_least_two_values():
    assert all(math.isnan(x) for x in bootstrap_mean_ci([0.3]))
    assert all(math.isnan(x) for x in bootstrap_mean_ci([math.nan, math.nan]))


@pytest.mark.parametrize("seed", range(5))
def test_bootstrap_ci_agrees_with_scipy_percentile_bootstrap(seed):
    values = np.random.default_rng(seed).beta(2, 5, size=150)
    ours = bootstrap_mean_ci(values, n_boot=20_000, seed=seed)
    reference = scipy_stats.bootstrap(
        (values,), np.mean, n_resamples=20_000, method="percentile", confidence_level=0.95, random_state=seed
    ).confidence_interval
    # two independent Monte Carlo runs of the same procedure: equal up to resampling noise
    assert ours[0] == pytest.approx(reference.low, abs=0.004)
    assert ours[1] == pytest.approx(reference.high, abs=0.004)


def test_bootstrap_ci_matches_the_normal_theory_interval_on_a_large_sample():
    values = np.random.default_rng(3).normal(0.5, 0.2, size=2000)
    low, high = bootstrap_mean_ci(values, seed=3)
    half_width = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    assert low == pytest.approx(values.mean() - half_width, abs=0.002)
    assert high == pytest.approx(values.mean() + half_width, abs=0.002)


def test_bootstrap_ci_covers_the_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(11)
    covered = 0
    for trial in range(300):
        sample = rng.normal(0.6, 0.25, size=60)
        low, high = bootstrap_mean_ci(sample, n_boot=1000, seed=trial)
        covered += low <= 0.6 <= high
    assert 0.90 <= covered / 300 <= 0.985


# -- paired comparison ---------------------------------------------------------------------


def test_paired_comparison_hand_example():
    a = [0.9, 0.8, 0.7, 0.6, 1.0]
    b = [0.5, 0.6, 0.7, 0.2, 0.9]
    result = paired_comparison(a, b)
    # differences: 0.4, 0.2, 0.0, 0.4, 0.1 -> mean 0.22
    assert result.n_pairs == 5
    assert (result.mean_a, result.mean_b, result.mean_diff) == pytest.approx((0.8, 0.58, 0.22))
    # sample sd of the differences: sqrt(((0.18^2 + 0.02^2 + 0.22^2 + 0.18^2 + 0.12^2) / 4)) = sqrt(0.032) = 0.178885
    assert result.sd_diff == pytest.approx(0.178885, abs=1e-6)
    assert result.ci_low < 0.22 < result.ci_high


def test_pairs_with_a_nan_on_either_side_are_dropped_together():
    a = [1.0, math.nan, 0.5, 0.2, 0.9]
    b = [0.0, 0.3, math.nan, 0.1, 0.4]
    result = paired_comparison(a, b)
    assert result.n_pairs == 3  # queries 0, 3 and 4
    assert result.mean_a == pytest.approx((1.0 + 0.2 + 0.9) / 3)
    assert result.mean_diff == pytest.approx((1.0 + 0.1 + 0.5) / 3)


def test_identical_configurations_are_not_a_difference():
    values = [0.2, 0.9, 0.4, 0.4, 1.0]
    result = paired_comparison(values, values)
    assert result.mean_diff == 0.0 and result.wilcoxon_p == 1.0
    assert (result.ci_low, result.ci_high) == (0.0, 0.0)


def test_wilcoxon_p_value_is_scipy_wilcoxon_on_the_kept_pairs():
    rng = np.random.default_rng(2)
    a, b = rng.random(40), rng.random(40)
    assert paired_comparison(a, b).wilcoxon_p == pytest.approx(scipy_stats.wilcoxon(a, b).pvalue)


def test_a_real_paired_effect_is_detected_and_pure_noise_usually_is_not():
    rng = np.random.default_rng(4)
    base = rng.random(150)
    shifted = paired_comparison(base + 0.1 + rng.normal(0, 0.05, 150), base)
    assert shifted.ci_low > 0 and shifted.wilcoxon_p < 0.001
    assert shifted.mean_diff == pytest.approx(0.1, abs=0.02)

    false_alarms = 0
    for trial in range(200):
        trial_rng = np.random.default_rng(1000 + trial)
        noise = paired_comparison(trial_rng.normal(0, 1, 60), trial_rng.normal(0, 1, 60), n_boot=500, seed=trial)
        false_alarms += noise.wilcoxon_p < 0.05
    assert false_alarms / 200 < 0.10  # nominal rate is 5%


def test_pairing_matters_shuffling_one_side_loses_the_effect_size_precision():
    rng = np.random.default_rng(6)
    base = rng.random(100)
    better = base + 0.03
    paired = paired_comparison(better, base)
    assert paired.sd_diff == pytest.approx(0.0, abs=1e-12)  # every query improved by exactly 0.03
    unpaired = paired_comparison(better, rng.permutation(base))
    assert unpaired.sd_diff > 0.2  # same means, but the per-query link is gone


def test_misaligned_inputs_are_rejected():
    with pytest.raises(ValueError):
        paired_comparison([0.1, 0.2], [0.1])


def test_too_few_pairs_gives_nan_not_a_verdict():
    result = paired_comparison([0.5, math.nan], [math.nan, 0.5])
    assert result.n_pairs == 0 and math.isnan(result.mean_diff) and math.isnan(result.wilcoxon_p)


# -- power -------------------------------------------------------------------------------------


def test_min_detectable_difference_and_required_pairs_hand_values():
    # (1.959964 + 0.841621) * 0.3 / sqrt(100) = 2.801585 * 0.03 = 0.084048
    assert min_detectable_difference(0.3, 100) == pytest.approx(0.084048, abs=1e-6)
    # to detect 0.05 with sd 0.3: (2.801585 * 0.3 / 0.05)^2 = 16.80951^2 = 282.56
    assert required_pairs(0.3, 0.05) == pytest.approx(282.56, abs=0.01)
    # the two are inverses of each other
    assert min_detectable_difference(0.3, 282.56) == pytest.approx(0.05, abs=1e-4)
    assert math.isnan(min_detectable_difference(0.3, 1)) and math.isnan(required_pairs(0.3, 0.0))


def test_min_detectable_difference_delivers_about_80_percent_power_in_simulation():
    rng = np.random.default_rng(9)
    n, sd = 80, 0.3
    effect = min_detectable_difference(sd, n)
    detected = 0
    for _ in range(600):
        diff = rng.normal(effect, sd, n)
        detected += scipy_stats.ttest_1samp(diff, 0).pvalue < 0.05
    assert 0.74 <= detected / 600 <= 0.86


# -- multiple comparisons ----------------------------------------------------------------------


def test_holm_adjustment_hand_example():
    # sorted p: 0.01, 0.03, 0.04 -> x3, x2, x1 = 0.03, 0.06, 0.04 -> made monotone: 0.03, 0.06, 0.06
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjustment_caps_at_one_and_skips_nan():
    adjusted = holm_adjust([0.5, math.nan, 0.6])
    assert adjusted[0] == pytest.approx(1.0) and adjusted[2] == pytest.approx(1.0)  # 0.5 * 2 capped
    assert math.isnan(adjusted[1])
    assert holm_adjust([0.02]) == pytest.approx([0.02])  # one test: unchanged
    assert holm_adjust([]) == []


def test_holm_never_lowers_a_p_value_and_never_exceeds_bonferroni():
    rng = np.random.default_rng(1)
    p = rng.random(12).tolist()
    adjusted = holm_adjust(p)
    assert all(a >= raw - 1e-15 for a, raw in zip(adjusted, p))
    assert all(a <= min(1.0, raw * len(p)) + 1e-15 for a, raw in zip(adjusted, p))

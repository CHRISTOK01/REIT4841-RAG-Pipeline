"""RQ4 rank correlations, checked against hand calculations and against
Spearman / Kendall written out from their definitions in this file."""

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from analysis.correlation import cluster_bootstrap_ci, rank_correlation, retrieval_generation_correlations


def average_ranks(values):
    """1-based ranks, tied values sharing the mean of their positions."""
    values = list(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for position in range(i, j + 1):
            ranks[order[position]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman_by_definition(x, y):
    """Pearson correlation of the average ranks."""
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def kendall_tau_b_by_definition(x, y):
    """(concordant - discordant) / sqrt((n0 - ties_x) (n0 - ties_y)), by counting every pair."""
    concordant = discordant = ties_x = ties_y = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0:
            ties_x += 1
        if dy == 0:
            ties_y += 1
        if dx * dy > 0:
            concordant += 1
        elif dx * dy < 0:
            discordant += 1
    n0 = len(x) * (len(x) - 1) / 2
    return (concordant - discordant) / math.sqrt((n0 - ties_x) * (n0 - ties_y))


def test_hand_worked_example():
    # one swapped pair: d = (0, 1, -1, 0), rho = 1 - 6 * 2 / (4 * 15) = 0.8;
    # 5 concordant and 1 discordant of 6 pairs, tau = 4 / 6
    result = rank_correlation([1, 2, 3, 4], [1, 3, 2, 4])
    assert result.n == 4
    assert result.spearman_rho == pytest.approx(0.8)
    assert result.kendall_tau == pytest.approx(4 / 6)


def test_perfect_and_reversed_orderings():
    x = [0.1, 0.4, 0.2, 0.9, 0.7]
    assert rank_correlation(x, [v * 10 + 3 for v in x]).spearman_rho == pytest.approx(1.0)
    assert rank_correlation(x, [-v for v in x]).kendall_tau == pytest.approx(-1.0)


@pytest.mark.parametrize("seed", range(10))
def test_matches_the_definitions_on_tied_data_like_the_real_metrics(seed):
    rng = np.random.default_rng(seed)
    # retrieval metrics take few distinct values (e.g. 0, 0.2, ..., 1), so ties are the normal case
    x = rng.integers(0, 6, size=60) / 5
    y = np.clip(x + rng.normal(0, 0.4, size=60), 0, 1).round(1)
    result = rank_correlation(x, y)
    assert result.spearman_rho == pytest.approx(spearman_by_definition(x, y), abs=1e-12)
    assert result.kendall_tau == pytest.approx(kendall_tau_b_by_definition(x, y), abs=1e-12)


def test_pairs_with_a_nan_are_dropped_not_counted_as_zero():
    with_nan = rank_correlation([1, 2, 3, 4, 5], [1, 2, math.nan, 4, 0])
    without = rank_correlation([1, 2, 4, 5], [1, 2, 4, 0])
    assert with_nan.n == 4
    assert with_nan.spearman_rho == pytest.approx(without.spearman_rho)


def test_undefined_when_a_side_is_constant_or_too_few_pairs():
    assert math.isnan(rank_correlation([1, 1, 1, 1], [1, 2, 3, 4]).spearman_rho)
    assert math.isnan(rank_correlation([1, 2], [1, 2]).kendall_tau)


def test_misaligned_inputs_are_refused():
    with pytest.raises(ValueError):
        rank_correlation([1, 2, 3], [1, 2])


def test_cluster_bootstrap_interval_is_degenerate_for_a_perfect_relation_and_brackets_a_noisy_one():
    perfect = pd.DataFrame({"query_id": [f"q{i % 20}" for i in range(60)], "x": np.arange(60.0), "y": np.arange(60.0) * 2})
    assert cluster_bootstrap_ci(perfect, "x", "y", n_boot=200) == pytest.approx((1.0, 1.0, 1.0, 1.0))

    rng = np.random.default_rng(1)
    x = rng.random(90)
    noisy = pd.DataFrame({"query_id": [f"q{i % 30}" for i in range(90)], "x": x, "y": x + rng.normal(0, 0.3, 90)})
    point = rank_correlation(noisy["x"], noisy["y"])
    rho_low, rho_high, tau_low, tau_high = cluster_bootstrap_ci(noisy, "x", "y", n_boot=500)
    assert rho_low < point.spearman_rho < rho_high
    assert tau_low < point.kendall_tau < tau_high


def test_cluster_bootstrap_resamples_whole_queries_not_rows():
    # 12 queries, each repeated as 30 identical rows (as if run under 30 configurations).
    # There are only 12 independent units of evidence. Resampling rows would draw
    # nearly every query in every resample and give an interval close to zero width;
    # resampling queries must give an interval about as wide as for the 12 queries alone.
    rng = np.random.default_rng(2)
    base = pd.DataFrame({"query_id": [f"q{i}" for i in range(12)], "x": rng.random(12)})
    base["y"] = base["x"] + rng.normal(0, 0.3, 12)
    repeated = pd.concat([base] * 30).reset_index(drop=True)
    alone = cluster_bootstrap_ci(base, "x", "y", n_boot=1000, seed=4)
    clustered = cluster_bootstrap_ci(repeated, "x", "y", n_boot=1000, seed=4)
    width_alone, width_clustered = alone[1] - alone[0], clustered[1] - clustered[0]
    assert width_alone > 0.3
    assert width_clustered == pytest.approx(width_alone, rel=0.25)


def per_query_rows(method, kb, ndcg, faithfulness):
    return pd.DataFrame(
        {
            "kb_variant": kb,
            "retrieval_method": method,
            "query_id": [f"q{i}" for i in range(len(ndcg))],
            "ndcg_at_k_all": ndcg,
            "faithfulness": faithfulness,
        }
    )


def test_table_pools_real_methods_only_and_has_one_scope_per_configuration():
    ndcg = [0.0, 0.2, 0.5, 0.7, 1.0]
    df = pd.concat(
        [
            per_query_rows("dense", "curated", ndcg, [0.5, 0.6, 0.7, 0.8, 0.9]),
            per_query_rows("lexical", "curated", ndcg, [0.9, 0.8, 0.7, 0.6, 0.5]),
            # baselines sit at the extremes and must not enter the pooled correlation
            per_query_rows("oracle", "curated", [1.0] * 5, [1.0] * 5),
            per_query_rows("random", "curated", [0.0] * 5, [0.0] * 5),
        ]
    )
    table = retrieval_generation_correlations(df, ["ndcg_at_k_all"], ["faithfulness"], n_boot=50)
    by_scope = table.set_index("scope")
    assert set(by_scope.index) == {"pooled", "curated/dense", "curated/lexical"}
    assert by_scope.loc["pooled", "n"] == 10
    # dense rows agree perfectly, lexical rows disagree perfectly: pooled they cancel out
    assert by_scope.loc["curated/dense", "spearman_rho"] == pytest.approx(1.0)
    assert by_scope.loc["curated/lexical", "spearman_rho"] == pytest.approx(-1.0)
    assert by_scope.loc["pooled", "spearman_rho"] == pytest.approx(
        spearman_by_definition(ndcg * 2, [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5])
    )
    assert not math.isnan(by_scope.loc["pooled", "spearman_ci95_low"])
    assert math.isnan(by_scope.loc["curated/dense", "spearman_ci95_low"])  # intervals for the pooled scope only


def test_empty_input_gives_an_empty_table():
    df = pd.DataFrame(columns=["kb_variant", "retrieval_method", "query_id", "ndcg_at_k_all", "faithfulness"])
    assert retrieval_generation_correlations(df, ["ndcg_at_k_all"], ["faithfulness"]).empty

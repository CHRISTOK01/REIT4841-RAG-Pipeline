"""The planned Study 1 / Study 2 analyses and the method x KB interaction,
checked against hand calculations and SciPy."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from analysis.aggregate import REL_DEGRADATION_MIN_BASELINE, relative_degradation
from analysis.stats import holm_adjust
from analysis.studies import (
    method_by_kb_interaction,
    method_ranks_by_kb,
    planned_comparisons,
    select_study2_method,
)


def rows(kb, method, ndcg, faithfulness=None):
    faithfulness = faithfulness if faithfulness is not None else [1.0] * len(ndcg)
    return pd.DataFrame(
        {
            "kb_variant": kb,
            "retrieval_method": method,
            "query_id": [f"q{i}" for i in range(len(ndcg))],
            "ndcg_at_k_all": ndcg,
            "faithfulness": faithfulness,
        }
    )


# -- Study 2 method selection -----------------------------------------------------------------------


def test_study2_method_is_the_highest_mean_ndcg_on_curated():
    df = pd.concat(
        [
            rows("curated", "lexical", [0.5, 0.5]),  # mean 0.5
            rows("curated", "dense", [0.9, 0.5]),  # mean 0.7
            rows("curated", "hybrid", [0.6, 0.6]),  # mean 0.6
            rows("mixed", "hybrid", [1.0, 1.0]),  # other KBs do not count
            rows("curated", "oracle", [1.0, 1.0]),  # nor do baselines
        ]
    )
    assert select_study2_method(df) == "dense"


def test_an_exact_ndcg_tie_is_broken_by_faithfulness():
    df = pd.concat(
        [
            rows("curated", "dense", [0.6, 0.6], [0.9, 0.9]),
            rows("curated", "hybrid", [0.6, 0.6], [0.95, 0.95]),
            rows("curated", "lexical", [0.1, 0.1], [1.0, 1.0]),
        ]
    )
    assert select_study2_method(df) == "hybrid"


def test_selection_needs_curated_rows():
    with pytest.raises(ValueError):
        select_study2_method(rows("mixed", "dense", [0.5]))


# -- planned comparisons ----------------------------------------------------------------------------


def grid(n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random(n)
    parts = []
    for kb, shift in [("curated", 0.0), ("mixed", -0.02), ("reduced", -0.3)]:
        parts.append(rows(kb, "dense", np.clip(base + 0.2 + shift, 0, 1)))
        parts.append(rows(kb, "hybrid", np.clip(base + 0.1 + shift + rng.normal(0, 0.05, n), 0, 1)))
        parts.append(rows(kb, "lexical", np.clip(base + shift, 0, 1)))
    parts.append(rows("curated", "oracle", np.ones(n)))
    return pd.concat(parts)


def test_planned_comparisons_cover_exactly_the_two_studies():
    table = planned_comparisons(grid(), ["ndcg_at_k_all"], "dense")
    study1 = table[table["family"] == "study1_methods_on_curated"]
    study2 = table[table["family"] == "study2_kbs_with_selected_method"]
    assert set(zip(study1["a"], study1["b"])) == {("dense", "hybrid"), ("dense", "lexical"), ("hybrid", "lexical")}
    assert set(study1["held_fixed"]) == {"curated"}
    assert set(zip(study2["a"], study2["b"])) == {("curated", "mixed"), ("curated", "reduced"), ("mixed", "reduced")}
    assert set(study2["held_fixed"]) == {"dense"}
    assert len(table) == 6  # no oracle, no other method in Study 2


def test_holm_is_applied_within_each_study_not_across_both():
    df = grid()
    table = planned_comparisons(df, ["ndcg_at_k_all"], "dense")
    study1 = table[table["family"] == "study1_methods_on_curated"]
    wide = df[df["kb_variant"] == "curated"].pivot(index="query_id", columns="retrieval_method", values="ndcg_at_k_all")
    raw = [scipy_stats.wilcoxon(wide[a], wide[b]).pvalue for a, b in zip(study1["a"], study1["b"])]
    assert study1["wilcoxon_p"].tolist() == pytest.approx(raw)
    assert study1["p_holm"].tolist() == pytest.approx(holm_adjust(raw))
    # a constant shift of 0.2 on every query is found
    dense_lexical = study1[(study1["a"] == "dense") & (study1["b"] == "lexical")].iloc[0]
    assert dense_lexical["verdict"] == "difference"


def test_study2_method_defaults_to_the_selection_rule():
    table = planned_comparisons(grid(), ["ndcg_at_k_all"])
    assert set(table[table["family"] == "study2_kbs_with_selected_method"]["held_fixed"]) == {"dense"}


# -- interaction -----------------------------------------------------------------------------------


def test_change_in_gap_is_computed_query_by_query():
    df = pd.concat(
        [
            rows("curated", "dense", [0.5] * 6),
            rows("curated", "lexical", [0.5] * 6),
            rows("curated", "hybrid", [0.5] * 6),
            rows("reduced", "dense", [0.8] * 6),  # dense pulls ahead by 0.3 on reduced
            rows("reduced", "lexical", [0.5] * 6),
            rows("reduced", "hybrid", [0.5] * 6),
        ]
    )
    table = method_by_kb_interaction(df, ["ndcg_at_k_all"]).set_index(["a", "b"])
    lexical_dense = table.loc[("lexical", "dense")]  # a - b = lexical - dense
    assert lexical_dense["gap_on_curated"] == pytest.approx(0.0)
    assert lexical_dense["gap_on_kb"] == pytest.approx(-0.3)
    assert lexical_dense["change_in_gap"] == pytest.approx(-0.3)
    assert (lexical_dense["ci95_low"], lexical_dense["ci95_high"]) == pytest.approx((-0.3, -0.3))
    lexical_hybrid = table.loc[("lexical", "hybrid")]
    assert lexical_hybrid["change_in_gap"] == 0.0 and lexical_hybrid["verdict"] == "could be noise"
    assert set(table["kb_variant"]) == {"reduced"}  # mixed absent from the data: skipped, not invented


def test_method_ranks_by_kb():
    df = pd.concat(
        [
            rows("curated", "dense", [0.9]),
            rows("curated", "lexical", [0.5]),
            rows("curated", "hybrid", [0.7]),
            rows("reduced", "dense", [0.4]),
            rows("reduced", "lexical", [0.6]),
            rows("reduced", "hybrid", [0.4]),
        ]
    )
    ranks = method_ranks_by_kb(df, ["ndcg_at_k_all"]).set_index(["kb_variant", "retrieval_method"])["rank"]
    assert ranks[("curated", "dense")] == 1 and ranks[("curated", "lexical")] == 3
    assert ranks[("reduced", "lexical")] == 1
    assert ranks[("reduced", "dense")] == ranks[("reduced", "hybrid")] == 2  # a tie shares the better rank


def test_abstention_ranks_lower_as_better():
    df = pd.concat([rows("curated", "dense", [0.0]).assign(abstained=0.1), rows("curated", "lexical", [0.0]).assign(abstained=0.3)])
    ranks = method_ranks_by_kb(df, ["abstained"]).set_index("retrieval_method")["rank"]
    assert ranks["dense"] == 1 and ranks["lexical"] == 2


# -- relative degradation (thesis Eq. 3.13) ---------------------------------------------------------


def test_relative_degradation_by_hand():
    assert relative_degradation(0.8, 0.6) == pytest.approx(0.25)  # (0.8 - 0.6) / 0.8
    assert relative_degradation(0.5, 0.75) == pytest.approx(-0.5)  # an improvement is negative
    assert relative_degradation(1.0, 1.0) == 0.0


def test_relative_degradation_is_undefined_near_a_zero_baseline():
    assert math.isnan(relative_degradation(0.0, 0.0))
    assert math.isnan(relative_degradation(REL_DEGRADATION_MIN_BASELINE / 2, 0.0))
    assert not math.isnan(relative_degradation(REL_DEGRADATION_MIN_BASELINE, 0.0))
    assert math.isnan(relative_degradation(math.nan, 0.5)) and math.isnan(relative_degradation(0.5, math.nan))

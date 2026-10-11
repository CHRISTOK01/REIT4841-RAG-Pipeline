"""The analysis tables, checked against values worked out by hand from small
hand-written result logs."""

import math

import pandas as pd
import pytest

from analysis.aggregate import (
    ResultsIntegrityError,
    aggregate_robustness_deltas,
    aggregate_standard_metrics,
    load_robustness_results,
    load_standard_results,
    summarize_with_uncertainty,
)
from analysis.compare import paired_comparisons
from experiments.results_store import ResultsStore

NAN = math.nan


def generation_metrics(faithfulness=1.0, label="answered", answer_relevance=0.8, context_relevance=0.6, reference_agreement=0.5):
    return {
        "faithfulness": faithfulness,
        "n_claims": 4,
        "n_supported": 4,
        "n_unsupported": 0,
        "n_no_information": 0,
        "abstention_label": label,
        "abstained": label == "abstained",
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "reference_agreement": reference_agreement,
    }


def standard(kb, method, query, precision, recall, mrr, ndcg, relevant=("x#0",), lenient=None, **gen):
    """`relevant=()` marks a query with no relevant chunk in this KB variant."""
    lenient_metrics = lenient or {"precision_at_k": precision, "recall_at_k": recall, "mrr": mrr, "ndcg_at_k": ndcg}
    return {
        "record_type": "standard",
        "run_key": f"{kb}:{method}:{query}",
        "kb_variant": kb,
        "retrieval_method": method,
        "query_id": query,
        "question_type": "information",
        "top_k": 5,
        "retrieved": [{"chunk_id": "x#0"}] * 5,
        "relevant_chunk_ids": list(relevant),
        "relevant_chunk_ids_lenient": list(relevant) if lenient is None else ["x#0", "y#0"],
        "retrieval_metrics": {"precision_at_k": precision, "recall_at_k": recall, "mrr": mrr, "ndcg_at_k": ndcg},
        "retrieval_metrics_lenient": lenient_metrics,
        "generation_metrics": generation_metrics(**gen),
    }


def write(tmp_path, records):
    store = ResultsStore(tmp_path / "results.jsonl")
    for record in records:
        store.append(record)
    return tmp_path / "results.jsonl"


# -- the two views of the retrieval metrics --------------------------------------------------------


def test_unanswerable_queries_count_as_zero_in_the_headline_and_are_excluded_from_the_answerable_view(tmp_path):
    path = write(
        tmp_path,
        [
            standard("reduced", "dense", "q1", 0.4, 1.0, 1.0, 1.0),
            standard("reduced", "dense", "q2", 0.2, 0.5, 0.5, 0.6),
            # q3's topic was dropped from `reduced`: the per-query log holds 0 / NaN / NaN / NaN
            standard("reduced", "dense", "q3", 0.0, NAN, NAN, NAN, relevant=()),
            standard("reduced", "dense", "q4", 0.0, NAN, NAN, NAN, relevant=()),
        ],
    )
    row = aggregate_standard_metrics(load_standard_results(path)).iloc[0]

    assert row["coverage"] == pytest.approx(0.5)  # 2 of 4 queries are answerable
    # headline (_all): unanswerable = 0 -> (1.0 + 0.5 + 0 + 0) / 4
    assert row["recall_at_k_all"] == pytest.approx(0.375)
    assert row["mrr_all"] == pytest.approx(0.375)
    assert row["ndcg_at_k_all"] == pytest.approx(0.4)  # (1.0 + 0.6) / 4
    assert row["precision_at_k_all"] == pytest.approx(0.15)  # (0.4 + 0.2) / 4
    assert row["recall_at_k_all_n"] == 4
    # answerable-only: (1.0 + 0.5) / 2, over 2 queries
    assert row["recall_at_k"] == pytest.approx(0.75)
    assert row["precision_at_k"] == pytest.approx(0.3)  # precision too: (0.4 + 0.2) / 2, not diluted by q3, q4
    assert row["recall_at_k_n"] == 2 and row["precision_at_k_n"] == 2
    assert row["n"] == 4


def test_a_kb_that_answers_fewer_queries_scores_lower_on_the_headline_even_if_it_ranks_equally_well(tmp_path):
    """The survivorship bias the audit found: on the answerable-only view
    `reduced` looks identical to `curated`; the headline shows the loss."""
    curated = [standard("curated", "dense", f"q{i}", 0.2, 1.0, 1.0, 1.0) for i in range(4)]
    reduced = [standard("reduced", "dense", f"q{i}", 0.2, 1.0, 1.0, 1.0) for i in range(2)]
    reduced += [standard("reduced", "dense", f"q{i}", 0.0, NAN, NAN, NAN, relevant=()) for i in range(2, 4)]
    agg = aggregate_standard_metrics(load_standard_results(write(tmp_path, curated + reduced))).set_index("kb_variant")

    assert agg.loc["curated", "recall_at_k"] == agg.loc["reduced", "recall_at_k"] == 1.0
    assert agg.loc["curated", "recall_at_k_all"] == 1.0
    assert agg.loc["reduced", "recall_at_k_all"] == 0.5


def test_strict_and_lenient_labels_are_reported_side_by_side(tmp_path):
    lenient = {"precision_at_k": 0.4, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 0.9}
    path = write(tmp_path, [standard("curated", "dense", "q1", 0.2, 0.5, 0.5, 0.4, lenient=lenient)])
    row = load_standard_results(path).iloc[0]
    assert (row["recall_at_k"], row["recall_at_k_lenient"]) == (0.5, 1.0)
    assert (row["precision_at_k_all"], row["precision_at_k_lenient_all"]) == (0.2, 0.4)
    assert (row["coverage"], row["coverage_lenient"]) == (1.0, 1.0)


def test_lenient_coverage_can_exceed_strict_coverage(tmp_path):
    """A query whose own document was dropped, but a verbatim copy of its answer survives."""
    record = standard("reduced", "dense", "q1", 0.0, NAN, NAN, NAN, relevant=(), lenient={"precision_at_k": 0.2, "recall_at_k": 1.0, "mrr": 0.5, "ndcg_at_k": 0.63})
    row = load_standard_results(write(tmp_path, [record])).iloc[0]
    assert (row["coverage"], row["coverage_lenient"]) == (0.0, 1.0)
    assert row["recall_at_k_all"] == 0.0 and math.isnan(row["recall_at_k"])
    assert row["recall_at_k_lenient_all"] == 1.0 and row["recall_at_k_lenient"] == 1.0


def test_no_retrieval_rows_have_no_retrieval_metrics_at_all_not_zeros(tmp_path):
    record = standard("none", "none", "q1", 0.0, NAN, NAN, NAN, relevant=())
    record["retrieval_metrics"] = None
    record["retrieval_metrics_lenient"] = None
    record["retrieved"] = []
    row = load_standard_results(write(tmp_path, [record])).iloc[0]
    for column in ("recall_at_k_all", "recall_at_k", "precision_at_k_all", "coverage", "ndcg_at_k_lenient_all"):
        assert math.isnan(row[column]), column
    assert row["n_retrieved"] == 0 and row["answer_relevance"] == 0.8


# -- generation metrics ---------------------------------------------------------------------------


def test_abstention_is_a_rate_and_undefined_scores_are_left_out_of_means(tmp_path):
    path = write(
        tmp_path,
        [
            standard("curated", "dense", "q1", 0.2, 1.0, 1.0, 1.0, faithfulness=1.0, answer_relevance=0.9),
            standard("curated", "dense", "q2", 0.2, 1.0, 1.0, 1.0, faithfulness=0.5, answer_relevance=0.7, label="partial"),
            # abstained: no factual claims (faithfulness NaN) and no answer relevance (NaN)
            standard("curated", "dense", "q3", 0.2, 1.0, 1.0, 1.0, faithfulness=NAN, answer_relevance=NAN, label="abstained"),
            standard("curated", "dense", "q4", 0.2, 1.0, 1.0, 1.0, faithfulness=NAN, answer_relevance=NAN, label="abstained"),
        ],
    )
    row = aggregate_standard_metrics(load_standard_results(path)).iloc[0]
    assert row["abstained"] == pytest.approx(0.5) and row["abstained_n"] == 4  # 2 of 4 answers
    assert row["partial"] == pytest.approx(0.25)
    assert row["faithfulness"] == pytest.approx(0.75) and row["faithfulness_n"] == 2  # (1.0 + 0.5) / 2, not / 4
    assert row["answer_relevance"] == pytest.approx(0.8) and row["answer_relevance_n"] == 2


def test_reference_agreement_is_only_defined_for_answers_that_were_given(tmp_path):
    """Found in the real sweep: the judge sometimes labels an abstention's "the
    context does not say" as unsupported by the reference answer, which scored
    refusals as wrong answers (reference agreement 0)."""
    path = write(
        tmp_path,
        [
            standard("reduced", "dense", "q1", 0.2, 1.0, 1.0, 1.0, reference_agreement=0.8),
            standard("reduced", "dense", "q2", 0.0, NAN, NAN, NAN, relevant=(), label="abstained", faithfulness=NAN, answer_relevance=NAN, reference_agreement=0.0),
            standard("reduced", "dense", "q3", 0.2, 1.0, 1.0, 1.0, label="partial", reference_agreement=0.4),
        ],
    )
    df = load_standard_results(path).set_index("query_id")
    assert math.isnan(df.loc["q2", "reference_agreement"])  # the logged 0.0 is not used
    assert df.loc["q3", "reference_agreement"] == 0.4  # partial answers are still scored
    row = aggregate_standard_metrics(load_standard_results(path)).iloc[0]
    assert row["reference_agreement"] == pytest.approx(0.6) and row["reference_agreement_n"] == 2


def test_faithfulness_is_only_defined_for_answers_that_were_given(tmp_path):
    """Found in the manual spot-check: for answers that abstain, the judge
    sometimes labels sentences such as "the resources listed do not specify
    treatments" as unsupported, giving a correct refusal a faithfulness of 0."""
    path = write(
        tmp_path,
        [
            standard("curated", "dense", "q1", 0.2, 1.0, 1.0, 1.0, faithfulness=1.0),
            standard("curated", "dense", "q2", 0.2, 1.0, 1.0, 1.0, label="abstained", faithfulness=0.0, answer_relevance=NAN),
            standard("curated", "dense", "q3", 0.2, 1.0, 1.0, 1.0, label="partial", faithfulness=0.5),
        ],
    )
    df = load_standard_results(path).set_index("query_id")
    assert math.isnan(df.loc["q2", "faithfulness"])
    assert df.loc["q3", "faithfulness"] == 0.5  # partial answers are still scored
    row = aggregate_standard_metrics(load_standard_results(path)).iloc[0]
    assert row["faithfulness"] == pytest.approx(0.75) and row["faithfulness_n"] == 2


def test_error_and_robustness_records_are_not_counted_as_results(tmp_path):
    path = write(
        tmp_path,
        [
            standard("curated", "dense", "q1", 0.2, 1.0, 1.0, 1.0),
            {"record_type": "error", "run_key": "curated:dense:q2:error:t", "failed_run_key": "curated:dense:q2"},
            {"record_type": "robustness_skipped", "run_key": "curated:dense:q1:robustness:inject_noise"},
        ],
    )
    assert len(load_standard_results(path)) == 1


def test_groups_are_kept_separate(tmp_path):
    path = write(
        tmp_path,
        [
            standard("curated", "dense", "q1", 0.2, 1.0, 1.0, 1.0),
            standard("curated", "lexical", "q1", 0.2, 0.0, 0.0, 0.0),
            standard("mixed", "dense", "q1", 0.2, 0.5, 0.5, 0.5),
        ],
    )
    agg = aggregate_standard_metrics(load_standard_results(path)).set_index(["kb_variant", "retrieval_method"])
    assert agg["recall_at_k_all"].to_dict() == {("curated", "dense"): 1.0, ("curated", "lexical"): 0.0, ("mixed", "dense"): 0.5}
    assert set(agg["n"]) == {1}


def test_empty_log_gives_empty_tables_with_the_right_columns(tmp_path):
    df = load_standard_results(tmp_path / "does_not_exist.jsonl")
    assert df.empty and {"recall_at_k_all", "coverage", "abstained", "reference_agreement"} <= set(df.columns)
    assert aggregate_standard_metrics(df).empty
    assert summarize_with_uncertainty(df, ["kb_variant"], ["recall_at_k_all"]).empty
    assert paired_comparisons(df, ["recall_at_k_all"]).empty
    assert load_robustness_results(tmp_path / "does_not_exist.jsonl").empty


def test_a_log_with_duplicate_or_unreadable_rows_is_refused(tmp_path):
    record = standard("curated", "dense", "q1", 0.2, 1.0, 1.0, 1.0)
    with pytest.raises(ResultsIntegrityError, match="duplicate"):
        load_standard_results(write(tmp_path, [record, record]))

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"record_type": "standard", "run_key": "a"\n', encoding="utf-8")
    with pytest.raises(ResultsIntegrityError, match="unreadable"):
        load_standard_results(bad)


# -- uncertainty table ------------------------------------------------------------------------------


def test_summary_reports_mean_sd_n_and_a_ci_that_brackets_the_mean(tmp_path):
    values = [1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    path = write(tmp_path, [standard("curated", "dense", f"q{i}", 0.2, v, v, v) for i, v in enumerate(values)])
    summary = summarize_with_uncertainty(load_standard_results(path), ["kb_variant", "retrieval_method"], ["recall_at_k_all"])
    row = summary.iloc[0]
    assert (row["kb_variant"], row["retrieval_method"], row["metric"]) == ("curated", "dense", "recall_at_k_all")
    assert row["mean"] == pytest.approx(0.5625) and row["n"] == 8
    # sample sd: sqrt(sum((v - 0.5625)^2) / 7) = sqrt(1.21875 / 7) = 0.417261
    assert row["sd"] == pytest.approx(0.417261, abs=1e-6)
    assert row["ci95_low"] < 0.5625 < row["ci95_high"]
    assert 0.2 < row["ci95_low"] and row["ci95_high"] < 0.9


# -- robustness ---------------------------------------------------------------------------------------


def robustness(kb, query, perturbation, baseline, perturbed, baseline_recall=1.0, perturbed_recall=0.5):
    return {
        "record_type": "robustness",
        "run_key": f"{kb}:dense:{query}:robustness:{perturbation}",
        "kb_variant": kb,
        "retrieval_method": "dense",
        "query_id": query,
        "perturbation": perturbation,
        "changed_chunk_id": "x#0",
        "noise_chunk_id": None,
        "relevant_chunk_ids": ["x#0"],
        "baseline_retrieval_metrics": {"recall_at_k": baseline_recall},
        "perturbed_retrieval_metrics": {"recall_at_k": perturbed_recall},
        "baseline_generation_metrics": baseline,
        "perturbed_generation_metrics": perturbed,
    }


def test_robustness_deltas_are_perturbed_minus_baseline_and_abstention_is_its_own_delta(tmp_path):
    path = write(
        tmp_path,
        [
            robustness("curated", "q1", "drop_top_relevant", generation_metrics(1.0, "answered", 0.9, 0.6, 0.8), generation_metrics(0.5, "answered", 0.7, 0.4, 0.3)),
            # after the drop the model abstains: faithfulness / answer relevance become undefined, not 0
            robustness("curated", "q2", "drop_top_relevant", generation_metrics(1.0, "answered", 0.8, 0.6, 0.8), generation_metrics(NAN, "abstained", NAN, 0.2, NAN)),
        ],
    )
    df = load_robustness_results(path).set_index("query_id")
    assert df.loc["q1", "faithfulness_delta"] == pytest.approx(-0.5)
    assert df.loc["q1", "answer_relevance_delta"] == pytest.approx(-0.2)
    assert df.loc["q1", "reference_agreement_delta"] == pytest.approx(-0.5)
    assert df.loc["q1", "recall_at_k_delta"] == pytest.approx(-0.5)
    assert df.loc["q1", "abstained_delta"] == 0.0
    assert math.isnan(df.loc["q2", "faithfulness_delta"]) and math.isnan(df.loc["q2", "answer_relevance_delta"])
    assert df.loc["q2", "abstained_delta"] == 1.0  # the effect shows up here instead
    assert df.loc["q2", "context_relevance_delta"] == pytest.approx(-0.4)

    agg = aggregate_robustness_deltas(load_robustness_results(path)).iloc[0]
    assert agg["abstained_delta"] == pytest.approx(0.5) and agg["abstained_delta_n"] == 2
    assert agg["faithfulness_delta"] == pytest.approx(-0.5) and agg["faithfulness_delta_n"] == 1
    assert agg["context_relevance_delta"] == pytest.approx(-0.3)


# -- paired comparisons ---------------------------------------------------------------------------------


def test_paired_comparisons_match_queries_and_cover_both_families(tmp_path):
    records = []
    hybrid = [1.0, 1.0, 0.5, 1.0, 0.5, 1.0]
    lexical = [0.5, 0.5, 0.0, 0.5, 0.0, 0.5]
    for i, (h, l) in enumerate(zip(hybrid, lexical)):
        records.append(standard("curated", "hybrid", f"q{i}", 0.2, h, h, h))
        records.append(standard("curated", "lexical", f"q{i}", 0.2, l, l, l))
        records.append(standard("mixed", "hybrid", f"q{i}", 0.2, h, h, h))
    # written in a different query order for one config: pairing must be by query id, not row order
    records = records[::-1]
    comparisons = paired_comparisons(load_standard_results(write(tmp_path, records)), ["recall_at_k_all"])

    methods = comparisons[(comparisons["family"] == "methods_within_kb") & (comparisons["held_fixed"] == "curated")].iloc[0]
    assert (methods["a"], methods["b"], methods["n_pairs"]) == ("hybrid", "lexical", 6)
    assert methods["mean_diff"] == pytest.approx(0.5)  # hybrid beats lexical by exactly 0.5 on every query
    assert methods["sd_diff"] == pytest.approx(0.0)
    assert (methods["ci95_low"], methods["ci95_high"]) == pytest.approx((0.5, 0.5))

    kbs = comparisons[(comparisons["family"] == "kbs_within_method") & (comparisons["held_fixed"] == "hybrid")].iloc[0]
    assert (kbs["a"], kbs["b"]) == ("curated", "mixed")
    assert kbs["mean_diff"] == 0.0 and kbs["verdict"] == "could be noise"
    assert set(comparisons["family"]) == {"methods_within_kb", "kbs_within_method"}


def test_comparison_verdict_requires_both_the_interval_and_the_adjusted_p_value(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    records = []
    for i in range(60):
        base = float(rng.random())
        records.append(standard("curated", "dense", f"q{i}", 0.2, min(1.0, base + 0.3), 0, 0))
        records.append(standard("curated", "lexical", f"q{i}", 0.2, base * 0.7, 0, 0))
        records.append(standard("curated", "hybrid", f"q{i}", 0.2, float(rng.random()), 0, 0))
        records.append(standard("curated", "random", f"q{i}", 0.2, float(rng.random()), 0, 0))
    comparisons = paired_comparisons(load_standard_results(write(tmp_path, records)), ["recall_at_k_all"])
    verdict = {(r["a"], r["b"]): r["verdict"] for _, r in comparisons.iterrows()}
    assert verdict[("dense", "lexical")] == "difference"
    assert verdict[("hybrid", "random")] == "could be noise"  # two sets of unrelated random numbers
    assert (comparisons["p_holm"] >= comparisons["wilcoxon_p"] - 1e-12).all()
    assert isinstance(comparisons, pd.DataFrame)

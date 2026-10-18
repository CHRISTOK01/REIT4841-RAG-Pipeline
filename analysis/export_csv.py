"""Export the canonical CSVs + all thesis figures from one sweep's results log.

    python -m analysis.export_csv --results results/full_sweep.jsonl --out-dir reports

Reads nothing but the results file. Writes:
    results_per_query.csv        -- one row per (kb_variant, retrieval_method, query)
    results_aggregated.csv       -- mean per (kb_variant, retrieval_method), with per-metric n:
                                    the canonical comparison table
    results_summary.csv          -- long form: mean, sd, n and 95% bootstrap CI per metric
    paired_comparisons.csv       -- paired tests between methods and between KB variants
    robustness_per_query.csv     -- one row per perturbation run (if any)
    robustness_aggregated.csv    -- mean deltas per (kb_variant, perturbation)
    robustness_summary.csv       -- long form with sd, n and 95% CI (deltas and relative degradation)
    study_selection.txt          -- the Study 2 retrieval method and the rule that chose it
    study_comparisons.csv        -- the planned Study 1 / Study 2 paired tests (Holm within each study)
    method_ranks_by_kb.csv       -- each method's mean and rank per KB variant (interaction)
    method_kb_interaction.csv    -- change in the gap between two methods from curated to mixed/reduced
    retrieval_generation_correlations.csv -- RQ4: Spearman and Kendall, pooled and per configuration
    figures/*.png                -- one bar chart per metric (with CI whiskers), a robustness figure,
                                    study1_distributions.png and rq4_correlations_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aggregate import (
    ROBUSTNESS_DELTA_COLUMNS,
    ROBUSTNESS_REL_DEGRADATION_COLUMNS,
    STANDARD_METRIC_COLUMNS,
    aggregate_robustness_deltas,
    aggregate_standard_metrics,
    load_robustness_results,
    load_standard_results,
    summarize_with_uncertainty,
)
from analysis.compare import paired_comparisons
from analysis.correlation import retrieval_generation_correlations
from analysis.plots import (
    plot_all_standard_metrics,
    plot_correlation_heatmap,
    plot_metric_distributions,
    plot_robustness_deltas,
)
from analysis.studies import (
    SELECTION_METRIC,
    TIE_BREAK_METRIC,
    method_by_kb_interaction,
    method_ranks_by_kb,
    planned_comparisons,
    select_study2_method,
)

# Metrics that get a paired significance test. Kept short on purpose: every
# extra test is another chance of a spurious "difference".
COMPARED_METRICS = [
    "recall_at_k_all",
    "mrr_all",
    "ndcg_at_k_all",
    "recall_at_k_lenient_all",
    "ndcg_at_k_lenient_all",
    "faithfulness",
    "abstained",
    "answer_relevance",
    "context_relevance",
    "reference_agreement",
]

STUDY1_DISTRIBUTION_METRICS = [
    "precision_at_k_all",
    "recall_at_k_all",
    "mrr_all",
    "ndcg_at_k_all",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "reference_agreement",
]


def export_results(results_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    standard_df = load_standard_results(results_path)
    standard_df.to_csv(out_dir / "results_per_query.csv", index=False)

    aggregate_standard_metrics(standard_df).to_csv(out_dir / "results_aggregated.csv", index=False)
    summary_df = summarize_with_uncertainty(standard_df, ["kb_variant", "retrieval_method"], STANDARD_METRIC_COLUMNS)
    summary_df.to_csv(out_dir / "results_summary.csv", index=False)
    paired_comparisons(standard_df, COMPARED_METRICS).to_csv(out_dir / "paired_comparisons.csv", index=False)
    plot_all_standard_metrics(summary_df, STANDARD_METRIC_COLUMNS, out_dir / "figures")

    has_real = not standard_df.empty and standard_df["retrieval_method"].isin(["lexical", "dense", "hybrid"]).any()
    if has_real:
        study2_method = select_study2_method(standard_df)
        (out_dir / "study_selection.txt").write_text(
            f"Study 2 retrieval method: {study2_method}\n"
            f"Rule (fixed in advance): highest mean {SELECTION_METRIC} on the curated KB among lexical, dense and hybrid; "
            f"an exact tie is broken by the higher mean {TIE_BREAK_METRIC}, then by name.\n",
            encoding="utf-8",
        )
        planned_comparisons(standard_df, COMPARED_METRICS, study2_method).to_csv(out_dir / "study_comparisons.csv", index=False)
        method_ranks_by_kb(standard_df, COMPARED_METRICS).to_csv(out_dir / "method_ranks_by_kb.csv", index=False)
        method_by_kb_interaction(standard_df, COMPARED_METRICS).to_csv(out_dir / "method_kb_interaction.csv", index=False)
        correlations_df = retrieval_generation_correlations(standard_df)
        correlations_df.to_csv(out_dir / "retrieval_generation_correlations.csv", index=False)
        plot_correlation_heatmap(correlations_df, out_dir / "figures" / "rq4_correlations_spearman.png", statistic="spearman_rho")
        plot_correlation_heatmap(correlations_df, out_dir / "figures" / "rq4_correlations_kendall.png", statistic="kendall_tau")
        plot_metric_distributions(standard_df, STUDY1_DISTRIBUTION_METRICS, out_dir / "figures" / "study1_distributions.png")

    robustness_df = load_robustness_results(results_path)
    robustness_rows = len(robustness_df)
    if not robustness_df.empty:
        robustness_df.to_csv(out_dir / "robustness_per_query.csv", index=False)
        aggregate_robustness_deltas(robustness_df).to_csv(out_dir / "robustness_aggregated.csv", index=False)
        robustness_summary_df = summarize_with_uncertainty(
            robustness_df, ["kb_variant", "perturbation"], ROBUSTNESS_DELTA_COLUMNS + ROBUSTNESS_REL_DEGRADATION_COLUMNS
        )
        robustness_summary_df.to_csv(out_dir / "robustness_summary.csv", index=False)
        plot_robustness_deltas(
            robustness_summary_df, out_dir / "figures" / "robustness_deltas.png", metrics=ROBUSTNESS_DELTA_COLUMNS
        )

    print(
        f"Wrote {len(standard_df)} standard row(s) and {robustness_rows} robustness row(s) "
        f"of results to {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSVs and figures from an experiment sweep's results.")
    parser.add_argument("--results", type=Path, default=Path("results/full_sweep.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    export_results(args.results, args.out_dir)


if __name__ == "__main__":
    main()

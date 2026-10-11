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
    robustness_summary.csv       -- long form with sd, n and 95% CI
    figures/*.png                -- one bar chart per metric (with CI whiskers) + a robustness figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aggregate import (
    ROBUSTNESS_DELTA_COLUMNS,
    STANDARD_METRIC_COLUMNS,
    aggregate_robustness_deltas,
    aggregate_standard_metrics,
    load_robustness_results,
    load_standard_results,
    summarize_with_uncertainty,
)
from analysis.compare import paired_comparisons
from analysis.plots import plot_all_standard_metrics, plot_robustness_deltas

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


def export_results(results_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    standard_df = load_standard_results(results_path)
    standard_df.to_csv(out_dir / "results_per_query.csv", index=False)

    aggregate_standard_metrics(standard_df).to_csv(out_dir / "results_aggregated.csv", index=False)
    summary_df = summarize_with_uncertainty(standard_df, ["kb_variant", "retrieval_method"], STANDARD_METRIC_COLUMNS)
    summary_df.to_csv(out_dir / "results_summary.csv", index=False)
    paired_comparisons(standard_df, COMPARED_METRICS).to_csv(out_dir / "paired_comparisons.csv", index=False)
    plot_all_standard_metrics(summary_df, STANDARD_METRIC_COLUMNS, out_dir / "figures")

    robustness_df = load_robustness_results(results_path)
    robustness_rows = len(robustness_df)
    if not robustness_df.empty:
        robustness_df.to_csv(out_dir / "robustness_per_query.csv", index=False)
        aggregate_robustness_deltas(robustness_df).to_csv(out_dir / "robustness_aggregated.csv", index=False)
        robustness_summary_df = summarize_with_uncertainty(
            robustness_df, ["kb_variant", "perturbation"], ROBUSTNESS_DELTA_COLUMNS
        )
        robustness_summary_df.to_csv(out_dir / "robustness_summary.csv", index=False)
        plot_robustness_deltas(robustness_summary_df, out_dir / "figures" / "robustness_deltas.png")

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

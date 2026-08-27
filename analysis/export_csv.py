"""Export the canonical CSVs + all thesis figures from one sweep's results.

    python -m analysis.export_csv --results results/full_sweep.jsonl --out-dir reports

Writes:
    reports/results_per_query.csv        -- one row per (kb_variant, retrieval_method, query)
    reports/results_aggregated.csv        -- mean per (kb_variant, retrieval_method), the
                                              canonical comparison table for the methodology chapter
    reports/robustness_per_query.csv      -- one row per robustness perturbation run (if any)
    reports/robustness_aggregated.csv     -- mean deltas per (kb_variant, perturbation) (if any)
    reports/figures/*.png                 -- one bar chart per metric, plus a robustness figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aggregate import (
    STANDARD_METRIC_COLUMNS,
    aggregate_robustness_deltas,
    aggregate_standard_metrics,
    load_robustness_results,
    load_standard_results,
)
from analysis.plots import plot_all_standard_metrics, plot_robustness_deltas


def export_results(results_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    standard_df = load_standard_results(results_path)
    standard_df.to_csv(out_dir / "results_per_query.csv", index=False)

    agg_df = aggregate_standard_metrics(standard_df)
    agg_df.to_csv(out_dir / "results_aggregated.csv", index=False)
    plot_all_standard_metrics(agg_df, STANDARD_METRIC_COLUMNS, out_dir / "figures")

    robustness_df = load_robustness_results(results_path)
    robustness_rows = len(robustness_df)
    if not robustness_df.empty:
        robustness_df.to_csv(out_dir / "robustness_per_query.csv", index=False)
        robustness_agg_df = aggregate_robustness_deltas(robustness_df)
        robustness_agg_df.to_csv(out_dir / "robustness_aggregated.csv", index=False)
        plot_robustness_deltas(robustness_agg_df, out_dir / "figures" / "robustness_deltas.png")

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

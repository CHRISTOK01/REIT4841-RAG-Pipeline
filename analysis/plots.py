"""Matplotlib figures for the thesis, saved directly to files (no interactive
display -- the Agg backend is forced so this runs headless on any machine).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_metric_by_kb_and_method(
    agg_df: pd.DataFrame, metric: str, out_path: Path, title: str | None = None
) -> None:
    """Grouped bar chart: KB variant on the x-axis, one bar per retrieval method."""
    if agg_df.empty:
        print(f"Skipping plot for {metric!r}: no data.")
        return

    pivot = agg_df.pivot(index="kb_variant", columns="retrieval_method", values=metric)
    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.set_ylabel(metric)
    ax.set_xlabel("KB variant")
    ax.set_title(title or metric)
    ax.legend(title="Retrieval method")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_all_standard_metrics(agg_df: pd.DataFrame, metrics: list[str], out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for metric in metrics:
        if metric not in agg_df.columns:
            continue
        out_path = out_dir / f"{metric}.png"
        plot_metric_by_kb_and_method(agg_df, metric, out_path)
        written.append(out_path)
    return written


def plot_robustness_deltas(robustness_agg_df: pd.DataFrame, out_path: Path) -> None:
    """One subplot per delta metric: KB variant on the x-axis, one bar per
    perturbation type, a zero line marking "no change from baseline"."""
    delta_columns = [
        c
        for c in ["faithfulness_delta", "answer_relevance_delta", "context_relevance_delta"]
        if c in robustness_agg_df.columns
    ]
    if robustness_agg_df.empty or not delta_columns:
        print("Skipping robustness plot: no data.")
        return

    fig, axes = plt.subplots(1, len(delta_columns), figsize=(5 * len(delta_columns), 4), squeeze=False)
    for ax, metric in zip(axes[0], delta_columns):
        pivot = robustness_agg_df.pivot(index="kb_variant", columns="perturbation", values=metric)
        pivot.plot(kind="bar", ax=ax, legend=(ax is axes[0][-1]))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("KB variant")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

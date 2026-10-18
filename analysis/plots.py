"""Matplotlib figures for the thesis, saved directly to files (no interactive
display -- the Agg backend is forced so this runs headless on any machine).

Bars show the mean over queries; the whiskers are the 95% bootstrap confidence
interval of that mean (from analysis.aggregate.summarize_with_uncertainty).
The three retrievers keep one colour each in every figure; the oracle and
random reference points are drawn in grey with hatching so they are never
mistaken for a method under test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KB_ORDER = ["curated", "mixed", "reduced", "none"]
METHOD_ORDER = ["lexical", "dense", "hybrid", "oracle", "random", "none"]
METHOD_STYLE = {
    "lexical": {"color": "#2a78d6"},
    "dense": {"color": "#eb6834"},
    "hybrid": {"color": "#1baf7a"},
    "oracle": {"color": "#8a8a85", "hatch": "//"},
    "random": {"color": "#c9c8c2", "hatch": ".."},
    "none": {"color": "#e4e3de", "hatch": "xx"},
}
PERTURBATION_ORDER = ["control_rerun", "drop_top_relevant", "inject_noise"]
PERTURBATION_STYLE = {
    "control_rerun": {"color": "#c9c8c2", "hatch": ".."},
    "drop_top_relevant": {"color": "#2a78d6"},
    "inject_noise": {"color": "#eb6834"},
}
_INK = "#33332f"
_GRID = "#e4e3de"


def _ordered(present, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(present))
    return [x for x in preferred if x in present] + sorted(x for x in present if x not in preferred)


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK, length=0)


def _grouped_bars(ax, summary: pd.DataFrame, x_col: str, series_col: str, x_order, series_order, styles) -> None:
    """`summary` holds one metric: columns x_col, series_col, mean, ci95_low, ci95_high."""
    width = 0.8 / max(len(series_order), 1)
    for s_idx, series in enumerate(series_order):
        rows = summary[summary[series_col] == series].set_index(x_col)
        xs, means, lows, highs = [], [], [], []
        for x_idx, x_value in enumerate(x_order):
            if x_value not in rows.index or pd.isna(rows.loc[x_value, "mean"]):
                continue
            mean = rows.loc[x_value, "mean"]
            low, high = rows.loc[x_value, "ci95_low"], rows.loc[x_value, "ci95_high"]
            xs.append(x_idx + (s_idx - (len(series_order) - 1) / 2) * width)
            means.append(mean)
            lows.append(0.0 if pd.isna(low) else mean - low)
            highs.append(0.0 if pd.isna(high) else high - mean)
        if not xs:
            continue
        ax.bar(
            xs,
            means,
            width=width * 0.9,
            label=series,
            edgecolor="white",
            linewidth=1.0,
            yerr=[lows, highs],
            error_kw={"ecolor": _INK, "elinewidth": 1.0, "capsize": 2.5},
            **styles.get(series, {}),
        )
    ax.set_xticks(np.arange(len(x_order)))
    ax.set_xticklabels(x_order, color=_INK)


def plot_metric_by_kb_and_method(
    summary_df: pd.DataFrame, metric: str, out_path: Path, title: str | None = None
) -> None:
    """Grouped bar chart: KB variant on the x-axis, one bar per retrieval
    method, 95% CI whiskers. `summary_df` is the long uncertainty table."""
    data = summary_df[summary_df["metric"] == metric] if not summary_df.empty else summary_df
    data = data.dropna(subset=["mean"]) if not data.empty else data
    if data.empty:
        print(f"Skipping plot for {metric!r}: no data.")
        return

    kb_order = _ordered(data["kb_variant"], KB_ORDER)
    method_order = _ordered(data["retrieval_method"], METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _grouped_bars(ax, data, "kb_variant", "retrieval_method", kb_order, method_order, METHOD_STYLE)
    ax.set_ylabel(f"{metric} (mean, 95% CI)", color=_INK)
    ax.set_xlabel("KB variant", color=_INK)
    ax.set_title(title or metric, color=_INK, loc="left")
    ax.legend(title="Retrieval method", frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_standard_metrics(summary_df: pd.DataFrame, metrics: list[str], out_dir: Path) -> list[Path]:
    written: list[Path] = []
    available = set(summary_df["metric"]) if not summary_df.empty else set()
    for metric in metrics:
        if metric not in available:
            continue
        out_path = out_dir / f"{metric}.png"
        plot_metric_by_kb_and_method(summary_df, metric, out_path)
        if out_path.exists():
            written.append(out_path)
    return written


def plot_robustness_deltas(robustness_summary_df: pd.DataFrame, out_path: Path, metrics: list[str] | None = None) -> None:
    """One panel per delta metric: KB variant on the x-axis, one bar per
    condition (including the no-change control), a zero line marking "no
    change from baseline". Each panel has its own y-axis."""
    if robustness_summary_df.empty:
        print("Skipping robustness plot: no data.")
        return
    data = robustness_summary_df.dropna(subset=["mean"])
    metrics = [m for m in (metrics or list(dict.fromkeys(data["metric"]))) if m in set(data["metric"])]
    if not metrics:
        print("Skipping robustness plot: no data.")
        return

    kb_order = _ordered(data["kb_variant"], KB_ORDER)
    perturbation_order = _ordered(data["perturbation"], PERTURBATION_ORDER)
    n_cols = min(3, len(metrics))
    n_rows = -(-len(metrics) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        _grouped_bars(
            ax, data[data["metric"] == metric], "kb_variant", "perturbation", kb_order, perturbation_order, PERTURBATION_STYLE
        )
        ax.axhline(0, color=_INK, linewidth=0.8)
        ax.set_title(metric, color=_INK, loc="left", fontsize=10)
        ax.set_xlabel("KB variant", color=_INK)
        _style_axes(ax)
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Condition", frameon=False, loc="lower center", ncol=len(labels))
    fig.suptitle("Change from baseline after perturbing the retrieved context (mean, 95% CI)", color=_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_distributions(df: pd.DataFrame, metrics: list[str], out_path: Path, kb_variant: str = "curated") -> None:
    """Study 1 figure: one panel per metric, a box plot of the per-query values
    for each real retrieval method on one KB variant (box = quartiles, line =
    median, whiskers = 1.5 IQR, white diamond = mean). `df` is the per-query
    table from analysis.aggregate.load_standard_results."""
    data = df[(df["kb_variant"] == kb_variant) & df["retrieval_method"].isin(["lexical", "dense", "hybrid"])] if not df.empty else df
    metrics = [m for m in metrics if not data.empty and data[m].notna().any()]
    if not metrics:
        print("Skipping distribution plot: no data.")
        return
    methods = _ordered(data["retrieval_method"], METHOD_ORDER)
    n_cols = min(3, len(metrics))
    n_rows = -(-len(metrics) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        values = [data.loc[data["retrieval_method"] == m, metric].dropna().to_numpy() for m in methods]
        box = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": _INK, "markersize": 5},
            medianprops={"color": _INK, "linewidth": 1.2},
            whiskerprops={"color": _INK, "linewidth": 0.8},
            capprops={"color": _INK, "linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": _GRID, "markeredgecolor": _INK, "alpha": 0.6},
        )
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(METHOD_STYLE[method]["color"])
            patch.set_edgecolor("white")
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels([f"{m}\n(n={len(v)})" for m, v in zip(methods, values)], color=_INK, fontsize=8)
        ax.set_title(metric, color=_INK, loc="left", fontsize=10)
        _style_axes(ax)
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle(f"Per-query distribution by retrieval method, {kb_variant} KB", color=_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(correlations_df: pd.DataFrame, out_path: Path, scope: str = "pooled", statistic: str = "spearman_rho") -> None:
    """RQ4 figure: retrieval metric (rows) x generation metric (columns),
    each cell the rank correlation, on a diverging scale fixed at [-1, 1] with a
    neutral midpoint at 0. `correlations_df` is from
    analysis.correlation.retrieval_generation_correlations."""
    data = correlations_df[correlations_df["scope"] == scope] if not correlations_df.empty else correlations_df
    if data.empty:
        print("Skipping correlation heatmap: no data.")
        return
    from matplotlib.colors import LinearSegmentedColormap

    table = data.pivot(index="retrieval_metric", columns="generation_metric", values=statistic)
    table = table.loc[list(dict.fromkeys(data["retrieval_metric"])), list(dict.fromkeys(data["generation_metric"]))]
    cmap = LinearSegmentedColormap.from_list("diverging", ["#eb6834", "#f2f1ec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(1.5 * table.shape[1] + 2.2, 0.75 * table.shape[0] + 1.6))
    image = ax.imshow(table.to_numpy(dtype=float), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iat[i, j]
            ax.text(j, i, "n/a" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", color=_INK, fontsize=9)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=30, ha="right", color=_INK)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index, color=_INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)
    label = {"spearman_rho": "Spearman's rho", "kendall_tau": "Kendall's tau-b"}.get(statistic, statistic)
    ax.set_title(f"{label}: retrieval vs generation metrics ({scope})", color=_INK, loc="left", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

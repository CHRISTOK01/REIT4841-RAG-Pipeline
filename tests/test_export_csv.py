"""CSV export and figures. (Rebuilding every report from a real harness log,
with all other data deleted, is tested in tests/test_run_sweep_integration.py.)"""

import pandas as pd
import pytest

from analysis.export_csv import COMPARED_METRICS, export_results
from analysis.plots import METHOD_STYLE, plot_metric_by_kb_and_method, plot_robustness_deltas
from tests.test_aggregate import generation_metrics, robustness, standard, write


def small_log(tmp_path):
    records = []
    for i, recall in enumerate([1.0, 0.5, 0.0, 1.0]):
        records.append(standard("curated", "dense", f"q{i}", 0.2, recall, recall, recall))
        records.append(standard("curated", "lexical", f"q{i}", 0.2, recall / 2, recall / 2, recall / 2))
        records.append(
            robustness("curated", f"q{i}", "drop_top_relevant", generation_metrics(1.0), generation_metrics(0.5 + 0.1 * i))
        )
        records.append(robustness("curated", f"q{i}", "control_rerun", generation_metrics(1.0), generation_metrics(1.0), 1.0, 1.0))
    return write(tmp_path, records)


def test_export_writes_every_table_with_the_expected_rows_and_values(tmp_path):
    out_dir = tmp_path / "reports"
    export_results(small_log(tmp_path), out_dir)

    per_query = pd.read_csv(out_dir / "results_per_query.csv")
    assert len(per_query) == 8
    assert per_query[per_query["retrieval_method"] == "dense"]["recall_at_k_all"].tolist() == [1.0, 0.5, 0.0, 1.0]

    aggregated = pd.read_csv(out_dir / "results_aggregated.csv").set_index("retrieval_method")
    assert aggregated.loc["dense", "recall_at_k_all"] == pytest.approx(0.625)  # (1 + 0.5 + 0 + 1) / 4
    assert aggregated.loc["lexical", "recall_at_k_all"] == pytest.approx(0.3125)
    assert aggregated.loc["dense", "n"] == 4

    summary = pd.read_csv(out_dir / "results_summary.csv")
    row = summary[(summary["retrieval_method"] == "dense") & (summary["metric"] == "recall_at_k_all")].iloc[0]
    assert row["mean"] == pytest.approx(0.625) and row["n"] == 4
    assert row["ci95_low"] <= 0.625 <= row["ci95_high"]

    comparisons = pd.read_csv(out_dir / "paired_comparisons.csv")
    assert set(comparisons["metric"]) <= set(COMPARED_METRICS)
    recall_row = comparisons[(comparisons["metric"] == "recall_at_k_all") & (comparisons["family"] == "methods_within_kb")].iloc[0]
    assert (recall_row["a"], recall_row["b"]) == ("dense", "lexical")
    assert recall_row["mean_diff"] == pytest.approx(0.3125)

    robustness_per_query = pd.read_csv(out_dir / "robustness_per_query.csv")
    assert len(robustness_per_query) == 8
    robustness_aggregated = pd.read_csv(out_dir / "robustness_aggregated.csv").set_index("perturbation")
    # perturbed faithfulness 0.5, 0.6, 0.7, 0.8 against a baseline of 1.0 -> mean delta -0.35
    assert robustness_aggregated.loc["drop_top_relevant", "faithfulness_delta"] == pytest.approx(-0.35)
    assert robustness_aggregated.loc["control_rerun", "faithfulness_delta"] == pytest.approx(0.0)
    assert (out_dir / "robustness_summary.csv").exists()

    # relative degradation (Eq. 3.13): (1.0 - p) / 1.0 for p = 0.5, 0.6, 0.7, 0.8 -> mean 0.35
    assert robustness_aggregated.loc["drop_top_relevant", "faithfulness_rel_degradation"] == pytest.approx(0.35)
    assert robustness_aggregated.loc["control_rerun", "faithfulness_rel_degradation"] == pytest.approx(0.0)

    # dense beats lexical on NDCG (0.625 vs 0.3125): it is the Study 2 method
    assert "Study 2 retrieval method: dense" in (out_dir / "study_selection.txt").read_text(encoding="utf-8")
    study = pd.read_csv(out_dir / "study_comparisons.csv")
    assert set(study["family"]) == {"study1_methods_on_curated"}  # one KB in this log, so Study 2 has no pairs
    correlations = pd.read_csv(out_dir / "retrieval_generation_correlations.csv")
    assert {"pooled", "curated/dense", "curated/lexical"} <= set(correlations["scope"])
    assert pd.read_csv(out_dir / "method_ranks_by_kb.csv").shape[0] > 0

    for figure in (
        "recall_at_k_all.png",
        "faithfulness.png",
        "abstained.png",
        "robustness_deltas.png",
        "study1_distributions.png",
        "rq4_correlations_spearman.png",
        "rq4_correlations_kendall.png",
    ):
        assert (out_dir / "figures" / figure).stat().st_size > 5000, figure


def test_export_on_empty_results_writes_empty_tables_and_no_figures(tmp_path):
    out_dir = tmp_path / "reports"
    export_results(tmp_path / "never_written.jsonl", out_dir)
    assert pd.read_csv(out_dir / "results_per_query.csv").empty
    assert pd.read_csv(out_dir / "results_aggregated.csv").empty
    assert not (out_dir / "robustness_per_query.csv").exists()
    assert not (out_dir / "figures").exists() or not list((out_dir / "figures").iterdir())


def test_every_method_and_baseline_has_a_fixed_chart_style():
    assert set(METHOD_STYLE) == {"lexical", "dense", "hybrid", "oracle", "random", "none"}
    colours = [style["color"] for style in METHOD_STYLE.values()]
    assert len(set(colours)) == len(colours)
    # reference points are hatched so they are never read as a method under test
    assert all("hatch" in METHOD_STYLE[m] for m in ("oracle", "random", "none"))
    assert all("hatch" not in METHOD_STYLE[m] for m in ("lexical", "dense", "hybrid"))


def test_plots_skip_quietly_when_there_is_nothing_to_draw(tmp_path):
    empty = pd.DataFrame(columns=["kb_variant", "retrieval_method", "metric", "mean", "ci95_low", "ci95_high"])
    plot_metric_by_kb_and_method(empty, "recall_at_k_all", tmp_path / "a.png")
    plot_robustness_deltas(pd.DataFrame(), tmp_path / "b.png")
    assert not (tmp_path / "a.png").exists() and not (tmp_path / "b.png").exists()


def test_bar_heights_and_whiskers_in_the_figure_are_the_table_values(tmp_path, monkeypatch):
    """Reads the numbers back out of the drawn chart."""
    import matplotlib.pyplot as plt

    summary = pd.DataFrame(
        [
            {"kb_variant": "curated", "retrieval_method": "dense", "metric": "m", "mean": 0.8, "ci95_low": 0.7, "ci95_high": 0.85},
            {"kb_variant": "curated", "retrieval_method": "lexical", "metric": "m", "mean": 0.6, "ci95_low": 0.5, "ci95_high": 0.75},
            {"kb_variant": "reduced", "retrieval_method": "dense", "metric": "m", "mean": 0.3, "ci95_low": 0.2, "ci95_high": 0.4},
            {"kb_variant": "reduced", "retrieval_method": "lexical", "metric": "m", "mean": 0.2, "ci95_low": 0.1, "ci95_high": 0.3},
        ]
    )
    captured = {}
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    plot_metric_by_kb_and_method(summary, "m", tmp_path / "m.png")

    ax = captured["fig"].axes[0]
    by_series = {container.get_label(): container for container in ax.containers if hasattr(container, "patches")}
    assert list(by_series) == ["lexical", "dense"]  # fixed method order
    assert [round(p.get_height(), 6) for p in by_series["dense"].patches] == [0.8, 0.3]  # curated, reduced
    assert [round(p.get_height(), 6) for p in by_series["lexical"].patches] == [0.6, 0.2]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["curated", "reduced"]
    segments = by_series["dense"].errorbar.lines[2][0].get_segments()
    assert [[round(y, 6) for _, y in seg] for seg in segments] == [[0.7, 0.85], [0.2, 0.4]]  # the CI, not mean +/- something
    plt.close("all")

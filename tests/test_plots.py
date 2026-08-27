import pandas as pd

from analysis.plots import plot_all_standard_metrics, plot_metric_by_kb_and_method, plot_robustness_deltas

AGG_DF = pd.DataFrame(
    [
        {"kb_variant": "curated", "retrieval_method": "dense", "precision_at_k": 0.8, "recall_at_k": 0.7, "n": 10},
        {"kb_variant": "curated", "retrieval_method": "lexical", "precision_at_k": 0.6, "recall_at_k": 0.5, "n": 10},
        {"kb_variant": "reduced", "retrieval_method": "dense", "precision_at_k": 0.3, "recall_at_k": 0.2, "n": 10},
        {"kb_variant": "reduced", "retrieval_method": "lexical", "precision_at_k": 0.2, "recall_at_k": 0.1, "n": 10},
    ]
)

ROBUSTNESS_AGG_DF = pd.DataFrame(
    [
        {"kb_variant": "curated", "perturbation": "drop_top_relevant", "faithfulness_delta": -0.3, "answer_relevance_delta": -0.1, "context_relevance_delta": -0.2, "n": 5},
        {"kb_variant": "curated", "perturbation": "inject_noise", "faithfulness_delta": -0.1, "answer_relevance_delta": 0.0, "context_relevance_delta": -0.05, "n": 5},
    ]
)


def test_plot_metric_by_kb_and_method_writes_file(tmp_path):
    out_path = tmp_path / "precision_at_k.png"
    plot_metric_by_kb_and_method(AGG_DF, "precision_at_k", out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_metric_by_kb_and_method_skips_empty_df(tmp_path):
    out_path = tmp_path / "should_not_exist.png"
    plot_metric_by_kb_and_method(pd.DataFrame(), "precision_at_k", out_path)
    assert not out_path.exists()


def test_plot_all_standard_metrics_writes_one_file_per_available_metric(tmp_path):
    written = plot_all_standard_metrics(AGG_DF, ["precision_at_k", "recall_at_k", "mrr"], tmp_path)
    # mrr isn't a column in AGG_DF, so only 2 files should be written
    assert len(written) == 2
    assert (tmp_path / "precision_at_k.png").exists()
    assert (tmp_path / "recall_at_k.png").exists()
    assert not (tmp_path / "mrr.png").exists()


def test_plot_robustness_deltas_writes_file(tmp_path):
    out_path = tmp_path / "robustness.png"
    plot_robustness_deltas(ROBUSTNESS_AGG_DF, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_robustness_deltas_skips_empty_df(tmp_path):
    out_path = tmp_path / "should_not_exist.png"
    plot_robustness_deltas(pd.DataFrame(), out_path)
    assert not out_path.exists()

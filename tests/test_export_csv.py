import pandas as pd

from analysis.export_csv import export_results
from experiments.results_store import ResultsStore


def standard_record(kb_variant, retrieval_method, query_id):
    return {
        "record_type": "standard",
        "run_key": f"{kb_variant}:{retrieval_method}:{query_id}",
        "kb_variant": kb_variant,
        "retrieval_method": retrieval_method,
        "query_id": query_id,
        "question_type": "information",
        "top_k": 5,
        "retrieval_metrics": {"precision_at_k": 0.6, "recall_at_k": 0.7, "mrr": 0.8, "ndcg_at_k": 0.9},
        "generation_metrics": {"faithfulness": 0.9, "answer_relevance": 0.8, "context_relevance": 0.7},
    }


def robustness_record(kb_variant, retrieval_method, query_id, perturbation):
    return {
        "record_type": "robustness",
        "run_key": f"{kb_variant}:{retrieval_method}:{query_id}:robustness:{perturbation}",
        "kb_variant": kb_variant,
        "retrieval_method": retrieval_method,
        "query_id": query_id,
        "perturbation": perturbation,
        "noise_chunk_id": None,
        "baseline_generation_metrics": {"faithfulness": 1.0, "answer_relevance": 1.0, "context_relevance": 1.0},
        "perturbed_generation_metrics": {"faithfulness": 0.7, "answer_relevance": 0.9, "context_relevance": 0.8},
        "faithfulness_delta": -0.3,
        "answer_relevance_delta": -0.1,
        "context_relevance_delta": -0.2,
    }


def test_export_results_writes_csvs_and_figures(tmp_path):
    results_path = tmp_path / "results.jsonl"
    store = ResultsStore(results_path)
    store.append(standard_record("curated", "dense", "q1"))
    store.append(standard_record("curated", "lexical", "q1"))
    store.append(robustness_record("curated", "dense", "q1", "drop_top_relevant"))

    out_dir = tmp_path / "reports"
    export_results(results_path, out_dir)

    per_query = pd.read_csv(out_dir / "results_per_query.csv")
    assert len(per_query) == 2

    aggregated = pd.read_csv(out_dir / "results_aggregated.csv")
    assert len(aggregated) == 2  # curated x dense, curated x lexical

    robustness_per_query = pd.read_csv(out_dir / "robustness_per_query.csv")
    assert len(robustness_per_query) == 1

    robustness_aggregated = pd.read_csv(out_dir / "robustness_aggregated.csv")
    assert len(robustness_aggregated) == 1

    figures_dir = out_dir / "figures"
    assert (figures_dir / "precision_at_k.png").exists()
    assert (figures_dir / "faithfulness.png").exists()
    assert (figures_dir / "robustness_deltas.png").exists()


def test_export_results_on_empty_results_does_not_crash(tmp_path):
    results_path = tmp_path / "results.jsonl"  # never written to
    out_dir = tmp_path / "reports"

    export_results(results_path, out_dir)

    per_query = pd.read_csv(out_dir / "results_per_query.csv")
    assert per_query.empty
    aggregated = pd.read_csv(out_dir / "results_aggregated.csv")
    assert aggregated.empty
    assert not (out_dir / "robustness_per_query.csv").exists()
    assert not (out_dir / "figures" / "robustness_deltas.png").exists()

"""The sweep harness on the REAL KB variants, indexes and queries, with only
the language model replaced by tests/fakes.py. This is where the oracle /
random bracket and the perturbation effects on retrieval are checked on real
data without spending anything."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from analysis.aggregate import load_robustness_results, load_standard_results
from data.query_set import save_query_set_jsonl
from experiments.config import BudgetConfig
from experiments.results_store import ResultsStore
from experiments.runner import run_sweep
from tests.conftest import REPO_ROOT
from tests.fakes import RuleBasedClient

pytestmark = pytest.mark.realdata

N_QUERIES = 40


class CacheOnlyClient(RuleBasedClient):
    """A fake LLM whose embedder must never be needed for corpus or query
    text: those come from the real cache. (Writing toy vectors into the real
    cache would poison it.)"""

    def embed_fn(self, model):
        def _fn(texts):
            raise AssertionError(f"cache miss for {len(texts)} text(s): the real embedding cache must not be written by a test")

        return _fn


def real_config(sweep_config, queries, tmp_path, name):
    out = tmp_path / name
    out.mkdir()
    save_query_set_jsonl(queries[:N_QUERIES], out / "query_set.jsonl")
    return replace(
        sweep_config,
        query_set_path=out / "query_set.jsonl",
        results_path=out / "results.jsonl",
        processed_data_dir=REPO_ROOT / sweep_config.processed_data_dir,
        artifacts_root=REPO_ROOT / sweep_config.artifacts_root,
        embedding_cache_path=REPO_ROOT / sweep_config.embedding_cache_path,
        max_workers=1,
        budget=BudgetConfig(max_usd=0.0, ledger_path=out / "ledger.json"),
    )


@pytest.fixture(scope="module")
def real_run(sweep_config, queries, real_retrievers, tmp_path_factory):
    config = real_config(sweep_config, queries, tmp_path_factory.mktemp("real_harness"), "run")
    summary = run_sweep(config, CacheOnlyClient())
    return config, summary


def test_real_sweep_writes_queries_times_configs_rows_with_no_errors(real_run):
    config, summary = real_run
    df = load_standard_results(config.results_path)
    assert summary.errors == 0
    assert len(df) == N_QUERIES * (3 * 5 + 1)  # 3 KB variants x (3 retrievers + oracle + random) + no-retrieval
    assert set(df.groupby(["kb_variant", "retrieval_method"]).size()) == {N_QUERIES}
    assert not df.duplicated(["kb_variant", "retrieval_method", "query_id"]).any()


def test_real_retrievers_fall_between_random_and_oracle_on_every_kb_variant(real_run):
    config, _ = real_run
    df = load_standard_results(config.results_path)
    means = df.groupby(["kb_variant", "retrieval_method"])[["recall_at_k_all", "mrr_all", "ndcg_at_k_all"]].mean()
    for kb in ("curated", "mixed", "reduced"):
        for metric in ("recall_at_k_all", "mrr_all", "ndcg_at_k_all"):
            oracle, random_ = means.loc[(kb, "oracle"), metric], means.loc[(kb, "random"), metric]
            assert random_ < 0.05  # 5 random chunks out of thousands
            for method in ("lexical", "dense", "hybrid"):
                value = means.loc[(kb, method), metric]
                assert random_ < value <= oracle + 1e-9, (kb, method, metric)
                assert value > 10 * max(random_, 0.01), (kb, method, metric)  # far above chance, not just above it


def test_oracle_is_perfect_on_answerable_queries_and_coverage_is_what_limits_it(real_run):
    config, _ = real_run
    df = load_standard_results(config.results_path)
    oracle = df[df["retrieval_method"] == "oracle"]
    answerable = oracle[oracle["coverage"] == 1.0]
    assert (answerable["mrr"] == 1.0).all()
    assert answerable["ndcg_at_k"].apply(lambda v: math.isclose(v, 1.0)).all()
    by_kb = oracle.groupby("kb_variant")
    assert by_kb["coverage"].mean()["curated"] == 1.0 and by_kb["coverage"].mean()["mixed"] == 1.0
    assert by_kb["coverage"].mean()["reduced"] < 1.0
    # on `reduced` the oracle's headline MRR is exactly the share of queries that still have an answer
    assert by_kb["mrr_all"].mean()["reduced"] == pytest.approx(by_kb["coverage"].mean()["reduced"])


def test_reduced_kb_lowers_headline_retrieval_scores_for_every_real_retriever(real_run):
    """The effect `reduced` exists to produce must be visible in the headline metrics."""
    config, _ = real_run
    df = load_standard_results(config.results_path)
    means = df.groupby(["kb_variant", "retrieval_method"])["recall_at_k_all"].mean()
    for method in ("lexical", "dense", "hybrid", "oracle"):
        assert means["reduced", method] < means["curated", method], method


def test_no_two_configurations_have_identical_retrieval_results(real_run):
    """If two methods (or two KB variants) returned the same chunks for every
    query, one of the settings would not actually be taking effect."""
    config, _ = real_run
    retrieved: dict[tuple[str, str], tuple] = {}
    for record in ResultsStore(config.results_path).iter_records():
        if record["record_type"] == "standard" and record["retrieval_method"] != "none":
            key = (record["kb_variant"], record["retrieval_method"])
            retrieved.setdefault(key, []).append((record["query_id"], tuple(c["chunk_id"] for c in record["retrieved"])))
    signatures = {key: tuple(sorted(rows)) for key, rows in retrieved.items()}
    assert len(signatures) == 15
    identical_pairs = {
        frozenset((a, b)) for a in signatures for b in signatures if a < b and signatures[a] == signatures[b]
    }
    # Exactly one pair coincides, and it must: the oracle returns each query's gold
    # chunks, and `mixed` only adds noise documents, which are gold for no query.
    assert identical_pairs == {frozenset((("curated", "oracle"), ("mixed", "oracle")))}


def test_real_perturbations_change_retrieval_metrics_in_the_expected_direction(real_run):
    config, _ = real_run
    df = load_robustness_results(config.results_path)
    assert set(df["perturbation"]) == {"drop_top_relevant", "inject_noise", "control_rerun"}
    drop = df[df["perturbation"] == "drop_top_relevant"]
    inject = df[df["perturbation"] == "inject_noise"]
    control = df[df["perturbation"] == "control_rerun"]

    assert len(control) == 3 * N_QUERIES and (control["recall_at_k_delta"].fillna(0) == 0).all()
    assert len(drop) > 0.5 * 3 * N_QUERIES  # most dense baselines retrieved something relevant to drop
    assert (drop["recall_at_k_delta"] < 0).all()  # removing a relevant chunk always lowers recall
    assert len(inject) > 0.9 * 3 * N_QUERIES
    assert (inject["recall_at_k_delta"].dropna() == 0).all()  # noise replaces a non-relevant chunk: recall untouched


def test_real_noise_chunks_are_off_topic_and_not_relevant(real_run, variant_chunks, queries):
    config, _ = real_run
    focus_of_query = {q.query_id: " ".join(q.focus.lower().split()) for q in queries}
    chunk = {v: {c.chunk_id: c for c in variant_chunks[v]} for v in variant_chunks}
    checked = 0
    for record in ResultsStore(config.results_path).iter_records():
        if record["record_type"] != "robustness" or record["perturbation"] != "inject_noise":
            continue
        noise = chunk[record["kb_variant"]][record["noise_chunk_id"]]
        assert " ".join(noise.focus.lower().split()) != focus_of_query[record["query_id"]]
        assert noise.doc_id != record["query_id"]
        assert record["noise_chunk_id"] not in record["relevant_chunk_ids"]
        assert len(record["perturbed_retrieved"]) == len(record["baseline_retrieved_chunk_ids"]) == 5
        checked += 1
    assert checked > 100


def test_a_second_real_run_retrieves_exactly_the_same_chunks_with_the_same_scores(real_run, sweep_config, queries, tmp_path):
    first_config, _ = real_run
    second_config = real_config(sweep_config, queries, tmp_path, "second")
    run_sweep(second_config, CacheOnlyClient())

    def retrieval_view(config):
        view = {}
        for record in ResultsStore(config.results_path).iter_records():
            if record["record_type"] == "standard":
                view[record["run_key"]] = (record["retrieved"], record["relevant_chunk_ids"], record["seed"])
            elif record["record_type"] == "robustness":
                view[record["run_key"]] = (record["perturbed_retrieved"], record["noise_chunk_id"], record["seed"])
        return view

    assert retrieval_view(first_config) == retrieval_view(second_config)

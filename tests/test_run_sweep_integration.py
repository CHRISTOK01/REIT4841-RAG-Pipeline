"""The sweep harness end to end: real KB files, real BM25/FAISS index building,
real retrieval, real orchestration, real results file. Only the OpenAI API is
simulated (tests/fakes.py), which is what lets these tests crash the run, time
it out and feed it malformed replies on demand."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

from analysis.aggregate import ResultsIntegrityError, load_standard_results, read_records
from analysis.export_csv import export_results
from data.chunking import load_chunks_jsonl
from data.kb_variants import normalize_topic
from data.query_set import load_query_set_jsonl
from experiments.config import GenerationMetricsConfig, RobustnessConfig
from experiments.results_store import ResultsStore
from experiments.runner import ResultsConfigMismatch, run_sweep
from generation.budget import BudgetExceeded
from generation.prompts import SYSTEM_PROMPT_CLOSED_BOOK_V1, SYSTEM_PROMPT_V1, prompt_hash
from tests.fakes import RuleBasedClient
from tests.harness_fixtures import expected_robustness_keys, expected_standard_keys, make_workspace


class SimulatedCrash(BaseException):
    """Stands in for the process being killed: not an Exception, so nothing in
    the runner can catch it and carry on."""


def records_by_type(config) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {"standard": [], "robustness": [], "robustness_skipped": [], "error": []}
    for record in ResultsStore(config.results_path).iter_records():
        grouped[record["record_type"]].append(record)
    return grouped


def query_ids(config) -> list[str]:
    return [q.query_id for q in load_query_set_jsonl(config.query_set_path)]


def stable_view(record: dict) -> dict:
    """A record minus the fields that legitimately differ between two runs."""
    view = json.loads(json.dumps(record))
    view.pop("generated_at", None)
    view.pop("code_version", None)
    for key in ("generation", "perturbed_generation"):
        if key in view:
            view[key].pop("generated_at", None)
    return view


@pytest.fixture
def finished(tmp_path):
    """A complete, uninterrupted sweep over the mini workspace."""
    config = make_workspace(tmp_path)
    client = RuleBasedClient()
    summary = run_sweep(config, client)
    return config, client, summary


# -- completeness ------------------------------------------------------------------------


def test_sweep_writes_exactly_one_row_per_query_per_config(finished):
    config, _, summary = finished
    records = records_by_type(config)
    ids = query_ids(config)
    assert len(ids) == 8

    standard_keys = [r["run_key"] for r in records["standard"]]
    assert set(standard_keys) == expected_standard_keys(config, ids)
    assert len(standard_keys) == len(set(standard_keys)) == 8 * (3 * 5 + 1)  # 3 KBs x 5 methods + no-retrieval

    robustness_keys = [r["run_key"] for r in records["robustness"] + records["robustness_skipped"]]
    assert set(robustness_keys) == expected_robustness_keys(config, ids)  # every perturbation ran or says why not
    assert len(robustness_keys) == len(set(robustness_keys)) == 3 * 8 * 3

    assert records["error"] == []
    assert summary.errors == 0 and summary.written == len(standard_keys) + len(robustness_keys)

    per_config = Counter((r["kb_variant"], r["retrieval_method"]) for r in records["standard"])
    assert set(per_config.values()) == {8}


def test_every_row_logs_what_is_needed_to_reproduce_and_audit_it(finished):
    config, _, _ = finished
    for record in records_by_type(config)["standard"]:
        assert record["query_id"] and record["question"]
        assert record["kb_variant"] in ("curated", "mixed", "reduced", "none")
        assert record["retrieval_method"] in ("lexical", "dense", "hybrid", "oracle", "random", "none")
        assert record["top_k"] == 3
        for chunk in record["retrieved"]:
            assert chunk["chunk_id"] and isinstance(chunk["score"], float) and chunk["text"]
        assert [c["rank"] for c in record["retrieved"]] == list(range(1, len(record["retrieved"]) + 1))
        generation = record["generation"]
        assert generation["raw_text"].startswith("## Summary")
        assert generation["model"] == "gen-model"  # requested
        assert generation["returned_model"] == "fake-model-2024-01-01"  # reported by the API
        assert generation["prompt_hash"] == prompt_hash(generation["prompt_template_id"])
        assert (generation["temperature"], generation["max_tokens"]) == (0.2, 500)
        assert isinstance(generation["seed"], int) and generation["seed"] == record["seed"]
        assert generation["retrieved_chunk_ids"] == [c["chunk_id"] for c in record["retrieved"]]
        metrics = record["generation_metrics"]
        assert metrics["judge_model"] == "judge-model" and metrics["metrics_version"] == "v2"
        assert set(metrics) >= {"faithfulness", "abstention_label", "answer_relevance", "context_relevance", "reference_agreement", "claim_labels"}
        settings = record["settings"]
        assert settings["model"] == "gen-model" and settings["embedding_model"] == "embed-model"
        assert settings["prompt_hash"] == prompt_hash("wellness_summary_v1")
        assert (settings["rrf_k"], settings["hybrid_k_per_retriever"], settings["seed"]) == (60, 10, 123)
        assert record["run_fingerprint"] and record["code_version"]
        assert record["generated_at"].endswith("+00:00")


def test_model_prompt_and_temperature_are_identical_across_all_nine_configs(finished):
    _, client, _ = finished
    system_prompts = Counter(call["messages"][0]["content"] for call in client.generation_calls)
    assert set(system_prompts) == {SYSTEM_PROMPT_V1, SYSTEM_PROMPT_CLOSED_BOOK_V1}
    assert system_prompts[SYSTEM_PROMPT_CLOSED_BOOK_V1] == 8  # only the no-retrieval baseline's 8 queries
    assert {call["model"] for call in client.generation_calls} == {"gen-model"}
    assert {call["temperature"] for call in client.generation_calls} == {0.2}


def test_the_fingerprint_and_settings_are_the_same_on_every_row(finished):
    config, _, _ = finished
    rows = records_by_type(config)
    all_rows = rows["standard"] + rows["robustness"]
    assert len({r["run_fingerprint"] for r in all_rows + rows["robustness_skipped"]}) == 1
    assert len({json.dumps(r["settings"], sort_keys=True) for r in all_rows}) == 1


# -- baselines ---------------------------------------------------------------------------------


def test_oracle_rows_contain_only_gold_chunks_and_random_rows_are_reproducible(finished):
    config, _, _ = finished
    rows = records_by_type(config)["standard"]
    for record in (r for r in rows if r["retrieval_method"] == "oracle"):
        retrieved = [c["chunk_id"] for c in record["retrieved"]]
        gold = set(record["relevant_chunk_ids"])
        assert set(retrieved) <= gold and len(retrieved) == min(3, len(gold))
        if gold:
            assert record["retrieval_metrics"]["mrr"] == 1.0
            assert record["retrieval_metrics"]["ndcg_at_k"] == pytest.approx(1.0)
    for record in (r for r in rows if r["retrieval_method"] == "random"):
        assert len(record["retrieved"]) == 3


def test_no_retrieval_rows_use_the_closed_book_prompt_and_have_no_context_metrics(finished):
    config, _, _ = finished
    none_rows = [r for r in records_by_type(config)["standard"] if r["retrieval_method"] == "none"]
    assert len(none_rows) == 8
    for record in none_rows:
        assert record["kb_variant"] == "none" and record["retrieved"] == []
        assert record["generation"]["prompt_template_id"] == "closed_book_v1"
        assert record["retrieval_metrics"] is None
        assert math.isnan(record["generation_metrics"]["faithfulness"])
        assert math.isnan(record["generation_metrics"]["context_relevance"])
        assert record["generation_metrics"]["reference_agreement"] == 1.0  # still comparable to the reference


def test_oracle_bounds_every_real_retriever_and_random_is_below_it(finished):
    config, _, _ = finished
    df = load_standard_results(config.results_path)
    means = df.groupby(["kb_variant", "retrieval_method"])["recall_at_k_all"].mean()
    for kb in ("curated", "mixed", "reduced"):
        for method in ("lexical", "dense", "hybrid", "random"):
            assert means[kb, method] <= means[kb, "oracle"] + 1e-12
        assert means[kb, "random"] < means[kb, "oracle"]


# -- robustness --------------------------------------------------------------------------------


def test_perturbed_contexts_are_exactly_what_each_perturbation_claims(finished):
    config, _, _ = finished
    rows = records_by_type(config)
    baselines = {r["run_key"]: r for r in rows["standard"]}
    chunks = {v: {c.chunk_id: c for c in load_chunks_jsonl(config.kb_variant_chunks_path(v))} for v in config.kb_variant_names}
    seen = Counter()
    for record in rows["robustness"]:
        baseline = baselines[record["baseline_run_key"]]
        before = [c["chunk_id"] for c in baseline["retrieved"]]
        after = [c["chunk_id"] for c in record["perturbed_retrieved"]]
        relevant = set(baseline["relevant_chunk_ids"])
        protected = relevant | set(baseline["relevant_chunk_ids_lenient"])
        assert record["baseline_retrieved_chunk_ids"] == before
        seen[record["perturbation"]] += 1

        if record["perturbation"] == "control_rerun":
            assert after == before
            assert record["perturbed_retrieved"] == baseline["retrieved"]

        elif record["perturbation"] == "drop_top_relevant":
            top_relevant = next(cid for cid in before if cid in relevant)
            assert record["changed_chunk_id"] == top_relevant
            assert after == [cid for cid in before if cid != top_relevant]  # the rest, same order
            assert record["perturbed_retrieval_metrics"]["recall_at_k"] < baseline["retrieval_metrics"]["recall_at_k"]

        elif record["perturbation"] == "inject_noise":
            noise_id, replaced = record["noise_chunk_id"], record["changed_chunk_id"]
            assert len(after) == len(before)
            assert replaced == [cid for cid in before if cid not in protected][-1]  # lowest-ranked non-relevant
            assert after == [noise_id if cid == replaced else cid for cid in before]  # one slot changed
            assert noise_id not in protected and noise_id not in before
            noise_chunk = chunks[record["kb_variant"]][noise_id]
            query = next(q for q in load_query_set_jsonl(config.query_set_path) if q.query_id == record["query_id"])
            assert normalize_topic(noise_chunk.focus) != normalize_topic(query.focus)
            assert record["perturbed_retrieval_metrics"]["recall_at_k"] == baseline["retrieval_metrics"]["recall_at_k"] or (
                math.isnan(record["perturbed_retrieval_metrics"]["recall_at_k"]) and math.isnan(baseline["retrieval_metrics"]["recall_at_k"])
            )
    assert seen["control_rerun"] == 24 and seen["drop_top_relevant"] > 0 and seen["inject_noise"] > 0


def test_a_perturbation_that_cannot_be_applied_is_logged_with_its_reason(finished):
    config, _, _ = finished
    rows = records_by_type(config)
    baselines = {r["run_key"]: r for r in rows["standard"]}
    skipped = rows["robustness_skipped"]
    assert skipped  # `reduced` leaves some queries with nothing relevant to drop
    for record in skipped:
        baseline = baselines[f"{record['kb_variant']}:dense:{record['query_id']}"]
        retrieved = {c["chunk_id"] for c in baseline["retrieved"]}
        if record["perturbation"] == "drop_top_relevant":
            assert record["reason"] == "no relevant chunk in the baseline context to drop"
            assert not retrieved & set(baseline["relevant_chunk_ids"])
        else:
            assert record["perturbation"] == "inject_noise"
            assert retrieved <= set(baseline["relevant_chunk_ids"]) | set(baseline["relevant_chunk_ids_lenient"])


def test_robustness_deltas_are_perturbed_minus_baseline(finished):
    config, _, _ = finished
    for record in records_by_type(config)["robustness"]:
        for metric in ("faithfulness", "answer_relevance", "context_relevance", "reference_agreement"):
            expected = record["perturbed_generation_metrics"][metric] - record["baseline_generation_metrics"][metric]
            delta = record[f"{metric}_delta"]
            assert delta == pytest.approx(expected) or (math.isnan(delta) and math.isnan(expected))


# -- resumability: kill the run part-way, resume, compare with an uninterrupted run ------------------


@pytest.mark.parametrize("crash_at_call", [1, 7, 60, 200, 333, 501, 640])
def test_crash_and_resume_gives_the_same_rows_with_nothing_duplicated_missing_or_rebilled(tmp_path, crash_at_call):
    reference_config = make_workspace(tmp_path / "reference")
    reference_client = RuleBasedClient()
    run_sweep(reference_config, reference_client)
    reference = {r["run_key"]: stable_view(r) for r in ResultsStore(reference_config.results_path).iter_records()}
    assert crash_at_call < reference_client.chat_call_count  # the crash really does land mid-run

    config = make_workspace(tmp_path / "crashed")
    first_client = RuleBasedClient(fail_on={crash_at_call: SimulatedCrash()})
    with pytest.raises(SimulatedCrash):
        run_sweep(config, first_client)
    after_crash = list(ResultsStore(config.results_path).iter_records())
    assert len(after_crash) < len(reference)  # it stopped early

    second_client = RuleBasedClient()
    run_sweep(config, second_client)
    resumed = list(ResultsStore(config.results_path).iter_records())

    keys = [r["run_key"] for r in resumed]
    assert len(keys) == len(set(keys))  # no duplicated rows
    assert set(keys) == set(reference)  # no missing rows, no extra rows
    assert [r for r in resumed if r["record_type"] == "error"] == []
    # rows finished before the crash were not touched again
    assert resumed[: len(after_crash)] == after_crash
    # no re-billing: the resumed run generated exactly once per row it had to write
    rows_written_on_resume = [r for r in resumed[len(after_crash) :] if r["record_type"] in ("standard", "robustness")]
    assert len(second_client.generation_calls) == len(rows_written_on_resume)
    # and the combined result is what an uninterrupted run produces (same contexts, seeds, noise chunks, scores)
    assert {r["run_key"]: stable_view(r) for r in resumed} == reference

    third_client = RuleBasedClient()
    run_sweep(config, third_client)  # a finished sweep re-run costs nothing
    assert third_client.chat_call_count == 0 and third_client.embed_call_count == 0
    assert len(list(ResultsStore(config.results_path).iter_records())) == len(resumed)


def test_a_write_cut_off_mid_line_is_discarded_and_that_row_is_run_again(tmp_path):
    config = make_workspace(tmp_path)
    run_sweep(config, RuleBasedClient())
    complete = list(ResultsStore(config.results_path).iter_records())

    raw = config.results_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    config.results_path.write_bytes(b"".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 2])  # torn final record

    client = RuleBasedClient()
    run_sweep(config, client)

    assert [r["run_key"] for r in read_records(config.results_path)] == [r["run_key"] for r in complete]
    assert len(client.generation_calls) == 1  # only the torn row was redone


def test_a_torn_line_is_never_silently_dropped_by_the_analysis(tmp_path):
    config = make_workspace(tmp_path)
    run_sweep(config, RuleBasedClient())
    with config.results_path.open("ab") as f:
        f.write(b'{"record_type": "standard", "run_key": "curated:dense:half-writ')
    with pytest.raises(ResultsIntegrityError, match="unreadable"):
        load_standard_results(config.results_path)


def test_duplicate_rows_are_refused_by_the_analysis(tmp_path):
    config = make_workspace(tmp_path)
    run_sweep(config, RuleBasedClient())
    first_line = config.results_path.read_bytes().splitlines(keepends=True)[0]
    with config.results_path.open("ab") as f:
        f.write(first_line)
    with pytest.raises(ResultsIntegrityError, match="duplicate run_key"):
        load_standard_results(config.results_path)


# -- failures are logged and retried, never recorded as results -----------------------------------


@pytest.mark.parametrize(
    "failure,error_type",
    [
        (TimeoutError("simulated API timeout"), "TimeoutError"),
        (ConnectionError("simulated rate limit / connection failure"), "ConnectionError"),
    ],
)
def test_an_api_failure_becomes_an_error_record_and_the_row_is_retried_next_run(tmp_path, failure, error_type):
    config = make_workspace(tmp_path)
    client = RuleBasedClient(fail_on={3: failure, 40: failure})  # one mid-record, one later
    summary = run_sweep(config, client)

    rows = records_by_type(config)
    assert summary.errors == len(rows["error"]) >= 2
    failed_keys = {r["failed_run_key"] for r in rows["error"]}
    assert all(r["error_type"] == error_type and "simulated" in r["error"] for r in rows["error"])
    result_keys = {r["run_key"] for r in rows["standard"] + rows["robustness"]}
    assert not failed_keys & result_keys  # a failed row was not also written as a result

    # error rows are invisible to the analysis and never count as completed
    df = load_standard_results(config.results_path)
    assert len(df) == len(rows["standard"])
    assert not failed_keys & ResultsStore(config.results_path).load_completed_keys()

    run_sweep(config, RuleBasedClient())  # the retry
    final = records_by_type(config)
    ids = query_ids(config)
    assert {r["run_key"] for r in final["standard"]} == expected_standard_keys(config, ids)
    assert {r["run_key"] for r in final["robustness"] + final["robustness_skipped"]} == expected_robustness_keys(config, ids)
    assert len(final["error"]) == len(rows["error"])  # the error log is kept as history


@pytest.mark.parametrize(
    "bad_reply",
    ["", "not json at all", '{"labels": ["supported"]}', '{"labels": ["false", "false", "false", "false"]}', '{"relevant": "yes"}'],
    ids=["empty", "not_json", "wrong_length", "wrong_label_values", "wrong_type"],
)
def test_malformed_judge_replies_are_errors_not_scores(tmp_path, bad_reply):
    config = make_workspace(tmp_path)
    # calls 2-4 are the first record's faithfulness judge and its two retries
    client = RuleBasedClient(fail_on={2: bad_reply, 3: bad_reply, 4: bad_reply})
    run_sweep(config, client)

    rows = records_by_type(config)
    assert len(rows["error"]) == 1
    error = rows["error"][0]
    assert error["error_type"] == "MalformedJudgeResponse" and error["stage"] == "standard"
    assert error["failed_run_key"] not in {r["run_key"] for r in rows["standard"]}
    for record in rows["standard"]:  # no row carries a score invented from the bad reply
        faithfulness = record["generation_metrics"]["faithfulness"]
        assert faithfulness == 1.0 or math.isnan(faithfulness)


@pytest.mark.parametrize(
    "bad_answer",
    ["", "I cannot help with that.", "## Summary\nOnly a summary, no key points.\n", "## Key Points\n1. No summary section. [1]\n"],
    ids=["empty", "refusal", "no_key_points", "no_summary"],
)
def test_unusable_generated_answers_are_errors_not_zero_scores(tmp_path, bad_answer):
    config = make_workspace(tmp_path, robustness=RobustnessConfig(enabled=False), baselines=[])
    client = RuleBasedClient(answer=bad_answer)
    summary = run_sweep(config, client)

    rows = records_by_type(config)
    assert rows["standard"] == [] and rows["robustness"] == []
    assert summary.written == 0 and summary.errors == len(rows["error"]) == 8 * 9
    assert {r["error_type"] for r in rows["error"]} == {"MalformedGenerationOutput"}
    assert load_standard_results(config.results_path).empty
    assert len(client.generation_calls) == 8 * 9 * config.generation.max_attempts  # each was retried, then given up on


def test_a_failed_baseline_row_means_no_robustness_rows_until_it_succeeds(tmp_path):
    config = make_workspace(tmp_path, kb_variant_names=["curated"], retrieval_methods=["dense"], baselines=[])
    client = RuleBasedClient(fail_on={1: TimeoutError("simulated")})  # the first dense baseline fails
    run_sweep(config, client)
    rows = records_by_type(config)
    failed_query = rows["error"][0]["query_id"]
    assert not [r for r in rows["robustness"] + rows["robustness_skipped"] if r["query_id"] == failed_query]

    run_sweep(config, RuleBasedClient())
    rows = records_by_type(config)
    assert len([r for r in rows["robustness"] + rows["robustness_skipped"] if r["query_id"] == failed_query]) == 3


def test_reaching_the_spend_cap_stops_the_sweep_instead_of_logging_errors(tmp_path):
    config = make_workspace(tmp_path)
    client = RuleBasedClient(fail_on={30: BudgetExceeded("cap reached")})
    with pytest.raises(BudgetExceeded):
        run_sweep(config, client)
    rows = records_by_type(config)
    assert rows["error"] == []  # a budget stop is not a per-query failure
    assert 0 < len(rows["standard"]) < 20
    assert client.chat_call_count == 30  # nothing more was attempted


# -- one results file never mixes two setups -----------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"top_k": 2},
        {"seed": 999},
        {"repeat": 1},
        {"generation_metrics": GenerationMetricsConfig(answer_relevance_num_questions=3)},
    ],
    ids=["top_k", "seed", "repeat", "metric_setting"],
)
def test_rerunning_with_a_changed_setting_into_the_same_results_file_is_refused(tmp_path, change):
    config = make_workspace(tmp_path)
    run_sweep(config, RuleBasedClient())
    before = config.results_path.read_bytes()

    client = RuleBasedClient()
    with pytest.raises(ResultsConfigMismatch):
        run_sweep(replace(config, **change), client)
    assert config.results_path.read_bytes() == before  # nothing appended
    assert client.chat_call_count == 0  # and nothing paid for


def test_changing_the_generation_model_or_the_query_set_changes_the_fingerprint(tmp_path):
    config = make_workspace(tmp_path)
    base = config.run_fingerprint("abc")
    assert config.run_fingerprint("abc") == base
    assert config.run_fingerprint("different-query-set-hash") != base
    assert replace(config, generation=replace(config.generation, model="other-model")).run_fingerprint("abc") != base
    assert replace(config, generation=replace(config.generation, judge_model="other-judge")).run_fingerprint("abc") != base
    assert replace(config, generation=replace(config.generation, temperature=0.0)).run_fingerprint("abc") != base


# -- determinism -----------------------------------------------------------------------------------


def test_two_independent_runs_and_a_parallel_run_produce_identical_rows(tmp_path):
    runs = []
    for name, workers in (("a", 1), ("b", 1), ("c", 4)):
        config = make_workspace(tmp_path / name, max_workers=workers)
        run_sweep(config, RuleBasedClient())
        runs.append({r["run_key"]: stable_view(r) for r in ResultsStore(config.results_path).iter_records()})
    assert runs[0] == runs[1]  # same config + same seed -> same retrieval, same noise chunks, same seeds
    assert runs[0] == runs[2]  # the number of worker threads changes nothing


def test_repeat_runs_keep_retrieval_fixed_but_change_the_generation_seed(tmp_path):
    base = make_workspace(tmp_path / "r0")
    repeat = make_workspace(tmp_path / "r1", repeat=1)
    run_sweep(base, RuleBasedClient())
    run_sweep(repeat, RuleBasedClient())
    first = {r["run_key"]: r for r in records_by_type(base)["standard"]}
    second = {r["run_key"]: r for r in records_by_type(repeat)["standard"]}
    assert first.keys() == second.keys()
    for key in first:
        assert first[key]["retrieved"] == second[key]["retrieved"]
        assert first[key]["seed"] != second[key]["seed"]


# -- every table and figure can be rebuilt from the log alone -------------------------------------


def test_reports_are_regenerated_from_the_results_log_with_all_other_data_deleted(finished, tmp_path):
    config, _, _ = finished
    log_only = tmp_path / "elsewhere" / "results.jsonl"
    log_only.parent.mkdir()
    shutil.copy(config.results_path, log_only)
    for path in (config.processed_data_dir, config.artifacts_root):
        shutil.rmtree(path)
    config.embedding_cache_path.unlink()
    config.query_set_path.unlink()

    out_dir = tmp_path / "reports"
    export_results(log_only, out_dir)

    for name in (
        "results_per_query.csv", "results_aggregated.csv", "results_summary.csv", "paired_comparisons.csv",
        "robustness_per_query.csv", "robustness_aggregated.csv", "robustness_summary.csv",
        "figures/recall_at_k_all.png", "figures/faithfulness.png", "figures/abstained.png", "figures/robustness_deltas.png",
    ):  # fmt: skip
        assert (out_dir / name).stat().st_size > 0, name

    # the aggregated table equals a from-scratch recomputation straight from the JSON lines
    raw = [json.loads(line) for line in log_only.read_text(encoding="utf-8").splitlines()]
    aggregated = pd.read_csv(out_dir / "results_aggregated.csv").set_index(["kb_variant", "retrieval_method"])
    assert len(aggregated) == 16
    for (kb, method), row in aggregated.iterrows():
        mine = [r for r in raw if r["record_type"] == "standard" and r["kb_variant"] == kb and r["retrieval_method"] == method]
        assert row["n"] == len(mine) == 8
        if method == "none":
            continue
        recalls = [r["retrieval_metrics"]["recall_at_k"] if r["relevant_chunk_ids"] else 0.0 for r in mine]
        assert row["recall_at_k_all"] == pytest.approx(sum(recalls) / 8)
        answerable = [r["retrieval_metrics"]["recall_at_k"] for r in mine if r["relevant_chunk_ids"]]
        assert row["recall_at_k_n"] == len(answerable)
        assert row["coverage"] == pytest.approx(len(answerable) / 8)
        if answerable:
            assert row["recall_at_k"] == pytest.approx(sum(answerable) / len(answerable))

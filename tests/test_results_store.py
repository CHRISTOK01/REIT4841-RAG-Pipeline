from experiments.results_store import ResultsStore


def test_append_and_iter_records(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append({"run_key": "a", "value": 1})
    store.append({"run_key": "b", "value": 2})

    records = list(store.iter_records())
    assert [r["run_key"] for r in records] == ["a", "b"]


def test_load_completed_keys_on_missing_file(tmp_path):
    store = ResultsStore(tmp_path / "does_not_exist.jsonl")
    assert store.load_completed_keys() == set()


def test_load_completed_keys_after_appends(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append({"run_key": "a"})
    store.append({"run_key": "b"})
    assert store.load_completed_keys() == {"a", "b"}


def test_resume_workflow_skips_already_completed(tmp_path):
    path = tmp_path / "results.jsonl"
    store1 = ResultsStore(path)
    store1.append({"run_key": "a"})
    store1.append({"run_key": "b"})

    # simulate a fresh process resuming
    store2 = ResultsStore(path)
    completed = store2.load_completed_keys()
    to_run = [k for k in ["a", "b", "c"] if k not in completed]
    assert to_run == ["c"]


def test_malformed_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write('{"run_key": "a"}\n')
        f.write("not valid json at all\n")
        f.write('{"run_key": "b"}\n')

    store = ResultsStore(path)
    assert store.load_completed_keys() == {"a", "b"}


def test_nan_metric_values_round_trip(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append({"run_key": "a", "metric": float("nan")})
    records = list(store.iter_records())
    assert records[0]["metric"] != records[0]["metric"]  # NaN != NaN

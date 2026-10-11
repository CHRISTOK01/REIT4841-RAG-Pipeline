import json
import math

from experiments.results_store import ResultsStore


def test_append_and_iter_records_in_order(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append({"run_key": "a", "value": 1})
    store.append({"run_key": "b", "value": 2})
    assert list(store.iter_records()) == [{"run_key": "a", "value": 1}, {"run_key": "b", "value": 2}]


def test_load_completed_keys_on_missing_file(tmp_path):
    assert ResultsStore(tmp_path / "does_not_exist.jsonl").load_completed_keys() == set()


def test_a_fresh_store_object_sees_what_an_earlier_one_wrote(tmp_path):
    path = tmp_path / "results.jsonl"
    ResultsStore(path).append({"run_key": "a"})
    ResultsStore(path).append({"run_key": "b"})
    assert ResultsStore(path).load_completed_keys() == {"a", "b"}


def test_each_record_is_one_line_ending_in_a_unix_newline(tmp_path):
    path = tmp_path / "results.jsonl"
    store = ResultsStore(path)
    store.append({"run_key": "a", "text": "line one\nline two — ünïcödé"})
    store.append({"run_key": "b"})
    raw = path.read_bytes()
    assert raw.count(b"\n") == 2 and raw.endswith(b"\n") and b"\r" not in raw
    assert json.loads(raw.splitlines()[0])["text"] == "line one\nline two — ünïcödé"


def test_malformed_line_is_skipped_and_counted(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"run_key": "a"}\nnot valid json at all\n{"run_key": "b"}\n', encoding="utf-8")
    store = ResultsStore(path)
    assert store.load_completed_keys() == {"a", "b"}
    assert store.malformed_line_count == 1


def test_malformed_count_resets_on_each_read(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('garbage\n{"run_key": "a"}\n', encoding="utf-8")
    store = ResultsStore(path)
    list(store.iter_records())
    list(store.iter_records())
    assert store.malformed_line_count == 1


def test_append_after_a_torn_final_line_does_not_glue_the_new_record_onto_it(tmp_path):
    """Regression: a crash mid-write leaves a fragment with no newline. The
    next record used to be written onto the same line, making both unreadable."""
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"run_key": "a"}\n{"run_key": "b", "half')
    store = ResultsStore(path)
    store.append({"run_key": "c"})
    assert store.load_completed_keys() == {"a", "c"}  # c survives; the fragment is skipped
    assert store.malformed_line_count == 1


def test_discard_torn_tail_removes_only_the_fragment(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"run_key": "a"}\n{"run_key": "b"}\n{"run_key": "c", "half')
    store = ResultsStore(path)
    assert store.discard_torn_tail() is True
    assert path.read_bytes() == b'{"run_key": "a"}\n{"run_key": "b"}\n'
    assert store.discard_torn_tail() is False  # nothing left to repair
    assert path.read_bytes() == b'{"run_key": "a"}\n{"run_key": "b"}\n'


def test_discard_torn_tail_on_a_file_that_is_only_a_fragment_or_missing_or_empty(tmp_path):
    only_fragment = tmp_path / "fragment.jsonl"
    only_fragment.write_bytes(b'{"run_key": "a", "ha')
    assert ResultsStore(only_fragment).discard_torn_tail() is True
    assert only_fragment.read_bytes() == b""

    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert ResultsStore(empty).discard_torn_tail() is False
    assert ResultsStore(tmp_path / "missing.jsonl").discard_torn_tail() is False


def test_nan_metric_values_round_trip_as_nan_not_zero_or_null(tmp_path):
    store = ResultsStore(tmp_path / "results.jsonl")
    store.append({"run_key": "a", "metric": float("nan"), "other": 0.0, "missing": None})
    record = next(store.iter_records())
    assert math.isnan(record["metric"])
    assert record["other"] == 0.0 and record["missing"] is None

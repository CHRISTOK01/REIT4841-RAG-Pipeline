import json
from pathlib import Path

import pytest

from generation.budget import (
    BudgetExceeded,
    SpendTracker,
    UnknownModelPrice,
    chat_cost_usd,
    embedding_cost_usd,
    load_pricing,
)

PRICING = {"chat": {"m": {"input": 0.15, "output": 0.60}}, "embedding": {"e": {"input": 0.02}}}


def test_chat_cost_arithmetic():
    # 2,000,000 prompt tokens at $0.15/M + 500,000 completion tokens at $0.60/M = 0.30 + 0.30
    assert chat_cost_usd(PRICING, "m", 2_000_000, 500_000) == pytest.approx(0.60)
    assert chat_cost_usd(PRICING, "m", 0, 0) == 0.0


def test_embedding_cost_arithmetic():
    assert embedding_cost_usd(PRICING, "e", 5_000_000) == pytest.approx(0.10)  # 5M tokens at $0.02/M


def test_unknown_model_has_no_price():
    with pytest.raises(UnknownModelPrice):
        chat_cost_usd(PRICING, "other", 1, 1)
    with pytest.raises(UnknownModelPrice):
        embedding_cost_usd(PRICING, "other", 1)


def test_tracker_accumulates_and_persists_across_instances(tmp_path):
    ledger = tmp_path / "sub" / "ledger.json"
    first = SpendTracker(max_usd=1.0, ledger_path=ledger, pricing=PRICING)
    first.record_chat("m", 1_000_000, 0)  # $0.15
    first.record_embedding("e", 1_000_000)  # $0.02
    first.save()
    assert json.loads(ledger.read_text())["spent_usd"] == pytest.approx(0.17)

    second = SpendTracker(max_usd=1.0, ledger_path=ledger, pricing=PRICING)
    assert second.spent_usd == pytest.approx(0.17)  # a new process starts from what was already spent
    second.record_chat("m", 0, 1_000_000)  # $0.60
    assert second.spent_usd == pytest.approx(0.77)


def test_cap_applies_to_cumulative_spend_including_previous_runs(tmp_path):
    ledger = tmp_path / "ledger.json"
    first = SpendTracker(max_usd=0.20, ledger_path=ledger, pricing=PRICING)
    first.check("m", "chat")  # nothing spent yet: allowed
    first.record_chat("m", 1_000_000, 100_000)  # $0.21
    first.save()

    second = SpendTracker(max_usd=0.20, ledger_path=ledger, pricing=PRICING)
    with pytest.raises(BudgetExceeded):
        second.check("m", "chat")
    with pytest.raises(BudgetExceeded):
        second.check("e", "embedding")


def test_ledger_is_written_periodically_without_an_explicit_save(tmp_path):
    ledger = tmp_path / "ledger.json"
    tracker = SpendTracker(max_usd=1.0, ledger_path=ledger, pricing=PRICING)
    for _ in range(20):
        tracker.record_chat("m", 1000, 0)
    assert json.loads(ledger.read_text())["calls"] == 20


def test_repo_pricing_file_covers_the_models_the_configs_use():
    import yaml

    pricing = load_pricing(Path("configs/pricing.yaml"))
    generation = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    assert generation["model"] in pricing["chat"]
    assert generation["judge_model"] in pricing["chat"]
    assert generation["embedding_model"] in pricing["embedding"]
    for price in pricing["chat"].values():
        assert 0 < price["input"] < price["output"]

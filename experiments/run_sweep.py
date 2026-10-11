"""Entry point for the full experiment sweep.

    python -m experiments.run_sweep --config configs/full_sweep.yaml --dry-run
    python -m experiments.run_sweep --config configs/full_sweep.yaml

`--dry-run` makes no API call: it prints how many records are still to run and
what they are estimated to cost. A real run prints the same estimate first and
stops if it would take the spend ledger past the cap in the config.

Resumable: rerunning after an interruption skips everything already logged in
the results file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import load_sweep_config
from experiments.cost_estimate import estimate_sweep
from experiments.runner import run_sweep
from generation.budget import SpendTracker
from generation.client import OpenAIClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval x KB-variant experiment sweep.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="print the plan and cost estimate; call nothing")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    estimate = estimate_sweep(config)
    budget = SpendTracker(config.budget.max_usd, config.budget.ledger_path)

    print(
        f"Sweeping {config.retrieval_methods + config.baselines} x {config.kb_variant_names}, "
        f"top_k={config.top_k}, kb_config_hash={config.kb_config_hash}"
    )
    print(f"Estimate: {estimate}")
    print(f"Spend ledger {config.budget.ledger_path}: ${budget.spent_usd:.2f} of ${config.budget.max_usd:.2f} cap used")
    if args.dry_run:
        print("Dry run: nothing was called.")
        return
    if budget.spent_usd + estimate.total_usd > config.budget.max_usd:
        raise SystemExit(
            "Refusing to start: the estimate would take the spend ledger past the cap. "
            "Raise budget.max_usd in the config if that is intended."
        )

    client = OpenAIClient(budget=budget)
    summary = run_sweep(config, client)
    print(f"Done: {summary}. Results: {config.results_path}. Spent so far: ${budget.spent_usd:.2f}")


if __name__ == "__main__":
    main()

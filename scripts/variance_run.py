"""Measure run-to-run variance: run the same configurations on the same
queries several times, changing nothing but the generation/judge seed.

    python -m scripts.variance_run --dry-run
    python -m scripts.variance_run --repeats 5 --queries 30

Each repeat is a normal sweep (same harness, same code path) restricted to the
curated KB and the three real retrievers, with robustness and baselines off,
written to results/variance/repeat_<i>.jsonl. Retrieval is deterministic, so
any difference between repeats comes from the generator (temperature as
configured) and the judge. `python -m scripts.validate_results` reads these
files and reports the spread.

Resumable, and under the same spend cap as everything else.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from data.query_set import load_query_set_jsonl, save_query_set_jsonl
from experiments.config import RobustnessConfig, load_sweep_config
from experiments.cost_estimate import estimate_sweep
from experiments.runner import run_sweep
from generation.budget import SpendTracker
from generation.client import OpenAIClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat runs of one setup to measure generation/judge variance.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--queries", type=int, default=30, help="use the first N queries of the query set")
    parser.add_argument("--out-dir", type=Path, default=Path("results/variance"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_sweep_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.out_dir / "query_subset.jsonl"
    save_query_set_jsonl(load_query_set_jsonl(base.query_set_path)[: args.queries], subset_path)

    configs = [
        replace(
            base,
            kb_variant_names=["curated"],
            baselines=[],
            robustness=RobustnessConfig(enabled=False),
            query_set_path=subset_path,
            results_path=args.out_dir / f"repeat_{i}.jsonl",
            repeat=i,
        )
        for i in range(args.repeats)
    ]
    budget = SpendTracker(base.budget.max_usd, base.budget.ledger_path)
    total = sum(estimate_sweep(c).total_usd for c in configs)
    pending = sum(estimate_sweep(c).pending_records for c in configs)
    print(f"{args.repeats} repeats x {args.queries} queries x {len(base.retrieval_methods)} retrievers on curated: {pending} records still to run, about ${total:.2f}")
    print(f"Spend ledger: ${budget.spent_usd:.2f} of ${base.budget.max_usd:.2f} cap used")
    if args.dry_run:
        print("Dry run: nothing was called.")
        return
    if budget.spent_usd + total > base.budget.max_usd:
        raise SystemExit("Refusing to start: the estimate would take the spend ledger past the cap.")

    client = OpenAIClient(budget=budget)
    for i, config in enumerate(configs):
        print(f"--- repeat {i} ---")
        run_sweep(config, client)
    print(f"Done. Spent so far: ${budget.spent_usd:.2f}")


if __name__ == "__main__":
    main()

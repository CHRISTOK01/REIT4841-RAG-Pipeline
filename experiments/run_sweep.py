"""Entry point for the full experiment sweep.

    python -m experiments.run_sweep --config configs/full_sweep.yaml

Sweeps retrieval_method x kb_variant (see configs/full_sweep.yaml) over the
fixed query set, holding the query set, LLM, prompt template, and top-k fixed
throughout. Resumable: rerunning after an interruption skips any
(kb_variant, retrieval_method, query) already logged in the results file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import load_sweep_config
from experiments.runner import run_sweep
from generation.client import OpenAIClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval x KB-variant experiment sweep.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    client = OpenAIClient()

    print(
        f"Sweeping {len(config.retrieval_methods)} retrieval method(s) x "
        f"{len(config.kb_variant_names)} KB variant(s), top_k={config.top_k}, "
        f"kb_config_hash={config.kb_config_hash}"
    )
    run_sweep(config, client)
    print(f"Done. Results: {config.results_path}")


if __name__ == "__main__":
    main()

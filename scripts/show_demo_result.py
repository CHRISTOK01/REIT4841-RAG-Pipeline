"""Pretty-print one record from a results JSONL file, so you can see the
actual question -> retrieved chunks -> generated summary -> metrics, not just
aggregated numbers.

    python scripts/show_demo_result.py
    python scripts/show_demo_result.py --results results/full_sweep.jsonl --method hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print one standard result record.")
    parser.add_argument("--results", type=Path, default=Path("results/demo_sweep.jsonl"))
    parser.add_argument("--method", default="dense", help="retrieval method to show (lexical/dense/hybrid)")
    args = parser.parse_args()

    with args.results.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("record_type") == "standard" and record["retrieval_method"] == args.method:
                _print_record(record)
                return

    print(f"No standard record found for retrieval_method={args.method!r} in {args.results}")


def _print_record(record: dict) -> None:
    print(f"KB variant:       {record['kb_variant']}")
    print(f"Retrieval method: {record['retrieval_method']}")
    print(f"Question:         {record['question']}")
    print()
    print("Retrieved chunks:")
    for c in record["retrieved"]:
        snippet = c["text"][:100].replace("\n", " ")
        print(f"  - {c['chunk_id']} (score {c['score']:.3f}): {snippet}...")
    print()
    print("Generated summary:")
    print(" ", record["generation"]["output"]["summary"])
    print()
    print("Key points:")
    for kp in record["generation"]["output"]["key_points"]:
        print("  -", kp["text"])
    print()
    print("Retrieval metrics:", record["retrieval_metrics"])
    print("Generation metrics:", record["generation_metrics"])


if __name__ == "__main__":
    main()

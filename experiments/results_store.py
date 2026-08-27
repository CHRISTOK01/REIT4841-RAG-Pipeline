"""Append-only JSONL results store, with resumability.

`load_completed_keys` reads whatever's already in the results file so the
runner can skip (kb_variant, retrieval_method, query) combinations already
logged, avoiding wasted -- and re-paid-for -- OpenAI calls on a rerun after
an interruption. A line that fails to parse (e.g. a partial write from a
process killed mid-flush) is skipped rather than crashing the resume.

Note: Python's json module serializes float('nan') as the bare token `NaN`,
which is not valid per strict RFC 8259 JSON but round-trips fine through
Python's own json.dumps/json.loads (the default `allow_nan=True` on both
ends) -- which is all this project needs it for. See
evaluation/retrieval_metrics.py for why NaN shows up here at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class ResultsStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_completed_keys(self) -> set[str]:
        keys: set[str] = set()
        for record in self.iter_records():
            key = record.get("run_key")
            if key is not None:
                keys.add(key)
        return keys

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def iter_records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping malformed results line in {self.path}: {line[:100]!r}...")
                    continue

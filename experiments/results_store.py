"""Append-only JSONL results store, with resumability.

One JSON object per line. `append` is safe against a torn final line: if a
previous process died mid-write and left the file without a trailing newline,
a newline is written first so the new record never gets glued onto the
fragment. Each append is flushed and fsynced, so a record that `append`
returned from is on disk.

A line that fails to parse is skipped when reading and counted in
`malformed_line_count`; callers that must not lose data (the analysis layer)
check that count.

Note: Python's json module serializes float('nan') as the bare token `NaN`,
which is not strict RFC 8259 JSON but round-trips through Python's own
json.dumps/json.loads -- which is all this project needs it for.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator


class ResultsStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.malformed_line_count = 0
        self._lock = threading.Lock()

    def load_completed_keys(self) -> set[str]:
        return {record["run_key"] for record in self.iter_records() if record.get("run_key") is not None}

    def _ends_mid_line(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"

    def discard_torn_tail(self) -> bool:
        """If the file ends in a partial line (a write that was cut off by a
        crash), remove that fragment. It is an incomplete record: its run is
        not marked completed, so it is simply run again. Returns True if a
        fragment was removed."""
        with self._lock:
            if not self._ends_mid_line():
                return False
            data = self.path.read_bytes()
            keep = data.rfind(b"\n") + 1  # 0 if there is no complete line at all
            with self.path.open("r+b") as f:
                f.truncate(keep)
            print(f"Removed a torn final line ({len(data) - keep} bytes) from {self.path}")
            return True

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            repair = self._ends_mid_line()
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                if repair:
                    f.write("\n")
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def iter_records(self) -> Iterator[dict[str, Any]]:
        self.malformed_line_count = 0
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
                    self.malformed_line_count += 1
                    print(f"Skipping malformed results line in {self.path}: {line[:100]!r}...")
                    continue

"""Common interface every retriever (lexical, dense, hybrid) implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int  # 1-based rank within this retriever's result list


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...

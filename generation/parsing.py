"""Parse a generated wellness-summary response into its structured sections.

Deterministic, LLM-independent logic -- this is what lets the evaluation
layer check individual claims against retrieved context without re-calling
an LLM to do the parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SECTION_RE = re.compile(
    r"##\s*(Summary|Key Points|Disclaimer)\s*\n(.*?)(?=\n##\s*\w|\Z)", re.IGNORECASE | re.DOTALL
)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_KEY_POINT_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$")


@dataclass(frozen=True)
class KeyPoint:
    text: str
    cited_passage_numbers: list[int]
    cited_chunk_ids: list[str]


@dataclass(frozen=True)
class GenerationOutput:
    raw_text: str
    summary: str
    key_points: list[KeyPoint]
    disclaimer: str


def parse_generation_output(raw_text: str, passage_map: dict[int, str]) -> GenerationOutput:
    sections: dict[str, str] = {}
    for match in _SECTION_RE.finditer(raw_text):
        heading = match.group(1).strip().lower()
        sections[heading] = match.group(2).strip()

    key_points: list[KeyPoint] = []
    for line in sections.get("key points", "").splitlines():
        line = line.strip()
        if not line:
            continue
        line_match = _KEY_POINT_LINE_RE.match(line)
        if not line_match:
            continue
        point_text = line_match.group(1)
        passage_numbers = sorted({int(n) for n in _CITATION_RE.findall(point_text)})
        chunk_ids = [passage_map[n] for n in passage_numbers if n in passage_map]
        key_points.append(
            KeyPoint(text=point_text, cited_passage_numbers=passage_numbers, cited_chunk_ids=chunk_ids)
        )

    return GenerationOutput(
        raw_text=raw_text,
        summary=sections.get("summary", ""),
        key_points=key_points,
        disclaimer=sections.get("disclaimer", ""),
    )

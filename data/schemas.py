"""Data structures shared across the data ingestion / chunking / KB-variant pipeline.

A `SourceDocument` corresponds to one MedQuAD QAPair: a single question paired
with its answer text, tagged with the topic (`focus`) and NIH source subset it
came from. `Chunk` is a sub-span of a `SourceDocument`'s answer text produced by
the chunker in `chunking.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str  # stable id: f"{subset}:{document_id}:{qid}"
    subset: str  # e.g. "CancerGov", "GARD"
    document_id: str  # MedQuAD <Document id=...>, groups QAPairs about one focus
    focus: str  # topic/entity, e.g. "Adult Acute Lymphoblastic Leukemia"
    question_id: str  # MedQuAD <Question qid=...>
    question_type: str  # e.g. "symptoms", "treatment", "information"
    question: str
    answer: str
    url: str | None
    umls_cui: str | None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}#{chunk_index}"
    doc_id: str
    chunk_index: int
    text: str
    subset: str
    focus: str
    question_type: str
    source_question: str  # the question whose answer this chunk was cut from


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_words: int = 120
    chunk_overlap_words: int = 20

    def __post_init__(self) -> None:
        if self.chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if self.chunk_overlap_words < 0:
            raise ValueError("chunk_overlap_words cannot be negative")
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

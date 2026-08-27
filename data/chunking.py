"""Split SourceDocument answer text into overlapping, fixed-size word-count chunks.

Word count (not tokens) is used deliberately: it needs no tokenizer dependency,
is trivial to reproduce/cite in a methodology chapter, and MedQuAD answers are
short enough (mostly single paragraphs) that the word/token distinction has
negligible effect on chunk boundaries at this corpus scale.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.schemas import Chunk, ChunkingConfig, SourceDocument


def chunk_document(doc: SourceDocument, config: ChunkingConfig) -> list[Chunk]:
    """Slide a `chunk_size_words`-word window over doc.answer with
    `chunk_overlap_words` words of overlap between consecutive chunks."""
    words = doc.answer.split()
    if not words:
        return []

    step = config.chunk_size_words - config.chunk_overlap_words
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while True:
        window = words[start : start + config.chunk_size_words]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{index}",
                doc_id=doc.doc_id,
                chunk_index=index,
                text=" ".join(window),
                subset=doc.subset,
                focus=doc.focus,
                question_type=doc.question_type,
                source_question=doc.question,
            )
        )
        index += 1
        if start + config.chunk_size_words >= len(words):
            break
        start += step
    return chunks


def chunk_documents(docs: list[SourceDocument], config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, config))
    return chunks


def save_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    return chunks

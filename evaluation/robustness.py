"""Robustness / sensitivity test: mechanically perturb a retrieved-context list
so the harness can re-generate on the perturbed context and measure how much
generation quality shifts.

Three conditions, each applied to the same baseline context:

- `drop_top_relevant`: remove the highest-ranked relevant chunk. The context
  shrinks from k to k-1 chunks; nothing else changes.
- `inject_noise`: replace the lowest-ranked chunk that is NOT relevant with an
  off-topic noise chunk, in the same slot. The context keeps its size and
  every relevant chunk keeps its place.
- `control_rerun`: no change at all. Re-generating on the identical context
  measures how much the scores move from generation/judge randomness alone;
  a perturbation effect smaller than that is noise.

This module only does the mechanical perturbation; it has no OpenAI dependency
and is fully deterministic given the caller's random generator.
"""

from __future__ import annotations

import random

from data.kb_variants import normalize_topic
from data.schemas import Chunk
from retrieval.base import RetrievedChunk

PERTURBATIONS = ("drop_top_relevant", "inject_noise", "control_rerun")


def _renumber_ranks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=c.score, rank=i)
        for i, c in enumerate(chunks, start=1)
    ]


def drop_top_relevant(
    retrieved: list[RetrievedChunk], relevant_chunk_ids: set[str]
) -> tuple[list[RetrievedChunk], str | None]:
    """Remove the highest-ranked retrieved chunk that is relevant. Returns
    (new list with ranks renumbered, dropped chunk_id), or (unchanged copy,
    None) if nothing relevant was retrieved."""
    for i, chunk in enumerate(retrieved):
        if chunk.chunk_id in relevant_chunk_ids:
            return _renumber_ranks(retrieved[:i] + retrieved[i + 1 :]), chunk.chunk_id
    return list(retrieved), None


def replace_lowest_nonrelevant_with_noise(
    retrieved: list[RetrievedChunk], noise_chunk: Chunk, protected_chunk_ids: set[str]
) -> tuple[list[RetrievedChunk], str | None]:
    """Swap the lowest-ranked chunk not in `protected_chunk_ids` for
    `noise_chunk`, in place. Returns (new list, replaced chunk_id), or
    (unchanged copy, None) if every retrieved chunk is protected."""
    if any(c.chunk_id == noise_chunk.chunk_id for c in retrieved):
        raise ValueError("noise chunk is already in the retrieved list")
    for i in range(len(retrieved) - 1, -1, -1):
        target = retrieved[i]
        if target.chunk_id not in protected_chunk_ids:
            noise_item = RetrievedChunk(
                chunk_id=noise_chunk.chunk_id,
                doc_id=noise_chunk.doc_id,
                text=noise_chunk.text,
                score=target.score,  # takes over the slot; it has no retrieval score of its own
                rank=target.rank,
            )
            return _renumber_ranks(retrieved[:i] + [noise_item] + retrieved[i + 1 :]), target.chunk_id
    return list(retrieved), None


def select_noise_chunk(
    chunks: list[Chunk],
    exclude_focus: str,
    rng: random.Random,
    exclude_chunk_ids: set[str] = frozenset(),
    exclude_texts: set[str] = frozenset(),
) -> Chunk | None:
    """Pick a random chunk to use as an injected distractor: its topic differs
    from `exclude_focus`, and it is neither one of `exclude_chunk_ids` (the
    query's relevant chunks and whatever was already retrieved) nor a verbatim
    copy of one of `exclude_texts`. Returns None if no such chunk exists."""
    excluded_topic = normalize_topic(exclude_focus)
    candidates = [
        c
        for c in chunks
        if normalize_topic(c.focus) != excluded_topic
        and c.chunk_id not in exclude_chunk_ids
        and c.text not in exclude_texts
    ]
    if not candidates:
        return None
    return rng.choice(candidates)

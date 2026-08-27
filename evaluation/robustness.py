"""Robustness / sensitivity test: mechanically perturb a retrieved-context list
(drop the top relevant chunk, inject an off-topic distractor chunk) so the
harness can re-generate on the perturbed context and measure how much
generation quality degrades. Ties directly to the "sensitivity to retrieval
variation" objective in the thesis proposal.

This module only does the mechanical perturbation + delta bookkeeping; it has
no OpenAI dependency and is fully deterministic/testable. Choosing *which*
noise chunk to inject and running generation twice (baseline vs perturbed) is
the experiment harness's job.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from data.schemas import Chunk
from retrieval.base import RetrievedChunk


def _renumber_ranks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=c.score, rank=i)
        for i, c in enumerate(chunks, start=1)
    ]


def drop_top_relevant(
    retrieved: list[RetrievedChunk], relevant_chunk_ids: set[str]
) -> list[RetrievedChunk]:
    """Remove the highest-ranked retrieved chunk that is actually relevant (if
    any), simulating loss of the single most useful retrieved passage. Ranks
    of the remaining chunks are renumbered contiguously."""
    for i, chunk in enumerate(retrieved):
        if chunk.chunk_id in relevant_chunk_ids:
            return _renumber_ranks(retrieved[:i] + retrieved[i + 1 :])
    return list(retrieved)  # nothing relevant was retrieved -- nothing to drop


def inject_noise_chunk(
    retrieved: list[RetrievedChunk], noise_chunk: Chunk, position: int = 1
) -> list[RetrievedChunk]:
    """Insert an unrelated chunk at 1-based `position`, pushing the rest down,
    simulating a distractor surfacing in the retrieved context."""
    noise_item = RetrievedChunk(
        chunk_id=noise_chunk.chunk_id,
        doc_id=noise_chunk.doc_id,
        text=noise_chunk.text,
        score=retrieved[0].score if retrieved else 0.0,
        rank=position,
    )
    index = max(0, min(position - 1, len(retrieved)))
    return _renumber_ranks(retrieved[:index] + [noise_item] + retrieved[index:])


def select_noise_chunk(chunks: list[Chunk], exclude_focus: str, rng: random.Random) -> Chunk | None:
    """Pick a random chunk whose topic differs from `exclude_focus`, to use as
    an injected distractor. Returns None if no such chunk exists."""
    candidates = [c for c in chunks if c.focus != exclude_focus]
    if not candidates:
        return None
    return rng.choice(candidates)


@dataclass(frozen=True)
class RobustnessComparison:
    baseline_faithfulness: float
    perturbed_faithfulness: float
    baseline_answer_relevance: float
    perturbed_answer_relevance: float
    baseline_context_relevance: float
    perturbed_context_relevance: float

    @property
    def faithfulness_delta(self) -> float:
        return self.perturbed_faithfulness - self.baseline_faithfulness

    @property
    def answer_relevance_delta(self) -> float:
        return self.perturbed_answer_relevance - self.baseline_answer_relevance

    @property
    def context_relevance_delta(self) -> float:
        return self.perturbed_context_relevance - self.baseline_context_relevance

"""Select a fixed, reproducible query set from the curated base corpus.

Each query is a MedQuAD question that is itself one QA pair's own
SourceDocument -- `query_id` doubles as that document's `doc_id`, so
evaluation/relevance_judgments.py can later find "the chunks this query's own
answer was cut into" inside any KB variant's chunk list, without a separate
mapping table.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from data.ingest_medquad import load_source_documents_jsonl
from data.schemas import SourceDocument


@dataclass(frozen=True)
class QueryItem:
    query_id: str  # == source SourceDocument.doc_id
    question: str
    subset: str
    focus: str
    question_type: str


@dataclass(frozen=True)
class QuerySetConfig:
    seed: int = 42
    num_queries: int = 40

    def __post_init__(self) -> None:
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive")


def sample_query_set(
    curated_docs: list[SourceDocument], config: QuerySetConfig
) -> list[QueryItem]:
    """Stratified sample by question_type, so the fixed query set spans a mix
    of question intents (symptoms, treatment, diagnosis, ...) rather than
    skewing toward whichever type happens to be most common in the corpus."""
    rng = random.Random(config.seed)
    by_type: dict[str, list[SourceDocument]] = defaultdict(list)
    for d in curated_docs:
        by_type[d.question_type].append(d)
    for group in by_type.values():
        group.sort(key=lambda d: d.doc_id)
        rng.shuffle(group)

    n = min(config.num_queries, len(curated_docs))
    total = len(curated_docs)
    quotas = {t: (len(g) / total) * n for t, g in by_type.items()}
    floor_quotas = {t: int(q) for t, q in quotas.items()}
    remainder = n - sum(floor_quotas.values())
    by_fractional_part_desc = sorted(quotas, key=lambda t: quotas[t] - floor_quotas[t], reverse=True)
    for t in by_fractional_part_desc[:remainder]:
        floor_quotas[t] += 1

    selected: list[SourceDocument] = []
    for t, group in by_type.items():
        take = min(floor_quotas.get(t, 0), len(group))
        selected.extend(group[:take])
    selected.sort(key=lambda d: d.doc_id)  # stable, deterministic final order

    return [
        QueryItem(
            query_id=d.doc_id,
            question=d.question,
            subset=d.subset,
            focus=d.focus,
            question_type=d.question_type,
        )
        for d in selected
    ]


def save_query_set_jsonl(queries: list[QueryItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def load_query_set_jsonl(path: Path) -> list[QueryItem]:
    queries: list[QueryItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(QueryItem(**json.loads(line)))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a fixed query set from the curated KB documents."
    )
    parser.add_argument(
        "--curated-docs", type=Path, default=Path("datasets/processed/kb_curated/documents.jsonl")
    )
    parser.add_argument("--num-queries", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("datasets/processed/query_set.jsonl"))
    args = parser.parse_args()

    curated_docs = load_source_documents_jsonl(args.curated_docs)
    config = QuerySetConfig(seed=args.seed, num_queries=args.num_queries)
    queries = sample_query_set(curated_docs, config)
    save_query_set_jsonl(queries, args.out)

    type_counts: dict[str, int] = defaultdict(int)
    for q in queries:
        type_counts[q.question_type] += 1

    print(f"Sampled {len(queries)} queries from {len(curated_docs)} curated documents.")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t}: {c}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

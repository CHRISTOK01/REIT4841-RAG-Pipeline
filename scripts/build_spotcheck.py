"""Build the manual spot-check pack: sampled result rows shown in full (query,
reference answer, retrieved chunks, generated answer, every score and the
judge's per-claim labels) for a human to read.

    python -m scripts.build_spotcheck --config configs/full_sweep.yaml

Sampling (seeded, so the pack is reproducible): 15 rows from the nine real
configurations --
  - 9 drawn at random, one per configuration;
  - 6 chosen because the automated scores disagree with each other, two of each kind:
      A. retrieval looks good (gold chunk at rank 1) but faithfulness is low;
      B. retrieval found nothing relevant, yet the answer is judged faithful and non-abstaining;
      C. the answer is judged fully faithful to the context but disagrees with the reference answer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from analysis.aggregate import read_records
from data.ingest_medquad import load_source_documents_jsonl
from experiments.config import load_sweep_config

REAL_METHODS = ("lexical", "dense", "hybrid")


def is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt(x) -> str:
    return "undefined" if is_nan(x) else f"{x:.2f}"


def pick(pool: list[dict], n: int, rng: random.Random, taken: set[str]) -> list[dict]:
    candidates = [r for r in pool if r["run_key"] not in taken]
    rng.shuffle(candidates)
    chosen = candidates[:n]
    taken.update(r["run_key"] for r in chosen)
    return chosen


def render(index: int, why: str, record: dict, reference: str) -> list[str]:
    gm, rm, gen = record["generation_metrics"], record["retrieval_metrics"], record["generation"]
    relevant = set(record["relevant_chunk_ids"])
    lenient = set(record["relevant_chunk_ids_lenient"])
    lines = [
        "",
        "---",
        "",
        f"## Case {index}: {record['kb_variant']} / {record['retrieval_method']}",
        "",
        f"*Why this case is here:* {why}",
        "",
        f"**Query** (`{record['query_id']}`): {record['question']}",
        "",
        f"**Reference answer (MedQuAD; never shown to the generator):** {reference[:1200]}{' …[truncated]' if len(reference) > 1200 else ''}",
        "",
        f"**Relevant chunks in this KB:** {len(relevant)} strict, {len(lenient)} lenient.",
        "",
        "**Retrieved context, as shown to the generator:**",
        "",
    ]
    for chunk, judged in zip(record["retrieved"], gm["context_labels"]):
        label = "GOLD" if chunk["chunk_id"] in relevant else ("lenient-relevant" if chunk["chunk_id"] in lenient else "not relevant")
        lines.append(
            f"{chunk['rank']}. `{chunk['chunk_id']}` — label: **{label}**; judge says passage is {'relevant' if judged else 'not relevant'}; score {chunk['score']:.4f}"
        )
        lines.append(f"   > {chunk['text']}")
    lines += ["", "**Generated answer:**", "", "```", gen["raw_text"].strip(), "```", "", "**Judge's label for each claim** (context / reference answer):", ""]
    claims = [s for s in _claims(gen["output"])]
    reference_labels = gm.get("reference_labels") or [""] * len(claims)
    for claim, label, ref_label in zip(claims, gm["claim_labels"] or [""] * len(claims), reference_labels):
        lines.append(f"- [{label or 'n/a'} / {ref_label or 'n/a'}] {claim}")
    lines += [
        "",
        "| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {fmt(rm['precision_at_k'])} | {fmt(rm['recall_at_k'])} | {fmt(rm['mrr'])} | {fmt(rm['ndcg_at_k'])} | {fmt(gm['faithfulness'])} | {gm['abstention_label']} | {fmt(gm['answer_relevance'])} | {fmt(gm['context_relevance'])} | {fmt(gm['reference_agreement'])} |",
        "",
        "**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:",
    ]
    return lines


def _claims(output: dict) -> list[str]:
    from evaluation.generation_metrics import extract_claims
    from generation.parsing import GenerationOutput, KeyPoint

    parsed = GenerationOutput(
        raw_text=output["raw_text"],
        summary=output["summary"],
        key_points=[KeyPoint(**kp) for kp in output["key_points"]],
        disclaimer=output["disclaimer"],
    )
    return extract_claims(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the manual spot-check pack from a results log.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/validation/spot_check_pack.md"))
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    rng = random.Random(args.seed)
    rows = [
        r
        for r in read_records(config.results_path)
        if r["record_type"] == "standard" and r["retrieval_method"] in REAL_METHODS
    ]
    references = {d.doc_id: d.answer for d in load_source_documents_jsonl(config.processed_data_dir / "kb_curated" / "documents.jsonl")}
    taken: set[str] = set()
    cases: list[tuple[str, dict]] = []

    def faith(r):
        return r["generation_metrics"]["faithfulness"]

    kind_a = [r for r in rows if r["retrieval_metrics"]["mrr"] == 1.0 and not is_nan(faith(r)) and faith(r) <= 0.6]
    kind_b = [
        r
        for r in rows
        if r["retrieval_metrics"]["precision_at_k"] == 0.0
        and r["generation_metrics"]["abstention_label"] == "answered"
        and not is_nan(faith(r))
        and faith(r) >= 0.99
    ]
    kind_c = [
        r
        for r in rows
        if not is_nan(faith(r))
        and faith(r) >= 0.99
        and not is_nan(r["generation_metrics"]["reference_agreement"])
        and r["generation_metrics"]["reference_agreement"] <= 0.34
    ]
    for why, pool in (
        (f"**metrics disagree (A):** the gold chunk was retrieved at rank 1, yet faithfulness is low. ({len(kind_a)} such rows in the sweep.)", kind_a),
        (f"**metrics disagree (B):** nothing relevant was retrieved (P@k = 0), yet the answer does not abstain and is judged fully faithful. ({len(kind_b)} such rows.)", kind_b),
        (f"**metrics disagree (C):** every claim is judged supported by the retrieved context, but at most a third agree with the reference answer. ({len(kind_c)} such rows.)", kind_c),
    ):
        for record in pick(pool, 2, rng, taken):
            cases.append((why, record))

    for kb in config.kb_variant_names:
        for method in REAL_METHODS:
            pool = [r for r in rows if r["kb_variant"] == kb and r["retrieval_method"] == method]
            for record in pick(pool, 1, rng, taken):
                cases.append(("random draw for this configuration.", record))

    lines = [
        "# Manual spot-check pack",
        "",
        f"{len(cases)} result rows from `{config.results_path.as_posix()}`, sampled with seed {args.seed} by `python -m scripts.build_spotcheck`.",
        "The first 6 were selected because the automated scores disagree with each other; the other 9 are one random row per real configuration.",
        "",
        "How to read a case: chunk labels are the pipeline's relevance labels (GOLD = cut from this query's own MedQuAD answer). Each claim line shows the judge's label against the retrieved context, then against the reference answer.",
    ]
    for i, (why, record) in enumerate(cases, start=1):
        lines += render(i, why, record, references[record["query_id"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.out} ({len(cases)} cases; disagreement pools: A={len(kind_a)}, B={len(kind_b)}, C={len(kind_c)})")
    summary = [{"case": i, "run_key": r["run_key"]} for i, (_, r) in enumerate(cases, start=1)]
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

# RAG Retrieval-Quality Evaluation Pipeline

Research codebase for a UQ REIT4841 thesis investigating:

> How does retrieval quality influence the factual accuracy, relevance, and reliability of
> outputs in a RAG-based pipeline for medical report/wellness summary generation?

This is a controlled experiment harness, not a product: a config-driven, reproducible pipeline
for comparing retrieval methods (lexical / dense / hybrid) and knowledge-base quality variations
(curated / mixed / reduced), and measuring the downstream effect on LLM-generated output quality.

Pure Python, CPU-only, no web framework. Generation and embeddings use the OpenAI API; vector
search uses FAISS (`faiss-cpu`); lexical search uses BM25 (`rank_bm25`).

Related documents:

- `AUDIT.md` — the audit of the original code (what was wrong and why it mattered).
- `TESTING.md` — what every test category checks, the requirement-to-test map, mutation results,
  known limitations.
- `reports/validation/` — data-validation report, spot-check pack, relevance-label check,
  mutation report, coverage.

## Data source

Corpus: [MedQuAD](https://github.com/abachaa/MedQuAD) (Ben Abacha & Demner-Fushman, 2019,
*"A Question-Entailment Approach to Question Answering,"* BMC Bioinformatics), CC BY 4.0.
Only the 9 subsets that retain full answer text are used (`CancerGov`, `GARD`, `GHR`,
`MPlusHealthTopics`, `NIDDK`, `NINDS`, `SeniorHealth`, `NHLBI`, `CDC`): 16,407 QA pairs. The other
3 (MedlinePlus ADAM, Drugs, Herbs/Supplements) have their answer text stripped in the source repo
for copyright reasons and are skipped. No other data source is used.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a file named `.env` in the repo root containing `OPENAI_API_KEY=...` (it is gitignored).

## Verifying the pipeline

```bash
python -m scripts.verify_all                  # every deterministic check + one pass/fail summary
python -m pytest                              # fast tests + real-data tests, no API calls
python -m pytest --run-api                    # also the real-API tests (a few cents, hard-capped)
python -m scripts.mutation_test               # inject deliberate bugs, confirm the tests catch them
```

## Running the experiment

```bash
# 1. Download MedQuAD and parse the usable subsets into one JSONL corpus
python -m data.ingest_medquad
# 2. Build the curated / mixed / reduced KB variants (configs/kb_variants.yaml)
python -m data.kb_variants
# 3. Sample the fixed query set from the curated corpus (configs/query_set.yaml)
python -m data.query_set
# 4. Build BM25 + FAISS indexes (embeds any text not already cached; a few cents)
python -m scripts.build_indexes
# 5. See what the sweep will cost, then run it (resumable; hard spend cap in the config)
python -m experiments.run_sweep --config configs/full_sweep.yaml --dry-run
python -m experiments.run_sweep --config configs/full_sweep.yaml
# 6. Validate the results, then export tables and figures
python -m scripts.validate_results
python -m analysis.export_csv --results results/full_sweep.jsonl --out-dir reports
python -m scripts.build_spotcheck
# Optional: run-to-run variance
python -m scripts.variance_run --dry-run
```

At the default config, from the real MedQuAD download:

| Variant | Documents | Chunks | How it is built |
|---|---|---|---|
| curated | 1,500 | 3,696 | quality-filtered (answer ≥ 15 words), stratified by source subset |
| mixed | 1,950 | 4,595 | curated + 450 noise documents (ratio 0.30) |
| reduced | 987 | 2,425 | curated with 35% of topics removed entirely |

## Design decisions to cite in the methodology

**Knowledge-base variants** (`data/kb_variants.py`)

- `mixed` noise is drawn only from topics that do not occur in `curated`, and a document is
  excluded if any of its chunks repeats a curated chunk word for word. Noise can therefore not be a
  second correct source for a query. The build fails rather than use less noise than configured.
- `reduced` removes whole topics (MedQuAD `Focus`, compared case-insensitively). 39 of the 150
  queries lose every relevant chunk in `reduced`.
- Builds are deterministic: each variant's manifest stores a SHA-256 of its chunk file, and a test
  rebuilds the variants in fresh processes and compares.

**Queries and relevance labels** (`data/query_set.py`, `evaluation/relevance_judgments.py`)

- 150 queries, stratified by MedQuAD question type, each the question of one curated QA pair.
- *Strict* label: a chunk is relevant iff it was cut from that query's own answer.
- *Lenient* label: strict, plus chunks with text identical to one of the query's answer chunks,
  plus chunks from a document with the same topic and question type.
- Every retrieval metric is reported under both. These are document-of-origin labels, not
  clinician-verified passage labels; `reports/validation/relevance_label_sample.md` shows a hand
  check of 60 pairs.

**Retrieval** (`retrieval/`)

- `lexical`: BM25 (`rank_bm25` defaults). Returns only chunks sharing a term with the query, so
  the list can be shorter than k. Ties: corpus order.
- `dense`: `text-embedding-3-small`, exact FAISS inner product on L2-normalised vectors (= cosine).
  Ties: corpus order.
- `hybrid`: reciprocal rank fusion of the top 50 from each, constant 60. Ties: chunk id.
- Embeddings are cached in SQLite keyed by (model, text). A saved index is reused only if its
  fingerprint matches the current chunks and embedding model.

**Retrieval metrics** (`evaluation/retrieval_metrics.py`, `analysis/aggregate.py`)

- Precision@5, Recall@5, MRR@5, NDCG@5, binary relevance, rank = position in the returned list.
- A query with no relevant chunk in a KB variant is reported two ways: counted as 0 in the
  headline `<metric>_all` columns, and excluded from the answerable-only `<metric>` columns.
  `coverage` is the share of answerable queries.

**Generation** (`generation/`)

- One model (`gpt-4o-mini-2024-07-18`, a dated snapshot), one prompt template
  (`wellness_summary_v1`), temperature 0.2, for every configuration. The template's SHA-256 is
  logged with every row and pinned by a snapshot test.
- An answer that is cut off, or lacks the Summary or Key Points section, is regenerated (up to 3
  attempts) and otherwise logged as an error. It is never scored.

**Generation metrics** (`evaluation/generation_metrics.py`; LLM judge at temperature 0, JSON mode)

- Claims = every Summary sentence + every Key Point.
- *Faithfulness* = supported / (supported + unsupported) claims against the retrieved context.
  Statements that only say the context lacks information are in neither count.
- *Abstention* = the judge's label for the whole answer (`answered` / `partial` / `abstained`),
  reported as a rate.
- *Answer relevance* = mean cosine similarity between the question and 3 questions the judge
  writes back from the answer; computed only for answers that did not abstain.
- *Context relevance* = share of retrieved passages the judge marks as helping to answer.
- *Reference agreement* = the faithfulness calculation against the MedQuAD reference answer
  instead of the retrieved context (the reference is never shown to the generator).
- Ragas itself is not used: version 0.4.3 would not import under a clean install when this was
  built. The judge prompts are inline in the module and pinned by a snapshot test.

**Baselines** (same harness): `oracle` (the query's gold chunks), `random` (5 random chunks),
`none` (no retrieval, closed-book prompt `closed_book_v1`).

**Robustness** (`evaluation/robustness.py`; dense retriever, every KB variant)

- `drop_top_relevant`: remove the highest-ranked relevant chunk (context becomes k−1).
- `inject_noise`: replace the lowest-ranked non-relevant chunk, in place, with an off-topic chunk.
- `control_rerun`: regenerate on the unchanged context, to measure noise.

**Harness** (`experiments/`)

- Every row logs: query id, configuration, retrieved chunk ids/scores/text, relevant ids under
  both label sets, the generated text, all metrics and the judge's per-claim labels, requested and
  returned model, prompt hash, temperature, seed, token counts, git commit, a hash of the source
  tree, and a UTC timestamp.
- Resumable by run key. A failed query is logged as an error and retried next run. A results file
  refuses rows from a different setup (run fingerprint).
- Spend is tracked from the token counts the API reports and capped (`budget.max_usd`).

## Repository layout

```
data/          ingestion, chunking, KB variant construction, query set sampling
retrieval/     retrievers, baselines, embedding cache, index persistence
generation/    prompt templates, OpenAI client, spend cap, generator, output parser
evaluation/    relevance labels, retrieval metrics, generation metrics, perturbations
experiments/   sweep config, runner, results store, cost estimate
analysis/      aggregation, statistics, paired comparisons, plots, CSV export
scripts/       verify_all, mutation_test, validate_results, build_spotcheck, variance_run, ...
configs/       YAML configs for every stage
tests/         fast tests, real-data tests, real-API tests (--run-api)
datasets/      gitignored: raw MedQuAD download + processed corpus / KB variants / query set
artifacts/     gitignored: indexes, embedding cache, spend ledger
results/       gitignored: JSONL experiment logs
reports/       CSVs, figures, validation reports
```

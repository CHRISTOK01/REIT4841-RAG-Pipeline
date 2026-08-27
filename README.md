# RAG Retrieval-Quality Evaluation Pipeline

Research codebase for a UQ REIT4841 thesis investigating:

> How does retrieval quality influence the factual accuracy, relevance, and reliability of
> outputs in a RAG-based pipeline for medical report/wellness summary generation?

This is a controlled experiment harness, not a product: a config-driven, reproducible pipeline
for comparing retrieval methods (lexical / dense / hybrid) and knowledge-base quality variations
(curated / mixed / reduced), and measuring the downstream effect on LLM-generated output quality.

Pure Python, CPU-only, no web framework. Generation and embeddings use the OpenAI API; vector
search uses FAISS (`faiss-cpu`); lexical search uses BM25 (`rank_bm25`).

## Data source

Corpus: [MedQuAD](https://github.com/abachaa/MedQuAD) (Ben Abacha & Demner-Fushman, 2019,
*"A Question-Entailment Approach to Question Answering,"* BMC Bioinformatics), CC BY 4.0.
47,457 medical QA pairs from 12 NIH source websites. Only the 9 subsets that retain full answer
text are used (`CancerGov`, `GARD`, `GHR`, `MPlusHealthTopics`, `NIDDK`, `NINDS`, `SeniorHealth`,
`NHLBI`, `CDC`); the other 3 (MedlinePlus ADAM, Drugs, Herbs/Supplements) have their answer text
stripped in the source repo for copyright reasons and are skipped.

If you use this data in the thesis, cite the MedQuAD paper.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in OPENAI_API_KEY
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Pipeline status

| Module | Status |
|---|---|
| `data/` — ingestion, chunking, KB variants, query set | done |
| `retrieval/` — lexical / dense / hybrid retrievers, FAISS+BM25 index store | done |
| `generation/` — prompt templates, OpenAI generation client | done |
| `evaluation/` — retrieval metrics, generation metrics, robustness tests | done |
| `experiments/` — sweep runner, resumable results store | done |
| `analysis/` — aggregation, plots, CSV export | done |

## Running the data layer

```bash
# 1. Download MedQuAD and parse all usable subsets into one JSONL corpus
python -m data.ingest_medquad
#   -> datasets/processed/medquad_documents.jsonl (~16.4k source QA documents)

# 2. Build the curated / mixed / reduced KB variants (config: configs/kb_variants.yaml)
python -m data.kb_variants
#   -> datasets/processed/kb_{curated,mixed,reduced}/{documents,chunks}.jsonl + manifest.json

# 3. Sample the fixed query set from the curated corpus
python -m data.query_set --num-queries 40
#   -> datasets/processed/query_set.jsonl
```

At the default config (`configs/kb_variants.yaml`), this produces (from the real MedQuAD
download):

| Variant | Documents | Chunks |
|---|---|---|
| curated | 1,500 | 3,715 |
| mixed | 1,950 | 4,776 |
| reduced | 976 | 2,451 |

### Reproducibility

- `datasets/raw/` and `datasets/processed/` are gitignored (regenerable from code + config, not
  committed as binary blobs).
- Every KB variant build is stamped with a `config_hash` (SHA-256 of the KB-variant + chunking
  config, truncated to 12 hex chars) recorded in each variant's `manifest.json`, so a given
  index/chunk set can always be traced back to the exact settings that produced it.
- All sampling (base corpus selection, noise injection, topic dropout, query set selection) is
  seeded (default `seed: 42`) and deterministic — rerunning `data.kb_variants` or `data.query_set`
  with the same config reproduces identical output.

## Retrieval module

`retrieval/` implements the common `Retriever` protocol (`retrieve(query, k) -> list[RetrievedChunk]`)
with three implementations, plus the plumbing to make them cheap and reproducible to rerun:

- `LexicalRetriever` (`retrieval/lexical.py`) — BM25 (`rank_bm25`) over chunk text.
- `DenseRetriever` (`retrieval/dense.py`) — OpenAI embeddings + FAISS `IndexFlatIP` cosine search.
  The embedding call is injected (`embed_fn`) rather than hardcoded to the OpenAI SDK, so the
  module and its tests have no network/API-key dependency; the real OpenAI-backed embedder gets
  wired in by the generation/experiments layer.
- `HybridRetriever` (`retrieval/hybrid.py`) — reciprocal rank fusion (RRF) over any set of
  sub-retrievers; the fusion function is swappable (pass a different callable matching `FusionFn`).
- `retrieval/embedding_cache.py` — a SQLite-backed cache keyed by `(model, text)`, so a given
  chunk or query is only ever sent to the embeddings API once across all runs.
- `retrieval/index_store.py` — persists/loads a KB variant's FAISS index, BM25 index, and chunk
  metadata under `artifacts/<config_hash>/<variant>/`; `build_or_load_variant_indexes` skips
  rebuilding (and re-embedding) an index that's already on disk for that config hash.

Scale check against the real `curated` KB variant (3,715 chunks): BM25 index build ~0.2s, FAISS
index build ~1s, a single retrieval call ~5ms — comfortably laptop-CPU scale.

## Generation module

`generation/` produces one structured wellness-summary response per query:

- `prompts.py` — versioned prompt template (`wellness_summary_v1`, config: `configs/generation.yaml`).
  Numbers retrieved passages for the model, instructs it to cite passage number(s) after every
  claim in a `## Key Points` list, and to say explicitly when the provided context doesn't fully
  answer the question rather than filling the gap from outside knowledge.
- `parsing.py` — deterministic Markdown-section + citation parser (no LLM call) that turns raw
  model output into `GenerationOutput(summary, key_points, disclaimer)`, resolving each citation
  back to a `chunk_id` via the passage-number map from `prompts.py`. A citation to a passage
  number that was never given to the model (a hallucinated citation) is kept as a visible flag
  (`cited_passage_numbers`) but drops out of `cited_chunk_ids` -- useful directly as a
  faithfulness/robustness signal later.
- `client.py` — `OpenAIClient`: chat completions + embeddings, both with explicit tenacity
  retry/backoff on `RateLimitError` / `APIConnectionError` / `APITimeoutError` /
  `InternalServerError` (the SDK's own retries are disabled so this is the single place retry
  policy is owned/cited). `embed_fn(model)` returns a callable that plugs directly into
  `retrieval.dense.DenseRetriever` / the embedding cache. Reads `OPENAI_API_KEY` from the
  environment only -- never hardcoded.
- `generator.py` — `generate_summary(client, question, retrieved_chunks, model, ...)` renders the
  prompt, calls the client, parses the response, and returns a `GenerationRecord` with everything
  the results store will need to log (question, model, prompt id, retrieved chunk ids, raw text,
  parsed output, UTC timestamp).

All of the above is tested with fake/injected clients (no network, no API key needed to run
`pytest`). Real OpenAI connectivity can be checked any time with:

```bash
python -m generation.client   # makes one real embedding + one real chat completion call
```

## Evaluation module

`evaluation/` implements the retrieval and generation quality metrics, ground-truth derivation,
and the robustness/perturbation test:

- `relevance_judgments.py` — for each query, the set of relevant `chunk_id`s in a given KB
  variant (see "Design decisions" below). Can legitimately be empty for a query under `reduced`.
- `retrieval_metrics.py` — Precision@k, Recall@k, MRR, NDCG@k, binary relevance. Metrics that are
  mathematically undefined when a query has zero relevant chunks (Recall, MRR, NDCG) return
  `float('nan')`, not `0.0` -- `0.0` would silently conflate "retrieval missed it" with "there was
  nothing to find," which are different findings for the thesis. Aggregate with
  `pandas`' nan-aware `.mean(skipna=True)`.
- `robustness.py` — mechanical context perturbation (`drop_top_relevant`, `inject_noise_chunk`,
  `select_noise_chunk`) plus a `RobustnessComparison` delta bookkeeping dataclass. No OpenAI
  dependency; fully deterministic. Choosing which query/noise chunk to perturb and re-running
  generation on the perturbed context is the experiment harness's job.
- `generation_metrics.py` — faithfulness, answer relevance, context relevance, computed via
  LLM-as-judge against the project's own `OpenAIClient` (see decision below), each independently
  testable with a fake client returning canned JSON.

**Decision: Ragas was dropped in favor of a self-built implementation of its scoring logic.**
The build spec asked for Ragas "or replicate its scoring logic if Ragas integration proves
awkward." As of this writing, `ragas` 0.4.3 (the current latest) fails to import at all after a
clean `pip install ragas`: `ragas/llms/base.py` unconditionally imports
`langchain_community.chat_models.vertexai.ChatVertexAI`, which no longer exists at that import
path in current `langchain-community`. Pinning `langchain-community` back to a version that still
has it breaks `langchain_openai`'s required `langchain-core` version instead -- verified there is
no consistent pin across that dependency tree at the time of writing (`langchain-classic`,
`langgraph`, and `langgraph-sdk` all also pin conflicting `langchain-core` ranges). Chasing that
further would mean building the thesis's evaluation pipeline on a dependency chain that may not
even be pip-installable months from now at defense time. `evaluation/generation_metrics.py`
reimplements Ragas' actual approach directly (claim-level faithfulness via LLM entailment
judgment, answer relevance via reverse-generated-question cosine similarity, context relevance
via per-passage LLM relevance judgment) with every judge prompt inline and quotable verbatim in
the methodology chapter, and zero dependency on the langchain ecosystem.

## Experiment harness

`experiments/` runs the full grid and is the single documented end-to-end entry point:

```bash
python -m experiments.run_sweep --config configs/full_sweep.yaml
```

- `config.py` — `load_sweep_config` assembles a `SweepConfig` from `configs/full_sweep.yaml` plus
  the KB-variant and generation configs it references, exposing `kb_config_hash` (used to locate
  `artifacts/<hash>/<variant>/`) and `kb_variant_chunks_path(variant)` (locates
  `datasets/processed/kb_<variant>/chunks.jsonl`, built earlier by `data.kb_variants`).
- `results_store.py` — `ResultsStore`: append-only JSONL, `load_completed_keys()` for resumability,
  tolerant of a malformed trailing line (e.g. from a process killed mid-write).
- `runner.py` — `run_sweep` iterates `kb_variant x retrieval_method x query`, holding the query
  set/LLM/prompt template/top_k fixed throughout (per `configs/full_sweep.yaml`). For each
  combination not already in the results file: retrieve, compute retrieval metrics, generate,
  compute generation metrics, log one record. A query that raises after the client's own
  retry/backoff is exhausted is logged as a `record_type: "error"` record and **not** marked
  completed, so the next run retries it rather than silently dropping it. Robustness perturbations
  (`evaluation.robustness`) run as an add-on against whichever single retrieval method
  `robustness.retrieval_method` names (default `dense`) -- a 2x cost add-on to that slice of the
  grid, not a 3x add-on to the whole sweep.
- `run_sweep.py` — the CLI entry point.

Tested at two levels: `run_standard_query` / `run_robustness_for_query` / `build_retriever` in
isolation against stub retrievers and a fake chat+embed client (`tests/test_runner.py`), and the
full `run_sweep` orchestration end-to-end against a small synthetic on-disk KB + real FAISS/BM25
index building + real resumability (`tests/test_run_sweep_integration.py`) -- the fake client there
responds correctly to any judge prompt by reading the required array length out of the prompt text
itself (`evaluation/generation_metrics.py`'s prompts literally say "exactly N boolean values"),
rather than needing a hand-scripted response sequence.

### Design decisions (see full rationale in-repo / thesis methodology chapter)

- `curated` = a quality-filtered (`min_answer_words`), subset-stratified sample of the corpus.
- `mixed` = `curated` + a configurable ratio of topically irrelevant "distractor" documents drawn
  from the rest of the corpus (simulates a diluted/noisy knowledge base).
- `reduced` = `curated` with a configurable fraction of topics (MedQuAD `Focus` values) dropped
  entirely (simulates deliberate coverage gaps).
- Relevance judgments (used later for retrieval metrics) derive from MedQuAD's native
  question→source-document link: a chunk is relevant to a query if it was cut from that query's
  own QA pair's answer text. This is document-level, not manually verified passage-level,
  relevance — a simplifying assumption worth stating explicitly in the methodology chapter.

## Analysis module

`analysis/` turns `results/*.jsonl` into thesis-ready output:

```bash
python -m analysis.export_csv --results results/full_sweep.jsonl --out-dir reports
```

- `aggregate.py` — `load_standard_results` / `load_robustness_results` flatten the JSONL into
  pandas DataFrames; `aggregate_standard_metrics` / `aggregate_robustness_deltas` group-mean them
  (default `kb_variant x retrieval_method`, and `kb_variant x perturbation` respectively). Each
  metric gets its own `<metric>_n` count column alongside the group's total row count `n` --
  these can differ, since `recall_at_k` / `mrr` / `ndcg_at_k` (and robustness deltas built from
  them) are `NaN`, not just low, for a query with zero relevant chunks, while `precision_at_k` is
  always defined. A single shared `n` would silently overstate the sample size behind those
  specific means.
- `plots.py` — grouped bar charts (KB variant on the x-axis, one bar per retrieval method or
  perturbation type) saved as PNG via matplotlib's non-interactive `Agg` backend.
- `export_csv.py` — the canonical output of a sweep: `results_per_query.csv`,
  `results_aggregated.csv` (the comparison table for the methodology chapter),
  `robustness_per_query.csv` / `robustness_aggregated.csv` if robustness was run, and
  `figures/*.png`.

Validated against a realistic synthetic 108-standard-row / 72-robustness-row sweep (3 KB variants
x 3 retrieval methods x 12 queries, with `reduced` deliberately including some zero-relevant-chunk
queries to exercise the NaN-handling path): the aggregated table came out in the expected
`curated > mixed > reduced` and `hybrid > dense > lexical` order, and the robustness figure showed
consistent negative deltas (`drop_top_relevant` more severe than `inject_noise`) across all three
KB variants -- both spot-checked visually, not just asserted in `pytest`.

## Repository layout

```
data/          ingestion, chunking, KB variant construction, query set sampling
retrieval/     Retriever implementations + FAISS/BM25 index persistence
generation/    Prompt templates + OpenAI generation client
evaluation/    Retrieval metrics, generation metrics, robustness tests
experiments/   Sweep runner, resumable append-only results store
analysis/      Aggregation, plots, CSV export for thesis figures
configs/       YAML configs for every stage, citeable in the methodology chapter
tests/         pytest suite for all deterministic (non-LLM) logic
datasets/      gitignored: raw MedQuAD download + processed corpus/KB-variant/query-set JSONL
artifacts/     gitignored: FAISS/BM25 indexes + embedding cache, per config hash
results/       gitignored: JSONL experiment run logs
reports/       aggregated CSVs + matplotlib figures for the thesis
```

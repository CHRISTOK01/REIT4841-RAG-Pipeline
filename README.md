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
| `retrieval/` — lexical / dense / hybrid retrievers, FAISS+BM25 index store | not yet built |
| `generation/` — prompt templates, OpenAI generation client | not yet built |
| `evaluation/` — retrieval metrics, Ragas generation metrics, robustness tests | not yet built |
| `experiments/` — sweep runner, resumable results store | not yet built |
| `analysis/` — aggregation, plots, CSV export | not yet built |

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

## Repository layout

```
data/          ingestion, chunking, KB variant construction, query set sampling (code)
retrieval/     Retriever implementations + FAISS/BM25 index persistence (pending)
generation/    Prompt templates + OpenAI generation client (pending)
evaluation/    Retrieval metrics, Ragas-based generation metrics, robustness tests (pending)
experiments/   Sweep runner, resumable append-only results store (pending)
analysis/      Aggregation, plots, CSV export for thesis figures (pending)
configs/       YAML configs for every stage, citeable in the methodology chapter
tests/         pytest suite for all deterministic (non-LLM) logic
datasets/      gitignored: raw MedQuAD download + processed corpus/KB-variant/query-set JSONL
artifacts/     gitignored: FAISS/BM25 indexes + embedding cache, per config hash
results/       gitignored: JSONL experiment run logs
reports/       aggregated CSVs + matplotlib figures for the thesis
```

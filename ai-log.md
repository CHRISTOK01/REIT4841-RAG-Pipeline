# AI use log: codebase

This log records how I used an AI coding assistant, Claude Code (Anthropic), on the code in this
repository. It covers the code, tests, data checks and analysis scripts only. My use of AI in
writing the thesis report is declared separately in the report.

The pipeline as it stood at commit `271788e` (27 August 2026) is where the work below starts:
data layer, retrievers, generator, evaluation metrics, sweep harness and analysis module. All
assisted work came after that commit and is in commits `5faf455`, `ace9ab1` and `258f489`.

## How I worked with it

- Claude Code ran on my laptop inside this repository. It read the code, ran commands and tests,
  and edited files.
- I set the rules it had to follow:
  - every test must be able to fail;
  - no expected value may come from the function being tested;
  - the code under test is never mocked;
  - tests use the real data wherever possible;
  - no paid OpenAI run starts without a cost estimate first.
- Every methodology decision was mine. Claude Code explained the options and their effects, and I
  chose (see "Decisions I made" below).
- I made all the commits myself after reviewing the changes.

## 1. Audit of the original pipeline (11 October 2026)

I asked for a review of the whole pipeline against the project specification before running any
real experiment. Nothing was changed in this step. The findings are in `AUDIT.md`. The main ones:

- the `reduced` knowledge base was scored only on the queries it could still answer, which hid
  the effect it was meant to show;
- part of the "noise" injected into the `mixed` knowledge base shared a topic with, or copied
  text from, the curated documents;
- faithfulness scored "the context does not provide this" as a faithful claim, so it was 1.0
  even with the relevant chunk removed;
- the run logs were missing settings needed to reproduce a result;
- there were no baselines and no negative controls for the generation metrics;
- many tests used toy inputs that could not catch realistic bugs.

## 2. Fixes and hardening (11 October 2026)

Claude Code fixed the problems found in the audit. The 17 bugs fixed, each with the test that now
guards it, are listed in `TESTING.md` section 6. The main changes:

- **Data:** unique document ids (34 MedQuAD rows had collided); topic and verbatim-text filters on
  `mixed` noise; case-insensitive topic matching; build fails instead of silently using fewer
  noise documents.
- **Queries and labels:** query set raised from 40 to 150 (stratified, seeded); a second, lenient
  relevance label added beside the strict one.
- **Retrieval:** BM25 no longer pads results with zero-score chunks; fixed tie-breaking; RRF
  constant and fusion depth moved into the config; indexes reused only when the chunks and
  embedding model match.
- **Generation and metrics:** pinned model snapshot (`gpt-4o-mini-2024-07-18`); judge at
  temperature 0; a separate "no information" claim label; abstention as its own metric; strict
  parsing of the judge's replies; empty or malformed answers regenerated or logged as errors.
- **Harness:** every record logs the full setup (seeds, prompt hash, models, code version and
  more); resuming works for robustness rows too; rows from a different setup are refused;
  per-record seeds; spend cap and cost estimator (`experiments/cost_estimate.py`,
  `generation/budget.py`).
- **Baselines:** oracle, random and no-retrieval (`retrieval/baselines.py`).

Changes that alter what a reported number means are listed in
`reports/validation/thesis_facing_changes.md`.

## 3. Tests and verification (11 October 2026)

- Rewrote and extended the test suite into three tiers (fast, real-data and real-API), described
  in `TESTING.md`. 825 tests run without the API; 26 more call the real API.
- Expected values are worked out by hand or come from an independent library (`ranx`,
  scikit-learn, SciPy, NumPy).
- Negative controls check that the generation metrics give lower scores to answers known to be
  bad.
- Mutation testing (`scripts/mutation_test.py`): 99 deliberate bugs injected, 99 caught
  (`reports/validation/mutation_report.md`).
- Data validation and results checks (`scripts/validate_results.py`,
  `reports/validation/data_validation.md`).
- One command runs everything that is deterministic: `python -m scripts.verify_all`.
- Line coverage 90.7% (`reports/validation/coverage.txt`).

## 4. Running the experiments (October 2026)

- Built the knowledge-base indexes and ran the full sweep: 150 queries, 3 retrievers, 3 knowledge
  bases, plus robustness perturbations (3,750 records). Each paid run started only after I saw the
  cost estimate. Total OpenAI spend for the project was US$2.91.
- Ran the variance check (`scripts/variance_run.py`).
- Prepared the manual-check packs: `scripts/sample_relevance_pairs.py` (60 relevance-label pairs)
  and `scripts/build_spotcheck.py` (15 generated answers). **The judgements in
  `reports/validation/relevance_label_sample.md` and `spot_check_assessment.md` were made by
  Claude Code, not by me or a clinician**, as the files and the thesis say.

## 5. Analysis code (18 October and 1 November 2026)

- Added the analyses the thesis needed:
  - `analysis/correlation.py`: Spearman and Kendall correlations between retrieval and generation
    metrics, with a bootstrap over queries;
  - `analysis/studies.py`: study comparisons with Holm correction, and the method × knowledge-base
    interaction;
  - relative degradation in `analysis/aggregate.py`.
- Each came with tests and mutants.
- Regenerated the CSVs and figures in `reports/` from the sweep results (no API calls).
- Wrote the scripts for the thesis figures and exploratory analyses in `reports/thesis_extras/`.

## Decisions I made

- Unanswerable queries count as 0 in the headline metrics, with an answerable-only view beside it.
- `mixed` noise comes only from topics absent from `curated`.
- Noise injection replaces the lowest-ranked non-relevant chunk.
- Both strict and lenient relevance labels are reported.
- 150 queries.
- Generator pinned to `gpt-4o-mini-2024-07-18`. The judge stays the same model, because the
  negative controls showed it can tell good answers from bad ones.
- Abstention is reported as its own metric.

## Known gaps

- The 26 real-API tests have not been re-run since 11 October 2026.
- `scripts/live_single_query.py` still calls an old version of `compute_generation_metrics` and
  fails at the grading step. The sweep does not use it.

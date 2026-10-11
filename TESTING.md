# TESTING.md — how this pipeline is verified

This file explains, in plain language, what is checked, how, and what is not. It is written so
that the descriptions can be reused in the thesis methodology and defended in an examination.

Run everything deterministic with one command:

```bash
python -m scripts.verify_all
```

## 1. The three tiers of tests

| Tier | Command | What it uses | Cost |
|---|---|---|---|
| Fast | `python -m pytest` | Small hand-built inputs with expected values worked out by hand, plus an independent library | none |
| Real data | `python -m pytest` (same command) | The real MedQuAD corpus, the real KB variants, the real 150 queries, the real cached embeddings and indexes | none (no API call is possible: a cache miss fails the test) |
| Real API | `python -m pytest --run-api` | The real OpenAI generator, judge and embedder | a few cents; capped by `--api-max-usd` (default US$0.50) and by the project spend cap |

Counts at the time of writing: 795 tests in the fast and real-data tiers (about 7 minutes) and 26 real-API tests (20 negative-control checks and 6 smoke checks; about 2 minutes and US$0.02).

Rules the tests follow:

- **The thing under test is never mocked.** Only the OpenAI API is replaced in the fast and
  real-data tiers (`tests/fakes.py`), because it is an external paid service. The real-API tier
  runs the same code against the real service.
- **No expected value comes from the function being tested.** Expected values are (a) literals
  worked out by hand with the arithmetic in a comment, (b) the output of an independent library
  (`ranx`, scikit-learn, SciPy, plain NumPy), or (c) a property that must hold for any input.
- **A skipped real-data test is treated as a failure** by `scripts.verify_all`: a check that did
  not run has verified nothing.

## 2. What each test category does

### A. Retrieval metrics (P@k, R@k, MRR, NDCG)

- *Hand-worked examples* (`tests/test_retrieval_metrics.py`): each expected number is written as a
  literal with its arithmetic beside it, e.g. NDCG@5 for hits at ranks 2 and 4 with 3 relevant
  chunks = (0.630930 + 0.430677) / (1 + 0.630930 + 0.5) = 0.498189. Cases: hits past the cutoff,
  fewer than k results, k larger than the list, more relevant chunks than k, empty retrieval, no
  relevant chunks, duplicate ids, stale rank fields, tied scores.
- *Independent implementations* (`tests/test_retrieval_metrics_reference.py`): all four metrics
  are compared with `ranx` on 2,500 random rankings (20 seeds × 5 cutoffs × 25 queries), and NDCG
  with scikit-learn's `ndcg_score` on 200 more.
- *Properties* (`tests/test_retrieval_metrics_properties.py`, Hypothesis, 300 generated cases
  each): every metric is in [0, 1]; relevant-first is a perfect ranking; reversing it never scores
  higher; appending non-relevant chunks below never changes MRR; recall and MRR never fall as k
  grows; moving a relevant chunk up never lowers NDCG or MRR; a metric is 0 exactly when the top k
  holds nothing relevant.
- *Zero relevant chunks* is handled on purpose: recall, MRR and NDCG are undefined (NaN), precision
  is 0, and the analysis layer reports both "counted as 0" and "excluded" views
  (`tests/test_aggregate.py`).
- *Graded relevance does not apply*: the labels are binary.

### B. Retrievers

- *FAISS against brute force* (`tests/test_retrievers_real.py`): for all 150 queries, on each of
  the three real embedded KB variants, at k = 5 and k = 50, the FAISS result is compared with
  cosine similarity computed from its definition in NumPy (float64). Scores must agree to 1e-5 at
  every rank and nothing with a clearly higher cosine may be missing.
- *Normalisation*: the vectors stored in the real index are checked to be unit length and to be
  the cached embeddings in the same row order. Because OpenAI's embeddings are already unit
  length (also checked), an inner-product/cosine mix-up could not show on real data, so
  `tests/test_dense.py` uses hand-built vectors of unequal length where the two rankings differ
  ((10, 0) has the larger inner product with (1, 1); (1, 1) has the larger cosine).
- *BM25 by hand* (`tests/test_lexical.py`): scores computed by hand from the Okapi formula on small
  corpora, covering term frequency, two-term queries and length normalisation. On the real corpus:
  a chunk's own text retrieves that chunk (≥ 95%; the exceptions are one recurring boilerplate
  paragraph), a topic keyword returns only chunks containing it, and no result ever shares zero
  terms with the query.
- *RRF by hand* (`tests/test_hybrid.py`): known rankings with fused scores computed by hand at
  k = 60 and k = 0 (where the order changes), ties, and documents in only one list. On real data
  the hybrid result must equal an RRF written independently in the test from the BM25 and dense
  lists.
- *Interface contract* (real data, every query, k ∈ {1, 5, 20}): same types, descending scores,
  no duplicate chunk ids, ranks 1..n, k respected, ids exist in the KB, identical on repeat.
- *Embedding cache* (`tests/test_embedding_cache.py`): same text = hit and no call; changed text or
  model name = miss; a truncated row, a wrong-dimension row, a NaN row, a garbage file and a file
  cut in half all raise or lose only whole rows — none yields a wrong vector.
- *Index reuse* (`tests/test_index_store.py`): an index is not reused if the chunks, their text,
  their order, or the embedding model changed, or if the directory is half-written.

### C. Data layer and KB variants (on the real data: `tests/test_data_real.py`)

- *Chunking*: stitching each real document's chunks back together (minus the 20-word overlap)
  reproduces the MedQuAD answer exactly, for all 1,950 documents; overlaps are exactly 20 words;
  no empty chunks; ids unique and contiguous. Hypothesis tests the same invariants on arbitrary
  text and configurations, including unicode. The real corpus is confirmed to contain very short
  (< 20 words), very long (> 3,000 words) and non-ASCII documents.
- *Variants*: `reduced` ⊂ `curated` ⊂ `mixed` (documents and chunks); noise is exactly 450 = 0.30 ×
  1,500; no noise document shares a topic with curated, repeats a curated chunk, or is relevant to
  any query under either label set; `reduced` drops exactly 35% of topics, whole.
- *Determinism*: rebuilding from the corpus reproduces the files on disk exactly, and three fresh
  Python processes with different hash seeds reproduce the SHA-256 recorded in each manifest.
- *Relevance labels*: every query has relevant chunks in `curated` and `mixed`; a query's strict
  relevant chunks stitch back into exactly the MedQuAD answer to that question; lenient labels
  follow their definition and contain the strict ones.
- *Leakage*: no query contains its reference answer; the question text is not part of the indexed
  chunk text (checked against the BM25 index's document lengths); at most 5% of answers repeat
  their own question (a property of MedQuAD).
- *Manual label check*: `reports/validation/relevance_label_sample.md`.

### D. Generation and its metrics

- *Negative controls* (`tests/test_api_negative_controls.py`, real judge): for three real queries,
  hand-written answers that are (i) fully grounded, (ii) half invented, (iii) about a different
  subject, plus an abstaining answer and a correct answer shown beside an unrelated passage.
  Result with the configured judge: faithfulness 1.00 (grounded) / 0.40 (half
  invented) / 0.00 (off-topic) in all three cases, with exactly the invented claims labelled
  unsupported; answer relevance about 0.7-0.85 on-topic against 0.05 off-topic; context relevance
  1.00 for the real passage against 0.00 for an unrelated one; the abstaining answer labelled
  `abstained` with faithfulness and answer relevance undefined; reference agreement 1.00 against
  0.40. So the judge does separate the three kinds of answer, and it was kept (the alternative
  judge tried, gpt-4.1-mini, was not better: it fixed one weakness and introduced another). The
  first version of the new metrics failed these controls, which is how defect 17 in section 6 was
  found.
- *Baselines*: oracle, random and no-retrieval run through the same harness
  (`tests/test_run_sweep_integration.py`, `tests/test_harness_real.py`, and the real sweep).
- *Prompt snapshot* (`tests/test_prompts.py`, `tests/test_generation_metrics.py`): the generation
  prompt, the closed-book prompt and the four judge prompts are pinned by SHA-256; an edit fails
  the test. The generation prompt was verified to be byte-identical to the originally committed
  one. A harness test confirms one model, one template and one temperature across all nine
  configurations.
- *Smoke test* (`tests/test_api_smoke.py`, real API): 18 records end to end; checks the estimate
  against actual spend, that the API returned the pinned snapshot, and answer structure.
- *Metric plumbing* (`tests/test_generation_metrics.py`): with scripted judge replies, the
  arithmetic, which claims are sent, when a call is skipped, and that every malformed reply shape
  raises instead of becoming a score.
- *Run-to-run variance*: `python -m scripts.variance_run`; results in the validation report.

### E. Robustness perturbations

- By hand (`tests/test_robustness.py`): drop removes exactly the top relevant chunk; inject
  replaces exactly the lowest-ranked non-relevant chunk, in place; nothing else moves.
- In the harness, on synthetic and real data: each logged perturbed context is re-derived from its
  logged baseline and compared; recall falls after every drop and is unchanged after every inject
  and control.

### F. Harness, logging, reproducibility (`tests/test_run_sweep_integration.py`)

- *Crash and resume*: the run is killed at 7 different points (the fake API raises a
  non-catchable error at a chosen call), resumed, and compared with an uninterrupted run: same
  rows, none duplicated, none missing, rows finished before the crash untouched, and exactly one
  generation call per row written after the resume. A half-written final line is discarded and
  only that row is redone.
- *Failures*: timeouts, connection errors, malformed judge replies (5 shapes) and unusable answers
  (4 shapes) each become an error record, never a result row or a zero, and are retried on the
  next run. Reaching the spend cap stops the run.
- *Logging*: every row carries the fields listed in the README; all reports are rebuilt from the
  log after deleting every other data file, and the aggregated table is recomputed from the raw
  JSON lines in the test.
- *Determinism*: two independent runs and a 4-thread run produce identical rows (timestamps
  aside). Generation text is not deterministic against the real API; see the variance results.
- *Mixing setups*: rerunning into the same results file with a changed setting is refused.

### Analysis (`tests/test_stats.py`, `tests/test_aggregate.py`, `tests/test_export_csv.py`)

- Bootstrap intervals are compared with SciPy's bootstrap, with the normal-theory interval on a
  large sample, and by simulation (they cover the true mean about 95% of the time). The paired
  test uses SciPy's Wilcoxon; a simulation confirms a real effect is found and pure noise is
  flagged about 5% of the time. Holm adjustment and the power formula are checked by hand and by
  simulation. Bar heights and whiskers are read back out of a drawn figure.

## 3. Requirement-to-test map

| Requirement | Where it is tested |
|---|---|
| P@k, R@k, MRR, NDCG correct | `test_retrieval_metrics*.py`; recomputed with `ranx` on the real log by `scripts.validate_results` |
| BM25 retriever | `test_lexical.py`, `test_retrievers_real.py` |
| Dense retriever (embeddings + FAISS, cosine) | `test_dense.py`, `test_retrievers_real.py` |
| Hybrid RRF | `test_hybrid.py`, `test_retrievers_real.py` |
| One retriever interface | `test_retrievers_real.py` (contract tests) |
| Embeddings cached | `test_embedding_cache.py`, `test_dense.py`; real-data tests fail on any cache miss |
| Three KB variants, deterministic, configured ratio | `test_kb_variants.py`, `test_data_real.py` |
| Chunking | `test_chunking.py`, `test_data_real.py` |
| Relevance labels | `test_relevance_judgments.py`, `test_data_real.py`, manual sample |
| One fixed model and prompt | `test_prompts.py`, `test_config.py`, `test_run_sweep_integration.py`, `test_api_smoke.py`, validation report §1.6 |
| 9-config sweep, fixed top-k, config-driven | `test_config.py`, `test_run_sweep_integration.py`, `test_harness_real.py`, validation report §1.2 |
| Fully logged; figures from logs alone | `test_run_sweep_integration.py` |
| Resumable | `test_run_sweep_integration.py`, `test_results_store.py` |
| Faithfulness, answer relevance, context relevance | `test_generation_metrics.py` (plumbing), `test_api_negative_controls.py` (does it discriminate) |
| Robustness test | `test_robustness.py`, `test_runner.py`, `test_run_sweep_integration.py`, `test_harness_real.py`, validation report §1.7 and §2.7 |
| Oracle / random / no-retrieval baselines | `test_retrievers_real.py`, `test_run_sweep_integration.py`, `test_harness_real.py`, validation report §2.4 |
| Failure handling | `test_client.py`, `test_generator.py`, `test_run_sweep_integration.py` |
| Cost cap and dry run | `test_budget.py`, `test_client.py`, `test_api_smoke.py` |
| Tables, plots, CSV, statistics | `test_aggregate.py`, `test_stats.py`, `test_export_csv.py` |
| Output data sound | `scripts.validate_results` → `reports/validation/data_validation.md` |

## 4. Mutation testing

`python -m scripts.mutation_test` injects one deliberate bug at a time into a copy of the code and
runs the tests that cover that module. A bug the tests do not notice is a "surviving mutant".

**Result: 90 bugs injected, 90 caught. Mutation score 100%.** The full list, with the test that
caught each one, is in `reports/validation/mutation_report.md`.

| Area | Bugs injected | Examples |
|---|---|---|
| Retrieval metrics | 14 | divide by k+1; ignore the cutoff; rank off by one; no NDCG discount; ideal ranking ignores k; undefined recall returned as 0; duplicates accepted |
| Fusion (RRF) | 9 | constant 61; rank off by one; raw scores added instead of ranks; ascending sort; tie-break removed; configured constant ignored |
| Dense retriever | 6 | chunk or query vector not normalised; k-1 results; reversed order; tie rule removed |
| BM25 | 4 | reversed order; zero-score padding restored; k+1 results; case-sensitive tokens |
| Chunking | 5 | no overlap; overlap one word short; extra trailing chunk; ids from 1; window one word short |
| KB variants | 10 | noise ratio 90% of configured; topic filter removed; verbatim-copy filter removed; half the topic drop; unseeded sampling; unsorted pool |
| Relevance labels | 3 | lenient ignores verbatim copies; any same-topic chunk counted; strict uses topic instead of document |
| Perturbations | 6 | drop removes two chunks; noise replaces the wrong chunk; noise inserted instead of replacing; noise on the query's topic |
| Generation metrics and generator | 9 | "no information" counted against or for faithfulness; quoted booleans accepted; judge at 0.7; summary not checked; cut-off answer accepted |
| Analysis and statistics | 9 | unanswerable queries dropped from the headline; delta sign flipped; duplicates undetected; 80% interval; unaligned pairs |
| Harness, store, cache, index | 15 | resume re-runs everything; failed row marked done; mixed setups accepted; shared seed; stale index reused; cache key ignores the model |

Honest limits of this number: the same author wrote the mutants and the tests, so 100% shows that
the listed kinds of bug are caught, not that every possible bug would be. One mutant (noise appended unshuffled instead of shuffled) got past the fast tests and was caught only by the
real-data tier, which rebuilds the KB and compares it with the files on disk. One candidate mutant was not included because it changes nothing:
swapping the logarithm base in NDCG cancels out in the ratio.

## 5. Coverage

From `python -m scripts.verify_all` (per-file table in `reports/validation/coverage.txt`):
**line coverage 90.1%** (1,910 of 2,120 statements); **line and branch combined 89%** (546
branches, 35 partly covered).

What is not covered by the default tiers, and why:

- `experiments/run_sweep.py` (0%) and the `main()` functions of the data modules: command-line
  entry points. They were exercised by actually running them to produce the data and results.
- `experiments/cost_estimate.py` (25%): exercised by the real-API smoke test, which compares the
  estimate with the actual spend, and by the real sweep (estimate US$3.55, actual US$2.47).
- `data/ingest_medquad.py` download step (a `git clone`).

Coverage says which lines ran, not whether a wrong result would be noticed; the mutation score
above is the better measure of that.

## 6. Bugs found and fixed during this work

None of these affected thesis results: the full sweep had never been run before the audit. The
demo outputs produced before the audit are in `results/_superseded/` and `reports/_superseded_demo/`
and must not be used.

| # | Bug | Effect if left | Fix | Regression test |
|---|---|---|---|---|
| 1 | 34 MedQuAD rows shared a document id (CancerGov files reuse `<Document id>` and qids) | Different documents would share relevance labels and chunk ids | ids keyed on file name; duplicate ids are a hard error | `test_ingest_medquad.py`, `test_data_real.py` |
| 2 | Unanswerable queries in `reduced` were dropped from retrieval means | `reduced` scored only on queries it could still answer | two views of every metric + coverage | `test_aggregate.py` |
| 3 | `mixed` noise could share a topic with, or copy text from, curated | Part of the "noise" was extra relevant material | topic and verbatim-chunk filters; build fails if too few eligible | `test_kb_variants.py`, `test_data_real.py` |
| 4 | "The context does not say" was scored as a faithful claim | Faithfulness of 1.0 for answers that said nothing | third claim label, excluded from the ratio | `test_generation_metrics.py`, negative controls |
| 5 | Answer relevance was 0 for abstentions and judged at temperature 0.7 | Mixed abstention into relevance; scores moved run to run | abstention is its own label; temperature 0 | `test_generation_metrics.py` |
| 6 | `bool("false")` is `True`: quoted booleans from the judge counted as supported/relevant | Inflated scores whenever the judge quoted its booleans | strict type check; malformed replies raise | `test_generation_metrics.py` |
| 7 | Empty or off-format model output was scored (0 or dropped) | Format failures recorded as results | regenerate, then error record | `test_generator.py`, `test_client.py`, `test_run_sweep_integration.py` |
| 8 | An empty `## Summary` captured the next heading as its text | A missing summary looked present | parser regex fixed | `test_parsing.py` |
| 9 | Robustness rows were lost if the run stopped between a baseline and its perturbations | Silent gaps in the robustness data | perturbations resume from the logged baseline; inapplicable ones are logged | `test_run_sweep_integration.py` |
| 10 | Noise injection pushed the last original chunk out of the context | "Add noise" also removed a (possibly relevant) chunk | replace the lowest-ranked non-relevant chunk in place | `test_robustness.py`, `test_run_sweep_integration.py` |
| 11 | One random generator shared across the sweep | Noise chunk depended on run order and interruptions | per-record seeds derived from the sweep seed and run key | `test_runner.py`, `test_run_sweep_integration.py` |
| 12 | Results file accepted rows from a different setup | Old and new settings silently averaged together | run fingerprint on every row; mismatch refused | `test_run_sweep_integration.py` |
| 13 | Index reused whenever its files existed | Retrieval could run on a stale corpus or embedding model | fingerprint of chunks + model | `test_index_store.py` |
| 14 | BM25 padded results with zero-score chunks | Arbitrary chunks earned rank credit in fusion | only chunks sharing a term are returned | `test_lexical.py`, `test_retrievers_real.py` |
| 15 | A record appended after a torn line was glued onto it | That result was lost and paid for again | newline guard + torn tail discarded on start | `test_results_store.py`, `test_run_sweep_integration.py` |
| 16 | `answer_relevance_num_questions` and `prompt_template_id` in the config were never read | Config and behaviour could disagree | both wired; unknown template id is an error | `test_config.py`, `test_generation_metrics.py` |
| 17 | Sentence fragments such as "Yes." were sent to the judge as claims; off-topic claims were labelled "no information"; off-topic answers were labelled abstentions | Found by the negative controls on the first version of the new metrics | fragments joined; claim and abstention prompts rewritten | `test_generation_metrics.py`, negative controls |

## 7. Known limitations

These are limits of the evidence, not of the code. Each should be stated in the thesis.

1. **Relevance labels are document-of-origin labels.** In a hand check of 30 "relevant" pairs,
   20 clearly helped answer the question, 5 partly and 5 not at all (boilerplate inside long
   answers). In 30 retrieved-but-unlabelled pairs, none clearly helped and 3 partly did. Recall and
   NDCG are therefore depressed for every retriever alike; MRR and "any gold chunk in the top 5"
   are the more trustworthy retrieval measures. The check was done by the AI assistant, not a
   clinician.
2. **Queries are MedQuAD questions that name their topic and whose answers are in the KB.** This
   is known-item search and is easier than real user questions. Results say nothing yet about
   lay-language or paraphrased queries.
3. **Some reference answers contain no answer.** For example, several GHR "treatment" answers are
   lists of links. For such queries the correct behaviour is to abstain, and the gold chunks are
   not informative.
4. **The judge is the same model family as the generator** (`gpt-4o-mini-2024-07-18`) and is not
   perfectly consistent. Known weaknesses, all observed: it sometimes labels a correct "the
   context does not say this" sentence as unsupported; it sometimes accepts a claim that takes a
   true sentence about a different condition and applies it to the query's condition; in the
   negative controls it labelled "there is no vaccine" as supported by an unrelated passage in
   about half of repeated runs. In the 15-case spot check, faithfulness matched the reader in 10
   cases.
5. **Faithfulness has a ceiling.** It is 0.94-0.97 for every real configuration on `curated` and
   `mixed`, so it cannot separate them. It measures agreement with the retrieved text, not
   correctness: an answer built from the wrong passages scores 1.0.
6. **Answer relevance does not respond to retrieval quality in this setup.** It is about 0.75 for
   every configuration including no retrieval at all, and stayed high for off-target answers in
   the spot check. It separates on-topic from off-topic (negative controls) and nothing finer.
7. **Reference agreement was added by the audit and depends on how complete the reference is.** It
   penalises correct statements that the reference does not happen to contain.
8. **Faithfulness, answer relevance and reference agreement are conditional on the model
   answering.** Their means are over different subsets of queries in different configurations
   (most of all on `reduced` and with random context). Abstention rate must be read alongside them.
9. **Generation is not deterministic.** Five repeats of 30 queries moved a configuration's mean
   faithfulness by an SD of about 0.01 and reference agreement by about 0.02; 20-37% of
   individual queries changed score between repeats. Retrieval is exactly reproducible.
10. **One model, one prompt, one top-k, one chunking, one corpus, one seed for the KB build and the
    query sample.** Nothing here shows that the findings carry over to other settings.
11. **Robustness was run on the dense retriever only**, and the noise chunk replaces a chunk the
    model was probably already ignoring (about two thirds of retrieved chunks are judged not
    relevant to begin with), so "noise injection has no effect" is a statement about this weak
    form of noise.
12. **Prices in `configs/pricing.yaml` were typed in by hand.** The spend cap is only as right as
    they are; check the figure against the OpenAI billing page.
13. **OpenAI served the run from six different `system_fingerprint` values.** The model snapshot was
    constant; the serving stack behind it was not, and that is outside anyone's control.

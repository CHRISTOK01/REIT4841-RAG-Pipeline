# Data validation report

Results log: `results/full_sweep.jsonl`. Config: `configs/full_sweep.yaml`. Produced by `python -m scripts.validate_results`; every number below is computed from the results log.

**Data checks: ALL PASS.**

## Part 1 — Is the data sound?


### 1.1 The log itself

- PASS — every line of the log is readable and no run key appears twice: 3750 records
- PASS — every row was produced under one setup, and it is the setup in the config file today: fingerprints in log: {'55c4419a2851ffcc': 3750}; config gives 55c4419a2851ffcc
- PASS — every failure logged during the run was later retried successfully: 0 error record(s) in the log; 0 still unresolved
- PASS — all rows were produced by one version of the code: source_tree_sha256: {'560bcaea2ca0073d': 3666}

### 1.2 Row counts

- PASS — standard rows = queries x configurations = 150 x 16 = 2400: found 2400; missing 0; unexpected 0
- PASS — every configuration has exactly one row per query: 16 configurations
- PASS — every robustness condition is accounted for (1350 = KB variants x queries x conditions): 1266 ran, 84 could not be applied (reason logged), 0 missing
  - not applicable: curated / drop_top_relevant: 14 queries (no relevant chunk in the baseline context to drop)
  - not applicable: curated / inject_noise: 3 queries (every chunk in the baseline context is relevant; nothing to replace)
  - not applicable: mixed / drop_top_relevant: 14 queries (no relevant chunk in the baseline context to drop)
  - not applicable: mixed / inject_noise: 3 queries (every chunk in the baseline context is relevant; nothing to replace)
  - not applicable: reduced / drop_top_relevant: 48 queries (no relevant chunk in the baseline context to drop)
  - not applicable: reduced / inject_noise: 2 queries (every chunk in the baseline context is relevant; nothing to replace)

### 1.3 Missing values and ranges

- PASS — no row where a required field is empty: 2400 rows checked
- PASS — no row where an answer is empty or missing a section: 2400 rows checked
- PASS — no row where an answer was cut off or filtered: 2400 rows checked
- PASS — no row where generation metrics are missing: 2400 rows checked
- PASS — faithfulness is NaN exactly when the answer made no factual claim (or had no context): 2400 rows checked
- PASS — answer relevance is NaN exactly when the answer abstained: 2400 rows checked
- PASS — context relevance is NaN exactly when no context was retrieved: 2400 rows checked
- PASS — retrieval metrics are absent exactly for the no-retrieval baseline: 2400 rows checked
- PASS — recall/MRR/NDCG are NaN exactly when the KB holds no relevant chunk for the query: 2400 rows checked
- PASS — no row where a metric is outside [0, 1]: 2400 rows checked
- PASS — no row where abstention label is not one of the three allowed values: 2400 rows checked
- PASS — one label per claim: 2400 rows checked
- PASS — one relevance label per retrieved passage: 2400 rows checked

### 1.4 Logged scores recomputed independently from the logged evidence

- PASS — P@5, R@5, MRR@5, NDCG@5 (strict labels) equal the values `ranx` computes from the logged chunk ids: 2055 answerable rows x 4 metrics compared; 0 mismatch(es)
- PASS — P@5, R@5, MRR@5, NDCG@5 (lenient labels) equal the values `ranx` computes from the logged chunk ids: 2090 answerable rows x 4 metrics compared; 0 mismatch(es)
- PASS — faithfulness equals supported / (supported + unsupported) recounted from the logged claim labels: 2400 rows; 0 mismatch(es)
- PASS — context relevance equals the share of `true` in the logged passage labels: 0 mismatch(es)
- PASS — reference agreement equals the recount from the logged reference labels: 0 mismatch(es)

### 1.5 Retrieved lists

- PASS — no list is longer than top_k = 5: 2250 lists
- PASS — no list contains a chunk twice
- PASS — every list is in descending score order
- PASS — dense, hybrid and random always return exactly 5 chunks
  - lists shorter than 5 (allowed for oracle = fewer gold chunks than 5, and BM25 = fewer matching chunks): {'oracle': 425}
- PASS — no two configurations retrieved identical chunks for every query (which would mean a setting is not being applied): the only identical pair is oracle on curated vs oracle on mixed, which must coincide: noise documents are gold for no query

| KB | pair | mean Jaccard overlap of top-k | share of queries with identical list |
|---|---|---|---|
| curated | lexical vs dense | 0.277 | 0.000 |
| curated | lexical vs hybrid | 0.440 | 0.007 |
| curated | dense vs hybrid | 0.439 | 0.007 |
| mixed | lexical vs dense | 0.284 | 0.000 |
| mixed | lexical vs hybrid | 0.446 | 0.013 |
| mixed | dense vs hybrid | 0.444 | 0.013 |
| reduced | lexical vs dense | 0.229 | 0.000 |
| reduced | lexical vs hybrid | 0.391 | 0.007 |
| reduced | dense vs hybrid | 0.400 | 0.007 |

### 1.6 Model, prompt and settings held constant

- PASS — the API reported the same, pinned model snapshot for every generation: {'gpt-4o-mini-2024-07-18': 3666}
- PASS — every retrieval configuration used the same prompt template (by hash): 3516 generations, template `wellness_summary_v1`
- PASS — the no-retrieval baseline, and only it, used the closed-book template: 150 generations
- PASS — temperature and max_tokens are the configured values on every row
- PASS — one judge model scored every answer: {'gpt-4o-mini-2024-07-18': 3666}
- PASS — every row has its own seed: 3666 distinct seeds for 3666 rows
  - generation attempts needed per answer (1 = first output was usable): {1: 3666}
  - OpenAI system fingerprints seen (a change means OpenAI changed its serving stack mid-run): {'fp_6a7e3d127b': 2719, 'fp_1d691afdd3': 662, 'fp_f673e0375a': 135, 'fp_3c075393d6': 137, 'fp_a562f08857': 9, 'fp_09148a4f1e': 4}
  - standard rows whose generated text is byte-identical to another row's: 167 of 2400 (expected when two configurations retrieve the same chunks for a query)

### 1.7 Perturbations did what they claim (recomputed from the log)

- PASS — control re-run: context identical to the baseline: 450 rows
- PASS — drop: exactly the highest-ranked relevant chunk is gone, the rest is unchanged and in order, recall fell: 374 rows
- PASS — inject: one non-relevant chunk (the lowest-ranked) replaced in place by a chunk that is not relevant; context size and recall unchanged: 442 rows

## Part 2 — What the results say, with uncertainty


### 2.1 Retrieval: headline metrics (every query; a query the KB cannot answer scores 0)

Cell format: mean ± SD [95% bootstrap CI] (n = queries behind the mean).

| KB | method | coverage | recall_at_k_all | mrr_all | ndcg_at_k_all | precision_at_k_all |
|---|---|---|---|---|---|---|
| curated | lexical | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.608 ± 0.41 [0.54, 0.67] (n=150) | 0.695 ± 0.42 [0.63, 0.76] (n=150) | 0.595 ± 0.38 [0.53, 0.66] (n=150) | 0.232 ± 0.18 [0.20, 0.26] (n=150) |
| curated | dense | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.732 ± 0.36 [0.67, 0.79] (n=150) | 0.871 ± 0.31 [0.82, 0.92] (n=150) | 0.749 ± 0.32 [0.70, 0.80] (n=150) | 0.277 ± 0.19 [0.25, 0.31] (n=150) |
| curated | hybrid | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.677 ± 0.38 [0.61, 0.74] (n=150) | 0.774 ± 0.37 [0.71, 0.83] (n=150) | 0.668 ± 0.35 [0.61, 0.72] (n=150) | 0.253 ± 0.18 [0.23, 0.28] (n=150) |
| curated | oracle | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.986 ± 0.08 [0.97, 1.00] (n=150) | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.401 ± 0.23 [0.36, 0.44] (n=150) |
| curated | random | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.000 ± 0.00 [0.00, 0.00] (n=150) | 0.000 ± 0.00 [0.00, 0.00] (n=150) | 0.000 ± 0.00 [0.00, 0.00] (n=150) | 0.000 ± 0.00 [0.00, 0.00] (n=150) |
| mixed | lexical | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.606 ± 0.41 [0.54, 0.67] (n=150) | 0.697 ± 0.42 [0.63, 0.76] (n=150) | 0.594 ± 0.38 [0.53, 0.65] (n=150) | 0.229 ± 0.18 [0.20, 0.26] (n=150) |
| mixed | dense | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.724 ± 0.36 [0.67, 0.78] (n=150) | 0.871 ± 0.31 [0.82, 0.92] (n=150) | 0.744 ± 0.32 [0.69, 0.79] (n=150) | 0.273 ± 0.19 [0.24, 0.30] (n=150) |
| mixed | hybrid | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.674 ± 0.38 [0.61, 0.73] (n=150) | 0.782 ± 0.36 [0.72, 0.84] (n=150) | 0.670 ± 0.35 [0.61, 0.73] (n=150) | 0.253 ± 0.18 [0.22, 0.28] (n=150) |
| mixed | oracle | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.986 ± 0.08 [0.97, 1.00] (n=150) | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.401 ± 0.23 [0.36, 0.44] (n=150) |
| mixed | random | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.002 ± 0.03 [0.00, 0.01] (n=150) | 0.001 ± 0.02 [0.00, 0.00] (n=150) | 0.001 ± 0.01 [0.00, 0.00] (n=150) | 0.001 ± 0.02 [0.00, 0.00] (n=150) |
| reduced | lexical | 0.740 ± 0.44 [0.67, 0.81] (n=150) | 0.474 ± 0.44 [0.41, 0.54] (n=150) | 0.540 ± 0.47 [0.47, 0.61] (n=150) | 0.466 ± 0.42 [0.40, 0.53] (n=150) | 0.181 ± 0.19 [0.15, 0.21] (n=150) |
| reduced | dense | 0.740 ± 0.44 [0.67, 0.81] (n=150) | 0.559 ± 0.45 [0.49, 0.63] (n=150) | 0.658 ± 0.47 [0.58, 0.73] (n=150) | 0.573 ± 0.43 [0.50, 0.64] (n=150) | 0.215 ± 0.21 [0.18, 0.25] (n=150) |
| reduced | hybrid | 0.740 ± 0.44 [0.67, 0.81] (n=150) | 0.520 ± 0.44 [0.45, 0.59] (n=150) | 0.594 ± 0.46 [0.52, 0.67] (n=150) | 0.517 ± 0.42 [0.45, 0.58] (n=150) | 0.196 ± 0.20 [0.16, 0.23] (n=150) |
| reduced | oracle | 0.740 ± 0.44 [0.67, 0.81] (n=150) | 0.727 ± 0.44 [0.66, 0.80] (n=150) | 0.740 ± 0.44 [0.67, 0.81] (n=150) | 0.740 ± 0.44 [0.67, 0.81] (n=150) | 0.297 ± 0.27 [0.26, 0.34] (n=150) |
| reduced | random | 0.740 ± 0.44 [0.67, 0.81] (n=150) | 0.002 ± 0.03 [0.00, 0.01] (n=150) | 0.007 ± 0.08 [0.00, 0.02] (n=150) | 0.003 ± 0.04 [0.00, 0.01] (n=150) | 0.001 ± 0.02 [0.00, 0.00] (n=150) |

### 2.2 Retrieval: answerable queries only, and lenient labels

`recall_at_k` and `mrr` here are over queries that still have a relevant chunk in that KB (ranking quality with coverage held out). The `_lenient_all` columns use the looser relevance labels over every query.

| KB | method | recall_at_k | mrr | recall_at_k_lenient_all | ndcg_at_k_lenient_all |
|---|---|---|---|---|---|
| curated | lexical | 0.608 ± 0.41 [0.54, 0.67] (n=150) | 0.695 ± 0.42 [0.63, 0.76] (n=150) | 0.584 ± 0.43 [0.52, 0.65] (n=150) | 0.589 ± 0.39 [0.53, 0.65] (n=150) |
| curated | dense | 0.732 ± 0.36 [0.67, 0.79] (n=150) | 0.871 ± 0.31 [0.82, 0.92] (n=150) | 0.711 ± 0.38 [0.65, 0.77] (n=150) | 0.746 ± 0.33 [0.69, 0.80] (n=150) |
| curated | hybrid | 0.677 ± 0.38 [0.61, 0.74] (n=150) | 0.774 ± 0.37 [0.71, 0.83] (n=150) | 0.653 ± 0.40 [0.59, 0.72] (n=150) | 0.663 ± 0.35 [0.60, 0.72] (n=150) |
| curated | oracle | 0.986 ± 0.08 [0.97, 1.00] (n=150) | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.870 ± 0.27 [0.82, 0.91] (n=150) | 0.942 ± 0.15 [0.92, 0.96] (n=150) |
| curated | random | 0.000 ± 0.00 [0.00, 0.00] (n=150) | 0.000 ± 0.00 [0.00, 0.00] (n=150) | 0.000 ± 0.00 [0.00, 0.00] (n=150) | 0.000 ± 0.00 [0.00, 0.00] (n=150) |
| mixed | lexical | 0.606 ± 0.41 [0.54, 0.67] (n=150) | 0.697 ± 0.42 [0.63, 0.76] (n=150) | 0.583 ± 0.42 [0.52, 0.65] (n=150) | 0.588 ± 0.39 [0.53, 0.65] (n=150) |
| mixed | dense | 0.724 ± 0.36 [0.67, 0.78] (n=150) | 0.871 ± 0.31 [0.82, 0.92] (n=150) | 0.703 ± 0.38 [0.64, 0.76] (n=150) | 0.741 ± 0.33 [0.69, 0.79] (n=150) |
| mixed | hybrid | 0.674 ± 0.38 [0.61, 0.73] (n=150) | 0.782 ± 0.36 [0.72, 0.84] (n=150) | 0.651 ± 0.40 [0.59, 0.71] (n=150) | 0.665 ± 0.35 [0.61, 0.72] (n=150) |
| mixed | oracle | 0.986 ± 0.08 [0.97, 1.00] (n=150) | 1.000 ± 0.00 [1.00, 1.00] (n=150) | 0.870 ± 0.27 [0.83, 0.91] (n=150) | 0.942 ± 0.15 [0.92, 0.96] (n=150) |
| mixed | random | 0.002 ± 0.03 [0.00, 0.01] (n=150) | 0.001 ± 0.02 [0.00, 0.00] (n=150) | 0.002 ± 0.02 [0.00, 0.01] (n=150) | 0.001 ± 0.01 [0.00, 0.00] (n=150) |
| reduced | lexical | 0.641 ± 0.39 [0.57, 0.71] (n=111) | 0.730 ± 0.40 [0.65, 0.80] (n=111) | 0.462 ± 0.44 [0.39, 0.53] (n=150) | 0.461 ± 0.42 [0.40, 0.53] (n=150) |
| reduced | dense | 0.755 ± 0.35 [0.69, 0.82] (n=111) | 0.889 ± 0.29 [0.83, 0.94] (n=111) | 0.547 ± 0.45 [0.47, 0.62] (n=150) | 0.572 ± 0.44 [0.50, 0.64] (n=150) |
| reduced | hybrid | 0.702 ± 0.37 [0.63, 0.77] (n=111) | 0.803 ± 0.35 [0.74, 0.86] (n=111) | 0.507 ± 0.44 [0.44, 0.58] (n=150) | 0.513 ± 0.42 [0.45, 0.58] (n=150) |
| reduced | oracle | 0.982 ± 0.09 [0.96, 1.00] (n=111) | 1.000 ± 0.00 [1.00, 1.00] (n=111) | 0.662 ± 0.44 [0.59, 0.73] (n=150) | 0.705 ± 0.44 [0.64, 0.77] (n=150) |
| reduced | random | 0.003 ± 0.03 [0.00, 0.01] (n=111) | 0.009 ± 0.09 [0.00, 0.03] (n=111) | 0.001 ± 0.02 [0.00, 0.00] (n=150) | 0.002 ± 0.03 [0.00, 0.01] (n=150) |

### 2.3 Generation

`abstained` and `partial` are rates. `faithfulness` is over answers that made at least one factual claim; `answer_relevance` over answers that did not abstain (see n).

| KB | method | faithfulness | abstained | partial | answer_relevance | context_relevance | reference_agreement |
|---|---|---|---|---|---|---|---|
| curated | lexical | 0.948 ± 0.15 [0.92, 0.97] (n=135) | 0.100 ± 0.30 [0.05, 0.15] (n=150) | 0.093 ± 0.29 [0.05, 0.14] (n=150) | 0.746 ± 0.10 [0.73, 0.76] (n=135) | 0.317 ± 0.22 [0.28, 0.35] (n=150) | 0.846 ± 0.30 [0.79, 0.89] (n=134) |
| curated | dense | 0.965 ± 0.15 [0.94, 0.99] (n=143) | 0.040 ± 0.20 [0.01, 0.07] (n=150) | 0.127 ± 0.33 [0.07, 0.18] (n=150) | 0.758 ± 0.09 [0.74, 0.77] (n=144) | 0.365 ± 0.24 [0.33, 0.40] (n=150) | 0.893 ± 0.25 [0.85, 0.93] (n=144) |
| curated | hybrid | 0.954 ± 0.17 [0.92, 0.98] (n=141) | 0.060 ± 0.24 [0.03, 0.10] (n=150) | 0.107 ± 0.31 [0.06, 0.16] (n=150) | 0.757 ± 0.09 [0.74, 0.77] (n=141) | 0.344 ± 0.22 [0.31, 0.38] (n=150) | 0.877 ± 0.26 [0.83, 0.92] (n=141) |
| curated | oracle | 0.986 ± 0.10 [0.97, 1.00] (n=148) | 0.007 ± 0.08 [0.00, 0.02] (n=150) | 0.113 ± 0.32 [0.07, 0.17] (n=150) | 0.758 ± 0.10 [0.74, 0.77] (n=149) | 0.809 ± 0.30 [0.76, 0.86] (n=150) | 0.990 ± 0.09 [0.97, 1.00] (n=148) |
| curated | random | 0.714 ± 0.49 [0.43, 1.00] (n=7) | 0.953 ± 0.21 [0.91, 0.99] (n=150) | 0.013 ± 0.12 [0.00, 0.03] (n=150) | 0.817 ± 0.08 [0.76, 0.86] (n=7) | 0.015 ± 0.06 [0.01, 0.03] (n=150) | 0.262 ± 0.38 [0.05, 0.55] (n=7) |
| mixed | lexical | 0.943 ± 0.18 [0.91, 0.97] (n=136) | 0.073 ± 0.26 [0.03, 0.12] (n=150) | 0.147 ± 0.35 [0.09, 0.21] (n=150) | 0.755 ± 0.10 [0.74, 0.77] (n=139) | 0.325 ± 0.23 [0.29, 0.36] (n=150) | 0.841 ± 0.30 [0.79, 0.89] (n=137) |
| mixed | dense | 0.965 ± 0.13 [0.94, 0.98] (n=142) | 0.047 ± 0.21 [0.01, 0.08] (n=150) | 0.133 ± 0.34 [0.08, 0.19] (n=150) | 0.757 ± 0.09 [0.74, 0.77] (n=143) | 0.377 ± 0.24 [0.34, 0.42] (n=150) | 0.887 ± 0.26 [0.84, 0.93] (n=143) |
| mixed | hybrid | 0.959 ± 0.16 [0.93, 0.98] (n=139) | 0.053 ± 0.23 [0.02, 0.09] (n=150) | 0.100 ± 0.30 [0.05, 0.15] (n=150) | 0.761 ± 0.10 [0.74, 0.78] (n=142) | 0.351 ± 0.23 [0.32, 0.39] (n=150) | 0.869 ± 0.26 [0.82, 0.91] (n=142) |
| mixed | oracle | 0.986 ± 0.07 [0.97, 1.00] (n=148) | 0.007 ± 0.08 [0.00, 0.02] (n=150) | 0.087 ± 0.28 [0.05, 0.13] (n=150) | 0.756 ± 0.10 [0.74, 0.77] (n=149) | 0.798 ± 0.31 [0.75, 0.85] (n=150) | 0.982 ± 0.08 [0.97, 0.99] (n=148) |
| mixed | random | 0.667 ± 0.41 [0.33, 0.92] (n=6) | 0.960 ± 0.20 [0.93, 0.99] (n=150) | 0.013 ± 0.12 [0.00, 0.03] (n=150) | 0.784 ± 0.16 [0.65, 0.86] (n=6) | 0.013 ± 0.05 [0.01, 0.02] (n=150) | 0.208 ± 0.40 [0.00, 0.54] (n=6) |
| reduced | lexical | 0.942 ± 0.19 [0.90, 0.97] (n=110) | 0.267 ± 0.44 [0.20, 0.34] (n=150) | 0.107 ± 0.31 [0.06, 0.16] (n=150) | 0.750 ± 0.10 [0.73, 0.77] (n=110) | 0.264 ± 0.24 [0.23, 0.30] (n=150) | 0.835 ± 0.33 [0.77, 0.89] (n=110) |
| reduced | dense | 0.911 ± 0.24 [0.86, 0.95] (n=116) | 0.220 ± 0.42 [0.15, 0.29] (n=150) | 0.093 ± 0.29 [0.05, 0.14] (n=150) | 0.750 ± 0.10 [0.73, 0.77] (n=117) | 0.285 ± 0.26 [0.25, 0.33] (n=150) | 0.838 ± 0.31 [0.78, 0.89] (n=115) |
| reduced | hybrid | 0.945 ± 0.18 [0.91, 0.97] (n=113) | 0.227 ± 0.42 [0.16, 0.29] (n=150) | 0.127 ± 0.33 [0.07, 0.18] (n=150) | 0.749 ± 0.10 [0.73, 0.77] (n=116) | 0.275 ± 0.24 [0.24, 0.31] (n=150) | 0.846 ± 0.29 [0.79, 0.90] (n=114) |
| reduced | oracle | 0.992 ± 0.04 [0.98, 1.00] (n=108) | 0.273 ± 0.45 [0.20, 0.35] (n=150) | 0.060 ± 0.24 [0.03, 0.10] (n=150) | 0.755 ± 0.10 [0.74, 0.77] (n=109) | 0.799 ± 0.31 [0.74, 0.86] (n=111) | 0.990 ± 0.05 [0.98, 1.00] (n=108) |
| reduced | random | 0.875 ± 0.25 [0.62, 1.00] (n=4) | 0.973 ± 0.16 [0.95, 0.99] (n=150) | 0.013 ± 0.12 [0.00, 0.03] (n=150) | 0.707 ± 0.12 [0.61, 0.81] (n=4) | 0.009 ± 0.04 [0.00, 0.02] (n=150) | 0.250 ± 0.50 [0.00, 0.75] (n=4) |
| none | none |  | 0.000 ± 0.00 [0.00, 0.00] (n=150) | 0.013 ± 0.12 [0.00, 0.03] (n=150) | 0.757 ± 0.09 [0.74, 0.77] (n=150) |  | 0.555 ± 0.32 [0.50, 0.61] (n=150) |

### 2.4 Do the real configurations fall between the random and oracle baselines?

- PASS — retrieval metrics and context relevance of every real configuration lie between random and oracle; so does the abstention rate on every KB where all queries are answerable: 3 KB variants x 3 retrievers

Faithfulness and reference agreement are reported but not required to be bracketed: they are conditional on the answer making factual claims, and with random context most answers abstain, leaving few (and unrepresentative) answers to score.

| KB | method | recall_at_k_all | mrr_all | ndcg_at_k_all | context_relevance | reference_agreement | abstained |
|---|---|---|---|---|---|---|---|
| curated | random | 0.000 | 0.000 | 0.000 | 0.015 | 0.262 | 0.953 |
| curated | lexical | 0.608 | 0.695 | 0.595 | 0.317 | 0.846 | 0.100 |
| curated | dense | 0.732 | 0.871 | 0.749 | 0.365 | 0.893 | 0.040 |
| curated | hybrid | 0.677 | 0.774 | 0.668 | 0.344 | 0.877 | 0.060 |
| curated | oracle | 0.986 | 1.000 | 1.000 | 0.809 | 0.990 | 0.007 |
| mixed | random | 0.002 | 0.001 | 0.001 | 0.013 | 0.208 | 0.960 |
| mixed | lexical | 0.606 | 0.697 | 0.594 | 0.325 | 0.841 | 0.073 |
| mixed | dense | 0.724 | 0.871 | 0.744 | 0.377 | 0.887 | 0.047 |
| mixed | hybrid | 0.674 | 0.782 | 0.670 | 0.351 | 0.869 | 0.053 |
| mixed | oracle | 0.986 | 1.000 | 1.000 | 0.798 | 0.982 | 0.007 |
| reduced | random | 0.002 | 0.007 | 0.003 | 0.009 | 0.250 | 0.973 |
| reduced | lexical | 0.474 | 0.540 | 0.466 | 0.264 | 0.835 | 0.267 |
| reduced | dense | 0.559 | 0.658 | 0.573 | 0.285 | 0.838 | 0.220 |
| reduced | hybrid | 0.520 | 0.594 | 0.517 | 0.275 | 0.846 | 0.227 |
| reduced | oracle | 0.727 | 0.740 | 0.740 | 0.799 | 0.990 | 0.273 |
| none | none |  |  |  |  | 0.555 | 0.000 |

### 2.4b What happens on queries the KB cannot answer

On a KB with missing topics the oracle is not a bound on the abstention rate: for a query whose answer was removed the oracle supplies no context and the model abstains, whereas a real retriever supplies its nearest wrong passages and the model sometimes answers from them. The table splits each configuration's queries by whether the KB still holds their answer. `answers given` = share of answers not labelled `abstained`; faithfulness and reference agreement are over the answers they are defined for (n in brackets).

| KB | method | queries | n | answers given | of which partial | faithfulness | reference agreement | context relevance |
|---|---|---|---|---|---|---|---|---|
| reduced | lexical | answer in KB | 111 | 0.910 | 0.099 | 0.957 (n=101) | 0.883 (n=101) | 0.326 |
| reduced | lexical | answer NOT in KB | 39 | 0.231 | 0.128 | 0.778 (n=9) | 0.296 (n=9) | 0.087 |
| reduced | dense | answer in KB | 111 | 0.946 | 0.090 | 0.943 (n=104) | 0.904 (n=103) | 0.360 |
| reduced | dense | answer NOT in KB | 39 | 0.308 | 0.103 | 0.633 (n=12) | 0.267 (n=12) | 0.072 |
| reduced | hybrid | answer in KB | 111 | 0.919 | 0.108 | 0.977 (n=100) | 0.901 (n=101) | 0.328 |
| reduced | hybrid | answer NOT in KB | 39 | 0.359 | 0.179 | 0.700 (n=13) | 0.419 (n=13) | 0.123 |
| reduced | oracle | answer in KB | 111 | 0.982 | 0.081 | 0.992 (n=108) | 0.990 (n=108) | 0.799 |
| reduced | oracle | answer NOT in KB | 39 | 0.000 | 0.000 |  |  |  |

### 2.5 Which differences are real? Paired comparisons on the same queries

Each row compares two configurations query by query: mean difference (a − b), its 95% paired-bootstrap interval, a Wilcoxon signed-rank p-value adjusted (Holm) for the number of pairs tested for that metric, and the smallest true difference this many queries could reliably detect (80% power). **Verdict `could be noise` means the ordering of those two configurations must not be reported as a finding.**

**Retrieval methods compared within each KB variant**

| held_fixed | metric | a | b | n_pairs | mean_a | mean_b | mean_diff | ci95_low | ci95_high | p_holm | min_detectable_diff | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curated | recall_at_k_all | dense | hybrid | 150 | 0.732 | 0.677 | 0.055 | 0.024 | 0.089 | 0.005 | 0.047 | difference |
| curated | recall_at_k_all | dense | lexical | 150 | 0.732 | 0.608 | 0.124 | 0.076 | 0.177 | 0.000 | 0.071 | difference |
| curated | recall_at_k_all | hybrid | lexical | 150 | 0.677 | 0.608 | 0.069 | 0.035 | 0.108 | 0.003 | 0.053 | difference |
| curated | mrr_all | dense | hybrid | 150 | 0.871 | 0.774 | 0.097 | 0.061 | 0.137 | 0.000 | 0.054 | difference |
| curated | mrr_all | dense | lexical | 150 | 0.871 | 0.695 | 0.176 | 0.122 | 0.233 | 0.000 | 0.078 | difference |
| curated | mrr_all | hybrid | lexical | 150 | 0.774 | 0.695 | 0.078 | 0.046 | 0.115 | 0.000 | 0.049 | difference |
| curated | ndcg_at_k_all | dense | hybrid | 150 | 0.749 | 0.668 | 0.081 | 0.049 | 0.115 | 0.000 | 0.047 | difference |
| curated | ndcg_at_k_all | dense | lexical | 150 | 0.749 | 0.595 | 0.153 | 0.105 | 0.205 | 0.000 | 0.070 | difference |
| curated | ndcg_at_k_all | hybrid | lexical | 150 | 0.668 | 0.595 | 0.073 | 0.044 | 0.106 | 0.000 | 0.045 | difference |
| curated | recall_at_k_lenient_all | dense | hybrid | 150 | 0.711 | 0.653 | 0.058 | 0.026 | 0.091 | 0.004 | 0.046 | difference |
| curated | recall_at_k_lenient_all | dense | lexical | 150 | 0.711 | 0.584 | 0.126 | 0.079 | 0.179 | 0.000 | 0.071 | difference |
| curated | recall_at_k_lenient_all | hybrid | lexical | 150 | 0.653 | 0.584 | 0.069 | 0.035 | 0.108 | 0.003 | 0.053 | difference |
| curated | ndcg_at_k_lenient_all | dense | hybrid | 150 | 0.746 | 0.663 | 0.083 | 0.052 | 0.117 | 0.000 | 0.047 | difference |
| curated | ndcg_at_k_lenient_all | dense | lexical | 150 | 0.746 | 0.589 | 0.157 | 0.109 | 0.208 | 0.000 | 0.070 | difference |
| curated | ndcg_at_k_lenient_all | hybrid | lexical | 150 | 0.663 | 0.589 | 0.073 | 0.044 | 0.106 | 0.000 | 0.045 | difference |
| curated | faithfulness | dense | hybrid | 139 | 0.973 | 0.957 | 0.016 | -0.007 | 0.043 | 1.000 | 0.035 | could be noise |
| curated | faithfulness | dense | lexical | 134 | 0.976 | 0.955 | 0.021 | -0.007 | 0.048 | 0.642 | 0.039 | could be noise |
| curated | faithfulness | hybrid | lexical | 135 | 0.956 | 0.948 | 0.008 | -0.023 | 0.039 | 1.000 | 0.045 | could be noise |
| curated | abstained | dense | hybrid | 150 | 0.040 | 0.060 | -0.020 | -0.053 | 0.007 | 0.786 | 0.042 | could be noise |
| curated | abstained | dense | lexical | 150 | 0.040 | 0.100 | -0.060 | -0.107 | -0.020 | 0.060 | 0.061 | could be noise |
| curated | abstained | hybrid | lexical | 150 | 0.060 | 0.100 | -0.040 | -0.073 | -0.013 | 0.114 | 0.045 | could be noise |
| curated | answer_relevance | dense | hybrid | 140 | 0.757 | 0.756 | 0.001 | -0.007 | 0.009 | 1.000 | 0.011 | could be noise |
| curated | answer_relevance | dense | lexical | 134 | 0.758 | 0.747 | 0.011 | 0.002 | 0.020 | 0.032 | 0.012 | difference |
| curated | answer_relevance | hybrid | lexical | 135 | 0.758 | 0.746 | 0.011 | 0.004 | 0.020 | 0.053 | 0.011 | could be noise |
| curated | context_relevance | dense | hybrid | 150 | 0.365 | 0.344 | 0.021 | -0.001 | 0.045 | 0.214 | 0.033 | could be noise |
| curated | context_relevance | dense | lexical | 150 | 0.365 | 0.317 | 0.048 | 0.024 | 0.073 | 0.002 | 0.035 | difference |
| curated | context_relevance | hybrid | lexical | 150 | 0.344 | 0.317 | 0.027 | 0.007 | 0.048 | 0.044 | 0.029 | difference |
| curated | reference_agreement | dense | hybrid | 140 | 0.903 | 0.884 | 0.019 | -0.003 | 0.042 | 0.620 | 0.032 | could be noise |
| curated | reference_agreement | dense | lexical | 133 | 0.914 | 0.852 | 0.062 | 0.030 | 0.098 | 0.004 | 0.048 | difference |
| curated | reference_agreement | hybrid | lexical | 134 | 0.886 | 0.846 | 0.040 | 0.011 | 0.073 | 0.096 | 0.044 | could be noise |
| mixed | recall_at_k_all | dense | hybrid | 150 | 0.724 | 0.674 | 0.050 | 0.019 | 0.083 | 0.009 | 0.046 | difference |
| mixed | recall_at_k_all | dense | lexical | 150 | 0.724 | 0.606 | 0.118 | 0.071 | 0.170 | 0.000 | 0.070 | difference |
| mixed | recall_at_k_all | hybrid | lexical | 150 | 0.674 | 0.606 | 0.068 | 0.036 | 0.106 | 0.002 | 0.051 | difference |
| mixed | mrr_all | dense | hybrid | 150 | 0.871 | 0.782 | 0.089 | 0.053 | 0.128 | 0.000 | 0.053 | difference |
| mixed | mrr_all | dense | lexical | 150 | 0.871 | 0.697 | 0.174 | 0.122 | 0.231 | 0.000 | 0.078 | difference |
| mixed | mrr_all | hybrid | lexical | 150 | 0.782 | 0.697 | 0.085 | 0.052 | 0.124 | 0.000 | 0.052 | difference |
| mixed | ndcg_at_k_all | dense | hybrid | 150 | 0.744 | 0.670 | 0.073 | 0.042 | 0.107 | 0.000 | 0.046 | difference |
| mixed | ndcg_at_k_all | dense | lexical | 150 | 0.744 | 0.594 | 0.150 | 0.103 | 0.200 | 0.000 | 0.069 | difference |
| mixed | ndcg_at_k_all | hybrid | lexical | 150 | 0.670 | 0.594 | 0.076 | 0.047 | 0.110 | 0.000 | 0.045 | difference |
| mixed | recall_at_k_lenient_all | dense | hybrid | 150 | 0.703 | 0.651 | 0.052 | 0.021 | 0.085 | 0.007 | 0.046 | difference |
| mixed | recall_at_k_lenient_all | dense | lexical | 150 | 0.703 | 0.583 | 0.120 | 0.073 | 0.172 | 0.000 | 0.070 | difference |
| mixed | recall_at_k_lenient_all | hybrid | lexical | 150 | 0.651 | 0.583 | 0.068 | 0.036 | 0.106 | 0.002 | 0.051 | difference |
| mixed | ndcg_at_k_lenient_all | dense | hybrid | 150 | 0.741 | 0.665 | 0.076 | 0.046 | 0.109 | 0.000 | 0.045 | difference |
| mixed | ndcg_at_k_lenient_all | dense | lexical | 150 | 0.741 | 0.588 | 0.153 | 0.107 | 0.202 | 0.000 | 0.068 | difference |
| mixed | ndcg_at_k_lenient_all | hybrid | lexical | 150 | 0.665 | 0.588 | 0.077 | 0.048 | 0.110 | 0.000 | 0.045 | difference |
| mixed | faithfulness | dense | hybrid | 138 | 0.968 | 0.966 | 0.002 | -0.015 | 0.021 | 1.000 | 0.026 | could be noise |
| mixed | faithfulness | dense | lexical | 134 | 0.969 | 0.954 | 0.015 | -0.007 | 0.041 | 1.000 | 0.035 | could be noise |
| mixed | faithfulness | hybrid | lexical | 134 | 0.965 | 0.954 | 0.011 | -0.009 | 0.034 | 1.000 | 0.031 | could be noise |
| mixed | abstained | dense | hybrid | 150 | 0.047 | 0.053 | -0.007 | -0.033 | 0.013 | 1.000 | 0.032 | could be noise |
| mixed | abstained | dense | lexical | 150 | 0.047 | 0.073 | -0.027 | -0.067 | 0.007 | 0.786 | 0.053 | could be noise |
| mixed | abstained | hybrid | lexical | 150 | 0.053 | 0.073 | -0.020 | -0.053 | 0.007 | 0.786 | 0.042 | could be noise |
| mixed | answer_relevance | dense | hybrid | 141 | 0.756 | 0.760 | -0.004 | -0.010 | 0.003 | 1.000 | 0.009 | could be noise |
| mixed | answer_relevance | dense | lexical | 137 | 0.759 | 0.756 | 0.002 | -0.005 | 0.010 | 1.000 | 0.011 | could be noise |
| mixed | answer_relevance | hybrid | lexical | 138 | 0.763 | 0.755 | 0.007 | 0.001 | 0.014 | 0.332 | 0.009 | could be noise |
| mixed | context_relevance | dense | hybrid | 150 | 0.377 | 0.351 | 0.027 | 0.003 | 0.051 | 0.083 | 0.035 | could be noise |
| mixed | context_relevance | dense | lexical | 150 | 0.377 | 0.325 | 0.052 | 0.025 | 0.080 | 0.001 | 0.039 | difference |
| mixed | context_relevance | hybrid | lexical | 150 | 0.351 | 0.325 | 0.025 | 0.000 | 0.052 | 0.141 | 0.036 | could be noise |
| mixed | reference_agreement | dense | hybrid | 141 | 0.889 | 0.875 | 0.014 | -0.016 | 0.043 | 1.000 | 0.042 | could be noise |
| mixed | reference_agreement | dense | lexical | 135 | 0.886 | 0.853 | 0.033 | -0.005 | 0.071 | 0.761 | 0.055 | could be noise |
| mixed | reference_agreement | hybrid | lexical | 136 | 0.876 | 0.847 | 0.029 | -0.007 | 0.067 | 0.761 | 0.052 | could be noise |
| reduced | recall_at_k_all | dense | hybrid | 150 | 0.559 | 0.520 | 0.039 | 0.016 | 0.065 | 0.009 | 0.036 | difference |
| reduced | recall_at_k_all | dense | lexical | 150 | 0.559 | 0.474 | 0.085 | 0.047 | 0.126 | 0.001 | 0.056 | difference |
| reduced | recall_at_k_all | hybrid | lexical | 150 | 0.520 | 0.474 | 0.045 | 0.018 | 0.079 | 0.009 | 0.043 | difference |
| reduced | mrr_all | dense | hybrid | 150 | 0.658 | 0.594 | 0.064 | 0.034 | 0.098 | 0.001 | 0.046 | difference |
| reduced | mrr_all | dense | lexical | 150 | 0.658 | 0.540 | 0.118 | 0.073 | 0.166 | 0.000 | 0.066 | difference |
| reduced | mrr_all | hybrid | lexical | 150 | 0.594 | 0.540 | 0.054 | 0.027 | 0.084 | 0.001 | 0.041 | difference |
| reduced | ndcg_at_k_all | dense | hybrid | 150 | 0.573 | 0.517 | 0.056 | 0.031 | 0.084 | 0.000 | 0.038 | difference |
| reduced | ndcg_at_k_all | dense | lexical | 150 | 0.573 | 0.466 | 0.108 | 0.069 | 0.150 | 0.000 | 0.058 | difference |
| reduced | ndcg_at_k_all | hybrid | lexical | 150 | 0.517 | 0.466 | 0.051 | 0.028 | 0.079 | 0.000 | 0.036 | difference |
| reduced | recall_at_k_lenient_all | dense | hybrid | 150 | 0.547 | 0.507 | 0.040 | 0.017 | 0.066 | 0.007 | 0.036 | difference |
| reduced | recall_at_k_lenient_all | dense | lexical | 150 | 0.547 | 0.462 | 0.085 | 0.048 | 0.127 | 0.000 | 0.056 | difference |
| reduced | recall_at_k_lenient_all | hybrid | lexical | 150 | 0.507 | 0.462 | 0.045 | 0.018 | 0.079 | 0.007 | 0.043 | difference |
| reduced | ndcg_at_k_lenient_all | dense | hybrid | 150 | 0.572 | 0.513 | 0.059 | 0.034 | 0.085 | 0.000 | 0.037 | difference |
| reduced | ndcg_at_k_lenient_all | dense | lexical | 150 | 0.572 | 0.461 | 0.110 | 0.072 | 0.152 | 0.000 | 0.058 | difference |
| reduced | ndcg_at_k_lenient_all | hybrid | lexical | 150 | 0.513 | 0.461 | 0.052 | 0.029 | 0.079 | 0.000 | 0.036 | difference |
| reduced | faithfulness | dense | hybrid | 108 | 0.951 | 0.953 | -0.002 | -0.029 | 0.027 | 1.000 | 0.040 | could be noise |
| reduced | faithfulness | dense | lexical | 108 | 0.935 | 0.950 | -0.015 | -0.054 | 0.021 | 1.000 | 0.054 | could be noise |
| reduced | faithfulness | hybrid | lexical | 106 | 0.959 | 0.954 | 0.005 | -0.026 | 0.037 | 1.000 | 0.045 | could be noise |
| reduced | abstained | dense | hybrid | 150 | 0.220 | 0.227 | -0.007 | -0.053 | 0.040 | 1.000 | 0.068 | could be noise |
| reduced | abstained | dense | lexical | 150 | 0.220 | 0.267 | -0.047 | -0.087 | -0.007 | 0.244 | 0.061 | could be noise |
| reduced | abstained | hybrid | lexical | 150 | 0.227 | 0.267 | -0.040 | -0.087 | 0.000 | 0.500 | 0.064 | could be noise |
| reduced | answer_relevance | dense | hybrid | 110 | 0.749 | 0.747 | 0.002 | -0.006 | 0.010 | 0.867 | 0.012 | could be noise |
| reduced | answer_relevance | dense | lexical | 108 | 0.749 | 0.750 | -0.001 | -0.010 | 0.007 | 1.000 | 0.012 | could be noise |
| reduced | answer_relevance | hybrid | lexical | 107 | 0.744 | 0.750 | -0.006 | -0.015 | 0.003 | 0.867 | 0.013 | could be noise |
| reduced | context_relevance | dense | hybrid | 150 | 0.285 | 0.275 | 0.011 | -0.011 | 0.033 | 0.608 | 0.032 | could be noise |
| reduced | context_relevance | dense | lexical | 150 | 0.285 | 0.264 | 0.021 | -0.003 | 0.045 | 0.344 | 0.034 | could be noise |
| reduced | context_relevance | hybrid | lexical | 150 | 0.275 | 0.264 | 0.011 | -0.011 | 0.032 | 0.608 | 0.029 | could be noise |
| reduced | reference_agreement | dense | hybrid | 108 | 0.867 | 0.862 | 0.005 | -0.025 | 0.033 | 1.000 | 0.041 | could be noise |
| reduced | reference_agreement | dense | lexical | 108 | 0.860 | 0.851 | 0.009 | -0.023 | 0.042 | 1.000 | 0.047 | could be noise |
| reduced | reference_agreement | hybrid | lexical | 106 | 0.870 | 0.862 | 0.008 | -0.019 | 0.038 | 1.000 | 0.041 | could be noise |

**KB variants compared within each retrieval method**

| held_fixed | metric | a | b | n_pairs | mean_a | mean_b | mean_diff | ci95_low | ci95_high | p_holm | min_detectable_diff | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | recall_at_k_all | curated | mixed | 150 | 0.732 | 0.724 | 0.008 | 0.000 | 0.018 | 0.307 | 0.013 | could be noise |
| dense | recall_at_k_all | curated | reduced | 150 | 0.732 | 0.559 | 0.173 | 0.116 | 0.235 | 0.000 | 0.085 | difference |
| dense | recall_at_k_all | mixed | reduced | 150 | 0.724 | 0.559 | 0.165 | 0.107 | 0.227 | 0.000 | 0.086 | difference |
| dense | mrr_all | curated | mixed | 150 | 0.871 | 0.871 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | could be noise |
| dense | mrr_all | curated | reduced | 150 | 0.871 | 0.658 | 0.213 | 0.150 | 0.280 | 0.000 | 0.094 | difference |
| dense | mrr_all | mixed | reduced | 150 | 0.871 | 0.658 | 0.213 | 0.150 | 0.280 | 0.000 | 0.094 | difference |
| dense | ndcg_at_k_all | curated | mixed | 150 | 0.749 | 0.744 | 0.005 | 0.001 | 0.011 | 0.129 | 0.007 | could be noise |
| dense | ndcg_at_k_all | curated | reduced | 150 | 0.749 | 0.573 | 0.175 | 0.118 | 0.236 | 0.000 | 0.084 | difference |
| dense | ndcg_at_k_all | mixed | reduced | 150 | 0.744 | 0.573 | 0.170 | 0.113 | 0.231 | 0.000 | 0.084 | difference |
| dense | recall_at_k_lenient_all | curated | mixed | 150 | 0.711 | 0.703 | 0.008 | 0.000 | 0.018 | 0.307 | 0.013 | could be noise |
| dense | recall_at_k_lenient_all | curated | reduced | 150 | 0.711 | 0.547 | 0.164 | 0.107 | 0.226 | 0.000 | 0.085 | difference |
| dense | recall_at_k_lenient_all | mixed | reduced | 150 | 0.703 | 0.547 | 0.156 | 0.098 | 0.219 | 0.000 | 0.086 | difference |
| dense | ndcg_at_k_lenient_all | curated | mixed | 150 | 0.746 | 0.741 | 0.005 | 0.001 | 0.011 | 0.129 | 0.007 | could be noise |
| dense | ndcg_at_k_lenient_all | curated | reduced | 150 | 0.746 | 0.572 | 0.174 | 0.117 | 0.235 | 0.000 | 0.085 | difference |
| dense | ndcg_at_k_lenient_all | mixed | reduced | 150 | 0.741 | 0.572 | 0.169 | 0.112 | 0.231 | 0.000 | 0.085 | difference |
| dense | faithfulness | curated | mixed | 141 | 0.974 | 0.965 | 0.009 | -0.006 | 0.024 | 0.770 | 0.022 | could be noise |
| dense | faithfulness | curated | reduced | 114 | 0.979 | 0.927 | 0.052 | 0.018 | 0.089 | 0.051 | 0.052 | could be noise |
| dense | faithfulness | mixed | reduced | 114 | 0.971 | 0.927 | 0.044 | 0.013 | 0.081 | 0.169 | 0.049 | could be noise |
| dense | abstained | curated | mixed | 150 | 0.040 | 0.047 | -0.007 | -0.033 | 0.013 | 0.635 | 0.032 | could be noise |
| dense | abstained | curated | reduced | 150 | 0.040 | 0.220 | -0.180 | -0.247 | -0.120 | 0.000 | 0.092 | difference |
| dense | abstained | mixed | reduced | 150 | 0.047 | 0.220 | -0.173 | -0.240 | -0.113 | 0.000 | 0.091 | difference |
| dense | answer_relevance | curated | mixed | 142 | 0.758 | 0.757 | 0.002 | -0.004 | 0.008 | 1.000 | 0.009 | could be noise |
| dense | answer_relevance | curated | reduced | 116 | 0.750 | 0.750 | -0.000 | -0.009 | 0.009 | 1.000 | 0.013 | could be noise |
| dense | answer_relevance | mixed | reduced | 116 | 0.749 | 0.750 | -0.001 | -0.009 | 0.008 | 1.000 | 0.012 | could be noise |
| dense | context_relevance | curated | mixed | 150 | 0.365 | 0.377 | -0.012 | -0.027 | 0.003 | 0.426 | 0.021 | could be noise |
| dense | context_relevance | curated | reduced | 150 | 0.365 | 0.285 | 0.080 | 0.053 | 0.109 | 0.000 | 0.040 | difference |
| dense | context_relevance | mixed | reduced | 150 | 0.377 | 0.285 | 0.092 | 0.063 | 0.123 | 0.000 | 0.043 | difference |
| dense | reference_agreement | curated | mixed | 142 | 0.901 | 0.889 | 0.011 | -0.014 | 0.036 | 1.000 | 0.035 | could be noise |
| dense | reference_agreement | curated | reduced | 115 | 0.901 | 0.838 | 0.064 | 0.023 | 0.111 | 0.034 | 0.063 | difference |
| dense | reference_agreement | mixed | reduced | 115 | 0.893 | 0.838 | 0.055 | 0.009 | 0.107 | 0.316 | 0.070 | could be noise |
| hybrid | recall_at_k_all | curated | mixed | 150 | 0.677 | 0.674 | 0.002 | -0.003 | 0.010 | 1.000 | 0.010 | could be noise |
| hybrid | recall_at_k_all | curated | reduced | 150 | 0.677 | 0.520 | 0.157 | 0.101 | 0.217 | 0.000 | 0.083 | difference |
| hybrid | recall_at_k_all | mixed | reduced | 150 | 0.674 | 0.520 | 0.155 | 0.098 | 0.216 | 0.000 | 0.084 | difference |
| hybrid | mrr_all | curated | mixed | 150 | 0.774 | 0.782 | -0.008 | -0.022 | 0.004 | 0.587 | 0.019 | could be noise |
| hybrid | mrr_all | curated | reduced | 150 | 0.774 | 0.594 | 0.180 | 0.119 | 0.244 | 0.000 | 0.089 | difference |
| hybrid | mrr_all | mixed | reduced | 150 | 0.782 | 0.594 | 0.188 | 0.126 | 0.254 | 0.000 | 0.092 | difference |
| hybrid | ndcg_at_k_all | curated | mixed | 150 | 0.668 | 0.670 | -0.002 | -0.011 | 0.007 | 0.665 | 0.013 | could be noise |
| hybrid | ndcg_at_k_all | curated | reduced | 150 | 0.668 | 0.517 | 0.151 | 0.099 | 0.208 | 0.000 | 0.078 | difference |
| hybrid | ndcg_at_k_all | mixed | reduced | 150 | 0.670 | 0.517 | 0.153 | 0.100 | 0.211 | 0.000 | 0.080 | difference |
| hybrid | recall_at_k_lenient_all | curated | mixed | 150 | 0.653 | 0.651 | 0.002 | -0.003 | 0.010 | 1.000 | 0.010 | could be noise |
| hybrid | recall_at_k_lenient_all | curated | reduced | 150 | 0.653 | 0.507 | 0.147 | 0.090 | 0.207 | 0.000 | 0.084 | difference |
| hybrid | recall_at_k_lenient_all | mixed | reduced | 150 | 0.651 | 0.507 | 0.144 | 0.088 | 0.206 | 0.000 | 0.085 | difference |
| hybrid | ndcg_at_k_lenient_all | curated | mixed | 150 | 0.663 | 0.665 | -0.002 | -0.011 | 0.007 | 0.665 | 0.013 | could be noise |
| hybrid | ndcg_at_k_lenient_all | curated | reduced | 150 | 0.663 | 0.513 | 0.150 | 0.097 | 0.207 | 0.000 | 0.079 | difference |
| hybrid | ndcg_at_k_lenient_all | mixed | reduced | 150 | 0.665 | 0.513 | 0.152 | 0.098 | 0.211 | 0.000 | 0.081 | difference |
| hybrid | faithfulness | curated | mixed | 139 | 0.961 | 0.959 | 0.002 | -0.018 | 0.024 | 1.000 | 0.030 | could be noise |
| hybrid | faithfulness | curated | reduced | 112 | 0.972 | 0.950 | 0.022 | -0.012 | 0.056 | 0.770 | 0.049 | could be noise |
| hybrid | faithfulness | mixed | reduced | 111 | 0.977 | 0.950 | 0.027 | -0.006 | 0.060 | 0.432 | 0.048 | could be noise |
| hybrid | abstained | curated | mixed | 150 | 0.060 | 0.053 | 0.007 | 0.000 | 0.020 | 0.635 | 0.019 | could be noise |
| hybrid | abstained | curated | reduced | 150 | 0.060 | 0.227 | -0.167 | -0.233 | -0.107 | 0.000 | 0.093 | difference |
| hybrid | abstained | mixed | reduced | 150 | 0.053 | 0.227 | -0.173 | -0.240 | -0.113 | 0.000 | 0.091 | difference |
| hybrid | answer_relevance | curated | mixed | 141 | 0.757 | 0.760 | -0.003 | -0.009 | 0.003 | 1.000 | 0.008 | could be noise |
| hybrid | answer_relevance | curated | reduced | 114 | 0.750 | 0.748 | 0.003 | -0.005 | 0.011 | 1.000 | 0.011 | could be noise |
| hybrid | answer_relevance | mixed | reduced | 115 | 0.754 | 0.749 | 0.005 | -0.002 | 0.013 | 1.000 | 0.011 | could be noise |
| hybrid | context_relevance | curated | mixed | 150 | 0.344 | 0.351 | -0.007 | -0.028 | 0.012 | 1.000 | 0.028 | could be noise |
| hybrid | context_relevance | curated | reduced | 150 | 0.344 | 0.275 | 0.069 | 0.041 | 0.100 | 0.000 | 0.042 | difference |
| hybrid | context_relevance | mixed | reduced | 150 | 0.351 | 0.275 | 0.076 | 0.043 | 0.111 | 0.000 | 0.048 | difference |
| hybrid | reference_agreement | curated | mixed | 141 | 0.877 | 0.872 | 0.006 | -0.015 | 0.028 | 1.000 | 0.031 | could be noise |
| hybrid | reference_agreement | curated | reduced | 112 | 0.905 | 0.849 | 0.056 | 0.022 | 0.095 | 0.033 | 0.053 | difference |
| hybrid | reference_agreement | mixed | reduced | 113 | 0.896 | 0.850 | 0.045 | 0.007 | 0.087 | 0.316 | 0.058 | could be noise |
| lexical | recall_at_k_all | curated | mixed | 150 | 0.608 | 0.606 | 0.002 | -0.017 | 0.017 | 1.000 | 0.023 | could be noise |
| lexical | recall_at_k_all | curated | reduced | 150 | 0.608 | 0.474 | 0.133 | 0.083 | 0.188 | 0.000 | 0.075 | difference |
| lexical | recall_at_k_all | mixed | reduced | 150 | 0.606 | 0.474 | 0.132 | 0.080 | 0.188 | 0.000 | 0.078 | difference |
| lexical | mrr_all | curated | mixed | 150 | 0.695 | 0.697 | -0.001 | -0.021 | 0.016 | 1.000 | 0.026 | could be noise |
| lexical | mrr_all | curated | reduced | 150 | 0.695 | 0.540 | 0.155 | 0.098 | 0.218 | 0.000 | 0.084 | difference |
| lexical | mrr_all | mixed | reduced | 150 | 0.697 | 0.540 | 0.157 | 0.097 | 0.222 | 0.000 | 0.088 | difference |
| lexical | ndcg_at_k_all | curated | mixed | 150 | 0.595 | 0.594 | 0.001 | -0.018 | 0.018 | 0.689 | 0.025 | could be noise |
| lexical | ndcg_at_k_all | curated | reduced | 150 | 0.595 | 0.466 | 0.130 | 0.082 | 0.182 | 0.000 | 0.071 | difference |
| lexical | ndcg_at_k_all | mixed | reduced | 150 | 0.594 | 0.466 | 0.128 | 0.078 | 0.183 | 0.000 | 0.075 | difference |
| lexical | recall_at_k_lenient_all | curated | mixed | 150 | 0.584 | 0.583 | 0.002 | -0.017 | 0.017 | 1.000 | 0.023 | could be noise |
| lexical | recall_at_k_lenient_all | curated | reduced | 150 | 0.584 | 0.462 | 0.123 | 0.073 | 0.177 | 0.000 | 0.075 | difference |
| lexical | recall_at_k_lenient_all | mixed | reduced | 150 | 0.583 | 0.462 | 0.121 | 0.069 | 0.178 | 0.000 | 0.078 | difference |
| lexical | ndcg_at_k_lenient_all | curated | mixed | 150 | 0.589 | 0.588 | 0.001 | -0.018 | 0.018 | 0.689 | 0.025 | could be noise |
| lexical | ndcg_at_k_lenient_all | curated | reduced | 150 | 0.589 | 0.461 | 0.128 | 0.079 | 0.181 | 0.000 | 0.072 | difference |
| lexical | ndcg_at_k_lenient_all | mixed | reduced | 150 | 0.588 | 0.461 | 0.127 | 0.075 | 0.182 | 0.000 | 0.076 | difference |
| lexical | faithfulness | curated | mixed | 134 | 0.951 | 0.946 | 0.005 | -0.018 | 0.031 | 1.000 | 0.034 | could be noise |
| lexical | faithfulness | curated | reduced | 109 | 0.956 | 0.951 | 0.005 | -0.022 | 0.035 | 1.000 | 0.041 | could be noise |
| lexical | faithfulness | mixed | reduced | 108 | 0.960 | 0.955 | 0.005 | -0.024 | 0.033 | 1.000 | 0.041 | could be noise |
| lexical | abstained | curated | mixed | 150 | 0.100 | 0.073 | 0.027 | 0.000 | 0.060 | 0.307 | 0.045 | could be noise |
| lexical | abstained | curated | reduced | 150 | 0.100 | 0.267 | -0.167 | -0.233 | -0.107 | 0.000 | 0.090 | difference |
| lexical | abstained | mixed | reduced | 150 | 0.073 | 0.267 | -0.193 | -0.260 | -0.127 | 0.000 | 0.098 | difference |
| lexical | answer_relevance | curated | mixed | 134 | 0.746 | 0.752 | -0.006 | -0.014 | 0.001 | 0.354 | 0.011 | could be noise |
| lexical | answer_relevance | curated | reduced | 109 | 0.738 | 0.750 | -0.012 | -0.023 | -0.003 | 0.005 | 0.014 | difference |
| lexical | answer_relevance | mixed | reduced | 108 | 0.745 | 0.750 | -0.005 | -0.014 | 0.004 | 1.000 | 0.013 | could be noise |
| lexical | context_relevance | curated | mixed | 150 | 0.317 | 0.325 | -0.008 | -0.021 | 0.004 | 1.000 | 0.019 | could be noise |
| lexical | context_relevance | curated | reduced | 150 | 0.317 | 0.264 | 0.053 | 0.023 | 0.084 | 0.006 | 0.045 | difference |
| lexical | context_relevance | mixed | reduced | 150 | 0.325 | 0.264 | 0.061 | 0.032 | 0.092 | 0.004 | 0.043 | difference |
| lexical | reference_agreement | curated | mixed | 133 | 0.848 | 0.850 | -0.002 | -0.023 | 0.019 | 1.000 | 0.031 | could be noise |
| lexical | reference_agreement | curated | reduced | 108 | 0.865 | 0.851 | 0.015 | -0.024 | 0.057 | 1.000 | 0.058 | could be noise |
| lexical | reference_agreement | mixed | reduced | 108 | 0.868 | 0.851 | 0.017 | -0.024 | 0.062 | 1.000 | 0.062 | could be noise |

Totals: {'difference': 95, 'could be noise': 85}.

### 2.6 Are there enough queries?

The sweep used 150 queries. For each metric, the table gives the typical (median) spread of per-query paired differences between two real configurations, the smallest difference detectable at that spread with this many pairs, and how many queries would be needed to detect a difference of 0.05 or 0.10.

| metric | median n pairs | median SD of paired difference | detectable difference at this n | queries needed to detect 0.05 | queries needed to detect 0.10 |
|---|---|---|---|---|---|
| recall_at_k_all | 150 | 0.238 | 0.054 | 178 | 45 |
| mrr_all | 150 | 0.263 | 0.060 | 217 | 55 |
| ndcg_at_k_all | 150 | 0.230 | 0.053 | 166 | 42 |
| recall_at_k_lenient_all | 150 | 0.237 | 0.054 | 178 | 45 |
| ndcg_at_k_lenient_all | 150 | 0.228 | 0.052 | 163 | 41 |
| faithfulness | 124 | 0.153 | 0.040 | 74 | 19 |
| abstained | 150 | 0.266 | 0.061 | 223 | 56 |
| answer_relevance | 125 | 0.045 | 0.011 | 7 | 2 |
| context_relevance | 150 | 0.153 | 0.035 | 74 | 19 |
| reference_agreement | 124 | 0.190 | 0.048 | 114 | 29 |

### 2.7 Robustness: change after perturbing the context, against the no-change control

`control_rerun` regenerates on the identical context, so its deltas are pure generation/judge noise. A perturbation effect is only credible where its interval excludes 0 **and** it is clearly larger than the control's.

| KB | condition | faithfulness_delta | answer_relevance_delta | context_relevance_delta | reference_agreement_delta | abstained_delta | recall_at_k_delta |
|---|---|---|---|---|---|---|---|
| curated | control_rerun | -0.001 [-0.01, +0.01] (n=142) | -0.002 [-0.01, +0.00] (n=143) | -0.000 [-0.00, +0.00] (n=150) | -0.016 [-0.03, +0.00] (n=143) | +0.007 [+0.00, +0.02] (n=150) | +0.000 [+0.00, +0.00] (n=150) |
| curated | drop_top_relevant | -0.065 [-0.11, -0.02] (n=91) | -0.013 [-0.03, +0.00] (n=94) | -0.098 [-0.12, -0.07] (n=136) | -0.277 [-0.35, -0.20] (n=92) | +0.309 [+0.24, +0.39] (n=136) | -0.636 [-0.69, -0.58] (n=136) |
| curated | inject_noise | -0.002 [-0.02, +0.01] (n=137) | -0.000 [-0.01, +0.01] (n=138) | -0.001 [-0.01, +0.01] (n=147) | -0.018 [-0.04, +0.00] (n=138) | +0.020 [+0.00, +0.05] (n=147) | +0.000 [+0.00, +0.00] (n=147) |
| mixed | control_rerun | -0.003 [-0.02, +0.01] (n=141) | -0.001 [-0.01, +0.00] (n=142) | -0.004 [-0.01, +0.00] (n=150) | -0.005 [-0.02, +0.02] (n=141) | +0.007 [+0.00, +0.02] (n=150) | +0.000 [+0.00, +0.00] (n=150) |
| mixed | drop_top_relevant | -0.061 [-0.11, -0.02] (n=94) | -0.014 [-0.03, +0.00] (n=96) | -0.083 [-0.11, -0.05] (n=136) | -0.356 [-0.43, -0.28] (n=95) | +0.265 [+0.18, +0.35] (n=136) | -0.636 [-0.69, -0.58] (n=136) |
| mixed | inject_noise | -0.001 [-0.02, +0.02] (n=138) | +0.005 [-0.00, +0.01] (n=140) | -0.011 [-0.02, +0.00] (n=147) | -0.006 [-0.02, +0.01] (n=140) | +0.000 [+0.00, +0.00] (n=147) | +0.000 [+0.00, +0.00] (n=147) |
| reduced | control_rerun | +0.018 [-0.01, +0.05] (n=115) | -0.001 [-0.01, +0.01] (n=116) | +0.000 [+0.00, +0.00] (n=150) | -0.007 [-0.03, +0.02] (n=114) | -0.013 [-0.04, +0.01] (n=150) | +0.000 [+0.00, +0.00] (n=111) |
| reduced | drop_top_relevant | -0.030 [-0.08, +0.01] (n=64) | -0.021 [-0.04, -0.00] (n=67) | -0.091 [-0.12, -0.06] (n=102) | -0.314 [-0.41, -0.22] (n=65) | +0.294 [+0.20, +0.39] (n=102) | -0.636 [-0.70, -0.58] (n=102) |
| reduced | inject_noise | +0.027 [+0.00, +0.06] (n=113) | +0.003 [-0.01, +0.01] (n=114) | -0.007 [-0.02, +0.00] (n=148) | -0.000 [-0.02, +0.02] (n=113) | -0.007 [-0.03, +0.01] (n=148) | +0.000 [+0.00, +0.00] (n=109) |

Paired test of each perturbation against the control, per query (delta under perturbation − delta under control):

| KB | perturbation | metric | n_pairs | effect beyond control | ci95_low | ci95_high | wilcoxon_p | verdict |
|---|---|---|---|---|---|---|---|---|
| curated | drop_top_relevant | abstained_delta | 136 | 0.301 | 0.221 | 0.382 | 0.000 | difference |
| curated | drop_top_relevant | faithfulness_delta | 91 | -0.067 | -0.112 | -0.025 | 0.002 | difference |
| curated | drop_top_relevant | answer_relevance_delta | 93 | -0.010 | -0.024 | 0.004 | 0.193 | could be noise |
| curated | drop_top_relevant | context_relevance_delta | 136 | -0.098 | -0.124 | -0.071 | 0.000 | difference |
| curated | drop_top_relevant | reference_agreement_delta | 91 | -0.277 | -0.353 | -0.204 | 0.000 | difference |
| curated | inject_noise | abstained_delta | 147 | 0.014 | -0.014 | 0.041 | 0.317 | could be noise |
| curated | inject_noise | faithfulness_delta | 137 | -0.002 | -0.017 | 0.014 | 0.893 | could be noise |
| curated | inject_noise | answer_relevance_delta | 137 | 0.000 | -0.006 | 0.007 | 0.948 | could be noise |
| curated | inject_noise | context_relevance_delta | 147 | -0.001 | -0.012 | 0.010 | 0.704 | could be noise |
| curated | inject_noise | reference_agreement_delta | 137 | -0.002 | -0.022 | 0.018 | 0.828 | could be noise |
| mixed | drop_top_relevant | abstained_delta | 136 | 0.257 | 0.184 | 0.338 | 0.000 | difference |
| mixed | drop_top_relevant | faithfulness_delta | 94 | -0.058 | -0.110 | -0.009 | 0.019 | difference |
| mixed | drop_top_relevant | answer_relevance_delta | 96 | -0.012 | -0.028 | 0.002 | 0.136 | could be noise |
| mixed | drop_top_relevant | context_relevance_delta | 136 | -0.078 | -0.107 | -0.048 | 0.000 | difference |
| mixed | drop_top_relevant | reference_agreement_delta | 95 | -0.347 | -0.425 | -0.272 | 0.000 | difference |
| mixed | inject_noise | abstained_delta | 147 | -0.007 | -0.020 | 0.000 | 0.317 | could be noise |
| mixed | inject_noise | faithfulness_delta | 137 | 0.006 | -0.017 | 0.028 | 0.512 | could be noise |
| mixed | inject_noise | answer_relevance_delta | 139 | 0.005 | -0.002 | 0.012 | 0.505 | could be noise |
| mixed | inject_noise | context_relevance_delta | 147 | -0.007 | -0.019 | 0.005 | 0.103 | could be noise |
| mixed | inject_noise | reference_agreement_delta | 138 | 0.002 | -0.017 | 0.019 | 0.808 | could be noise |
| reduced | drop_top_relevant | abstained_delta | 102 | 0.314 | 0.216 | 0.412 | 0.000 | difference |
| reduced | drop_top_relevant | faithfulness_delta | 64 | -0.030 | -0.083 | 0.022 | 0.152 | could be noise |
| reduced | drop_top_relevant | answer_relevance_delta | 67 | -0.020 | -0.040 | -0.002 | 0.092 | could be noise |
| reduced | drop_top_relevant | context_relevance_delta | 102 | -0.091 | -0.120 | -0.060 | 0.000 | difference |
| reduced | drop_top_relevant | reference_agreement_delta | 65 | -0.303 | -0.397 | -0.211 | 0.000 | difference |
| reduced | inject_noise | abstained_delta | 148 | 0.007 | -0.014 | 0.034 | 0.564 | could be noise |
| reduced | inject_noise | faithfulness_delta | 112 | 0.013 | -0.015 | 0.045 | 0.530 | could be noise |
| reduced | inject_noise | answer_relevance_delta | 113 | 0.005 | -0.002 | 0.012 | 0.074 | could be noise |
| reduced | inject_noise | context_relevance_delta | 148 | -0.007 | -0.016 | 0.003 | 0.210 | could be noise |
| reduced | inject_noise | reference_agreement_delta | 112 | 0.005 | -0.015 | 0.026 | 0.626 | could be noise |

### 2.8 Run-to-run variance (same configuration, same queries, different generation seed)

5 repeats of 30 queries per configuration. Retrieval is identical across repeats (checked below); only generation and judging vary.
- PASS — retrieval metrics are identical in every repeat

| KB | method | metric | mean over repeats | SD of the config mean across repeats | min repeat mean | max repeat mean | median per-query SD | share of queries whose value changed |
|---|---|---|---|---|---|---|---|---|
| curated | dense | faithfulness | 0.974 | 0.014 | 0.958 | 0.989 | 0.000 | 0.200 |
| curated | dense | abstained | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| curated | dense | answer_relevance | 0.793 | 0.004 | 0.787 | 0.796 | 0.018 | 1.000 |
| curated | dense | context_relevance | 0.391 | 0.004 | 0.387 | 0.393 | 0.000 | 0.033 |
| curated | dense | reference_agreement | 0.926 | 0.025 | 0.897 | 0.961 | 0.000 | 0.233 |
| curated | hybrid | faithfulness | 0.954 | 0.009 | 0.941 | 0.964 | 0.000 | 0.233 |
| curated | hybrid | abstained | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| curated | hybrid | answer_relevance | 0.791 | 0.004 | 0.785 | 0.795 | 0.014 | 1.000 |
| curated | hybrid | context_relevance | 0.333 | 0.000 | 0.333 | 0.333 | 0.000 | 0.000 |
| curated | hybrid | reference_agreement | 0.914 | 0.014 | 0.900 | 0.932 | 0.000 | 0.300 |
| curated | lexical | faithfulness | 0.946 | 0.014 | 0.928 | 0.968 | 0.000 | 0.333 |
| curated | lexical | abstained | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| curated | lexical | answer_relevance | 0.785 | 0.005 | 0.780 | 0.790 | 0.016 | 0.967 |
| curated | lexical | context_relevance | 0.327 | 0.000 | 0.327 | 0.327 | 0.000 | 0.000 |
| curated | lexical | reference_agreement | 0.881 | 0.018 | 0.869 | 0.911 | 0.000 | 0.367 |

Reading this: a difference between two configurations in Part 2.5 that is smaller than about twice the `SD of the config mean across repeats` for that metric could be produced by re-running the same configuration.

# Thesis-facing settings that changed during the audit

Everything below changes what a number in the thesis means or how it was produced. "Before" is
the code at commit `271788e` (the state audited in `AUDIT.md`); "now" is the audited pipeline.
No full-sweep results existed before these changes, so no reported result is affected, but any
methodology text written against the old behaviour needs updating.

## Corpus and knowledge bases

| Setting | Before | Now | Where |
|---|---|---|---|
| Document ids | `subset:document_id:qid`; 34 MedQuAD rows collided | file name is used where it differs from the document id; all 16,407 ids unique | `data/ingest_medquad.py` |
| `mixed` noise source | any leftover document | only documents whose topic is absent from `curated`, with no chunk identical to a curated chunk, and no two with the same answer | `data/kb_variants.py::eligible_noise_docs` |
| `mixed` noise shortfall | silently used fewer documents | build fails | `data/kb_variants.py::build_mixed` |
| Topic matching (`reduced`, noise) | exact string | case- and whitespace-insensitive | `normalize_topic` |
| KB sizes | curated 1,500 docs / 3,715 chunks; mixed 1,950 / 4,776; reduced 976 / 2,451 | curated 1,500 / 3,696; mixed 1,950 / 4,595; reduced 987 / 2,425 | manifests in `datasets/processed/kb_*/` |
| KB config hash | `5b4983b42ed5` | `b1bc54c28398` (includes a build-rule version) | `config_hash` |

Unchanged: seed 42, base corpus 1,500, minimum answer length 15 words, noise ratio 0.30, topic
drop fraction 0.35, chunk size 120 words, overlap 20 words.

## Queries and relevance labels

| Setting | Before | Now |
|---|---|---|
| Number of queries | 40 | 150 (seed 42, stratified by question type; `configs/query_set.yaml`) |
| Queries unanswerable in `reduced` | 15 of 40 | 39 of 150 |
| Relevance label | one: chunk came from the query's own answer | two, both reported: **strict** (as before) and **lenient** (strict + chunks with identical text + chunks from a document with the same topic and question type) |
| Manual label check | none | 60 pairs read by hand (`relevance_label_sample.md`) |

## Retrieval

| Setting | Before | Now |
|---|---|---|
| BM25 results | always k, padded with zero-score chunks | only chunks sharing a term with the query (can be fewer than k) |
| Tie-breaking | unspecified | BM25 and dense: corpus order; RRF: chunk id |
| RRF rank | the item's `rank` field | position in the list (same thing today, robust to slicing) |
| RRF constant, fusion depth | 60 hardcoded; 50 | 60 and 50, both in the config and logged |
| Index reuse | whenever index files existed | only when chunks and embedding model match a stored fingerprint |

## Retrieval metrics

| Setting | Before | Now |
|---|---|---|
| Query with no relevant chunk in the KB | recall/MRR/NDCG = NaN and dropped from the mean; precision = 0 | reported two ways: `_all` (counted as 0, headline) and answerable-only (excluded); plus `coverage` |
| Precision for such a query | 0 in the only view | 0 in `_all`, excluded in the answerable-only view (so all four metrics share one query set per view) |
| MRR rank | item's `rank` field | list position |
| Duplicate chunk ids in a result | double-counted | rejected |
| Uncertainty | none | SD, 95% bootstrap interval, paired bootstrap + Wilcoxon (Holm-adjusted) per comparison |

## Generation

| Setting | Before | Now |
|---|---|---|
| Model | `gpt-4o-mini` (floating alias) | `gpt-4o-mini-2024-07-18` (dated snapshot); the model name the API returns is logged |
| Prompt template | `wellness_summary_v1` | unchanged, byte for byte; its SHA-256 is logged and pinned |
| Unusable output (empty, cut off, missing section) | scored | regenerated up to 3 times, then logged as an error |
| Seed | none | per-record seed sent to the API and logged |
| Parser | numbered key points and `##` headings only | also `-`/`*` bullets and other heading levels; empty sections parse as empty |

## Generation metrics (definitions version `v2`)

| Metric | Before | Now |
|---|---|---|
| Claims checked | Key Points only | every Summary sentence + every Key Point (fragments under 3 words joined to a neighbour) |
| Faithfulness | supported claims / all claims; "the context does not say" counted as supported | supported / (supported + unsupported); statements that only say the context lacks information are in neither count; undefined if no factual claim |
| Abstention | not measured (appeared as answer relevance = 0) | own metric: judge labels the answer `answered` / `partial` / `abstained` (seeing the answer only) |
| Answer relevance | on the Summary only; 0 for abstentions; judge temperature 0.7; 3 questions regardless of config | on Summary + Key Points; computed only for non-abstaining answers; judge temperature 0; number of questions from config (3) |
| Context relevance | fraction of passages judged relevant | unchanged in definition; strict parsing |
| Reference agreement | did not exist | **new, added by the audit**: faithfulness computed against the MedQuAD reference answer |
| Judge model | same as generator, implicit | `gpt-4o-mini-2024-07-18`, explicit in the config and logged |
| Judge reply format | free text, first JSON-looking span, `bool()` coercion | JSON mode, exact shape and types required, otherwise retried then an error |

## Baselines (new)

`oracle` (gold chunks), `random` (5 random chunks), `none` (no retrieval; separate closed-book
prompt `closed_book_v1`). The `none` baseline uses a different prompt from the nine experimental
configurations by necessity; it is a reference point, not a tenth configuration.

## Robustness test

| Setting | Before | Now |
|---|---|---|
| Inject noise | noise at rank 1, list cut to k (last chunk lost) | noise replaces the lowest-ranked chunk that is not relevant (strict or lenient), in place; context size unchanged |
| Noise chunk | any chunk with a different topic label | different topic, not relevant, not already retrieved, not a verbatim copy of a retrieved chunk |
| Noise chunk selection | one generator shared across the sweep | seeded per (sweep seed, KB variant, query) |
| Drop top relevant | context of k−1 | unchanged |
| Control | none | `control_rerun`: regenerate on the unchanged context |
| Skipped perturbations | silent | logged with the reason |
| Sweep seed | `robustness.seed: 123` | `seed: 123` for the whole sweep |

## Harness and analysis

Run fingerprint on every row; resumable robustness; error records; spend cap; headline/answerable
columns and `coverage` in every table; confidence intervals on every figure; `paired_comparisons.csv`.

# AUDIT.md — Phase 1 audit of the RAG evaluation pipeline

Date: 2026-10-11. Scope: every file under `data/`, `retrieval/`, `generation/`, `evaluation/`,
`experiments/`, `analysis/`, `scripts/`, `configs/`, `tests/`, plus the data and results on disk.
Nothing was changed. All numbers below were measured on the real files in `datasets/processed/`
and `results/`.

## 0. The five things to know first

1. **No thesis results exist yet.** The only run on disk is the demo (`results/demo_sweep.jsonl`:
   3 queries × 3 retrievers × `curated` only). The full 9-config sweep has never been run, and
   only the `curated` index has been built. The CSVs and figures in `reports/` are that demo
   output, sitting where the full-sweep output is documented to go. So nothing needs to be
   rerun, and every fix below can land before any real result is generated.
2. **The `reduced` variant hides its own effect.** 15 of the 40 queries lose all their relevant
   chunks in `reduced`. Recall, MRR and NDCG are set to NaN for those and dropped from the mean,
   so `reduced` is scored only on the 25 queries it can still answer. It will look about as good
   as `curated` on those three metrics, which is the opposite of what the variant is for.
3. **The relevance labels are weaker than they look.** A chunk counts as relevant only if it
   was cut from the query's own MedQuAD answer. Other chunks that answer the same question are
   labelled wrong. 368 of 3,715 curated chunks (10%) have text identical to another chunk, and
   10 of 40 queries have other curated documents on the same topic. Also, 15 of 40 queries have
   exactly one relevant chunk, so the best possible mean Precision@5 is 0.455, not 1.0.
4. **The "noise" in `mixed` is not reliably noise.** Of the 450 injected documents, 140 share a
   topic with a curated document, 10 share a topic with one of the 40 queries, and 7 have answer
   text identical to a curated answer. The code never checks topic; the README says it does.
5. **Faithfulness may not be measuring anything.** It is 1.0 on all 15 demo records, including
   after the relevant chunk was removed. A key point that says "the context does not provide
   this information" is scored as a faithful claim. Nothing in the test suite checks that any
   of the three generation metrics can tell a good answer from a bad one.

## 1. Requirement-by-requirement

"Tested" means a test that could fail if the behaviour were wrong, not merely a test that runs
the code.

| Spec item | Implemented? | Where | Really tested? |
|---|---|---|---|
| Pure Python, laptop-only | Yes | whole repo | n/a. `requirements.txt` has no version pins; `tests/test_client.py` imports `httpx2`, which is not listed |
| MedQuAD corpus | Yes, 9 of 12 subsets, 16,407 QA pairs; no other data source used | `data/ingest_medquad.py` | Synthetic XML only. The real corpus has 8 doc IDs that repeat (34 surplus rows); nothing checks this |
| BM25 retriever | Yes (`rank_bm25`) | `retrieval/lexical.py` | Weak: 3-document toy corpus, "right doc is first" |
| Dense retriever, `text-embedding-3-small` + FAISS | Yes, exact `IndexFlatIP` on L2-normalised vectors | `retrieval/dense.py` | Weak: toy bag-of-words embedder; never compared with brute force; normalisation never tested |
| Hybrid RRF | Yes | `retrieval/hybrid.py` | Partial: one symmetric hand case; the constant 60, ties, and one-list-only scores are not checked numerically |
| One retriever interface | Yes (`Retriever` protocol) | `retrieval/base.py` | No contract test across the three |
| Embeddings cached | Yes (SQLite, key = model + text) | `retrieval/embedding_cache.py` | Hit/miss tested; corruption and model change through the full path are not |
| `curated` variant | Yes: 1,500 docs / 3,715 chunks | `data/kb_variants.py:91` | Toy data only |
| `mixed` variant, configurable ratio | Ratio is exact (450 = 0.30 × 1,500). **Noise is not guaranteed off-topic** | `data/kb_variants.py:105` | Toy data where every doc has its own topic, so overlap cannot occur |
| `reduced` variant | Yes: 433 topics dropped, 976 docs left | `data/kb_variants.py:117` | Toy data only. Effect on the query set is untested |
| Deterministic builds | Seeded; looks deterministic | `data/kb_variants.py` | Only "two calls in one process agree". No hash of the real output |
| One fixed model and prompt | **Partial.** `gpt-4o-mini` is a floating alias, not a dated snapshot. `prompt_template_id` in the config is never read; the template is hardcoded | `generation/prompts.py:16`, `generator.py:51` | No snapshot test; a prompt edit would go unnoticed |
| 9-config sweep, fixed top-k | Code yes. **Never run.** | `experiments/runner.py:208` | Wiring only, with fakes |
| Fully logged | **Partial.** Missing: seed, prompt hash, temperature, max tokens, embedding model, judge model, RRF constant, hybrid depth, returned model version, token counts, code version | `experiments/runner.py:106` | No test of log completeness |
| Resumable | **Partial.** Standard rows resume. Robustness rows do not (see R7) | `experiments/runner.py`, `results_store.py` | Only "rerun after a complete run makes no calls". No mid-run crash |
| Config-driven | Mostly. Two dead settings: `answer_relevance_num_questions` and `prompt_template_id` | `experiments/config.py` | Loading tested; dead settings not noticed |
| P@k, R@k, MRR, NDCG | Yes, binary relevance. MRR is cut off at k | `evaluation/retrieval_metrics.py` | Weak: NDCG is never compared with a non-trivial number |
| Faithfulness, answer relevance, context relevance | Self-built LLM judge, not Ragas | `evaluation/generation_metrics.py` | Judge is fully mocked. Only arithmetic and JSON parsing are tested |
| Robustness: drop top relevant, inject noise | Yes, dense retriever only | `evaluation/robustness.py`, `runner.py:125` | Mechanics tested by hand. Two side effects untested (R8) |
| Tables, plots, canonical CSV | Yes; means only, no spread or intervals | `analysis/` | Aggregation arithmetic is hand-checked. Plots: "file exists" |
| Oracle / random / no-retrieval baselines | **Missing** | — | — |
| Negative controls for generation metrics | **Missing** | — | — |
| Run-to-run variance | **Missing** | — | — |
| Real-API smoke test with cost cap and dry run | **Missing** (`python -m generation.client` is a manual ping) | — | — |
| Mutation testing, coverage, `verify` command | **Missing** | — | — |

## 2. Correctness risks

Ordered by how much they could distort thesis conclusions.

### High

**R1. Survivorship bias in `reduced`.** `evaluation/retrieval_metrics.py:32,42,54` return NaN when
a query has no relevant chunk, and `analysis/aggregate.py:123` skips NaN. With 15 of 40 queries
affected, `reduced` recall/MRR/NDCG are averaged over a different, easier set of 25 queries than
`curated`. Precision counts those queries as 0, so the four metrics are not even on the same
query set as each other. The NaN choice is defensible on its own; comparing the resulting means
across variants is not.

**R2. Relevance labels are document-of-origin, not actual relevance.**
`evaluation/relevance_judgments.py:31`. Consequences measured on the real data:
- 368 curated chunks duplicate another chunk's text; 49 curated documents share their whole
  answer with another. A retriever that returns the identical twin is marked wrong.
- 10 of 40 queries have 1–5 other curated documents on the same topic; 2 have another document
  with the same topic and the same question type (4 in `mixed`).
- All of a document's chunks are relevant regardless of content. One query has 18 relevant
  chunks, so its best possible Recall@5 is 0.28.
- Precision@5 cannot exceed 0.455 on average.
- All 40 queries name their topic verbatim and are questions taken from the knowledge base
  itself. This is "find the known document", which favours BM25 and is easier than real use.

**R3. `mixed` noise overlaps the curated topics.** `data/kb_variants.py:105-114` samples noise
from all leftover documents with no topic filter. Numbers in section 0. Some of the "noise"
gives the model extra correct material, so `mixed` is partly a larger KB, not a noisier one.

**R4. Faithfulness is saturated and scored by the model that wrote the answer.**
`evaluation/generation_metrics.py:91`. Only the numbered Key Points are checked; the Summary
paragraph is never checked. Abstention sentences count as supported claims. The judge is the
same `gpt-4o-mini` that generated the text. Demo: 1.0 on 15 of 15.

**R5. Answer relevance mostly records whether the model abstained.**
`generation_metrics.py:176-190` returns exactly 0.0 when the judge decides the answer is
non-committal. In the demo, the one large robustness effect (−0.83) is this switch flipping.
A correct "the context does not say" is scored the same as an irrelevant answer. The judge call
also runs at temperature 0.7 (line 183), so the score changes from run to run.

**R6. Bad output is silently recorded as a valid score.**
- `generation/client.py:89`: an empty or filtered completion becomes `""` and is logged as a
  normal result. Answer relevance is then 0.0.
- `generation/parsing.py:13-17`: if the model formats headings or bullets slightly differently,
  Summary is empty (answer relevance 0.0) or Key Points is empty (faithfulness NaN, dropped
  from the mean). Nothing flags a parse failure. Truncation at `max_tokens` is not detected.
- `generation_metrics.py:68`: `bool(v)` turns the string `"false"` into `True`. If the judge
  ever returns quoted booleans, every claim is counted as supported.

**R7. Robustness rows are not resumable and can vanish.** `experiments/runner.py:264-298`. The
standard row is saved and marked done before perturbations run. If the process dies in between,
or a perturbation raises (line 290 prints and continues, with no error record), the resume
skips that query and its robustness rows are never produced. Nothing reports the gap.

**R8. The perturbations do more than they claim.**
- Inject: the noise chunk goes in at rank 1 and the list is cut back to k (`runner.py:154`), so
  the last original chunk is pushed out. Seen in the demo. That is "add noise and remove a
  chunk".
- Drop: leaves k−1 chunks. Fine, but baseline and perturbed contexts differ in size.
- The noise chunk only has to have a different topic label (`robustness.py:60`), and is given a
  made-up score copied from the top result.
- Baseline and perturbed answers are each generated once at temperature 0.2. With no repeat of
  the baseline, a delta cannot be separated from generation noise.
- One random generator is shared across the whole sweep (`runner.py:215`), so which noise chunk
  a query gets depends on run order and on whether the run was interrupted.

**R9. One results file can mix runs with different settings.** The resume key is
`variant:method:query_id` (`runner.py:48`). It does not include top-k, model, prompt, or KB
hash. Change a setting, rerun to the same file, and old rows are kept and silently combined
with new ones. The analysis also never checks for duplicate keys or for the expected row count.

### Medium

**R10. Stale index reuse.** `retrieval/index_store.py:80` loads an index if four files exist. The
directory hash covers KB and chunking settings only, not the embedding model, the corpus
content, or the chunk list. Change the embedding model or re-ingest MedQuAD and the old index
is used without warning. Retrieval reads chunks from `artifacts/`; relevance labels read them
from `datasets/processed/`; nothing checks they match.

**R11. BM25 padding leaks into fusion.** `retrieval/lexical.py:37` always returns k results, even
with a score of 0, in corpus order. The hybrid asks for 50, and RRF (`hybrid.py:29`) gives
credit by rank alone, so zero-score filler can earn fusion score. Probably rare for full
questions; I will measure it.

**R12. Ties are broken by accident, not by rule.** BM25 ties fall back to corpus order. RRF ties
fall back to "BM25's list first" (`hybrid.py:32`). Deterministic, but undocumented and slightly
biased. With 10% duplicate chunk text, dense ties are real.

**R13. MRR and NDCG read rank differently.** `reciprocal_rank` trusts the `rank` field
(`retrieval_metrics.py:46`); NDCG and precision use list position. They agree today. A future
slice or filter that forgets to renumber would make them disagree silently. Duplicate chunk IDs
in a result list are not rejected and would push recall above 1.

**R14. Run settings are not in the log.** See the "Fully logged" row. As it stands the figures
can be redrawn from the log, but you could not prove from the log alone which model version,
prompt or temperature produced them.

**R15. Duplicate document IDs in the raw corpus.** 16,407 rows, 16,373 unique IDs. None reached
`curated` under seed 42, but another seed or size could produce duplicate chunk IDs.

**R16. Torn last line.** `results_store.py:37`. If a crash leaves a partial line with no newline,
the next append is glued onto it; both are unreadable, the new result is lost and paid for
again on the following resume.

### Low

- R17. Embedding responses are not checked for count, order or dimension
  (`client.py:60`, `embedding_cache.py:79`). A short response fails loudly later; a reordered
  one would not.
- R18. A failed embedding batch discards earlier batches from the same call (cost only).
- R19. 4 curated documents have topic "Unknown" and are treated as one topic.
- R20. `.env.example` is deleted but the README tells you to copy it. Branch is `master`; the
  configured main branch is `main`.
- R21. Statistical power. 40 queries, 25 usable in `reduced`, and no measure of spread anywhere
  in the analysis. Differences between configs will often be within noise.

### Checked and found sound

Chunking boundaries (window, overlap, final chunk) by reading; cosine via normalise-then-inner-
product; the retry wrapper returns one result per call, so no double counting; variant set
relations on the real data (`reduced` ⊂ `curated` ⊂ `mixed`, ratio exactly 0.30); chunk IDs
unique in all three variants; no query text inside chunk text; seeded sampling with sorted
inputs. These still get real tests in Phase 2.

## 3. Existing tests: 152 tests, all pass in 16 s

No test touches real MedQuAD data, real embeddings or the real API. No expected value comes
from an independent library. There are no property-based tests.

| File | What it proves | Verdict |
|---|---|---|
| `test_retrieval_metrics.py` | P, R, MRR on one 5-item list with hand values; NDCG only as "= 1" and "worse < perfect" | Real but thin. A wrong log base or discount in NDCG would pass. No ties, duplicates, short lists, or k > list |
| `test_hybrid.py` | RRF score on one symmetric pair; truncation length | Thin. Order of results is barely checked; a wrong constant passes 4 of 6 tests |
| `test_dense.py` | Toy keyword-count embedder, 3 chunks | Weak. Would pass with normalisation removed. No brute-force comparison |
| `test_lexical.py` | Tokeniser; obvious doc ranks first of 3 | Sanity only |
| `test_embedding_cache.py` | Real SQLite round trip, hit/miss, reopen, large batch | Good. Missing corruption and model-change cases |
| `test_chunking.py` | One hand-worked overlap example; empty input; config validation | Good but small. No reconstruction, unicode, or real documents |
| `test_kb_variants.py` | Counts and set relations on synthetic docs | Mechanics only. Cannot detect R3 because every toy doc has a unique topic |
| `test_query_set.py` | Size, stratification, determinism | Fine |
| `test_relevance_judgments.py` | Chunks group by document ID | Mechanics only. Says nothing about whether the labels are right |
| `test_robustness.py` | Drop and inject on a 3-item list, by hand | Good for the functions. The truncation in the runner (R8) is outside it |
| `test_generation_metrics.py` | Judge replaced by canned JSON | Proves the division and JSON parsing. Proves nothing about the metrics. Misses the `"false"` bug |
| `test_generator.py`, `test_parsing.py` | Deterministic assembly and parsing | Fine. Parsing lacks off-format model output |
| `test_prompts.py` | Substrings appear in the prompt | Weak. Not a snapshot |
| `test_client.py` | Batching; retry on rate limit with the SDK endpoint faked | Legitimate mocking of an external service. Only one error type; no empty or malformed response |
| `test_runner.py` | Record fields with stub retriever and scripted client | Wiring only |
| `test_run_sweep_integration.py` | Real FAISS/BM25 build on 3 chunks; rerun makes no calls | Weaker than it looks: the fake embedder returns the same vector for every text, so dense retrieval is all ties; the fake judge always says "true"; no crash mid-run |
| `test_results_store.py` | Append, reload, skip a bad middle line | Fine. Not the torn last line |
| `test_config.py` | YAML loads; bad values rejected | Fine. Does not notice dead settings |
| `test_index_store.py` | Save/load gives the same retrieval | Fine. No staleness case |
| `test_aggregate.py`, `test_export_csv.py` | Hand-checked means and NaN counts; files written | Good for arithmetic |
| `test_plots.py` | PNG exists | Decorative |
| `test_ingest_medquad.py` | Parses synthetic XML | Fine. Not the real corpus |

The README's "validated against a realistic synthetic 108-row sweep … came out in the expected
order" is not evidence about the pipeline: the numbers were invented to have that order.

## 4. Decisions I need from you before Phase 2

Recommendation first in each case.

1. **`reduced` scoring (R1).** Report every retrieval metric two ways: over all 40 queries with
   unanswerable ones counted as 0, and over answerable queries only, plus the coverage rate.
   The first is the headline. Alternative: keep NaN-only and state the caveat.
2. **`mixed` noise (R3).** Draw noise only from topics absent from `curated`, and exclude exact
   duplicate answers. This changes the `mixed` KB; no results are affected. Alternative: keep
   it and describe `mixed` as "unfiltered extra documents".
3. **Inject perturbation (R8).** Replace the lowest-ranked non-relevant chunk with the noise
   chunk, so context size stays k and nothing relevant is lost. Alternative: allow k+1.
4. **Relevance labels (R2).** Keep document-of-origin as the primary label, and add a second,
   looser label (identical text, or same topic and question type) so every retrieval metric has
   a strict and a lenient value. Manual labelling of a sample is the stronger option if you
   have the time.
5. **Query count (R21).** Raise from 40 to about 150–200. The cost is small (item 7).
6. **Model pinning.** Pin a dated `gpt-4o-mini` snapshot and log the returned model name. Keep
   the same model as judge unless you want to pay for a stronger judge; I will measure whether
   the current judge discriminates before recommending either way.
7. **Spend and installs.** My rough estimate for one full sweep at 40 queries is under US$1,
   and a few dollars at 200 queries; I will compute it from real token counts and show you
   before any paid run. I propose a hard cap of US$5 for all of Phases 2–3 unless you set
   another. I also need to install test-only packages: `hypothesis`, `ranx` or `scikit-learn`,
   `pytest-cov`, and a mutation tool (`mutmut` does not run natively on Windows, so I expect
   to use a small manual mutation harness instead).

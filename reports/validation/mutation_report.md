# Mutation testing report

Mutants injected: 90. Killed: 90. Survived: 0. **Mutation score: 100%**.

Each mutant is one deliberate bug injected into a copy of the code. "Killed" means at least one
test failed with the bug in place. "Tier" says whether the fast unit tests caught it or it took
the slower real-data / harness tests. Produced by `python -m scripts.mutation_test`.

| ID | File | Injected bug | Result | Tier | Caught by |
|---|---|---|---|---|---|
| M01 | `evaluation/retrieval_metrics.py` | precision divides by k+1 (off-by-one in k) | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_two_relevant_at_ranks_2_and_4` |
| M02 | `evaluation/retrieval_metrics.py` | precision looks at the top k+1 results | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_cutoff_excludes_hits_below_k` |
| M03 | `evaluation/retrieval_metrics.py` | precision divides by the number returned instead of k | killed | fast | `tests/test_retrieval_metrics.py::test_fewer_than_k_results_still_divides_precision_by_k` |
| M04 | `evaluation/retrieval_metrics.py` | recall divides by k instead of the number of relevant chunks | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_two_relevant_at_ranks_2_and_4` |
| M05 | `evaluation/retrieval_metrics.py` | recall ignores the cutoff k | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_cutoff_excludes_hits_below_k` |
| M06 | `evaluation/retrieval_metrics.py` | recall of a query with no relevant chunks is 0 instead of undefined | killed | fast | `tests/test_retrieval_metrics.py::test_query_with_no_relevant_chunks_is_undefined_not_zero_or_one` |
| M07 | `evaluation/retrieval_metrics.py` | MRR is 1/(rank+1) (off-by-one in rank) | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_two_relevant_at_ranks_2_and_4` |
| M08 | `evaluation/retrieval_metrics.py` | MRR ignores the cutoff k | killed | fast | `tests/test_retrieval_metrics.py::test_first_relevant_just_past_k_scores_zero_not_one_over_rank` |
| M09 | `evaluation/retrieval_metrics.py` | MRR uses the item's stale rank field instead of list position | killed | fast | `tests/test_retrieval_metrics.py::test_rank_means_list_position_not_the_rank_field` |
| M10 | `evaluation/retrieval_metrics.py` | NDCG discount is 1/log2(rank+2) in DCG only | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_two_relevant_at_ranks_2_and_4` |
| M11 | `evaluation/retrieval_metrics.py` | NDCG has no rank discount (plain hit count ratio) | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_two_relevant_at_ranks_2_and_4` |
| M12 | `evaluation/retrieval_metrics.py` | NDCG ideal ranking ignores the cutoff k | killed | fast | `tests/test_retrieval_metrics.py::test_more_relevant_chunks_than_k_caps_the_ideal_at_k` |
| M13 | `evaluation/retrieval_metrics.py` | NDCG ideal ranking assumes k relevant chunks always exist | killed | fast | `tests/test_retrieval_metrics.py::test_worked_example_two_relevant_at_ranks_2_and_4` |
| M14 | `evaluation/retrieval_metrics.py` | duplicate chunk ids are no longer rejected | killed | fast | `tests/test_retrieval_metrics.py::test_duplicate_chunk_ids_are_rejected_not_double_counted[precision_at_k]` |
| M15 | `retrieval/hybrid.py` | wrong RRF constant (61 instead of 60) | killed | fast | `tests/test_hybrid.py::test_default_constant_is_60` |
| M16 | `retrieval/hybrid.py` | RRF rank off by one | killed | fast | `tests/test_hybrid.py::test_fused_scores_and_order_match_hand_calculation_k60` |
| M17 | `retrieval/hybrid.py` | fusion adds raw retriever scores instead of reciprocal ranks (no normalisation) | killed | fast | `tests/test_hybrid.py::test_fused_scores_and_order_match_hand_calculation_k60` |
| M18 | `retrieval/hybrid.py` | fused list sorted ascending (worst first) | killed | fast | `tests/test_hybrid.py::test_fused_scores_and_order_match_hand_calculation_k60` |
| M19 | `retrieval/hybrid.py` | fusion tie-break removed (order depends on which retriever is listed first) | killed | fast | `tests/test_hybrid.py::test_the_k_constant_changes_scores_and_can_change_the_order` |
| M20 | `retrieval/hybrid.py` | hybrid returns k+1 results | killed | fast | `tests/test_hybrid.py::test_hybrid_retriever_returns_the_top_k_of_the_fused_ranking` |
| M21 | `retrieval/hybrid.py` | hybrid fuses only the top k of each retriever, not k_per_retriever | killed | fast | `tests/test_hybrid.py::test_hybrid_retriever_passes_its_rrf_constant_through` |
| M22 | `retrieval/hybrid.py` | configured RRF constant is ignored | killed | fast | `tests/test_hybrid.py::test_hybrid_retriever_passes_its_rrf_constant_through` |
| M23 | `retrieval/hybrid.py` | RRF uses the item's rank field instead of its position in the list | killed | fast | `tests/test_hybrid.py::test_fusion_uses_list_position_and_ignores_raw_scores_and_rank_fields` |
| M24 | `retrieval/dense.py` | chunk vectors are not normalised (inner product instead of cosine) | killed | fast | `tests/test_dense.py::test_ranking_is_by_cosine_not_raw_inner_product` |
| M25 | `retrieval/dense.py` | query vector is not normalised | killed | fast | `tests/test_dense.py::test_ranking_is_by_cosine_not_raw_inner_product` |
| M26 | `retrieval/dense.py` | dense returns k-1 results | killed | fast | `tests/test_dense.py::test_ranking_is_by_cosine_not_raw_inner_product` |
| M27 | `retrieval/dense.py` | dense ranking reversed among candidates | killed | fast | `tests/test_dense.py::test_ranking_is_by_cosine_not_raw_inner_product` |
| M28 | `retrieval/dense.py` | dense tie rule removed (ties left in FAISS order) | killed | fast | `tests/test_dense.py::test_identical_vectors_tie_and_are_returned_in_corpus_order` |
| M29 | `retrieval/dense.py` | dense ranks start at 0 | killed | fast | `tests/test_dense.py::test_identical_vectors_tie_and_are_returned_in_corpus_order` |
| M30 | `retrieval/lexical.py` | BM25 ranking reversed | killed | fast | `tests/test_lexical.py::test_term_frequency_ranking_matches_hand_computed_scores` |
| M31 | `retrieval/lexical.py` | BM25 returns chunks that share no term with the query (zero-score padding) | killed | fast | `tests/test_lexical.py::test_term_frequency_ranking_matches_hand_computed_scores` |
| M32 | `retrieval/lexical.py` | BM25 returns k+1 results | killed | fast | `tests/test_lexical.py::test_k_truncates_the_ranked_list` |
| M33 | `retrieval/lexical.py` | BM25 tokeniser keeps case | killed | fast | `tests/test_lexical.py::test_tokenize_lowercases_and_strips_punctuation` |
| M34 | `data/chunking.py` | chunks do not overlap | killed | fast | `tests/test_chunking.py::test_long_answer_splits_with_overlap` |
| M35 | `data/chunking.py` | overlap is one word short | killed | fast | `tests/test_chunking.py::test_long_answer_splits_with_overlap` |
| M36 | `data/chunking.py` | extra trailing chunk when the document ends exactly at a window edge | killed | fast | `tests/test_chunking.py::test_chunk_count_at_window_boundaries[10-1]` |
| M37 | `data/chunking.py` | chunk ids start at 1 | killed | fast | `tests/test_chunking.py::test_short_answer_produces_single_chunk` |
| M38 | `data/chunking.py` | window is one word short | killed | fast | `tests/test_chunking.py::test_long_answer_splits_with_overlap` |
| M39 | `data/kb_variants.py` | wrong noise ratio in mixed (90% of configured) | killed | fast | `tests/test_kb_variants.py::test_build_mixed_adds_exactly_the_configured_ratio_of_off_topic_documents` |
| M40 | `data/kb_variants.py` | noise may share a topic with curated | killed | fast | `tests/test_kb_variants.py::test_noise_excludes_documents_on_a_curated_topic_even_with_different_spelling_case` |
| M41 | `data/kb_variants.py` | noise may contain a verbatim copy of a curated chunk | killed | fast | `tests/test_kb_variants.py::test_noise_excludes_duplicate_answers` |
| M42 | `data/kb_variants.py` | reduced drops half the configured share of topics | killed | fast | `tests/test_kb_variants.py::test_build_reduced_drops_whole_topics_and_every_document_on_them` |
| M43 | `data/kb_variants.py` | base corpus sampling is unseeded | killed | fast | `tests/test_kb_variants.py::test_sampling_is_deterministic_and_independent_of_input_order` |
| M44 | `data/kb_variants.py` | pool is not sorted before shuffling (result depends on input order) | killed | fast | `tests/test_kb_variants.py::test_sampling_is_deterministic_and_independent_of_input_order` |
| M45 | `data/kb_variants.py` | reduced keeps some documents of dropped topics | killed | fast | `tests/test_kb_variants.py::test_build_reduced_drops_whole_topics_and_every_document_on_them` |
| M46 | `data/kb_variants.py` | quality filter uses > instead of >= | killed | fast | `tests/test_kb_variants.py::test_answer_exactly_at_the_minimum_length_is_kept` |
| M47 | `data/kb_variants.py` | noise is appended unshuffled (always the first eligible documents) | killed | real-data/harness | `tests/test_data_real.py::test_rebuilding_from_the_corpus_reproduces_the_files_on_disk_exactly` |
| M48 | `data/kb_variants.py` | topic comparison is case-sensitive | killed | fast | `tests/test_kb_variants.py::test_noise_excludes_documents_on_a_curated_topic_even_with_different_spelling_case` |
| M49 | `evaluation/relevance_judgments.py` | lenient labels ignore verbatim copies of the answer | killed | fast | `tests/test_relevance_judgments.py::test_lenient_adds_chunks_with_text_identical_to_the_query_own_answer` |
| M50 | `evaluation/relevance_judgments.py` | lenient labels treat any chunk on the same topic as relevant | killed | fast | `tests/test_relevance_judgments.py::test_lenient_adds_chunks_on_the_same_topic_and_question_type_only` |
| M51 | `evaluation/relevance_judgments.py` | strict labels use the query's topic instead of its own document | killed | fast | `tests/test_relevance_judgments.py::test_strict_covers_every_query_even_unknown_ones` |
| M52 | `evaluation/robustness.py` | drop removes the relevant chunk and the one after it | killed | fast | `tests/test_robustness.py::test_drop_removes_exactly_the_highest_ranked_relevant_chunk` |
| M53 | `evaluation/robustness.py` | noise replaces the highest-ranked non-relevant chunk instead of the lowest | killed | fast | `tests/test_robustness.py::test_noise_replaces_the_lowest_ranked_non_relevant_chunk_in_place` |
| M54 | `evaluation/robustness.py` | noise chunk may share the query's topic | killed | fast | `tests/test_robustness.py::test_noise_chunk_never_shares_the_query_topic` |
| M55 | `evaluation/robustness.py` | noise is inserted instead of replacing (context grows by one) | killed | fast | `tests/test_robustness.py::test_noise_replaces_the_lowest_ranked_non_relevant_chunk_in_place` |
| M56 | `experiments/runner.py` | noise chunk choice is not keyed by query (same chunk for all) | killed | fast | `tests/test_runner.py::test_inject_perturbation_picks_the_same_noise_chunk_every_time_for_a_query` |
| M57 | `experiments/runner.py` | inject does not protect lenient-relevant chunks | killed | fast | `tests/test_runner.py::test_inject_perturbation_protects_lenient_relevant_chunks_and_avoids_retrieved_ones` |
| M58 | `evaluation/generation_metrics.py` | 'no information' statements count against faithfulness | killed | fast | `tests/test_generation_metrics.py::test_faithfulness_is_supported_over_supported_plus_unsupported[judge_labels3-0.3333333333333333]` |
| M59 | `evaluation/generation_metrics.py` | 'no information' statements count as supported (the pre-audit behaviour) | killed | fast | `tests/test_generation_metrics.py::test_faithfulness_is_supported_over_supported_plus_unsupported[judge_labels3-0.3333333333333333]` |
| M60 | `evaluation/generation_metrics.py` | quoted booleans from the judge are accepted | killed | fast | `tests/test_generation_metrics.py::test_quoted_booleans_are_rejected_not_read_as_true` |
| M61 | `evaluation/generation_metrics.py` | abstained answers are still scored for answer relevance | killed | fast | `tests/test_generation_metrics.py::test_abstained_answer_gets_no_answer_relevance_call_and_no_zero` |
| M62 | `evaluation/generation_metrics.py` | judge runs at temperature 0.7 | killed | fast | `tests/test_generation_metrics.py::test_judge_is_called_at_temperature_zero_in_json_mode_with_every_claim_and_passage` |
| M63 | `evaluation/generation_metrics.py` | summary sentences are not checked as claims | killed | fast | `tests/test_generation_metrics.py::test_claims_are_the_summary_sentences_then_the_key_points_without_citations_or_disclaimer` |
| M64 | `evaluation/generation_metrics.py` | wrong-length judge reply is accepted | killed | fast | `tests/test_generation_metrics.py::test_malformed_claim_labels_raise_after_retrying[{"labels": ["supported"]}]` |
| M65 | `generation/generator.py` | an answer cut off at max_tokens is accepted | killed | fast | `tests/test_generator.py::test_unusable_output_is_an_error_never_a_record[bad_reply0-finish_reason='length']` |
| M66 | `generation/generator.py` | an answer with no key points is accepted | killed | fast | `tests/test_generator.py::test_unusable_output_is_an_error_never_a_record[## Summary\nA summary.\n\n## Disclaimer\nd\n-no key points]` |
| M67 | `analysis/aggregate.py` | unanswerable queries are dropped from the headline metric (survivorship bias) | killed | fast | `tests/test_aggregate.py::test_unanswerable_queries_count_as_zero_in_the_headline_and_are_excluded_from_the_answerable_view` |
| M68 | `analysis/aggregate.py` | unanswerable queries count as 0 in the answerable-only view | killed | fast | `tests/test_aggregate.py::test_unanswerable_queries_count_as_zero_in_the_headline_and_are_excluded_from_the_answerable_view` |
| M69 | `analysis/aggregate.py` | abstention rate counts partial answers too | killed | fast | `tests/test_aggregate.py::test_abstention_is_a_rate_and_undefined_scores_are_left_out_of_means` |
| M70 | `analysis/aggregate.py` | duplicate rows are not detected | killed | fast | `tests/test_aggregate.py::test_a_log_with_duplicate_or_unreadable_rows_is_refused` |
| M71 | `analysis/aggregate.py` | robustness delta is baseline minus perturbed (sign flipped) | killed | fast | `tests/test_aggregate.py::test_robustness_deltas_are_perturbed_minus_baseline_and_abstention_is_its_own_delta` |
| M72 | `analysis/stats.py` | paired difference sign flipped | killed | fast | `tests/test_stats.py::test_paired_comparison_hand_example` |
| M73 | `analysis/stats.py` | Holm correction replaced by plain Bonferroni | killed | fast | `tests/test_stats.py::test_holm_adjustment_hand_example` |
| M74 | `analysis/stats.py` | bootstrap interval is 80% instead of 95% | killed | fast | `tests/test_stats.py::test_bootstrap_ci_agrees_with_scipy_percentile_bootstrap[0]` |
| M75 | `analysis/stats.py` | pairs are not aligned: NaN removed from each side separately | killed | fast | `tests/test_stats.py::test_pairs_with_a_nan_on_either_side_are_dropped_together` |
| M76 | `experiments/runner.py` | resume ignores completed rows (everything is re-run and duplicated) | killed | fast | `tests/test_run_sweep_integration.py::test_crash_and_resume_gives_the_same_rows_with_nothing_duplicated_missing_or_rebilled[1]` |
| M77 | `experiments/runner.py` | a failed row is also marked completed (never retried) | killed | fast | `tests/test_run_sweep_integration.py::test_an_api_failure_becomes_an_error_record_and_the_row_is_retried_next_run[failure0-TimeoutError]` |
| M78 | `experiments/runner.py` | rows from a different setup are accepted into the same results file | killed | fast | `tests/test_run_sweep_integration.py::test_rerunning_with_a_changed_setting_into_the_same_results_file_is_refused[top_k]` |
| M79 | `experiments/runner.py` | per-record seed ignores the query (all records share a seed) | killed | fast | `tests/test_runner.py::test_record_seed_depends_only_on_sweep_seed_repeat_and_key` |
| M80 | `experiments/runner.py` | random baseline is seeded by repeat as well (retrieval moves between repeat runs) | killed | fast | `tests/test_runner.py::test_keyed_rng_is_reproducible_and_ignores_repeat` |
| M81 | `experiments/runner.py` | no-retrieval baseline uses the normal RAG prompt | killed | fast | `tests/test_run_sweep_integration.py::test_model_prompt_and_temperature_are_identical_across_all_nine_configs` |
| M82 | `experiments/runner.py` | random baseline draws k+1 chunks | killed | fast | `tests/test_run_sweep_integration.py::test_oracle_rows_contain_only_gold_chunks_and_random_rows_are_reproducible` |
| M83 | `experiments/results_store.py` | a record appended after a torn line is glued onto it | killed | fast | `tests/test_results_store.py::test_append_after_a_torn_final_line_does_not_glue_the_new_record_onto_it` |
| M84 | `experiments/results_store.py` | torn-tail repair deletes the last complete record too | killed | fast | `tests/test_results_store.py::test_discard_torn_tail_removes_only_the_fragment` |
| M85 | `retrieval/index_store.py` | stale index is reused when the chunks changed | killed | fast | `tests/test_index_store.py::test_same_ids_but_edited_text_is_not_reused` |
| M86 | `retrieval/index_store.py` | index built with another embedding model is reused | killed | fast | `tests/test_index_store.py::test_index_built_with_a_different_embedding_model_is_not_reused` |
| M87 | `retrieval/embedding_cache.py` | cache key ignores the model name | killed | fast | `tests/test_embedding_cache.py::test_changed_model_name_is_a_miss` |
| M88 | `retrieval/embedding_cache.py` | truncated cached vector is not detected | killed | fast | `tests/test_embedding_cache.py::test_truncated_vector_in_cache_raises` |
| M89 | `retrieval/embedding_cache.py` | mixed-dimension vectors are not detected | killed | fast | `tests/test_embedding_cache.py::test_shortened_vector_with_whole_floats_is_caught_as_a_dimension_mismatch` |
| M90 | `generation/parsing.py` | an empty section swallows the next heading (pre-audit parser) | killed | fast | `tests/test_parsing.py::test_an_empty_section_is_empty_and_does_not_swallow_the_next_heading` |

"""Manual mutation testing: inject one deliberate bug at a time and check that
the test suite notices.

    python -m scripts.mutation_test                 # all mutants
    python -m scripts.mutation_test --only M13,M21  # a subset
    python -m scripts.mutation_test --list

How it works: the repository's code, tests, configs and the real data the
tests read are copied to a scratch directory. For each mutant, one exact piece
of source text in that copy is replaced by a buggy version, the tests that
cover that module are run, and the file is restored. A mutant is KILLED if any
test fails and SURVIVED if every test still passes -- a survivor is a bug the
suite cannot see. Nothing in the real repository is modified.

Each mutant names the tests expected to catch it in two tiers: fast unit tests
first, then (only if those pass) the slower real-data tests. The report says
which tier and which test killed it.

Results are written to reports/validation/mutation_report.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("data", "retrieval", "generation", "evaluation", "experiments", "analysis", "configs", "tests", "scripts")
DATA_PATHS = ("datasets/processed", "artifacts")  # real data the `realdata` tests read (copied, never linked)
REPORT_PATH = REPO_ROOT / "reports" / "validation" / "mutation_report.md"


@dataclass(frozen=True)
class Mutant:
    id: str
    file: str
    description: str
    old: str
    new: str
    fast: tuple[str, ...]
    slow: tuple[str, ...] = ()


T_METRICS = ("tests/test_retrieval_metrics.py", "tests/test_retrieval_metrics_properties.py")
T_METRICS_REF = ("tests/test_retrieval_metrics_reference.py",)
T_HYBRID = ("tests/test_hybrid.py",)
T_RETR_REAL = ("tests/test_retrievers_real.py",)
T_DENSE = ("tests/test_dense.py",)
T_LEX = ("tests/test_lexical.py",)
T_CHUNK = ("tests/test_chunking.py",)
T_KB = ("tests/test_kb_variants.py",)
T_DATA_REAL = ("tests/test_data_real.py",)
T_REL = ("tests/test_relevance_judgments.py",)
T_ROB = ("tests/test_robustness.py", "tests/test_runner.py")
T_GEN = ("tests/test_generation_metrics.py",)
T_AGG = ("tests/test_aggregate.py", "tests/test_export_csv.py")
T_STATS = ("tests/test_stats.py",)
T_SWEEP = ("tests/test_run_sweep_integration.py",)
T_HARNESS_REAL = ("tests/test_harness_real.py",)
T_STORE = ("tests/test_results_store.py",)
T_INDEX = ("tests/test_index_store.py",)
T_CACHE = ("tests/test_embedding_cache.py",)

RM = "evaluation/retrieval_metrics.py"
HY = "retrieval/hybrid.py"
DE = "retrieval/dense.py"
LE = "retrieval/lexical.py"
CH = "data/chunking.py"
KB = "data/kb_variants.py"
RJ = "evaluation/relevance_judgments.py"
RO = "evaluation/robustness.py"
GM = "evaluation/generation_metrics.py"
AG = "analysis/aggregate.py"
ST = "analysis/stats.py"
RU = "experiments/runner.py"
RS = "experiments/results_store.py"
IS = "retrieval/index_store.py"
EC = "retrieval/embedding_cache.py"
GE = "generation/generator.py"
PA = "generation/parsing.py"

MUTANTS: list[Mutant] = [
    # -- retrieval metrics ------------------------------------------------------------
    Mutant("M01", RM, "precision divides by k+1 (off-by-one in k)",
           "    hits = sum(1 for cid in _ranked_ids(retrieved)[:k] if cid in relevant_chunk_ids)\n    return hits / k",
           "    hits = sum(1 for cid in _ranked_ids(retrieved)[:k] if cid in relevant_chunk_ids)\n    return hits / (k + 1)",
           T_METRICS, T_METRICS_REF),
    Mutant("M02", RM, "precision looks at the top k+1 results",
           "hits = sum(1 for cid in _ranked_ids(retrieved)[:k] if cid in relevant_chunk_ids)",
           "hits = sum(1 for cid in _ranked_ids(retrieved)[: k + 1] if cid in relevant_chunk_ids)",
           T_METRICS, T_METRICS_REF),
    Mutant("M03", RM, "precision divides by the number returned instead of k",
           "    hits = sum(1 for cid in _ranked_ids(retrieved)[:k] if cid in relevant_chunk_ids)\n    return hits / k",
           "    hits = sum(1 for cid in _ranked_ids(retrieved)[:k] if cid in relevant_chunk_ids)\n    return hits / max(1, min(k, len(retrieved)))",
           T_METRICS, T_METRICS_REF),
    Mutant("M04", RM, "recall divides by k instead of the number of relevant chunks",
           "    return hits / len(relevant_chunk_ids)", "    return hits / k", T_METRICS, T_METRICS_REF),
    Mutant("M05", RM, "recall ignores the cutoff k",
           "    hits = sum(1 for cid in ids[:k] if cid in relevant_chunk_ids)\n    return hits / len(relevant_chunk_ids)",
           "    hits = sum(1 for cid in ids if cid in relevant_chunk_ids)\n    return hits / len(relevant_chunk_ids)",
           T_METRICS, T_METRICS_REF),
    Mutant("M06", RM, "recall of a query with no relevant chunks is 0 instead of undefined",
           "    ids = _ranked_ids(retrieved)\n    if not relevant_chunk_ids:\n        return math.nan\n    hits = sum(",
           "    ids = _ranked_ids(retrieved)\n    if not relevant_chunk_ids:\n        return 0.0\n    hits = sum(",
           T_METRICS),
    Mutant("M07", RM, "MRR is 1/(rank+1) (off-by-one in rank)",
           "            return 1.0 / position", "            return 1.0 / (position + 1)", T_METRICS, T_METRICS_REF),
    Mutant("M08", RM, "MRR ignores the cutoff k",
           "    for position, cid in enumerate(ids[:k], start=1):\n        if cid in relevant_chunk_ids:\n            return 1.0 / position",
           "    for position, cid in enumerate(ids, start=1):\n        if cid in relevant_chunk_ids:\n            return 1.0 / position",
           T_METRICS, T_METRICS_REF),
    Mutant("M09", RM, "MRR uses the item's stale rank field instead of list position",
           "    for position, cid in enumerate(ids[:k], start=1):\n        if cid in relevant_chunk_ids:\n            return 1.0 / position",
           "    for position, cid in enumerate(ids[:k], start=1):\n        if cid in relevant_chunk_ids:\n            return 1.0 / getattr(retrieved[position - 1], 'rank', position)",
           T_METRICS),
    Mutant("M10", RM, "NDCG discount is 1/log2(rank+2) in DCG only",
           "        1.0 / math.log2(position + 1)\n        for position, cid in enumerate(ids[:k], start=1)",
           "        1.0 / math.log2(position + 2)\n        for position, cid in enumerate(ids[:k], start=1)",
           T_METRICS, T_METRICS_REF),
    Mutant("M11", RM, "NDCG has no rank discount (plain hit count ratio)",
           "        1.0 / math.log2(position + 1)\n        for position, cid in enumerate(ids[:k], start=1)",
           "        1.0\n        for position, cid in enumerate(ids[:k], start=1)",
           T_METRICS, T_METRICS_REF),
    Mutant("M12", RM, "NDCG ideal ranking ignores the cutoff k",
           "    ideal_hits = min(k, len(relevant_chunk_ids))", "    ideal_hits = len(relevant_chunk_ids)", T_METRICS, T_METRICS_REF),
    Mutant("M13", RM, "NDCG ideal ranking assumes k relevant chunks always exist",
           "    ideal_hits = min(k, len(relevant_chunk_ids))", "    ideal_hits = k", T_METRICS, T_METRICS_REF),
    Mutant("M14", RM, "duplicate chunk ids are no longer rejected",
           "    if len(set(ids)) != len(ids):", "    if False:", T_METRICS),
    # -- fusion ----------------------------------------------------------------------------
    Mutant("M15", HY, "wrong RRF constant (61 instead of 60)", "DEFAULT_RRF_K = 60", "DEFAULT_RRF_K = 61", T_HYBRID, T_RETR_REAL),
    Mutant("M16", HY, "RRF rank off by one",
           "fused_scores[item.chunk_id] += 1.0 / (k_constant + position)",
           "fused_scores[item.chunk_id] += 1.0 / (k_constant + position + 1)", T_HYBRID, T_RETR_REAL),
    Mutant("M17", HY, "fusion adds raw retriever scores instead of reciprocal ranks (no normalisation)",
           "fused_scores[item.chunk_id] += 1.0 / (k_constant + position)",
           "fused_scores[item.chunk_id] += item.score", T_HYBRID, T_RETR_REAL),
    Mutant("M18", HY, "fused list sorted ascending (worst first)",
           "key=lambda cid: (-fused_scores[cid], cid)", "key=lambda cid: (fused_scores[cid], cid)", T_HYBRID, T_RETR_REAL),
    Mutant("M19", HY, "fusion tie-break removed (order depends on which retriever is listed first)",
           "key=lambda cid: (-fused_scores[cid], cid)", "key=lambda cid: -fused_scores[cid]", T_HYBRID, T_RETR_REAL),
    Mutant("M20", HY, "hybrid returns k+1 results",
           "return self.fusion_fn(ranked_lists)[:k]", "return self.fusion_fn(ranked_lists)[: k + 1]", T_HYBRID, T_RETR_REAL),
    Mutant("M21", HY, "hybrid fuses only the top k of each retriever, not k_per_retriever",
           "retriever.retrieve(query, max(k, self.k_per_retriever))", "retriever.retrieve(query, k)", T_HYBRID, T_RETR_REAL),
    Mutant("M22", HY, "configured RRF constant is ignored",
           "self.fusion_fn = fusion_fn or partial(reciprocal_rank_fusion, k_constant=rrf_k)",
           "self.fusion_fn = fusion_fn or reciprocal_rank_fusion", T_HYBRID),
    Mutant("M23", HY, "RRF uses the item's rank field instead of its position in the list",
           "fused_scores[item.chunk_id] += 1.0 / (k_constant + position)",
           "fused_scores[item.chunk_id] += 1.0 / (k_constant + item.rank)", T_HYBRID),
    # -- dense -----------------------------------------------------------------------------
    Mutant("M24", DE, "chunk vectors are not normalised (inner product instead of cosine)",
           "        vectors = l2_normalize(vectors.astype(np.float32))", "        vectors = vectors.astype(np.float32)", T_DENSE, T_RETR_REAL),
    Mutant("M25", DE, "query vector is not normalised",
           "        query_vec = l2_normalize(query_vec.astype(np.float32))", "        query_vec = query_vec.astype(np.float32)", T_DENSE, T_RETR_REAL),
    Mutant("M26", DE, "dense returns k-1 results", "enumerate(candidates[:k], start=1)", "enumerate(candidates[: k - 1], start=1)", T_DENSE, T_RETR_REAL),
    Mutant("M27", DE, "dense ranking reversed among candidates",
           "candidates.sort(key=lambda pair: (-pair[0], pair[1]))", "candidates.sort(key=lambda pair: (pair[0], pair[1]))", T_DENSE, T_RETR_REAL),
    Mutant("M28", DE, "dense tie rule removed (ties left in FAISS order)",
           "candidates.sort(key=lambda pair: (-pair[0], pair[1]))", "candidates.sort(key=lambda pair: -pair[0])", T_DENSE, T_RETR_REAL),
    Mutant("M29", DE, "dense ranks start at 0", "enumerate(candidates[:k], start=1)", "enumerate(candidates[:k], start=0)", T_DENSE, T_RETR_REAL),
    # -- lexical ---------------------------------------------------------------------------
    Mutant("M30", LE, "BM25 ranking reversed", "matching.sort(key=lambda i: (-scores[i], i))", "matching.sort(key=lambda i: (scores[i], i))", T_LEX, T_RETR_REAL),
    Mutant("M31", LE, "BM25 returns chunks that share no term with the query (zero-score padding)",
           "i for i, term_freqs in enumerate(self.bm25.doc_freqs) if not unique_terms.isdisjoint(term_freqs)",
           "i for i, term_freqs in enumerate(self.bm25.doc_freqs) if True", T_LEX, T_RETR_REAL),
    Mutant("M32", LE, "BM25 returns k+1 results", "top_indices = matching[:k]", "top_indices = matching[: k + 1]", T_LEX, T_RETR_REAL),
    Mutant("M33", LE, "BM25 tokeniser keeps case", "return _TOKEN_RE.findall(text.lower())", "return _TOKEN_RE.findall(text)", T_LEX, T_RETR_REAL),
    # -- chunking --------------------------------------------------------------------------
    Mutant("M34", CH, "chunks do not overlap", "step = config.chunk_size_words - config.chunk_overlap_words", "step = config.chunk_size_words", T_CHUNK, T_DATA_REAL),
    Mutant("M35", CH, "overlap is one word short", "step = config.chunk_size_words - config.chunk_overlap_words", "step = config.chunk_size_words - config.chunk_overlap_words + 1", T_CHUNK, T_DATA_REAL),
    Mutant("M36", CH, "extra trailing chunk when the document ends exactly at a window edge",
           "if start + config.chunk_size_words >= len(words):", "if start + config.chunk_size_words > len(words):", T_CHUNK, T_DATA_REAL),
    Mutant("M37", CH, "chunk ids start at 1", 'chunk_id=f"{doc.doc_id}#{index}"', 'chunk_id=f"{doc.doc_id}#{index + 1}"', T_CHUNK, T_DATA_REAL),
    Mutant("M38", CH, "window is one word short", "window = words[start : start + config.chunk_size_words]", "window = words[start : start + config.chunk_size_words - 1]", T_CHUNK, T_DATA_REAL),
    # -- KB variants -----------------------------------------------------------------------
    Mutant("M39", KB, "wrong noise ratio in mixed (90% of configured)",
           "n_noise = round(len(curated_docs) * config.mixed_noise_ratio)", "n_noise = round(len(curated_docs) * config.mixed_noise_ratio * 0.9)", T_KB, T_DATA_REAL),
    Mutant("M40", KB, "noise may share a topic with curated",
           "if normalize_topic(doc.focus) in curated_topics or doc.answer in seen_answers:", "if doc.answer in seen_answers:", T_KB, T_DATA_REAL),
    Mutant("M41", KB, "noise may contain a verbatim copy of a curated chunk",
           "if any(c.text in curated_chunk_texts for c in chunk_document(doc, chunking_config)):", "if False:", T_KB, T_DATA_REAL),
    Mutant("M42", KB, "reduced drops half the configured share of topics",
           "n_drop = round(len(topics) * config.reduced_drop_topic_fraction)", "n_drop = round(len(topics) * config.reduced_drop_topic_fraction / 2)", T_KB, T_DATA_REAL),
    Mutant("M43", KB, "base corpus sampling is unseeded",
           "    rng = random.Random(config.seed)\n    quality_pool", "    rng = random.Random()\n    quality_pool", T_KB, T_DATA_REAL),
    Mutant("M44", KB, "pool is not sorted before shuffling (result depends on input order)",
           "    quality_pool.sort(key=lambda d: d.doc_id)  # deterministic order before shuffling\n", "", T_KB, T_DATA_REAL),
    Mutant("M45", KB, "reduced keeps some documents of dropped topics",
           "reduced = [d for d in curated_docs if normalize_topic(d.focus) not in dropped]",
           'reduced = [d for d in curated_docs if normalize_topic(d.focus) not in dropped or d.question_type == "information"]', T_KB, T_DATA_REAL),
    Mutant("M46", KB, "quality filter uses > instead of >=",
           "len(d.answer.split()) >= min_answer_words", "len(d.answer.split()) > min_answer_words", T_KB, T_DATA_REAL),
    Mutant("M47", KB, "noise is appended unshuffled (always the first eligible documents)", "    rng.shuffle(pool)\n    return curated_docs + pool[:n_noise]", "    return curated_docs + pool[:n_noise]", T_KB, T_DATA_REAL),
    Mutant("M48", KB, "topic comparison is case-sensitive", 'return " ".join(focus.lower().split())', 'return " ".join(focus.split())', T_KB, T_DATA_REAL),
    # -- relevance labels ------------------------------------------------------------------
    Mutant("M49", RJ, "lenient labels ignore verbatim copies of the answer", "            relevant |= by_text.get(text, set())", "            pass", T_REL, T_DATA_REAL),
    Mutant("M50", RJ, "lenient labels treat any chunk on the same topic as relevant",
           "by_topic_and_type[(normalize_topic(chunk.focus), chunk.question_type)].add(chunk.chunk_id)",
           "by_topic_and_type[(normalize_topic(chunk.focus), q_type)].add(chunk.chunk_id) if False else [by_topic_and_type[(normalize_topic(chunk.focus), t)].add(chunk.chunk_id) for t in {c.question_type for c in chunks}]",
           T_REL, T_DATA_REAL),
    Mutant("M51", RJ, "strict labels use the query's topic instead of its own document",
           "return {q.query_id: set(doc_to_chunk_ids.get(q.query_id, set())) for q in queries}",
           "return {q.query_id: {c.chunk_id for c in chunks if c.focus == q.focus} for q in queries}", T_REL, T_DATA_REAL),
    # -- robustness ------------------------------------------------------------------------
    Mutant("M52", RO, "drop removes the relevant chunk and the one after it",
           "return _renumber_ranks(retrieved[:i] + retrieved[i + 1 :]), chunk.chunk_id", "return _renumber_ranks(retrieved[:i] + retrieved[i + 2 :]), chunk.chunk_id", T_ROB, T_SWEEP),
    Mutant("M53", RO, "noise replaces the highest-ranked non-relevant chunk instead of the lowest",
           "for i in range(len(retrieved) - 1, -1, -1):", "for i in range(len(retrieved)):", T_ROB, T_SWEEP),
    Mutant("M54", RO, "noise chunk may share the query's topic",
           "        if normalize_topic(c.focus) != excluded_topic\n        and c.chunk_id not in exclude_chunk_ids",
           "        if c.chunk_id not in exclude_chunk_ids", T_ROB, T_SWEEP),
    Mutant("M55", RO, "noise is inserted instead of replacing (context grows by one)",
           "return _renumber_ranks(retrieved[:i] + [noise_item] + retrieved[i + 1 :]), target.chunk_id",
           "return _renumber_ranks(retrieved[:i] + [noise_item] + retrieved[i:]), target.chunk_id", T_ROB, T_SWEEP),
    Mutant("M56", RU, "noise chunk choice is not keyed by query (same chunk for all)",
           'rng=keyed_rng(config, "noise", kb_variant, query.query_id),', 'rng=keyed_rng(config, "noise"),', T_ROB, T_SWEEP),
    Mutant("M57", RU, "inject does not protect lenient-relevant chunks",
           "        protected = strict_relevant | lenient_relevant", "        protected = set(strict_relevant)", T_ROB, T_SWEEP),
    # -- generation metrics ----------------------------------------------------------------
    Mutant("M58", GM, "'no information' statements count against faithfulness",
           '    factual = labels.count("supported") + labels.count("unsupported")', "    factual = len(labels)", T_GEN),
    Mutant("M59", GM, "'no information' statements count as supported (the pre-audit behaviour)",
           '    return labels.count("supported") / factual',
           '    return (labels.count("supported") + labels.count("no_information")) / len(labels)', T_GEN),
    Mutant("M60", GM, "quoted booleans from the judge are accepted",
           "    if not all(isinstance(v, bool) for v in values):  # \"false\" (a string) is rejected, not truthy", "    if False:", T_GEN),
    Mutant("M61", GM, "abstained answers are still scored for answer relevance", "        math.nan\n        if abstained\n", "        math.nan\n        if False\n", T_GEN),
    Mutant("M62", GM, "judge runs at temperature 0.7", "JUDGE_TEMPERATURE = 0.0", "JUDGE_TEMPERATURE = 0.7", T_GEN),
    Mutant("M63", GM, "summary sentences are not checked as claims",
           "    claims = [strip_citations(s) for s in split_sentences(output.summary)]", "    claims = []", T_GEN),
    Mutant("M64", GM, "wrong-length judge reply is accepted",
           "    if not isinstance(values, list) or len(values) != expected_length:", "    if not isinstance(values, list):", T_GEN),
    Mutant("M65", GE, "an answer cut off at max_tokens is accepted", '    if result.finish_reason != "stop":\n        return f"finish_reason', '    if False:\n        return f"finish_reason', ("tests/test_generator.py",)),
    Mutant("M66", GE, "an answer with no key points is accepted", "    if not output.key_points:", "    if False:", ("tests/test_generator.py",), T_SWEEP),
    # -- analysis --------------------------------------------------------------------------
    Mutant("M67", AG, "unanswerable queries are dropped from the headline metric (survivorship bias)",
           'columns[f"{m}{suffix}_all"] = value if answerable else 0.0', 'columns[f"{m}{suffix}_all"] = value if answerable else math.nan', T_AGG, T_SWEEP),
    Mutant("M68", AG, "unanswerable queries count as 0 in the answerable-only view",
           'columns[f"{m}{suffix}"] = value if answerable else math.nan', 'columns[f"{m}{suffix}"] = value if answerable else 0.0', T_AGG, T_SWEEP),
    Mutant("M69", AG, "abstention rate counts partial answers too",
           '"abstained": 1.0 if metrics["abstention_label"] == "abstained" else 0.0,', '"abstained": 1.0 if metrics["abstention_label"] != "answered" else 0.0,', T_AGG),
    Mutant("M70", AG, "duplicate rows are not detected", "    if len(set(keys)) != len(keys):", "    if False:", T_AGG, T_SWEEP),
    Mutant("M71", AG, "robustness delta is baseline minus perturbed (sign flipped)",
           'row[f"{metric}_delta"] = perturbed[metric] - baseline[metric]', 'row[f"{metric}_delta"] = baseline[metric] - perturbed[metric]', T_AGG),
    Mutant("M72", ST, "paired difference sign flipped", "    diff = a_arr - b_arr", "    diff = b_arr - a_arr", T_STATS, T_AGG),
    Mutant("M73", ST, "Holm correction replaced by plain Bonferroni", "min(1.0, (m - rank) * p[idx])", "min(1.0, m * p[idx])", T_STATS),
    Mutant("M74", ST, "bootstrap interval is 80% instead of 95%", "    tail = (1 - confidence) / 2 * 100", "    tail = 10.0", T_STATS),
    Mutant("M75", ST, "pairs are not aligned: NaN removed from each side separately",
           "    keep = ~(np.isnan(a_arr) | np.isnan(b_arr))\n    a_arr, b_arr = a_arr[keep], b_arr[keep]",
           "    a_arr, b_arr = a_arr[~np.isnan(a_arr)], b_arr[~np.isnan(b_arr)]\n    n_min = min(a_arr.size, b_arr.size)\n    a_arr, b_arr = a_arr[:n_min], b_arr[:n_min]", T_STATS),
    # -- harness ---------------------------------------------------------------------------
    Mutant("M76", RU, "resume ignores completed rows (everything is re-run and duplicated)",
           '        completed.add(record["run_key"])\n        if record["record_type"] == "standard" and', '        if record["record_type"] == "standard" and', T_SWEEP),
    Mutant("M77", RU, "a failed row is also marked completed (never retried)",
           "            summary.errors += 1\n            return", "            summary.errors += 1\n            completed.add(job.run_key)\n            store.append({'record_type': 'robustness_skipped', 'run_key': job.run_key, 'run_fingerprint': '', 'reason': 'x'})\n            return", T_SWEEP),
    Mutant("M78", RU, "rows from a different setup are accepted into the same results file",
           '        if record.get("run_fingerprint") != fingerprint:', "        if False:", T_SWEEP),
    Mutant("M79", RU, "per-record seed ignores the query (all records share a seed)",
           'digest = hashlib.sha256(f"{config.seed}|{config.repeat}|{run_key}".encode("utf-8")).digest()',
           'digest = hashlib.sha256(f"{config.seed}|{config.repeat}".encode("utf-8")).digest()', ("tests/test_runner.py",), T_SWEEP),
    Mutant("M80", RU, "random baseline is seeded by repeat as well (retrieval moves between repeat runs)",
           '    return random.Random("|".join([str(config.seed), *parts]))', '    return random.Random("|".join([str(config.seed), str(config.repeat), *parts]))', ("tests/test_runner.py",), T_SWEEP),
    Mutant("M81", RU, "no-retrieval baseline uses the normal RAG prompt",
           "        template_id=CLOSED_BOOK_TEMPLATE_ID if closed_book else None,", "        template_id=None,", T_SWEEP),
    Mutant("M82", RU, "random baseline draws k+1 chunks",
           "chunks, config.top_k, keyed_rng(config, \"random\", kb_variant, query.query_id)", "chunks, config.top_k + 1, keyed_rng(config, \"random\", kb_variant, query.query_id)", T_SWEEP, T_HARNESS_REAL),
    Mutant("M83", RS, "a record appended after a torn line is glued onto it", "                if repair:\n", "                if False:\n", T_STORE),
    Mutant("M84", RS, "torn-tail repair deletes the last complete record too",
           '            keep = data.rfind(b"\\n") + 1', '            keep = data.rfind(b"\\n", 0, max(0, data.rfind(b"\\n"))) + 1', T_STORE, T_SWEEP),
    Mutant("M85", IS, "stale index is reused when the chunks changed",
           '        and meta.get("chunks_fingerprint") == chunks_fingerprint(chunks)\n', "", T_INDEX),
    Mutant("M86", IS, "index built with another embedding model is reused", '        meta.get("model") == dense_model\n        and ', "        ", T_INDEX),
    Mutant("M87", EC, "cache key ignores the model name", 'hashlib.sha256(f"{model}\\n{text}".encode("utf-8")).hexdigest()', 'hashlib.sha256(f"{text}".encode("utf-8")).hexdigest()', T_CACHE),
    Mutant("M88", EC, "truncated cached vector is not detected", "                if len(blob) % _FLOAT32_BYTES != 0:", "                if False:", T_CACHE),
    Mutant("M89", EC, "mixed-dimension vectors are not detected", "    if len(dims) > 1:", "    if False:", T_CACHE),
    Mutant("M90", PA, "an empty section swallows the next heading (pre-audit parser)",
           r'r"^#{1,6}[ \t]*(Summary|Key Points|Disclaimer)[ \t]*\n(.*?)(?=^#{1,6}[ \t]*\w|\Z)"',
           r'r"^#{1,6}\s*(Summary|Key Points|Disclaimer)\s*\n(.*?)(?=\n#{1,6}\s*\w|\Z)"', ("tests/test_parsing.py", "tests/test_generation_metrics.py")),
]


def copy_workspace(work: Path) -> None:
    for name in CODE_DIRS:
        shutil.copytree(REPO_ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(REPO_ROOT / "pyproject.toml", work / "pyproject.toml")
    for rel in DATA_PATHS:
        src = REPO_ROOT / rel
        if src.exists():
            shutil.copytree(src, work / rel, ignore=shutil.ignore_patterns("api_spend.json", "5b4983b42ed5"))


def run_tests(work: Path, tests: tuple[str, ...], timeout: int) -> tuple[bool, str]:
    """(all passed?, name of the first failing test or a short reason)."""
    env = {**os.environ, "PYTHONPATH": str(work), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *tests, "-x", "-q", "-p", "no:cacheprovider", "--no-header", "-rf"],
            cwd=work, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timed out (counted as killed: the suite did not pass)"
    if result.returncode == 0:
        if " skipped" in result.stdout.splitlines()[-1]:
            return True, "passed, but with skipped tests"
        return True, ""
    for line in result.stdout.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            return False, line.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "")
    return False, (result.stdout.strip().splitlines() or ["no output"])[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual mutation testing (see module docstring).")
    parser.add_argument("--only", default=None, help="comma-separated mutant ids")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per test run")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    selected = MUTANTS if not args.only else [m for m in MUTANTS if m.id in set(args.only.split(","))]
    if args.list:
        for m in selected:
            print(f"{m.id}  {m.file:40s} {m.description}")
        return

    work = Path(tempfile.mkdtemp(prefix="rag_mutation_"))
    print(f"Copying workspace to {work} ...")
    copy_workspace(work)

    # Sanity: every mutant's target text exists exactly once, and the unmutated copy passes.
    for m in selected:
        count = (work / m.file).read_text(encoding="utf-8").count(m.old)
        if count != 1:
            raise SystemExit(f"{m.id}: target text occurs {count} times in {m.file} (need exactly 1)")
    all_tests = tuple(sorted({t for m in selected for t in (*m.fast, *m.slow)}))
    print(f"Baseline: running {len(all_tests)} test files on unmutated code ...")
    ok, detail = run_tests(work, all_tests, args.timeout * 3)
    if not ok:
        raise SystemExit(f"Baseline tests fail on unmutated code ({detail}); mutation results would be meaningless.")
    if detail:
        raise SystemExit(f"Baseline: {detail}. Real-data tests must run, not skip, for mutation testing.")

    rows = []
    for m in selected:
        path = work / m.file
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace(m.old, m.new), encoding="utf-8", newline="\n")
        started = time.time()
        try:
            passed, killer = run_tests(work, m.fast, args.timeout)
            tier = "fast"
            if passed and m.slow:
                passed, killer = run_tests(work, m.slow, args.timeout)
                tier = "real-data/harness"
        finally:
            path.write_text(original, encoding="utf-8", newline="\n")
        status = "SURVIVED" if passed else "killed"
        rows.append((m, status, tier if not passed else "-", killer))
        print(f"{m.id} {status:8s} [{time.time() - started:5.0f}s] {m.description}" + (f"  <- {killer}" if not passed else ""))

    killed = sum(1 for _, status, _, _ in rows if status == "killed")
    score = killed / len(rows) if rows else float("nan")
    print(f"\nMutation score: {killed}/{len(rows)} = {score:.0%}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mutation testing report",
        "",
        f"Mutants injected: {len(rows)}. Killed: {killed}. Survived: {len(rows) - killed}. "
        f"**Mutation score: {score:.0%}**.",
        "",
        "Each mutant is one deliberate bug injected into a copy of the code. \"Killed\" means at least one",
        "test failed with the bug in place. \"Tier\" says whether the fast unit tests caught it or it took",
        "the slower real-data / harness tests. Produced by `python -m scripts.mutation_test`.",
        "",
        "| ID | File | Injected bug | Result | Tier | Caught by |",
        "|---|---|---|---|---|---|",
    ]
    for m, status, tier, killer in rows:
        lines.append(f"| {m.id} | `{m.file}` | {m.description} | {status} | {tier} | `{killer}` |" if killer else f"| {m.id} | `{m.file}` | {m.description} | {status} | {tier} | |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Report: {args.report}")
    shutil.rmtree(work, ignore_errors=True)
    if killed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

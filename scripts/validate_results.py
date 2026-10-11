"""Interrogate a finished sweep's results log and write a data-validation report.

    python -m scripts.validate_results --config configs/full_sweep.yaml

Part 1 checks the data (every check is PASS or FAIL, with the evidence).
Part 2 summarises the findings with uncertainty: per-configuration mean, SD
and 95% bootstrap interval, paired comparisons across the same queries, the
oracle/random bracket, and whether the query set is large enough.

The retrieval metrics are recomputed here with `ranx`, an independent IR
evaluation library, from the logged chunk ids -- not with this project's own
metric code -- and compared with the logged values.

Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ranx import Qrels, Run, evaluate

from analysis.aggregate import (
    GENERATION_METRIC_COLUMNS,
    RETRIEVAL_ALL_COLUMNS,
    load_robustness_results,
    load_standard_results,
    read_records,
    summarize_with_uncertainty,
)
from analysis.compare import paired_comparisons
from analysis.export_csv import COMPARED_METRICS
from analysis.stats import paired_comparison, required_pairs
from data.query_set import load_query_set_jsonl
from experiments.config import NO_KB, load_sweep_config
from experiments.runner import file_sha256, robustness_run_key, standard_run_key
from generation.prompts import CLOSED_BOOK_TEMPLATE_ID, prompt_hash

REAL_METHODS = ["lexical", "dense", "hybrid"]
KB_ORDER = ["curated", "mixed", "reduced"]


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []

    def h(self, level: int, text: str) -> None:
        self.lines += ["", "#" * level + " " + text, ""]

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "**FAIL**"
        self.lines.append(f"- {status} — {name}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def table(self, df: pd.DataFrame, floatfmt: str = "{:.3f}") -> None:
        if df.empty:
            self.lines.append("(no rows)")
            return
        cols = list(df.columns)
        if self.lines and self.lines[-1].strip():
            self.lines.append("")  # Markdown needs a blank line before a table
        self.lines.append("| " + " | ".join(cols) + " |")
        self.lines.append("|" + "|".join("---" for _ in cols) + "|")
        for _, row in df.iterrows():
            cells = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    cells.append("" if math.isnan(value) else floatfmt.format(value))
                else:
                    cells.append(str(value))
            self.lines.append("| " + " | ".join(cells) + " |")


def nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def same(a, b, tol: float = 1e-9) -> bool:
    if nan(a) and nan(b):
        return True
    if nan(a) or nan(b):
        return False
    return abs(a - b) <= tol


# -- Part 1: data quality ---------------------------------------------------------------


def check_integrity(report: Report, config, records, queries) -> None:
    report.h(3, "1.1 The log itself")
    fingerprint = config.run_fingerprint(file_sha256(config.query_set_path))
    non_error = [r for r in records if r["record_type"] != "error"]
    fingerprints = Counter(r.get("run_fingerprint") for r in non_error)
    report.check("every line of the log is readable and no run key appears twice", True, f"{len(records)} records")
    report.check(
        "every row was produced under one setup, and it is the setup in the config file today",
        set(fingerprints) == {fingerprint},
        f"fingerprints in log: {dict(fingerprints)}; config gives {fingerprint}",
    )
    completed = {r["run_key"] for r in non_error}
    errors = [r for r in records if r["record_type"] == "error"]
    unresolved = sorted({r["failed_run_key"] for r in errors} - completed)
    report.check(
        "every failure logged during the run was later retried successfully",
        not unresolved,
        f"{len(errors)} error record(s) in the log; {len(unresolved)} still unresolved" + (f", e.g. {unresolved[:3]}" if unresolved else ""),
    )
    if errors:
        report.p(f"  - error types: {dict(Counter(r['error_type'] for r in errors))}")
    trees = Counter(r.get("source_tree_sha256") for r in non_error if r["record_type"] != "robustness_skipped")
    report.check("all rows were produced by one version of the code", len(trees) == 1, f"source_tree_sha256: {dict(trees)}")


def check_row_counts(report: Report, config, records, queries) -> None:
    report.h(3, "1.2 Row counts")
    ids = [q.query_id for q in queries]
    methods = config.retrieval_methods + [b for b in config.baselines if b != "none"]
    expected = {standard_run_key(v, m, q) for v in config.kb_variant_names for m in methods for q in ids}
    if "none" in config.baselines:
        expected |= {standard_run_key(NO_KB, "none", q) for q in ids}
    standard = [r for r in records if r["record_type"] == "standard"]
    got = {r["run_key"] for r in standard}
    n_configs = len(config.kb_variant_names) * len(methods) + ("none" in config.baselines)
    report.check(
        f"standard rows = queries x configurations = {len(ids)} x {n_configs} = {len(expected)}",
        got == expected and len(standard) == len(expected),
        f"found {len(standard)}; missing {len(expected - got)}; unexpected {len(got - expected)}",
    )
    per_config = Counter((r["kb_variant"], r["retrieval_method"]) for r in standard)
    report.check("every configuration has exactly one row per query", set(per_config.values()) == {len(ids)}, f"{len(per_config)} configurations")

    if config.robustness.enabled:
        expected_rob = {
            robustness_run_key(v, config.robustness.retrieval_method, q, p)
            for v in config.kb_variant_names
            for q in ids
            for p in config.robustness.perturbations
        }
        ran = {r["run_key"] for r in records if r["record_type"] == "robustness"}
        skipped = {r["run_key"] for r in records if r["record_type"] == "robustness_skipped"}
        report.check(
            f"every robustness condition is accounted for ({len(expected_rob)} = KB variants x queries x conditions)",
            ran | skipped == expected_rob and not ran & skipped,
            f"{len(ran)} ran, {len(skipped)} could not be applied (reason logged), {len(expected_rob - ran - skipped)} missing",
        )
        reasons = Counter((r["kb_variant"], r["perturbation"], r["reason"]) for r in records if r["record_type"] == "robustness_skipped")
        for (kb, perturbation, reason), count in sorted(reasons.items()):
            report.p(f"  - not applicable: {kb} / {perturbation}: {count} queries ({reason})")


def check_values(report: Report, config, records) -> None:
    report.h(3, "1.3 Missing values and ranges")
    standard = [r for r in records if r["record_type"] == "standard"]
    problems: dict[str, list[str]] = defaultdict(list)
    for r in standard:
        key = r["run_key"]
        closed_book = r["retrieval_method"] == "none"
        gm, gen = r["generation_metrics"], r["generation"]
        for field in ("query_id", "question", "generated_at", "seed", "run_fingerprint", "code_version", "source_tree_sha256"):
            if r.get(field) in (None, ""):
                problems["a required field is empty"].append(f"{key}.{field}")
        if not gen["raw_text"].strip() or not gen["output"]["summary"].strip() or not gen["output"]["key_points"]:
            problems["an answer is empty or missing a section"].append(key)
        if gen["finish_reason"] != "stop":
            problems["an answer was cut off or filtered"].append(key)
        if gm is None:
            problems["generation metrics are missing"].append(key)
            continue

        # NaN is allowed only where the metric is undefined by its definition.
        factual = gm["n_supported"] + gm["n_unsupported"]
        if nan(gm["faithfulness"]) != (factual == 0):
            problems["faithfulness is NaN exactly when the answer made no factual claim (or had no context)"].append(key)
        if nan(gm["answer_relevance"]) != (gm["abstention_label"] == "abstained"):
            problems["answer relevance is NaN exactly when the answer abstained"].append(key)
        if nan(gm["context_relevance"]) != (len(r["retrieved"]) == 0):
            problems["context relevance is NaN exactly when no context was retrieved"].append(key)
        if closed_book != (r["retrieval_metrics"] is None):
            problems["retrieval metrics are absent exactly for the no-retrieval baseline"].append(key)
        if not closed_book:
            unanswerable = len(r["relevant_chunk_ids"]) == 0
            rm = r["retrieval_metrics"]
            if any(nan(rm[m]) != unanswerable for m in ("recall_at_k", "mrr", "ndcg_at_k")) or nan(rm["precision_at_k"]):
                problems["recall/MRR/NDCG are NaN exactly when the KB holds no relevant chunk for the query"].append(key)
        values = {name: gm[name] for name in ("faithfulness", "answer_relevance", "context_relevance", "reference_agreement")}
        if r["retrieval_metrics"]:
            values.update({f"retrieval.{k}": v for k, v in r["retrieval_metrics"].items()})
            values.update({f"lenient.{k}": v for k, v in r["retrieval_metrics_lenient"].items()})
        for name, value in values.items():
            if not nan(value) and not (-1e-9 <= value <= 1 + 1e-9):
                problems["a metric is outside [0, 1]"].append(f"{key}.{name}={value}")
        if gm["abstention_label"] not in ("answered", "partial", "abstained"):
            problems["abstention label is not one of the three allowed values"].append(key)
        if len(gm["claim_labels"]) not in (0, gm["n_claims"]) or (len(r["retrieved"]) > 0 and len(gm["claim_labels"]) != gm["n_claims"]):
            problems["one label per claim"].append(key)
        if len(gm["context_labels"]) != len(r["retrieved"]):
            problems["one relevance label per retrieved passage"].append(key)

    names = [
        "a required field is empty",
        "an answer is empty or missing a section",
        "an answer was cut off or filtered",
        "generation metrics are missing",
        "faithfulness is NaN exactly when the answer made no factual claim (or had no context)",
        "answer relevance is NaN exactly when the answer abstained",
        "context relevance is NaN exactly when no context was retrieved",
        "retrieval metrics are absent exactly for the no-retrieval baseline",
        "recall/MRR/NDCG are NaN exactly when the KB holds no relevant chunk for the query",
        "a metric is outside [0, 1]",
        "abstention label is not one of the three allowed values",
        "one label per claim",
        "one relevance label per retrieved passage",
    ]
    for name in names:
        bad = problems.get(name, [])
        positive = name if "exactly" in name or name.startswith("one ") else f"no row where {name}"
        report.check(positive, not bad, f"{len(standard)} rows checked" if not bad else f"{len(bad)} violation(s), e.g. {bad[:3]}")


def check_recomputed(report: Report, config, records) -> None:
    report.h(3, "1.4 Logged scores recomputed independently from the logged evidence")
    standard = [r for r in records if r["record_type"] == "standard" and r["retrieval_metrics"] is not None]
    k = config.top_k

    for label, rel_field, metric_field in (
        ("strict", "relevant_chunk_ids", "retrieval_metrics"),
        ("lenient", "relevant_chunk_ids_lenient", "retrieval_metrics_lenient"),
    ):
        answerable = [r for r in standard if r[rel_field]]
        qrels = Qrels({r["run_key"]: {cid: 1 for cid in r[rel_field]} for r in answerable})
        run_dict = {}
        for r in answerable:
            ids = [c["chunk_id"] for c in r["retrieved"]]
            # ranx needs a non-empty run per query; a placeholder that is in no qrels stands for "nothing retrieved"
            run_dict[r["run_key"]] = {cid: float(len(ids) - i) for i, cid in enumerate(ids)} or {"__nothing_retrieved__": 0.0}
        names = [f"precision@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]
        reference = evaluate(qrels, Run(run_dict), names, return_mean=False)
        order = sorted(run_dict)  # ranx reports per-query scores in sorted query-id order
        logged_by_key = {r["run_key"]: r[metric_field] for r in answerable}
        mismatches = 0
        for ranx_name, ours in zip(names, ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")):
            for key, ref_value in zip(order, reference[ranx_name]):
                if not same(logged_by_key[key][ours], float(ref_value), 1e-9):
                    mismatches += 1
        report.check(
            f"P@{k}, R@{k}, MRR@{k}, NDCG@{k} ({label} labels) equal the values `ranx` computes from the logged chunk ids",
            mismatches == 0,
            f"{len(answerable)} answerable rows x 4 metrics compared; {mismatches} mismatch(es)",
        )

    all_standard = [r for r in records if r["record_type"] == "standard"]
    bad_faith = bad_ctx = bad_ref = 0
    for r in all_standard:
        gm = r["generation_metrics"]
        labels = gm["claim_labels"]
        factual = labels.count("supported") + labels.count("unsupported")
        expected = labels.count("supported") / factual if factual else math.nan
        bad_faith += not same(gm["faithfulness"], expected)
        ctx = gm["context_labels"]
        bad_ctx += not same(gm["context_relevance"], sum(ctx) / len(ctx) if ctx else math.nan)
        ref = gm.get("reference_labels") or []
        ref_factual = ref.count("supported") + ref.count("unsupported")
        bad_ref += not same(gm["reference_agreement"], ref.count("supported") / ref_factual if ref_factual else math.nan)
    report.check("faithfulness equals supported / (supported + unsupported) recounted from the logged claim labels", bad_faith == 0, f"{len(all_standard)} rows; {bad_faith} mismatch(es)")
    report.check("context relevance equals the share of `true` in the logged passage labels", bad_ctx == 0, f"{bad_ctx} mismatch(es)")
    report.check("reference agreement equals the recount from the logged reference labels", bad_ref == 0, f"{bad_ref} mismatch(es)")


def check_retrieval_lists(report: Report, config, records) -> None:
    report.h(3, "1.5 Retrieved lists")
    standard = [r for r in records if r["record_type"] == "standard" and r["retrieval_method"] != "none"]
    k = config.top_k
    too_long = dup = unsorted = wrong_k = 0
    short = Counter()
    for r in standard:
        ids = [c["chunk_id"] for c in r["retrieved"]]
        scores = [c["score"] for c in r["retrieved"]]
        too_long += len(ids) > k
        dup += len(set(ids)) != len(ids)
        unsorted += scores != sorted(scores, reverse=True)
        if r["retrieval_method"] in ("dense", "hybrid", "random") and len(ids) != k:
            wrong_k += 1
        if len(ids) < k:
            short[r["retrieval_method"]] += 1
    report.check(f"no list is longer than top_k = {k}", too_long == 0, f"{len(standard)} lists")
    report.check("no list contains a chunk twice", dup == 0)
    report.check("every list is in descending score order", unsorted == 0)
    report.check(f"dense, hybrid and random always return exactly {k} chunks", wrong_k == 0)
    report.p(f"  - lists shorter than {k} (allowed for oracle = fewer gold chunks than {k}, and BM25 = fewer matching chunks): {dict(short)}")

    signatures: dict[tuple, tuple] = {}
    by_config: dict[tuple, list] = defaultdict(list)
    for r in standard:
        by_config[(r["kb_variant"], r["retrieval_method"])].append((r["query_id"], tuple(c["chunk_id"] for c in r["retrieved"])))
    signatures = {key: tuple(sorted(rows)) for key, rows in by_config.items()}
    identical = sorted(
        (a, b) for a in signatures for b in signatures if a < b and signatures[a] == signatures[b]
    )
    expected_identical = [(("curated", "oracle"), ("mixed", "oracle"))] if {"curated", "mixed"} <= set(config.kb_variant_names) and "oracle" in config.baselines else []
    report.check(
        "no two configurations retrieved identical chunks for every query (which would mean a setting is not being applied)",
        identical == expected_identical,
        "the only identical pair is oracle on curated vs oracle on mixed, which must coincide: noise documents are gold for no query"
        if identical == expected_identical
        else f"identical pairs: {identical}",
    )
    # how different are the real retrievers from each other?
    rows = []
    for kb in config.kb_variant_names:
        for a, b in (("lexical", "dense"), ("lexical", "hybrid"), ("dense", "hybrid")):
            la, lb = dict(by_config[(kb, a)]), dict(by_config[(kb, b)])
            common = [q for q in la if q in lb]
            if not common:
                continue
            overlap = np.mean([len(set(la[q]) & set(lb[q])) / max(len(set(la[q]) | set(lb[q])), 1) for q in common])
            same_list = np.mean([la[q] == lb[q] for q in common])
            rows.append({"KB": kb, "pair": f"{a} vs {b}", "mean Jaccard overlap of top-k": overlap, "share of queries with identical list": same_list})
    report.p()
    report.table(pd.DataFrame(rows))


def check_settings(report: Report, config, records) -> None:
    report.h(3, "1.6 Model, prompt and settings held constant")
    rows = [r for r in records if r["record_type"] in ("standard", "robustness")]
    generations = [(r, r["generation"] if r["record_type"] == "standard" else r["perturbed_generation"]) for r in rows]
    models = Counter(g["returned_model"] for _, g in generations)
    report.check("the API reported the same, pinned model snapshot for every generation", set(models) == {config.generation.model}, str(dict(models)))
    rag = [(r, g) for r, g in generations if r["retrieval_method"] != "none"]
    report.check(
        "every retrieval configuration used the same prompt template (by hash)",
        {g["prompt_hash"] for _, g in rag} == {prompt_hash(config.generation.prompt_template_id)},
        f"{len(rag)} generations, template `{config.generation.prompt_template_id}`",
    )
    closed = [g for r, g in generations if r["retrieval_method"] == "none"]
    report.check("the no-retrieval baseline, and only it, used the closed-book template", all(g["prompt_hash"] == prompt_hash(CLOSED_BOOK_TEMPLATE_ID) for g in closed), f"{len(closed)} generations")
    report.check("temperature and max_tokens are the configured values on every row", {(g["temperature"], g["max_tokens"]) for _, g in generations} == {(config.generation.temperature, config.generation.max_tokens)})
    judges = Counter((r["generation_metrics"] if r["record_type"] == "standard" else r["perturbed_generation_metrics"])["judge_model"] for r in rows)
    report.check("one judge model scored every answer", set(judges) == {config.generation.effective_judge_model}, str(dict(judges)))
    seeds = [r["seed"] for r in rows]
    report.check("every row has its own seed", len(set(seeds)) == len(seeds), f"{len(set(seeds))} distinct seeds for {len(seeds)} rows")
    attempts = Counter(g["attempts"] for _, g in generations)
    report.p(f"  - generation attempts needed per answer (1 = first output was usable): {dict(sorted(attempts.items()))}")
    fingerprints = Counter(g.get("system_fingerprint") for _, g in generations)
    report.p(f"  - OpenAI system fingerprints seen (a change means OpenAI changed its serving stack mid-run): {dict(fingerprints)}")
    texts = Counter(g["raw_text"] for r, g in generations if r["record_type"] == "standard")
    repeated = sum(c for c in texts.values() if c > 1)
    report.p(f"  - standard rows whose generated text is byte-identical to another row's: {repeated} of {sum(texts.values())} (expected when two configurations retrieve the same chunks for a query)")


def check_robustness(report: Report, config, records) -> None:
    if not config.robustness.enabled:
        return
    report.h(3, "1.7 Perturbations did what they claim (recomputed from the log)")
    baselines = {r["run_key"]: r for r in records if r["record_type"] == "standard"}
    bad = Counter()
    seen = Counter()
    for r in (r for r in records if r["record_type"] == "robustness"):
        base = baselines[r["baseline_run_key"]]
        before = [c["chunk_id"] for c in base["retrieved"]]
        after = [c["chunk_id"] for c in r["perturbed_retrieved"]]
        relevant = set(base["relevant_chunk_ids"])
        protected = relevant | set(base["relevant_chunk_ids_lenient"])
        p = r["perturbation"]
        seen[p] += 1
        if p == "control_rerun":
            bad[p] += after != before
        elif p == "drop_top_relevant":
            top = next((c for c in before if c in relevant), None)
            ok = top is not None and r["changed_chunk_id"] == top and after == [c for c in before if c != top]
            ok = ok and r["perturbed_retrieval_metrics"]["recall_at_k"] < base["retrieval_metrics"]["recall_at_k"]
            bad[p] += not ok
        elif p == "inject_noise":
            replaceable = [c for c in before if c not in protected]
            ok = (
                bool(replaceable)
                and r["changed_chunk_id"] == replaceable[-1]
                and after == [r["noise_chunk_id"] if c == r["changed_chunk_id"] else c for c in before]
                and r["noise_chunk_id"] not in protected
                and r["noise_chunk_id"] not in before
                and same(r["perturbed_retrieval_metrics"]["recall_at_k"], base["retrieval_metrics"]["recall_at_k"])
            )
            bad[p] += not ok
    report.check("control re-run: context identical to the baseline", bad["control_rerun"] == 0, f"{seen['control_rerun']} rows")
    report.check("drop: exactly the highest-ranked relevant chunk is gone, the rest is unchanged and in order, recall fell", bad["drop_top_relevant"] == 0, f"{seen['drop_top_relevant']} rows")
    report.check("inject: one non-relevant chunk (the lowest-ranked) replaced in place by a chunk that is not relevant; context size and recall unchanged", bad["inject_noise"] == 0, f"{seen['inject_noise']} rows")


# -- Part 2: findings --------------------------------------------------------------------


def findings(report: Report, config, df: pd.DataFrame, rob: pd.DataFrame, n_queries: int) -> None:
    report.h(2, "Part 2 — What the results say, with uncertainty")

    report.h(3, "2.1 Retrieval: headline metrics (every query; a query the KB cannot answer scores 0)")
    summary = summarize_with_uncertainty(df, ["kb_variant", "retrieval_method"], ["coverage", *RETRIEVAL_ALL_COLUMNS, "recall_at_k", "mrr", "recall_at_k_lenient_all", "ndcg_at_k_lenient_all", *GENERATION_METRIC_COLUMNS])

    def wide(metrics: list[str], methods: list[str], kbs: list[str]) -> pd.DataFrame:
        rows = []
        for kb in kbs:
            for method in methods:
                row = {"KB": kb, "method": method}
                for metric in metrics:
                    s = summary[(summary["kb_variant"] == kb) & (summary["retrieval_method"] == method) & (summary["metric"] == metric)]
                    if s.empty or s.iloc[0]["n"] == 0:
                        row[metric] = ""
                    else:
                        s = s.iloc[0]
                        ci = "" if math.isnan(s["ci95_low"]) else f" [{s['ci95_low']:.2f}, {s['ci95_high']:.2f}]"
                        sd = "" if math.isnan(s["sd"]) else f" ± {s['sd']:.2f}"
                        row[metric] = f"{s['mean']:.3f}{sd}{ci} (n={int(s['n'])})"
                rows.append(row)
        return pd.DataFrame(rows)

    all_methods = [m for m in ["lexical", "dense", "hybrid", "oracle", "random"] if m in set(df["retrieval_method"])]
    kbs = [k for k in KB_ORDER if k in set(df["kb_variant"])]
    report.p("Cell format: mean ± SD [95% bootstrap CI] (n = queries behind the mean).")
    report.p()
    report.table(wide(["coverage", "recall_at_k_all", "mrr_all", "ndcg_at_k_all", "precision_at_k_all"], all_methods, kbs))

    report.h(3, "2.2 Retrieval: answerable queries only, and lenient labels")
    report.p("`recall_at_k` and `mrr` here are over queries that still have a relevant chunk in that KB (ranking quality with coverage held out). The `_lenient_all` columns use the looser relevance labels over every query.")
    report.p()
    report.table(wide(["recall_at_k", "mrr", "recall_at_k_lenient_all", "ndcg_at_k_lenient_all"], all_methods, kbs))

    report.h(3, "2.3 Generation")
    report.p("`abstained` and `partial` are rates. `faithfulness` is over answers that made at least one factual claim; `answer_relevance` over answers that did not abstain (see n).")
    report.p()
    gen_kbs = kbs + (["none"] if "none" in set(df["kb_variant"]) else [])
    report.table(wide(GENERATION_METRIC_COLUMNS, all_methods + ["none"], gen_kbs).query("faithfulness != '' or abstained != ''"))

    # -- bracket -------------------------------------------------------------------
    report.h(3, "2.4 Do the real configurations fall between the random and oracle baselines?")
    means = df.groupby(["kb_variant", "retrieval_method"])[["recall_at_k_all", "mrr_all", "ndcg_at_k_all", "context_relevance", "reference_agreement", "abstained"]].mean()
    violations = []
    for kb in kbs:
        full_coverage = df[df["kb_variant"] == kb]["coverage"].min() == 1.0
        for metric, higher_is_better in (("recall_at_k_all", True), ("mrr_all", True), ("ndcg_at_k_all", True), ("context_relevance", True), ("abstained", False)):
            if (kb, "oracle") not in means.index or (kb, "random") not in means.index:
                continue
            if metric == "abstained" and not full_coverage:
                continue  # the oracle has no context for unanswerable queries, so it is not a bound here (see 2.4b)
            lo, hi = means.loc[(kb, "random"), metric], means.loc[(kb, "oracle"), metric]
            if not higher_is_better:
                lo, hi = hi, lo
            for method in REAL_METHODS:
                value = means.loc[(kb, method), metric]
                if not (min(lo, hi) - 1e-9 <= value <= max(lo, hi) + 1e-9):
                    violations.append(f"{kb}/{method} {metric}={value:.3f} outside [random={means.loc[(kb, 'random'), metric]:.3f}, oracle={means.loc[(kb, 'oracle'), metric]:.3f}]")
    report.check(
        "retrieval metrics and context relevance of every real configuration lie between random and oracle; so does the abstention rate on every KB where all queries are answerable",
        not violations,
        "; ".join(violations) if violations else f"{len(kbs)} KB variants x 3 retrievers",
    )
    report.p()
    report.p("Faithfulness and reference agreement are reported but not required to be bracketed: they are conditional on the answer making factual claims, and with random context most answers abstain, leaving few (and unrepresentative) answers to score.")
    bracket_rows = []
    for kb in gen_kbs:
        for method in ["random", *REAL_METHODS, "oracle", "none"]:
            if (kb, method) in means.index:
                bracket_rows.append({"KB": kb, "method": method, **{m: means.loc[(kb, method), m] for m in means.columns}})
    report.table(pd.DataFrame(bracket_rows))

    # -- unanswerable queries ---------------------------------------------------------
    partial_kbs = [kb for kb in kbs if df[df["kb_variant"] == kb]["coverage"].min() < 1.0]
    if partial_kbs:
        report.h(3, "2.4b What happens on queries the KB cannot answer")
        report.p(
            "On a KB with missing topics the oracle is not a bound on the abstention rate: for a query whose answer was removed the oracle "
            "supplies no context and the model abstains, whereas a real retriever supplies its nearest wrong passages and the model sometimes "
            "answers from them. The table splits each configuration's queries by whether the KB still holds their answer. "
            "`answers given` = share of answers not labelled `abstained`; faithfulness and reference agreement are over the answers they are defined for (n in brackets)."
        )
        rows = []
        for kb in partial_kbs:
            for method in [*REAL_METHODS, "oracle"]:
                sub = df[(df["kb_variant"] == kb) & (df["retrieval_method"] == method)]
                for flag, name in ((1.0, "answer in KB"), (0.0, "answer NOT in KB")):
                    part = sub[sub["coverage"] == flag]
                    if part.empty:
                        continue
                    rows.append(
                        {
                            "KB": kb,
                            "method": method,
                            "queries": name,
                            "n": len(part),
                            "answers given": 1 - part["abstained"].mean(),
                            "of which partial": part["partial"].mean(),
                            "faithfulness": f"{part['faithfulness'].mean():.3f} (n={int(part['faithfulness'].count())})" if part["faithfulness"].count() else "",
                            "reference agreement": f"{part['reference_agreement'].mean():.3f} (n={int(part['reference_agreement'].count())})" if part["reference_agreement"].count() else "",
                            "context relevance": part["context_relevance"].mean(),
                        }
                    )
        report.table(pd.DataFrame(rows))

    # -- paired comparisons ----------------------------------------------------------
    report.h(3, "2.5 Which differences are real? Paired comparisons on the same queries")
    report.p("Each row compares two configurations query by query: mean difference (a − b), its 95% paired-bootstrap interval, a Wilcoxon signed-rank p-value adjusted (Holm) for the number of pairs tested for that metric, and the smallest true difference this many queries could reliably detect (80% power). **Verdict `could be noise` means the ordering of those two configurations must not be reported as a finding.**")
    comparisons = paired_comparisons(df[df["retrieval_method"].isin(REAL_METHODS)], COMPARED_METRICS)
    for family, title in (("methods_within_kb", "Retrieval methods compared within each KB variant"), ("kbs_within_method", "KB variants compared within each retrieval method")):
        report.p()
        report.p(f"**{title}**")
        report.p()
        sub = comparisons[comparisons["family"] == family][["held_fixed", "metric", "a", "b", "n_pairs", "mean_a", "mean_b", "mean_diff", "ci95_low", "ci95_high", "p_holm", "min_detectable_diff", "verdict"]]
        report.table(sub)
    verdicts = Counter(comparisons["verdict"])
    report.p()
    report.p(f"Totals: {dict(verdicts)}.")

    # -- power -----------------------------------------------------------------------
    report.h(3, "2.6 Are there enough queries?")
    report.p(f"The sweep used {n_queries} queries. For each metric, the table gives the typical (median) spread of per-query paired differences between two real configurations, the smallest difference detectable at that spread with this many pairs, and how many queries would be needed to detect a difference of 0.05 or 0.10.")
    rows = []
    for metric in COMPARED_METRICS:
        sub = comparisons[(comparisons["metric"] == metric) & comparisons["sd_diff"].notna()]
        if sub.empty:
            continue
        sd = float(sub["sd_diff"].median())
        n = int(sub["n_pairs"].median())
        rows.append(
            {
                "metric": metric,
                "median n pairs": n,
                "median SD of paired difference": sd,
                "detectable difference at this n": float(sub["min_detectable_diff"].median()),
                "queries needed to detect 0.05": math.ceil(required_pairs(sd, 0.05)) if sd > 0 else 0,
                "queries needed to detect 0.10": math.ceil(required_pairs(sd, 0.10)) if sd > 0 else 0,
            }
        )
    report.table(pd.DataFrame(rows))

    # -- robustness ------------------------------------------------------------------
    if not rob.empty:
        report.h(3, "2.7 Robustness: change after perturbing the context, against the no-change control")
        report.p("`control_rerun` regenerates on the identical context, so its deltas are pure generation/judge noise. A perturbation effect is only credible where its interval excludes 0 **and** it is clearly larger than the control's.")
        rsum = summarize_with_uncertainty(rob, ["kb_variant", "perturbation"], [c for c in rob.columns if c.endswith("_delta")])
        rows = []
        for (kb, perturbation), group in rsum.groupby(["kb_variant", "perturbation"]):
            row = {"KB": kb, "condition": perturbation}
            for _, s in group.iterrows():
                if s["n"] == 0:
                    row[s["metric"]] = ""
                else:
                    ci = "" if math.isnan(s["ci95_low"]) else f" [{s['ci95_low']:+.2f}, {s['ci95_high']:+.2f}]"
                    row[s["metric"]] = f"{s['mean']:+.3f}{ci} (n={int(s['n'])})"
            rows.append(row)
        report.table(pd.DataFrame(rows))

        report.p()
        report.p("Paired test of each perturbation against the control, per query (delta under perturbation − delta under control):")
        rows = []
        for kb in sorted(set(rob["kb_variant"])):
            control = rob[(rob["kb_variant"] == kb) & (rob["perturbation"] == "control_rerun")].set_index("query_id")
            for perturbation in ("drop_top_relevant", "inject_noise"):
                treated = rob[(rob["kb_variant"] == kb) & (rob["perturbation"] == perturbation)].set_index("query_id")
                common = treated.index.intersection(control.index)
                for metric in ("abstained_delta", "faithfulness_delta", "answer_relevance_delta", "context_relevance_delta", "reference_agreement_delta"):
                    result = paired_comparison(treated.loc[common, metric].to_numpy(), control.loc[common, metric].to_numpy())
                    if result.n_pairs < 2:
                        continue
                    noise = result.ci_low <= 0 <= result.ci_high or result.wilcoxon_p >= 0.05
                    rows.append({"KB": kb, "perturbation": perturbation, "metric": metric, "n_pairs": result.n_pairs, "effect beyond control": result.mean_diff, "ci95_low": result.ci_low, "ci95_high": result.ci_high, "wilcoxon_p": result.wilcoxon_p, "verdict": "could be noise" if noise else "difference"})
        report.table(pd.DataFrame(rows))


def variance_section(report: Report, variance_dir: Path) -> None:
    files = sorted(variance_dir.glob("repeat_*.jsonl")) if variance_dir.exists() else []
    if len(files) < 2:
        return
    report.h(3, "2.8 Run-to-run variance (same configuration, same queries, different generation seed)")
    frames = []
    for i, path in enumerate(files):
        frame = load_standard_results(path)
        frame["repeat"] = i
        frames.append(frame)
    data = pd.concat(frames)
    n_queries = data["query_id"].nunique()
    report.p(f"{len(files)} repeats of {n_queries} queries per configuration. Retrieval is identical across repeats (checked below); only generation and judging vary.")
    retrieval_cols = ["recall_at_k_all", "mrr_all", "ndcg_at_k_all"]
    spread = data.groupby(["kb_variant", "retrieval_method", "query_id"])[retrieval_cols].nunique().max().max()
    report.check("retrieval metrics are identical in every repeat", spread == 1)
    rows = []
    for (kb, method), group in data.groupby(["kb_variant", "retrieval_method"]):
        for metric in ("faithfulness", "abstained", "answer_relevance", "context_relevance", "reference_agreement"):
            per_repeat_mean = group.groupby("repeat")[metric].mean()
            per_query_sd = group.groupby("query_id")[metric].std(ddof=1)
            flips = (group.groupby("query_id")[metric].nunique(dropna=False) > 1).mean()
            rows.append(
                {
                    "KB": kb,
                    "method": method,
                    "metric": metric,
                    "mean over repeats": per_repeat_mean.mean(),
                    "SD of the config mean across repeats": per_repeat_mean.std(ddof=1),
                    "min repeat mean": per_repeat_mean.min(),
                    "max repeat mean": per_repeat_mean.max(),
                    "median per-query SD": per_query_sd.median(),
                    "share of queries whose value changed": flips,
                }
            )
    report.table(pd.DataFrame(rows))
    report.p()
    report.p("Reading this: a difference between two configurations in Part 2.5 that is smaller than about twice the `SD of the config mean across repeats` for that metric could be produced by re-running the same configuration.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a sweep's results log and write a report.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/validation/data_validation.md"))
    parser.add_argument("--variance-dir", type=Path, default=Path("results/variance"))
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    queries = load_query_set_jsonl(config.query_set_path)
    report = Report()
    report.lines.append("# Data validation report")
    report.p()
    report.p(f"Results log: `{config.results_path.as_posix()}`. Config: `{args.config.as_posix()}`. Produced by `python -m scripts.validate_results`; every number below is computed from the results log.")

    report.h(2, "Part 1 — Is the data sound?")
    records = read_records(config.results_path)  # raises on unreadable lines or duplicate keys
    check_integrity(report, config, records, queries)
    check_row_counts(report, config, records, queries)
    check_values(report, config, records)
    check_recomputed(report, config, records)
    check_retrieval_lists(report, config, records)
    check_settings(report, config, records)
    check_robustness(report, config, records)

    df = load_standard_results(config.results_path)
    rob = load_robustness_results(config.results_path)
    findings(report, config, df, rob, len(queries))
    variance_section(report, args.variance_dir)

    report.lines.insert(3, "")
    report.lines.insert(4, f"**Data checks: {'ALL PASS' if not report.failures else str(len(report.failures)) + ' FAILED'}.**" + ("" if not report.failures else " Failed: " + "; ".join(report.failures)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report.lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.out}: {'all data checks pass' if not report.failures else str(len(report.failures)) + ' check(s) FAILED: ' + '; '.join(report.failures)}")
    if report.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Run everything deterministic and print one pass/fail summary.

    python -m scripts.verify_all                  # tests + coverage + results validation
    python -m scripts.verify_all --with-mutation  # also the mutation run (slow, ~1 hour)

Steps:
  1. the full default test suite (fast tests + real-data tests), with line and
     branch coverage. A SKIPPED real-data test counts as a failure here: a check
     that did not run has not verified anything.
  2. validation of the results log (scripts.validate_results), if a results
     file exists for the config.
  3. optionally, mutation testing (scripts.mutation_test).

Nothing here calls the OpenAI API. The real-API tests are run separately with
`python -m pytest --run-api`.

Exit code 0 only if every step passed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from experiments.config import load_sweep_config
from experiments.runner import code_version, source_tree_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_REPORT = REPO_ROOT / "reports" / "validation" / "coverage.txt"


def run(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all deterministic verification and summarise.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_sweep.yaml"))
    parser.add_argument("--with-mutation", action="store_true")
    args = parser.parse_args()

    steps: list[tuple[str, bool, str]] = []
    started = time.time()

    print("[1/3] test suite with coverage ...", flush=True)
    code, out = run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", "--cov", "--cov-branch", "--cov-report=term"]
    )
    summary_line = next((line for line in reversed(out.splitlines()) if re.search(r"\d+ passed", line)), "no pytest summary found")
    counts = {name: int(n) for n, name in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", summary_line)}
    skipped_reasons = [line for line in out.splitlines() if line.startswith("SKIPPED")]
    unexpected_skips = [line for line in skipped_reasons if "pass --run-api" not in line]
    tests_ok = code == 0 and counts.get("failed", 0) == 0 and not counts.get("error", 0) and not counts.get("errors", 0)
    steps.append(("tests pass", tests_ok, summary_line.strip("= ")))
    steps.append(
        (
            "no real-data test was skipped",
            not unexpected_skips,
            "only the paid real-API tests were skipped (run them with --run-api)" if not unexpected_skips else f"{len(unexpected_skips)} skipped: {unexpected_skips[0][:160]}",
        )
    )
    coverage_lines = [line for line in out.splitlines() if re.match(r"^(Name|-+|TOTAL|\S+\.py)\s", line)]
    total = next((line for line in coverage_lines if line.startswith("TOTAL")), None)
    if total:
        COVERAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        COVERAGE_REPORT.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")
        # TOTAL  stmts  miss  branch  brpart  cover
        parts = total.split()
        stmts, miss, branches, partial = (int(x) for x in parts[1:5])
        line_cov = 100 * (stmts - miss) / stmts if stmts else 0.0
        steps.append(("coverage measured", True, f"line {line_cov:.1f}% ({stmts - miss}/{stmts}), line+branch combined {parts[5]}, {branches} branches of which {partial} partially covered; per-file table in {COVERAGE_REPORT.relative_to(REPO_ROOT).as_posix()}"))
    else:
        steps.append(("coverage measured", False, "pytest-cov output not found (pip install pytest-cov)"))

    print("[2/3] results validation ...", flush=True)
    config = load_sweep_config(args.config)
    if config.results_path.exists():
        code, out = run([sys.executable, "-m", "scripts.validate_results", "--config", str(args.config)])
        steps.append(("results log validates", code == 0, out.strip().splitlines()[-1] if out.strip() else "no output"))
    else:
        steps.append(("results log validates", False, f"{config.results_path} does not exist yet: run the sweep first"))

    if args.with_mutation:
        print("[3/3] mutation testing (slow) ...", flush=True)
        code, out = run([sys.executable, "-m", "scripts.mutation_test"])
        score = next((line for line in out.splitlines() if line.startswith("Mutation score")), "no mutation score found")
        steps.append(("every injected bug is caught", code == 0, score))
    else:
        print("[3/3] mutation testing skipped (pass --with-mutation)", flush=True)

    width = max(len(name) for name, _, _ in steps)
    print("\n" + "=" * 78)
    print(f"VERIFY  code {code_version()}  source tree {source_tree_sha256()}  ({time.time() - started:.0f}s)")
    print("=" * 78)
    for name, ok, detail in steps:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    print("=" * 78)
    all_ok = all(ok for _, ok, _ in steps)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

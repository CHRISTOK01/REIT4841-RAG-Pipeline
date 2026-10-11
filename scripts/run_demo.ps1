# Run the small demo sweep end-to-end and show what it produced.
# Usage (from the repo root, in PowerShell):
#   .\scripts\run_demo.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n=== 1/4: running tests (no API calls) ===`n" -ForegroundColor Cyan
python -m pytest -q

Write-Host "`n=== 2/4: running demo sweep (real API calls, a few cents) ===`n" -ForegroundColor Cyan
python -m experiments.run_sweep --config configs/demo_sweep.yaml

Write-Host "`n=== 3/4: exporting CSVs + figures ===`n" -ForegroundColor Cyan
python -m analysis.export_csv --results results/demo_sweep.jsonl --out-dir reports/demo

Write-Host "`n=== 4/4: sample result (dense retrieval) ===`n" -ForegroundColor Cyan
python scripts/show_demo_result.py --results results/demo_sweep.jsonl --method dense

Write-Host "`nDone. Full results: results\demo_sweep.jsonl" -ForegroundColor Green
Write-Host "Aggregated table:   reports\demo\results_aggregated.csv" -ForegroundColor Green
Write-Host "Figures:            reports\demo\figures\" -ForegroundColor Green

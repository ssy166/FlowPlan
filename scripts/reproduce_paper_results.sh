#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-reports}"

run_reports() {
  mkdir -p results data/processed/compute_efficiency

  python scripts/write_main_experiment_results.py \
    --root . \
    --out-md results/MAIN_EXPERIMENT_RESULTS.md \
    --out-json results/MAIN_EXPERIMENT_RESULTS.json

  python scripts/write_ablation_cost_report.py \
    --root . \
    --out-md results/ABLATION_PARAMETER_COSTS.md \
    --out-json results/ABLATION_PARAMETER_COSTS.json

  python scripts/write_significance_report.py \
    --root . \
    --out-md results/SIGNIFICANCE_REPORT.md \
    --out-json results/SIGNIFICANCE_REPORT.json

  python scripts/write_unseen_tool_generalization_report.py \
    --root . \
    --split all \
    --mode native \
    --out-md results/UNSEEN_TOOL_GENERALIZATION.native.all.md \
    --out-json results/UNSEEN_TOOL_GENERALIZATION.native.all.json

  python scripts/write_compute_efficiency_report.py \
    --root . \
    --out-dir data/processed/compute_efficiency
}

case "$MODE" in
  reports)
    run_reports
    ;;
  compute)
    mkdir -p data/processed/compute_efficiency
    python scripts/write_compute_efficiency_report.py \
      --root . \
      --out-dir data/processed/compute_efficiency
    ;;
  main)
    mkdir -p results
    python scripts/write_main_experiment_results.py \
      --root . \
      --out-md results/MAIN_EXPERIMENT_RESULTS.md \
      --out-json results/MAIN_EXPERIMENT_RESULTS.json
    ;;
  *)
    echo "Usage: bash scripts/reproduce_paper_results.sh [reports|compute|main]" >&2
    exit 2
    ;;
esac

echo "Done: $MODE"

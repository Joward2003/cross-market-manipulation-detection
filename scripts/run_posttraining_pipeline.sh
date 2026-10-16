#!/usr/bin/env bash
# Reproduce all post-training analyses from the 15 frozen OpenBayes runs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/posttraining_analysis.yaml}"
WINDOWSHAP_DEVICE="${WINDOWSHAP_DEVICE:-auto}"

echo "[1/8] Difficult same-period controls"
"$PYTHON_BIN" scripts/build_difficult_controls.py --config "$CONFIG"

echo "[2/8] Full 15-run and three-seed inference"
"$PYTHON_BIN" scripts/analyze_completed_runs.py --config "$CONFIG"

echo "[3/8] Same-stock time-placebo tests"
"$PYTHON_BIN" scripts/run_time_placebo.py --config "$CONFIG"

echo "[4/8] Start/end-aligned event-time trajectories"
"$PYTHON_BIN" scripts/build_event_time_trajectories.py --config "$CONFIG"

echo "[5/8] Fixed WindowSHAP on frozen representative models"
"$PYTHON_BIN" scripts/run_fixed_windowshap.py --config "$CONFIG" --device "$WINDOWSHAP_DEVICE"

echo "[6/8] Publication figures"
"$PYTHON_BIN" scripts/plot_posttraining_results.py --config "$CONFIG"

echo "[7/8] Paper tables, result digest, and SHA-256 manifest"
"$PYTHON_BIN" scripts/finalize_posttraining_archive.py --config "$CONFIG"

echo "[8/8] Archive validation"
"$PYTHON_BIN" scripts/validate_posttraining_archive.py --config "$CONFIG"

echo "Post-training pipeline complete. See results/final_20260819/README.md"

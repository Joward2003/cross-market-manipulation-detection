#!/usr/bin/env bash
# Label-free cross-market training. Run this from the repository root on Ubuntu.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/model_training_memory_efficient.yaml}"
RUN_ROOT="${RUN_ROOT:-runs}"

"${PYTHON_BIN}" scripts/verify_cuda.py

if [[ ! -f data/analysis/sequence_store/metadata.json ]]; then
  "${PYTHON_BIN}" scripts/build_sequence_store.py --config "${CONFIG}"
fi

# Cross-market models always use an explicit domain-alignment objective.  Plain
# direct pooling is intentionally excluded: it cannot distinguish a genuine
# cross-market representation benefit from a change in sample composition.
EXPERIMENTS="${EXPERIMENTS:-a_only a_us_coral a_hk_coral a_hk_us_coral a_us_dann a_hk_dann a_hk_us_dann}"
SEEDS="${SEEDS:-20260727 20260728 20260729}"
for DATASET in ${EXPERIMENTS}; do
  # shellcheck disable=SC2086
  for SEED in ${SEEDS}; do
    RUN_DIR="${RUN_ROOT}/${DATASET}/seed_${SEED}"
    "${PYTHON_BIN}" scripts/train_cross_market.py \
      --config "${CONFIG}" --dataset "${DATASET}" --seed "${SEED}" --run-dir "${RUN_DIR}" --resume
    "${PYTHON_BIN}" scripts/score_anomalies.py \
      --checkpoint "${RUN_DIR}/best.pt" --output-dir "${RUN_DIR}/scores"
    "${PYTHON_BIN}" scripts/evaluate_regulatory_events.py \
      --scores "${RUN_DIR}/scores/scores_with_percentile.parquet" --output-dir "${RUN_DIR}/event_evaluation"
  done
done

#!/bin/bash
# ============================================================
# CMNIST Baseline (Phase 1) — GMOE leave-one-out across 3 envs
#
# Usage:
#   # Run all envs (0, 1, 2):
#   bash scripts/train_cmnist.sh
#
#   # Run a single env only:
#   ENVS=0 bash scripts/train_cmnist.sh
#
#   # Override conda env:
#   CONDA_ENV=myenv bash scripts/train_cmnist.sh
# ============================================================

set -e

DATA_DIR="${DATA_DIR:-/home/hungnt/hungnt/DG_MoE/Generalizable-Mixture-of-Experts/domainbed/data}"
CONDA_ENV="${CONDA_ENV:-gmoe}"
ENVS="${ENVS:-0 1 2}"

LOG_DIR="logs/cmnist_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

for ENV in $ENVS; do
    LOG_FILE="$LOG_DIR/env${ENV}.log"
    echo "============================================================"
    echo "Training CMNIST env=${ENV}  (test on env${ENV}, train on others)"
    echo "  log → $LOG_FILE"

    conda run -n "$CONDA_ENV" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m domainbed.scripts.train \
        --dataset ColoredMNIST \
        --algorithm GMOE \
        --test_envs $ENV \
        --data_dir "$DATA_DIR" \
        --output_dir "train_output/cmnist_env${ENV}" \
        --hparams_seed 0 \
        --batch_size "${BATCH_SIZE:-16}" \
        2>&1 | tee "$LOG_FILE" | grep -E "Epoch|step|acc|loss|Saving|ERROR|Error|Traceback" || true

    echo "Done env=${ENV}"
    echo ""
done

echo "============================================================"
echo "CMNIST baseline training complete."
echo "Checkpoints: train_output/cmnist_env{0,1,2}/model.pkl"
echo "============================================================"

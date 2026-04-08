#!/bin/bash
# ============================================================
# PACS GMoE+OMoE training (Phase 1)
#
# Trains ViT-S/16 + GMoE with OMoE (Gram-Schmidt orthogonalization)
# at MoE blocks 8 and 10. Default leave-one-out: test_env=0 (Art_painting),
# train on Cartoon+Photo+Sketch.
#
# Usage:
#   bash scripts/phase1/train_pacs_gmoe_omoe.sh
#
#   # Override test env (0=A, 1=C, 2=P, 3=S):
#   TEST_ENV=2 bash scripts/phase1/train_pacs_gmoe_omoe.sh
#
#   # Override conda env / batch size / steps:
#   CONDA_ENV=myenv BATCH_SIZE=24 STEPS=5000 bash scripts/phase1/train_pacs_gmoe_omoe.sh
#
#   # Train apples-to-apples baseline (same setup, OMoE off):
#   USE_OMOE=false bash scripts/phase1/train_pacs_gmoe_omoe.sh
#
# Output checkpoint:
#   train_output/phase1/pacs_omoe_env${TEST_ENV}/model.pkl       (default)
#   train_output/phase1/pacs_baseline_env${TEST_ENV}/model.pkl   (when USE_OMOE=false)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/domainbed/data}"
CONDA_ENV="${CONDA_ENV:-gmoe}"
TEST_ENV="${TEST_ENV:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-5000}"
USE_OMOE="${USE_OMOE:-true}"

# Validate data dir contains PACS — fail fast with a clear message instead of
# letting train.py blow up with a cryptic FileNotFoundError.
if [ ! -d "${DATA_DIR}/PACS" ]; then
    echo "ERROR: PACS dataset not found at ${DATA_DIR}/PACS"
    echo "  Resolved DATA_DIR=${DATA_DIR}"
    echo "  REPO_ROOT=${REPO_ROOT}"
    echo "  Hint: unset stale DATA_DIR env var, or pass DATA_DIR=<path> explicitly."
    echo "  e.g.  DATA_DIR=${REPO_ROOT}/domainbed/data bash $0"
    exit 1
fi

if [ "$USE_OMOE" = "true" ]; then
    RUN_TAG="pacs_omoe_env${TEST_ENV}"
else
    RUN_TAG="pacs_baseline_env${TEST_ENV}"
fi
OUTPUT_DIR="train_output/phase1/${RUN_TAG}"
LOG_DIR="logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/env${TEST_ENV}.log"

# MoE hparams. gate_k=2 is required for Gram-Schmidt to be a non-trivial
# operation (k=1 makes the GS loop body never execute). For the baseline
# (use_omoe=false) we keep all OTHER hparams identical to the OMoE run so the
# only difference is the Gram-Schmidt step itself — apples-to-apples.
HPARAMS="{
  \"use_omoe\": ${USE_OMOE},
  \"gate_k\": 2,
  \"num_experts\": 6,
  \"expert_depth\": 2,
  \"mlp_ratio\": 4.0,
  \"model\": \"deit_small_patch16_224\"
}"

echo "============================================================"
echo "PACS GMoEOMoE training (use_omoe=${USE_OMOE})"
echo "  test_env  = ${TEST_ENV}  (0=A, 1=C, 2=P, 3=S)"
echo "  batch     = ${BATCH_SIZE}"
echo "  steps     = ${STEPS}"
echo "  output    = ${OUTPUT_DIR}"
echo "  log       = ${LOG_FILE}"
echo "============================================================"

conda run -n "$CONDA_ENV" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    python -u -m domainbed.scripts.train \
    --dataset PACS \
    --algorithm GMoEOMoE \
    --test_envs "$TEST_ENV" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --hparams "$HPARAMS" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --hparams_seed 0 \
    --seed 0 \
    2>&1 | tee "$LOG_FILE"

echo "============================================================"
echo "Done. Checkpoint: ${OUTPUT_DIR}/model.pkl"
echo "============================================================"

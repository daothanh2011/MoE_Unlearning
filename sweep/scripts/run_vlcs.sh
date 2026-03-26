#!/bin/bash
# ============================================================
# GMoE Expert Sweep — VLCS (4 environments)
#
# Usage:
#   bash sweep/scripts/run_vlcs.sh
#   PARALLEL=4 bash sweep/scripts/run_vlcs.sh
#   DRY_RUN=1  bash sweep/scripts/run_vlcs.sh
#   RESUME=1   bash sweep/scripts/run_vlcs.sh
#   FILTER="num_experts=6" bash sweep/scripts/run_vlcs.sh
# ============================================================

set -e

CONDA_ENV="${CONDA_ENV:-gmoe}"
PARALLEL="${PARALLEL:-1}"
FILTER="${FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

EXTRA_FLAGS=""
[ "$DRY_RUN" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --dry-run"
[ "$RESUME"  = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --resume"
[ -n "$FILTER" ]     && EXTRA_FLAGS="$EXTRA_FLAGS --filter $FILTER"

conda run -n "$CONDA_ENV" python sweep/run_sweep.py \
    --dataset VLCS \
    --parallel "$PARALLEL" \
    $EXTRA_FLAGS

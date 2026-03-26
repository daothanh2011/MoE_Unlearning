#!/bin/bash
# ============================================================
# Phase 2 ablation sweep — k (clusters) × x (samples/cluster)
# across all 4 Terra Incognita leave-one-out experiments.
#
# Usage:
#   # Run all envs, all k/x combos:
#   bash scripts/phase2/ablation_sweep.sh
#
#   # Run a single env only:
#   ENV=0 bash scripts/phase2/ablation_sweep.sh
#
#   # Skip embedding extraction (re-use saved embeddings):
#   SKIP_EXTRACT=1 bash scripts/phase2/ablation_sweep.sh
#
#   # Skip clustering + D-hat sampling too:
#   SKIP_EXTRACT=1 SKIP_CLUSTER=1 bash scripts/phase2/ablation_sweep.sh
# ============================================================

set -e  # exit on first error

# ── Configurable via environment variables ───────────────────
CONFIG="${CONFIG:-configs/phase2_config.yaml}"
CONDA_ENV="${CONDA_ENV:-gmoe}"

# Which environments to sweep (space-separated, default all 4)
ENVS="${ENVS:-0 1 2 3}"

# Ablation grid
K_VALUES="${K_VALUES:-5 7 10 13 15}"
X_VALUES="${X_VALUES:-5 10 20 30 50}"

# Skip flags (set to 1 to re-use saved artifacts)
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_CLUSTER="${SKIP_CLUSTER:-0}"

# ── Build optional flags ─────────────────────────────────────
EXTRA_FLAGS=""
if [ "$SKIP_EXTRACT" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip_extract"
fi
if [ "$SKIP_CLUSTER" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip_cluster"
fi

# ── Logging ──────────────────────────────────────────────────
LOG_DIR="logs/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

TOTAL=$(echo "$ENVS" | wc -w)
TOTAL=$((TOTAL * $(echo "$K_VALUES" | wc -w) * $(echo "$X_VALUES" | wc -w)))
DONE=0

# ── Sweep ────────────────────────────────────────────────────
for ENV in $ENVS; do
    # First run (k=first, x=first): extract embeddings unless skipped
    FIRST_IN_ENV=1

    for K in $K_VALUES; do
        for X in $X_VALUES; do
            DONE=$((DONE + 1))
            LOG_FILE="$LOG_DIR/env${ENV}_k${K}_x${X}.log"

            # After first run per env, always skip extraction
            RUN_FLAGS="$EXTRA_FLAGS"
            if [ "$SKIP_EXTRACT" != "1" ] && [ "$FIRST_IN_ENV" = "0" ]; then
                RUN_FLAGS="$RUN_FLAGS --skip_extract"
            fi
            FIRST_IN_ENV=0

            echo "──────────────────────────────────────────────"
            echo "[$DONE/$TOTAL] env=$ENV  k=$K  x=$X"
            echo "  log → $LOG_FILE"

            conda run -n "$CONDA_ENV" python -m domainbed.scripts.phase2_finetune \
                --config "$CONFIG" \
                --env "$ENV" \
                --k   "$K" \
                --x   "$X" \
                $RUN_FLAGS \
                2>&1 | tee "$LOG_FILE" | grep -E "^\[|D-hat total|Best test_acc|saved:|ERROR|Error|Traceback"

            echo ""
        done
    done
done

echo "============================================================"
echo "Ablation sweep complete. Results in $LOG_DIR"
echo "============================================================"

# ── Summary table ────────────────────────────────────────────
echo ""
echo "Best test_acc per run:"
echo "env   k    x    best_acc"
echo "----  ---  ---  --------"
grep -h "Best test_acc" "$LOG_DIR"/*.log 2>/dev/null | \
    sed 's|.*env\([0-9]\)_k\([0-9]*\)_x\([0-9]*\).*Best test_acc.*: \([0-9.]*\).*|\1  \2  \3  \4|' || \
    awk 'FNR==1{split(FILENAME,f,"/"); split(f[length(f)],g,"[._]"); env=g[2]; k=g[3]; x=g[4]}
         /Best test_acc/{print env, k, x, $NF}' "$LOG_DIR"/*.log 2>/dev/null || true

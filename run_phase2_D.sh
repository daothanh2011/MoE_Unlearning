#!/bin/bash
# Phase 2 — Fix D: rebalance Lo + Lv contributions
# No code changes needed — hyperparams only
# Run sequentially on single GPU

PY="/home/hungnt/anaconda3/envs/gmoe/bin/python3"
DATA="/media/hungnt/domainbed/data/"
BASE="--dataset TerraIncognita --algorithm GMOE --test_envs 0 --data_dir $DATA --steps 5000"
cd /home/hungnt/hungnt/DG_MoE/Generalizable-Mixture-of-Experts

echo "=== D1: k=2, β=1e-4, γ=2e-3 (balanced contributions) ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_D1 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":2e-3}' && \
echo "=== D1 done ===" && \

echo "=== D2: k=2, β=1e-4, γ=5e-3 (Lv dominates) ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_D2 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":5e-3}' && \
echo "=== D2 done ===" && \

echo "=== D3: k=2, β=5e-5, γ=1e-3 (Lo halved) ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_D3 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":5e-5,"variance_loss_weight":1e-3}' && \
echo "=== D3 done ===" && \

echo "=== D4: k=2, β=1e-4, γ=1e-2 (Lv strongly dominates) ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_D4 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":1e-2}' && \
echo "=== D4 done. Phase 2-D complete ==="

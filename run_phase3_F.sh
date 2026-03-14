#!/bin/bash
# Phase 3 — Larger experts (E > 6) and larger top-k (k > 2)
# Requires the 2-line code change in algorithms.py and hparams_registry.py (already done)
# Run sequentially on single GPU

PY="/home/hungnt/anaconda3/envs/gmoe/bin/python3"
DATA="/media/hungnt/domainbed/data/"
BASE="--dataset TerraIncognita --algorithm GMOE --test_envs 0 --data_dir $DATA --steps 5000"
cd /home/hungnt/hungnt/DG_MoE/Generalizable-Mixture-of-Experts

echo "=== F1: E=8, k=1, Lv-only ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_F1 \
  --hparams '{"num_experts":8,"moe_top_k":1,"ortho_loss_weight":0,"variance_loss_weight":1e-3}' && \
echo "=== F1 done ===" && \

echo "=== F2: E=12, k=1, Lv-only ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_F2 \
  --hparams '{"num_experts":12,"moe_top_k":1,"ortho_loss_weight":0,"variance_loss_weight":1e-3}' && \
echo "=== F2 done ===" && \

echo "=== F3: E=12, k=2, Lo-only ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_F3 \
  --hparams '{"num_experts":12,"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":0}' && \
echo "=== F3 done ===" && \

echo "=== F4: E=12, k=3, Lo-only ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_F4 \
  --hparams '{"num_experts":12,"moe_top_k":3,"ortho_loss_weight":1e-4,"variance_loss_weight":0}' && \
echo "=== F4 done ===" && \

echo "=== F5: E=16, k=3, Lo+Lv balanced ===" && \
$PY -m domainbed.scripts.train $BASE \
  --output_dir train_output/sweep_F5 \
  --hparams '{"num_experts":16,"moe_top_k":3,"ortho_loss_weight":1e-4,"variance_loss_weight":2e-3}' && \
echo "=== F5 done. Phase 3-F complete ==="

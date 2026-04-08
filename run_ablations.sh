#!/bin/bash
PY="/home/hungnt/anaconda3/envs/gmoe/bin/python3"
DATA="/media/hungnt/domainbed/data/"
COMMON="--dataset TerraIncognita --algorithm GMOE --test_envs 0 --data_dir $DATA --steps 5000"
cd /home/hungnt/hungnt/DG_MoE/Generalizable-Mixture-of-Experts

echo "=== Starting Ablation B: k=2, no losses ===" && \
$PY -m domainbed.scripts.train $COMMON \
  --output_dir train_output/ablation_B \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":0,"variance_loss_weight":0}' && \
echo "=== Ablation B done ===" && \

echo "=== Starting Ablation C: k=1, Lv only ===" && \
$PY -m domainbed.scripts.train $COMMON \
  --output_dir train_output/ablation_C \
  --hparams '{"moe_top_k":1,"ortho_loss_weight":0,"variance_loss_weight":1e-3}' && \
echo "=== Ablation C done ===" && \

echo "=== Starting Ablation E: k=2, Lo only ===" && \
$PY -m domainbed.scripts.train $COMMON \
  --output_dir train_output/ablation_E \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":0}' && \
echo "=== Ablation E done. All ablations complete ==="

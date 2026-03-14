#!/bin/bash
# Multi-dataset sweep: Baseline + D2 config on all DomainBed datasets
# Run AFTER download is complete
# D2 config: k=2, β=1e-4, γ=5e-3 (best performing combined config)
# Each dataset uses test_envs=0 as a representative run

PY="/home/hungnt/anaconda3/envs/gmoe/bin/python3"
DATA="/media/hungnt/domainbed/data/"
ALGO="--algorithm GMOE"
STEPS="--steps 5000"

cd /home/hungnt/hungnt/DG_MoE/Generalizable-Mixture-of-Experts

BASELINE='{"moe_top_k":1,"ortho_loss_weight":0,"variance_loss_weight":0}'
D2='{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":5e-3}'

# ─── PACS ──────────────────────────────────────────────────────────────────
echo "=== [PACS] Baseline ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset PACS --test_envs 0 --data_dir $DATA \
  --output_dir train_output/pacs_baseline \
  --hparams "$BASELINE" && \
echo "=== [PACS] Baseline done ===" && \

echo "=== [PACS] D2 ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset PACS --test_envs 0 --data_dir $DATA \
  --output_dir train_output/pacs_D2 \
  --hparams "$D2" && \
echo "=== [PACS] D2 done ===" && \

# ─── OfficeHome ────────────────────────────────────────────────────────────
echo "=== [OfficeHome] Baseline ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset OfficeHome --test_envs 0 --data_dir $DATA \
  --output_dir train_output/officehome_baseline \
  --hparams "$BASELINE" && \
echo "=== [OfficeHome] Baseline done ===" && \

echo "=== [OfficeHome] D2 ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset OfficeHome --test_envs 0 --data_dir $DATA \
  --output_dir train_output/officehome_D2 \
  --hparams "$D2" && \
echo "=== [OfficeHome] D2 done ===" && \

# ─── VLCS ──────────────────────────────────────────────────────────────────
echo "=== [VLCS] Baseline ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset VLCS --test_envs 0 --data_dir $DATA \
  --output_dir train_output/vlcs_baseline \
  --hparams "$BASELINE" && \
echo "=== [VLCS] Baseline done ===" && \

echo "=== [VLCS] D2 ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset VLCS --test_envs 0 --data_dir $DATA \
  --output_dir train_output/vlcs_D2 \
  --hparams "$D2" && \
echo "=== [VLCS] D2 done ===" && \

# ─── DomainNet ─────────────────────────────────────────────────────────────
echo "=== [DomainNet] Baseline ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset DomainNet --test_envs 0 --data_dir $DATA \
  --output_dir train_output/domainnet_baseline \
  --hparams "$BASELINE" && \
echo "=== [DomainNet] Baseline done ===" && \

echo "=== [DomainNet] D2 ===" && \
$PY -m domainbed.scripts.train $ALGO $STEPS \
  --dataset DomainNet --test_envs 0 --data_dir $DATA \
  --output_dir train_output/domainnet_D2 \
  --hparams "$D2" && \
echo "=== [DomainNet] D2 done ===" && \

echo "=== All dataset sweeps complete ==="

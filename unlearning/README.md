# MoE Unlearning — Training & Unlearning Guide

Run all commands from the **repository root** (`MoE_Unlearning/`):

```bash
cd /path/to/MoE_Unlearning
```

Scripts in this folder:

| Script | Purpose |
|--------|---------|
| `train.py` | Train (or retrain) a model before unlearning |
| `unlearn.py` | Unlearn on a saved checkpoint |
| `analyze_routing.py` | Inspect expert routing per class/domain |
| `metrics.py` | Accuracy / MIA helpers (used by train & unlearn) |

Outputs go under `unlearning/train_output/`.

---

## Pipeline overview

```text
1. Train (modular MoE)     train.py  --algorithm GMOE_ModularLearn --train_setting origin
        ↓
   model_final.pt
        ↓
2. (Optional) Routing check   analyze_routing.py
        ↓
3. Modular unlearn          unlearn.py  --algorithm GMOE_Full_Unlearn --unlearn_algo modular
        ↓
   model_unlearned_final.pt
```

**Important:** Use the **same** `--hparams` JSON for train, unlearn, and routing analysis (model, `num_experts`, `expert_depth`, etc.). Mismatched architecture will load weights incorrectly or silently misalign layers.

---

## Data split

Both `train.py` and `unlearn.py` use the same protocol:

| Split | Fraction | Use |
|-------|----------|-----|
| Train | 80% | Full train (`origin`) or split into retain/forget |
| Test | 10% | `test_acc` (domain generalization holdout) |
| Unseen | 10% | MIA score |

**Forget / retain** (within the 80% train block):

- **`random`**: first `unlearn_random_ratio` of train indices → forget; rest → retain.
- **`class`**: samples whose label is in the first `unlearn_num_class` classes → forget.

Example (PACS, ratio 0.4): ~4796 retain, ~3196 forget.

---

## Algorithms

| `--algorithm` (train) | Description |
|---------------------|-------------|
| `GMOE_ModularLearn` | **Recommended** — \(\mathcal L_{\mathrm{learn}} = \mathcal L_{\mathrm{task}} + \lambda_{\mathrm{sp}}\mathcal L_{\mathrm{sp}} + \lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}} + \lambda_{\mathrm{div}}\mathcal L_{\mathrm{div}}\) |
| `GMOE_Full` | Task + domain invariance (OT/MMD/…) + modular aux losses |
| `ERM` | Baseline (no MoE) |

| `--algorithm` (unlearn) | Description |
|-------------------------|-------------|
| `GMOE_Full_Unlearn` | Required for `--unlearn_algo modular` (MoE head + routing) |

| `--unlearn_algo` | Description |
|------------------|-------------|
| `modular` | Expert-level unlearning (forget + retain + distill) |
| `finetune`, `ga`, `rl`, … | Other baselines (see `unlearn.py` choices) |

---

## Hyperparameters (`--hparams` JSON)

Pass a single JSON string. It is merged into DomainBed defaults from `domainbed/hparams_registry.py`.

### Architecture (must match train ↔ unlearn)

| Key | Example | Meaning |
|-----|---------|---------|
| `model` | `deit_tiny_patch16_224` | ViT backbone |
| `num_experts` | `12` | Number of experts \(M\) |
| `mlp_ratio` | `4` | Expert MLP width ratio |
| `expert_depth` | `2` | Expert MLP depth |
| `gate_k` | `4` | **Ignored** for soft routing (logged warning only) |

### Modular **training** (`GMOE_ModularLearn`)

| Key | Default (class) | Meaning |
|-----|-----------------|--------|
| `lambda_sp` | `0.001` | Weight on routing sparsity (entropy) |
| `lambda_bal` | `0.1` | Weight on load balancing |
| `lambda_div` | `0.01` | Weight on expert diversity |
| `motion_fro_normalize` | `true` | Frobenius-normalized \(\mathcal L_{\mathrm{div}}\) (`motion_fro_normalize` is an accepted typo alias) |
| `use_batch_balance_loss` | `true` | Batch MSE balance (vs EMA) |
| `balance_loss_type` | `mse_switch` | `mse`, `switch`, or `mse_switch` |
| `router_temperature` | `2.0` | Softmax temperature on router logits |

**Healthy training logs:** `loss_bal` well below `0.1` (not ~`0.92`), `loss_sp` not ~`0`, experts used evenly (see routing analysis).

### Modular **unlearning** (CLI flags → hparams)

| Key / flag | Default | Meaning |
|------------|---------|---------|
| `modular_unlearn_beta` | `1.0` | \(\beta\) on retain CE |
| `modular_unlearn_gamma` | `1.0` | \(\gamma\) on KL distillation |
| `modular_unlearn_topk` | `M/4` | Select top-\(k\) experts by mean forget routing |
| `modular_unlearn_tau` | — | Alternative: experts with score \(> \tau\) |
| `modular_unlearn_lr` | training `lr` | LR for selected experts only |
| `modular_unlearn_use_modular_reg` | off | Optional extra \(\lambda_{\mathrm{div}}\mathcal L_{\mathrm{div}}\) on selected experts |
| `modular_unlearn_lambda_div` | `0.0` | Weight for optional unlearn diversity |

**Unlearn objective (default):**

\[
\mathcal L_{\mathrm{unlearn}}
= \mathcal L_{\mathrm{forget}}
+ \beta\,\mathcal L_{\mathrm{retain}}
+ \gamma\,\mathcal L_{\mathrm{distill}}
\]

- \(\mathcal L_{\mathrm{forget}} = -\mathrm{CE}(\hat y, y)\) on forget batch  
- \(\mathcal L_{\mathrm{retain}} = \mathrm{CE}(\hat y, y)\) on retain batch  
- \(\mathcal L_{\mathrm{distill}} = \mathrm{KL}(p_{\mathrm{old}}\|p_{\mathrm{new}})\) on retain vs frozen teacher  

Router, backbone, classifier, and non-selected experts are **frozen**.

### Logged metrics (unlearn)

| Metric | In loss? | Meaning |
|--------|----------|---------|
| `forget_acc` | — | Accuracy on forget test set (lower = better unlearning) |
| `retain_acc` | — | Accuracy on retain test set (keep high) |
| `test_acc` | — | Accuracy on 10% test split |
| `mia_score` | — | Loss-based MIA attack accuracy (forget vs unseen); **~0.5** = hard to tell member from non-member |
| `loss_forget` | yes | Negative CE on forget |
| `loss_retain` | yes | CE on retain |
| `loss_distill` | — | Raw KL |
| `loss_distill_weighted` | yes | \(\gamma \times\) KL |
| `loss_sp`, `loss_bal` | **no** | **Diagnostics only** (routing stats; router frozen) |
| `loss_div` | optional | Raw diversity if `--modular_unlearn_use_modular_reg` |
| `loss_div_weighted` | optional | \(\lambda_{\mathrm{div}} \times\) `loss_div` |

---

## Commands

### 1. Train — modular MoE (recommended)

```bash
export CUDA_VISIBLE_DEVICES=0

HPARAMS='{
  "model": "deit_tiny_patch16_224",
  "num_experts": 12,
  "mlp_ratio": 4,
  "expert_depth": 2,
  "lambda_sp": 0.001,
  "lambda_bal": 0.1,
  "lambda_div": 0.01,
  "div_fro_normalize": true,
  "use_batch_balance_loss": true,
  "balance_loss_type": "mse_switch",
  "router_temperature": 2.0
}'
```

Use **`div_fro_normalize`** instead of `motion_fro_normalize` in JSON (typo is auto-aliased in code).

```bash
python unlearning/train.py \
  --algorithm GMOE_ModularLearn \
  --train_setting origin \
  --dataset PACS \
  --data_dir ./domainbed/data \
  --test_envs 0 \
  --unlearn_setting random \
  --unlearn_random_ratio 0.4 \
  --seed 0 \
  --debug False \
  --steps 15000 \
  --hparams "$HPARAMS"
```

**Output directory:**

`unlearning/train_output/GMOE_ModularLearn_origin_PACS_random_0.4_seed_0/`

**Checkpoint:** `model_final.pt`

### 2. Train — full GMOE (with invariance)

```bash
python unlearning/train.py \
  --algorithm GMOE_Full \
  --train_setting origin \
  --dataset PACS \
  --data_dir ./domainbed/data \
  --test_envs 0 \
  --unlearn_setting random \
  --unlearn_random_ratio 0.4 \
  --seed 0 \
  --debug False \
  --steps 15000 \
  --hparams '{"model":"deit_tiny_patch16_224","num_experts":12,"mlp_ratio":4,"expert_depth":2,"lambda_inv":0.001,"lambda_sp":0.01,"lambda_bal":0.01,"lambda_div":0.01,"inv_type":"OT"}'
```

### 3. Retrain on retain only (optional)

```bash
CKPT=unlearning/train_output/GMOE_ModularLearn_origin_PACS_random_0.4_seed_0/model_final.pt

python unlearning/train.py \
  --algorithm GMOE_ModularLearn \
  --train_setting retrained \
  --dataset PACS \
  --data_dir ./domainbed/data \
  --test_envs 0 \
  --unlearn_setting random \
  --unlearn_random_ratio 0.4 \
  --seed 0 \
  --debug False \
  --steps 6000 \
  --checkpoint_path "$CKPT" \
  --num_step_per_evaluate 200 \
  --hparams "$HPARAMS"
```

### 4. Analyze routing (before / after unlearn)

```bash
CKPT=unlearning/train_output/GMOE_ModularLearn_origin_PACS_random_0.4_seed_0/model_final.pt

python unlearning/analyze_routing.py \
  --algorithm GMOE_Full_Unlearn \
  --checkpoint_path "$CKPT" \
  --dataset PACS \
  --data_dir ./domainbed/data \
  --test_envs 0 \
  --unlearn_setting random \
  --unlearn_random_ratio 0.4 \
  --subsets full retain forget \
  --hparams '{"model":"deit_tiny_patch16_224","num_experts":12,"mlp_ratio":4,"expert_depth":2}'
```

CSV/heatmaps: `<ckpt_dir>/routing_analysis/`.

**Good routing:** ~`1/M` per expert on average (~0.083 for 12 experts), not one column ≈ `1.000`.

### 5. Modular unlearn

```bash
CKPT=unlearning/train_output/GMOE_ModularLearn_origin_PACS_random_0.4_seed_0/model_final.pt

python unlearning/unlearn.py \
  --algorithm GMOE_Full_Unlearn \
  --unlearn_algo modular \
  --checkpoint_path "$CKPT" \
  --dataset PACS \
  --data_dir ./domainbed/data \
  --test_envs 0 \
  --unlearn_setting random \
  --unlearn_random_ratio 0.4 \
  --seed 0 \
  --debug False \
  --steps 20000 \
  --modular_unlearn_topk 2 \
  --modular_unlearn_beta 5 \
  --modular_unlearn_gamma 5 \
  --modular_unlearn_lr 4e-5 \
  --hparams '{"model":"deit_tiny_patch16_224","num_experts":12,"mlp_ratio":4,"expert_depth":2}'
```

**Output directory:**

`unlearning/train_output/unlearn_modular_GMOE_Full_Unlearn_PACS_random_0.4_seed_0/`

**Checkpoint:** `model_unlearned_final.pt`

**Optional** diversity during unlearn (usually not needed):

```bash
  --modular_unlearn_use_modular_reg \
  --modular_unlearn_lambda_div 0.01
```

**Early stopping (MIA):** by default unlearn stops when `mia_score` is in **`[0.48, 0.52]`** (configurable). A **narrow** `[0.50, 0.51]` band often never fires on `deit_small` because early epochs sit at ~**0.48–0.49**, then later overshoot to **0.35–0.70** once `forget_acc` drops.

```bash
# Wider band (default since fix)
--mia_stop_low 0.48 --mia_stop_high 0.52

# Old narrow band
--mia_stop_low 0.50 --mia_stop_high 0.51

# Run full steps; pick checkpoint from results.jsonl by best tradeoff
--no_mia_early_stop
```

---

## CLI reference

### `train.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | `ERM` | Training algorithm |
| `--train_setting` | `origin` | `origin` = full train; `retrained` = retain only |
| `--dataset` | `RotatedMNIST` | e.g. `PACS` |
| `--data_dir` | `./domainbed/data` | Dataset root |
| `--test_envs` | `0` | Left-out domain(s) for DG |
| `--unlearn_setting` | `random` | How forget set is defined |
| `--unlearn_random_ratio` | `0.1` if random | Fraction of train → forget |
| `--unlearn_num_class` | `1` if class | Number of classes to forget |
| `--steps` | `1000000` | Training steps |
| `--seed` | `0` | Random seed |
| `--hparams` | — | JSON hyperparameters |
| `--checkpoint_path` | — | Warm-start (e.g. retrained) |
| `--num_step_per_evaluate` | 1 epoch | Eval interval (`retrained` only) |
| `--batch_size`, `--lr`, `--weight_decay` | registry | Override registry defaults |

### `unlearn.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | `ERM` | Use `GMOE_Full_Unlearn` for modular |
| `--unlearn_algo` | `finetune` | `modular` for expert unlearning |
| `--checkpoint_path` | **required** | Pretrained `.pt` state dict |
| `--steps` | `1000000` | Unlearning steps |
| `--modular_unlearn_topk` | — | Experts to edit (top-\(k\) on forget) |
| `--modular_unlearn_tau` | — | Expert threshold (alternative to top-\(k\)) |
| `--modular_unlearn_beta` | `1.0` | Retain CE weight |
| `--modular_unlearn_gamma` | `1.0` | Distillation weight |
| `--modular_unlearn_lr` | training `lr` | Expert optimizer LR |
| `--modular_unlearn_use_modular_reg` | off | Optional \(\mathcal L_{\mathrm{div}}\) |
| `--modular_unlearn_lambda_div` | `0.0` | Optional diversity weight |
| `--num_step_per_evaluate` | 1 epoch | Eval print interval |
| `--mia_stop_low` / `--mia_stop_high` | `0.48` / `0.52` | MIA early-stop band |
| `--no_mia_early_stop` | off | Run all `--steps`; choose checkpoint manually |

(Same dataset / forget / `hparams` arguments as `train.py`.)

---

## Output layout

```text
unlearning/train_output/
├── GMOE_ModularLearn_origin_PACS_random_0.4_seed_0/
│   ├── model_final.pt
│   ├── results.jsonl
│   ├── out.txt
│   ├── err.txt
│   ├── done
│   └── routing_analysis/          # after analyze_routing.py
└── unlearn_modular_GMOE_Full_Unlearn_PACS_random_0.4_seed_0/
    ├── model_unlearned_final.pt
    ├── results.jsonl
    ├── out.txt
    └── err.txt
```

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `loss_bal` → `0.92` in **training** | Expert collapse | Raise `lambda_bal`, lower `lambda_sp`, use `balance_loss_type: mse_switch`, `router_temperature: 2` |
| All routing on one expert | Same | Retrain; verify with `analyze_routing.py` |
| `loss` ~ 1e4 in **unlearn** | Old bug: huge \(\mathcal L_{\mathrm{div}}\) in unlearn loss | Update code; do **not** use large `lambda_div` without normalization; default reg is off |
| `loss_sp` / `loss_bal` = 0 in unlearn | Diagnostics only / not in loss | Expected; check printed values after latest code (~2.4 / ~0.01) |
| `forget_acc` stays high | Weak forget signal or wrong experts | More steps, higher LR, tune \(\beta,\gamma\), check expert selection |
| `retain_acc` / `test_acc` drop a lot | Too aggressive unlearn | Lower LR, raise \(\beta,\gamma\), fewer steps, smaller `topk` |
| MIA stuck at ~**0.48–0.49**, never hits 0.50–0.51 | Band too narrow; strong model still separable by loss | Use `--mia_stop_low 0.47 --mia_stop_high 0.53` or `--no_mia_early_stop` |
| MIA **0.35** or **0.65** after unlearn | Forget/unseen losses too separable (or inverted) | Stop earlier when MIA enters band; lower LR; match train/unlearn backbone |
| Train **tiny**, unlearn **small** hparams | Architecture mismatch | Same `model` in train, checkpoint, and unlearn `--hparams` |
| `TypeError: router_temperature` | Wrong `ExplicitMoEHead` import | Ensure `algorithms.py` re-imports from `gmoe_utils` after `deit_transformer` |

---

## End-to-end copy-paste (PACS, deit-tiny, 12 experts)

```bash
cd /path/to/MoE_Unlearning
export CUDA_VISIBLE_DEVICES=0

HPARAMS='{"model":"deit_tiny_patch16_224","num_experts":12,"mlp_ratio":4,"expert_depth":2,"lambda_sp":0.001,"lambda_bal":0.1,"lambda_div":0.01,"div_fro_normalize":true,"use_batch_balance_loss":true,"balance_loss_type":"mse_switch","router_temperature":2.0}'

python unlearning/train.py \
  --algorithm GMOE_ModularLearn --train_setting origin \
  --dataset PACS --data_dir ./domainbed/data --test_envs 0 \
  --unlearn_setting random --unlearn_random_ratio 0.4 --seed 0 --debug False \
  --steps 15000 --hparams "$HPARAMS"

CKPT=unlearning/train_output/GMOE_ModularLearn_origin_PACS_random_0.4_seed_0/model_final.pt

python unlearning/unlearn.py \
  --algorithm GMOE_Full_Unlearn --unlearn_algo modular \
  --checkpoint_path "$CKPT" \
  --dataset PACS --data_dir ./domainbed/data --test_envs 0 \
  --unlearn_setting random --unlearn_random_ratio 0.4 --seed 0 --debug False \
  --steps 20000 --modular_unlearn_topk 2 --modular_unlearn_beta 5 \
  --modular_unlearn_gamma 5 --modular_unlearn_lr 4e-5 \
  --hparams '{"model":"deit_tiny_patch16_224","num_experts":12,"mlp_ratio":4,"expert_depth":2}'
```


## End-to-end copy-paste (PACS, deit-tiny, 12 experts)

```bash
cd /path/to/MoE_Unlearning
export CUDA_VISIBLE_DEVICES=0

HPARAMS='{"model":"deit_small_patch16_224","num_experts":12,"mlp_ratio":4,"expert_depth":2,"lambda_sp":0.001,"lambda_bal":0.1,"lambda_div":0.01,"div_fro_normalize":true,"use_batch_balance_loss":true,"balance_loss_type":"mse_switch","router_temperature":2.0}'

python unlearning/train.py \
  --algorithm GMOE_ModularLearn --train_setting origin \
  --dataset PACS --data_dir ./domainbed/data --test_envs 0 \
  --unlearn_setting random --unlearn_random_ratio 0.4 --seed 0 --debug False \
  --steps 15000 --hparams "$HPARAMS"

CKPT=unlearning/train_output/GMOE_ModularLearn_origin_PACS_random_0.4_seed_0/model_final.pt

python3.11 unlearning/unlearn.py   --algorithm GMOE_Full_Unlearn   --unlearn_algo modular   --checkpoint_path "$CKPT"   --dataset PACS   --data_dir ./domainbed/data   --test_envs 0   --unlearn_setting random   --unlearn_random_ratio 0.4   --seed 0   --debug False   --steps 20000   --modular_unlearn_topk 2   --modular_unlearn_beta 2  --modular_unlearn_gamma 3   --modular_unlearn_lr 2e-5   --hparams "$HPARAMS" --modular_unlearn_use_modular_reg --modular_unlearn_lambda_div 0.05  --mia_stop_low 0.49   --mia_stop_high 0.51
```
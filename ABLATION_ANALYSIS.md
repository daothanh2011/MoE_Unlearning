# Ablation Analysis: GMoE + Lo/Lv Auxiliary Losses

**Dataset**: TerraIncognita, test_env=0, 5000 steps, seed=0

---

## Phase 1 Results

### Full Results Table

| ID | k | β (ortho) | γ (variance) | Peak env0_out | Step | Δ vs Baseline |
|----|---|-----------|--------------|--------------|------|--------------|
| **A** — Baseline | 1 | 0 | 0 | 0.5682 | 1500 | — |
| **B** — k=2 only | 2 | 0 | 0 | 0.6092 | 3300 | **+4.1%** |
| **C** — Lv only | 1 | 0 | 1e-3 | **0.6174** | 3300 | **+4.9%** |
| **D** — Lo+Lv (prior) | 2 | 1e-4 | 1e-3 | 0.6062 | 3900 | +3.8% |
| **E** — Lo only | 2 | 1e-4 | 0 | **0.6174** | 4500 | **+4.9%** |

### Accuracy Trajectories (env0_out_acc per step)

| Step | A (baseline) | B (k=2) | C (Lv) | D (Lo+Lv) | E (Lo) |
|------|-------------|---------|--------|-----------|--------|
| 0 | 0.0697 | 0.0615 | 0.0451 | 0.0533 | 0.0615 |
| 300 | 0.4369 | 0.4431 | 0.4533 | 0.4759 | 0.4297 |
| 600 | 0.5282 | 0.5344 | 0.5200 | 0.5344 | 0.5405 |
| 900 | 0.4164 | 0.4595 | 0.4769 | 0.5026 | 0.4687 |
| 1200 | 0.5497 | 0.5805 | 0.5282 | 0.5518 | 0.5477 |
| 1500 | **0.5682** | 0.5723 | 0.5774 | 0.5600 | 0.5764 |
| 1800 | 0.5538 | 0.5651 | 0.5087 | 0.5785 | 0.5426 |
| 2100 | 0.5395 | 0.5415 | 0.5395 | 0.5826 | 0.5467 |
| 2400 | 0.5210 | 0.5487 | 0.5241 | 0.5508 | 0.5497 |
| 2700 | 0.5159 | 0.5251 | 0.5138 | 0.5795 | 0.5077 |
| 3000 | 0.5518 | 0.5569 | 0.5662 | 0.5518 | 0.5590 |
| 3300 | 0.5662 | **0.6092** | **0.6174** | 0.5846 | 0.6051 |
| 3600 | 0.5364 | 0.5826 | 0.5405 | 0.5990 | 0.5990 |
| 3900 | 0.5159 | 0.5600 | 0.5487 | **0.6062** | 0.5518 |
| 4200 | 0.5641 | 0.5897 | 0.5621 | 0.5990 | 0.5518 |
| 4500 | 0.5641 | 0.6000 | 0.5867 | 0.5928 | **0.6174** |
| 4800 | 0.5200 | 0.5579 | 0.5487 | 0.5846 | 0.5754 |
| 4999 | 0.5128 | 0.5590 | 0.5282 | 0.5518 | 0.5651 |

---

## Analysis: What is Good, What is Bad, and Why

### B (k=2, no losses): GOOD but not best (+4.1%)

k=2 forces each token to aggregate outputs from 2 expert FFNs instead of 1. This provides:
- Richer gradient feedback to the router (2 experts are updated per token per step)
- More diverse feature aggregation in the model output

**Why only +4.1%**: Without an explicit diversity constraint, the model can still settle into lazy routing where 2 "similar" experts dominate. k=2 increases capacity but not specialization pressure. The 6 experts (E=6) remain free to learn overlapping representations.

### C (k=1, Lv only): BEST — tied at +4.9%, simplest config

Lv (variance loss) directly penalizes routing collapse. The router is forced to maintain a high-variance probability distribution over all 6 experts for each token. This means:
- Even with k=1 (only 1 expert active per token), the full pre-top-k softmax vector `[N, 6]` is subject to variance maximization
- More discriminative routing → implicit expert specialization emerges organically
- `loss_variance = -0.0055` at peak confirms the router is actively spreading probability mass

**Why this works with k=1**: Lv acts on the raw softmax scores before top-k selection. It directly attacks routing collapse — the most common failure mode of MoE models — without any architectural change.

**Key advantage**: Requires no architecture change (k stays at 1), less memory, less compute.

### D (k=2, Lo+Lv combined): BAD — 0.6062, worse than both C and E

The combined configuration underperforms its individual components (0.6062 < 0.6174).

**Root cause 1 — Gradient interference**: Lo and Lv push in partially conflicting directions:
- Lv says: "spread routing probability mass evenly → more diverse routing"
- Lo says: "route to experts with orthogonal outputs → tokens cluster near their best expert"
- When experts specialize (via Lo), each token has a clearly preferred expert → routing variance naturally *decreases*, directly opposing Lv's objective

**Root cause 2 — Loss scale imbalance**:
- β × Lo = 1e-4 × 0.12 = **1.2e-5**
- γ × |Lv| = 1e-3 × 0.0055 = **5.5e-6**
- Lo contributes ~2× more to total loss than Lv → Lv's signal is partially drowned out

### E (k=2, Lo only): BEST — tied at +4.9%, most principled for k≥2

Lo explicitly penalizes the projection of each expert's output onto other experts. With k=2:
- Each token's combined output comes from 2 structurally orthogonal expert representations
- `loss_ortho = 0.120544` at peak — non-trivial, well-scaled, confirms active orthogonalization
- The 6 experts are forced to learn distinct, non-redundant feature transformations

**Why not better than C**: Lo requires k≥2 (added complexity, more FFN compute for all E=6 experts per token), but achieves identical accuracy to Lv's simpler k=1 intervention.

---

## Conclusions

| Finding | Implication |
|---------|-------------|
| C = E > B > A | Both individual losses work; combining them (D) currently hurts |
| Lv works at k=1 | No architecture change needed for the best single-loss result |
| Lo requires k≥2 | Added complexity (more compute, more memory) with no extra accuracy gain over Lv alone |
| D < C and D < E | Interference at current weights; combination is not beneficial as configured |
| Loss scale in D | β×Lo = 1.2e-5 is only ~2× larger than γ×\|Lv\| = 5.5e-6 — fixable by adjusting γ |

**Recommended best config so far: C (k=1, β=0, γ=1e-3)** — simplest, highest accuracy, least memory overhead.

**Can D be fixed?** Yes. The ~2× imbalance is fixable. The deeper issue is gradient interference, which can be mitigated by making Lv's contribution larger (higher γ), reducing Lv's relative signal cost, or by reducing β so Lo exerts less counterpressure.

---

## Phase 2 Plan: Fix D (Rebalance Lo + Lv)

**Target**: balance contributions so β×Lo ≈ γ×|Lv|.

To equalize exactly: γ = (β × Lo) / |Lv| = (1e-4 × 0.12) / 0.0055 ≈ **2.2e-3**

| Run | k | β | γ | Hypothesis |
|-----|---|---|---|-----------|
| D1 | 2 | 1e-4 | 2e-3 | Balance Lo ≈ Lv contributions |
| D2 | 2 | 1e-4 | 5e-3 | Lv dominates slightly |
| D3 | 2 | 5e-5 | 1e-3 | Lo halved, original Lv |
| D4 | 2 | 1e-4 | 1e-2 | Lv strongly dominates |

```bash
PY="/home/hungnt/anaconda3/envs/gmoe/bin/python3"
DATA="/media/hungnt/domainbed/data/"
BASE="--dataset TerraIncognita --algorithm GMOE --test_envs 0 --data_dir $DATA --steps 5000"

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_D1 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":2e-3}'

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_D2 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":5e-3}'

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_D3 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":5e-5,"variance_loss_weight":1e-3}'

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_D4 \
  --hparams '{"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":1e-2}'
```

---

## Phase 3 Plan: Larger Experts (E > 6) and Larger Top-k (k > 2)

### Motivation

- **More experts**: finer-grained specialization, more Lo pairwise terms, larger routing distribution for Lv
- **Larger k**: richer gradient signal to router; Lo becomes more powerful (k pairwise orthogonality constraints per token)
- **Capacity constraint**: `capacity = floor(1.5 × N / E) × k` — must keep k proportional to E to avoid token overflow

| E | k | Capacity/expert | Overflow risk |
|---|---|-----------------|--------------|
| 6 | 1 | N/4 | Low |
| 6 | 2 | N/2 | Very low |
| 8 | 2 | N/2.7 | Low |
| 12 | 2 | N/4 | Low (same as E=6, k=1) |
| 12 | 3 | N/2.7 | Low |
| 16 | 3 | N/3.6 | Low |

### Code changes required (2 lines only)

**[algorithms.py](domainbed/algorithms.py) ~line 200** — change hardcoded `num_experts=6`:
```python
num_experts=int(self.hparams.get('num_experts', 6)),
```

**[hparams_registry.py](domainbed/hparams_registry.py)** — add inside GMOE block:
```python
_hparam('num_experts', 6, lambda r: r.choice([6, 8, 12, 16]))
```

No changes needed to `moe_layer.py`, `vision_transformer.py`, or loss files — they already accept `n_experts` dynamically.

### Experiment Grid

| Run | E | k | β | γ | Purpose |
|-----|---|---|---|---|---------|
| F1 | 8 | 1 | 0 | 1e-3 | Lv-only with 8 experts |
| F2 | 12 | 1 | 0 | 1e-3 | Lv-only with 12 experts |
| F3 | 12 | 2 | 1e-4 | 0 | Lo-only, 12 experts, more pairwise terms |
| F4 | 12 | 3 | 1e-4 | 0 | Lo with k=3, 3 pairwise constraints per token |
| F5 | 16 | 3 | 1e-4 | 2e-3 | Large E + large k + balanced Lo+Lv |

```bash
$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_F1 \
  --hparams '{"num_experts":8,"moe_top_k":1,"ortho_loss_weight":0,"variance_loss_weight":1e-3}'

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_F2 \
  --hparams '{"num_experts":12,"moe_top_k":1,"ortho_loss_weight":0,"variance_loss_weight":1e-3}'

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_F3 \
  --hparams '{"num_experts":12,"moe_top_k":2,"ortho_loss_weight":1e-4,"variance_loss_weight":0}'

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_F4 \
  --hparams '{"num_experts":12,"moe_top_k":3,"ortho_loss_weight":1e-4,"variance_loss_weight":0}'

$PY -m domainbed.scripts.train $BASE --output_dir train_output/sweep_F5 \
  --hparams '{"num_experts":16,"moe_top_k":3,"ortho_loss_weight":1e-4,"variance_loss_weight":2e-3}'
```

> Note: F1–F5 require the 2-line code change above (to make `num_experts` a hyperparameter). D1–D4 require **no code change**.

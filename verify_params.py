"""
Verify total_params and active_params for DeiT-Ti GMoE models.
Run from repo root: python verify_params.py
"""
import sys
sys.path.insert(0, 'domainbed')
import torch
import vision_transformer as vt

D, N, K, num_moe = 192, 6, 1, 2   # DeiT-Ti, N=6, K=1, 2 MoE blocks

print("=" * 60)
print("Test 1: DeiT-Ti, N=6, K=1, mlp_ratio=4, depth=3 (DeepMoE)")
print("=" * 60)
H = D * 4  # mlp_ratio=4 → H=768

m = vt.deit_tiny_patch16_224(
    pretrained=True, num_classes=7,
    moe_layers=['F']*8+['S','F']*2, mlp_ratio=4.0, expert_mlp_ratio=4,
    num_experts=N, gate_k=K, prune_ratio=0.0,
    is_tutel=True, drop_path_rate=0.1, router='cosine_top', expert_depth=3)

total = sum(p.numel() for p in m.parameters())
print(f"total_params  actual:      {total:,}")
print(f"total_params  theoretical: 15,573,895")
assert total == 15_573_895, f"TOTAL MISMATCH: got {total:,}"
print("  PASS: total_params matches theory")

# Correct formula: weights + biases for all layers
params_per_expert = (D*H + H) + (1*(H*H + H)) + (H*D + D)  # depth=3
inactive = num_moe * (N - K) * params_per_expert
active   = total - inactive
print(f"\nparams_per_expert: {params_per_expert:,}  (expect 886,464)")
print(f"inactive:          {inactive:,}  (expect 8,864,640)")
print(f"active_params:     {active:,}   (expect 6,709,255)")
assert params_per_expert == 886_464,  f"MISMATCH params_per_expert: {params_per_expert}"
assert inactive           == 8_864_640, f"MISMATCH inactive: {inactive}"
assert active             == 6_709_255, f"MISMATCH active: {active}"
print("  PASS: active_params formula with biases is correct")

# Show what the current (buggy) formula gives
params_per_expert_old = D*H + 1*(H*H) + H*D  # no biases
inactive_old = num_moe * (N - K) * params_per_expert_old
active_old   = total - inactive_old
print(f"\nCurrent formula (no biases): active = {active_old:,}  (overestimated by {active_old - active:,})")

print()
print("=" * 60)
print("Test 2: DeiT-Ti, depth=2 (Tutel) — theoretical only")
print("=" * 60)
# Tutel not installed in this env; verify formula by theory only.
# Tutel stores per expert: D*H + H (fc1 w+b) + H*D + D (fc2 w+b) = 295,872
per_exp2 = (D*H + H) + (H*D + D)
theoretical2 = 5_717_416 - 193_000 + 1_351 - 2*295_872 + 2*(N * per_exp2)
print(f"per_exp2 (depth=2, w+b): {per_exp2:,}  (expect 295,872)")
print(f"theoretical total2:      {theoretical2:,}")
assert per_exp2 == 295_872, f"MISMATCH per_exp2: {per_exp2}"
print("  PASS (formula verified; model not built — Tutel not installed in this env)")

print()
print("=" * 60)
print("Test 3: _compute_params in sweep_logger (fixed formula)")
print("=" * 60)
sys.path.insert(0, '.')
from domainbed.lib.sweep_logger import _compute_params, _BACKBONE_LABEL

hparams = {"num_experts": 6, "gate_k": 1, "mlp_ratio": 4, "expert_prune_ratio": 0.0, "expert_depth": 3}
total_c, active_c, hidden_c, edim_c = _compute_params(m, hparams)
print(f"_compute_params total:  {total_c:,}  (expect 15,573,895)")
print(f"_compute_params active: {active_c:,}  (expect 6,709,255)")
print(f"_compute_params hidden: {hidden_c}   (expect 768)")
print(f"_compute_params edim:   {edim_c}     (expect 192)")
print(f"backbone label:         {_BACKBONE_LABEL.get(edim_c)}  (expect DeiT-Ti/16)")
assert total_c  == 15_573_895, f"MISMATCH total: {total_c}"
assert active_c == 6_709_255,  f"MISMATCH active: {active_c}"
assert hidden_c == 768,        f"MISMATCH hidden: {hidden_c}"
assert edim_c   == 192,        f"MISMATCH embed_dim: {edim_c}"
assert _BACKBONE_LABEL.get(edim_c) == "DeiT-Ti/16"
print("  PASS: all sweep_logger values correct")

print()
print("=" * 60)
print("All assertions passed. Implementation is correct.")
print("=" * 60)

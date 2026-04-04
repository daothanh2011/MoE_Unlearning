"""
test_mlp8_depth4.py — Verify exact expected values for expert_mlp_ratio=8, expert_depth=4.

Expected:
  Expert layers:   [0] Linear(192→1536), [1] Linear(1536→1536), [2] Linear(1536→1536), [3] Linear(1536→192)
  Init layers[0]:  rows[0:768]=pretrained fc1, rows[768:]=Kaiming, bias[768:]=0
  Init layers[1]:  identity weight, zero bias
  Init layers[2]:  identity weight, zero bias
  Init layers[3]:  cols[0:768]=pretrained fc2, cols[768:]=0, bias=pretrained
  Dense blocks:    fc1=(768,192), fc2=(192,768)  — unchanged

  params_per_expert = 296,448 + 2,360,832 + 2,360,832 + 295,104 = 5,313,216
  total_params      = 68,694,919
  inactive          = 2 × 5 × 5,313,216 = 53,132,160
  active_params     = 15,562,759
"""
import sys, math, torch, torch.nn as nn
sys.path.insert(0, 'domainbed')
import vision_transformer as vt

EMBED_DIM    = 192
PRETRAINED_H = 768     # DeiT-Ti pretrained fc1 hidden
EXPERT_H     = 1536    # 192 * 8
DEPTH        = 4
NUM_EXPERTS  = 6
GATE_K       = 1
MOE_IDX      = [8, 10]
DENSE_IDX    = [i for i in range(12) if i not in MOE_IDX]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(cond, msg, got=None):
    suffix = f"  (got {got})" if got is not None else ""
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  {msg}{suffix}")
        raise AssertionError(msg + (str(suffix) if got else ""))

# Load pretrained state dict once
url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
print("Loading pretrained DeiT-Ti weights...")
sd_pre = torch.hub.load_state_dict_from_url(url, map_location='cpu')
if 'model' in sd_pre:
    sd_pre = sd_pre['model']
print("Done.\n")

# Build model
print("Building DeiT-Ti GMoE (expert_mlp_ratio=8, expert_depth=4)...")
model = vt.deit_tiny_patch16_224(
    pretrained=True, num_classes=7,
    moe_layers=['F']*8 + ['S','F']*2,
    mlp_ratio=4.0, expert_mlp_ratio=8.0,
    num_experts=NUM_EXPERTS, gate_k=GATE_K, prune_ratio=0.0,
    is_tutel=False, drop_path_rate=0.1, router='cosine_top',
    expert_depth=DEPTH)
print()

# -----------------------------------------------------------------------
print("=" * 60)
print("Test 1 — Dense blocks: fc1=(768,192), fc2=(192,768)")
print("=" * 60)
for i in DENSE_IDX:
    b = model.blocks[i]
    check(b.mlp.fc1.weight.shape == (PRETRAINED_H, EMBED_DIM),
          f"block {i} fc1 shape", b.mlp.fc1.weight.shape)
    check(b.mlp.fc2.weight.shape == (EMBED_DIM, PRETRAINED_H),
          f"block {i} fc2 shape", b.mlp.fc2.weight.shape)

# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2 — Expert layer shapes: [192→1536, 1536→1536, 1536→1536, 1536→192]")
print("=" * 60)
expected_shapes = [(EXPERT_H, EMBED_DIM), (EXPERT_H, EXPERT_H), (EXPERT_H, EXPERT_H), (EMBED_DIM, EXPERT_H)]
for i in MOE_IDX:
    for ei, expert in enumerate(model.blocks[i].mlp.experts):
        check(len(expert.layers) == DEPTH,
              f"block {i} expert {ei} depth={len(expert.layers)} (expect {DEPTH})")
        for li, (layer, exp_shape) in enumerate(zip(expert.layers, expected_shapes)):
            check(tuple(layer.weight.shape) == exp_shape,
                  f"block {i} expert {ei} layers[{li}] shape={layer.weight.shape} (expect {exp_shape})")

# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3 — layers[0]: rows[0:768]=pretrained fc1, bias[768:]=0")
print("=" * 60)
for i in MOE_IDX:
    fc1_pre = sd_pre[f'blocks.{i}.mlp.fc1.weight']  # (768, 192)
    fc1_b_pre = sd_pre[f'blocks.{i}.mlp.fc1.bias']  # (768,)
    for ei, expert in enumerate(model.blocks[i].mlp.experts):
        w0 = expert.layers[0].weight.data
        b0 = expert.layers[0].bias.data
        check(torch.allclose(w0[:PRETRAINED_H], fc1_pre, atol=1e-5),
              f"block {i} expert {ei} layers[0].weight[0:768] == pretrained fc1")
        check(b0[PRETRAINED_H:].abs().max().item() < 1e-6,
              f"block {i} expert {ei} layers[0].bias[768:] == 0",
              b0[PRETRAINED_H:].abs().max().item())
        check(torch.allclose(b0[:PRETRAINED_H], fc1_b_pre, atol=1e-5),
              f"block {i} expert {ei} layers[0].bias[0:768] == pretrained fc1 bias")
        # Extra rows should be non-zero Kaiming (not random trunc_normal garbage)
        extra_mean = w0[PRETRAINED_H:].abs().mean().item()
        check(extra_mean > 0 and extra_mean < 0.5,
              f"block {i} expert {ei} layers[0].weight[768:] is Kaiming (mean={extra_mean:.4f}, expect 0<x<0.5)")

# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4 — layers[1] and layers[2]: identity weight, zero bias")
print("=" * 60)
for i in MOE_IDX:
    for ei, expert in enumerate(model.blocks[i].mlp.experts):
        for li in [1, 2]:
            layer = expert.layers[li]
            eye = torch.eye(EXPERT_H)
            check(torch.allclose(layer.weight.data, eye, atol=1e-6),
                  f"block {i} expert {ei} layers[{li}].weight == I_{EXPERT_H}")
            check(layer.bias.data.abs().max().item() < 1e-6,
                  f"block {i} expert {ei} layers[{li}].bias == 0")

# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 5 — layers[3]: cols[0:768]=pretrained fc2, cols[768:]=0, bias=pretrained")
print("=" * 60)
for i in MOE_IDX:
    fc2_pre = sd_pre[f'blocks.{i}.mlp.fc2.weight']  # (192, 768)
    fc2_b_pre = sd_pre[f'blocks.{i}.mlp.fc2.bias']  # (192,)
    for ei, expert in enumerate(model.blocks[i].mlp.experts):
        wn = expert.layers[-1].weight.data  # (192, 1536)
        bn = expert.layers[-1].bias.data    # (192,)
        check(torch.allclose(wn[:, :PRETRAINED_H], fc2_pre, atol=1e-5),
              f"block {i} expert {ei} layers[-1].weight[:,0:768] == pretrained fc2")
        check(wn[:, PRETRAINED_H:].abs().max().item() < 1e-6,
              f"block {i} expert {ei} layers[-1].weight[:,768:] == 0",
              wn[:, PRETRAINED_H:].abs().max().item())
        check(torch.allclose(bn, fc2_b_pre, atol=1e-5),
              f"block {i} expert {ei} layers[-1].bias == pretrained fc2 bias")

# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 6 — Param counts")
print("=" * 60)
EXPECTED_PER_EXPERT = 296_448 + 2_360_832 + 2_360_832 + 295_104  # = 5,313,216
EXPECTED_TOTAL      = 68_694_919
EXPECTED_INACTIVE   = 2 * (NUM_EXPERTS - GATE_K) * EXPECTED_PER_EXPERT  # = 53,132,160
EXPECTED_ACTIVE     = EXPECTED_TOTAL - EXPECTED_INACTIVE                 # = 15,562,759

# Verify per-layer breakdown
for i in MOE_IDX:
    expert = model.blocks[i].mlp.experts[0]
    per_layer = [sum(p.numel() for p in l.parameters()) for l in expert.layers]
    check(per_layer[0] == 296_448,  f"block {i} expert 0 layers[0] params={per_layer[0]} (expect 296,448)")
    check(per_layer[1] == 2_360_832, f"block {i} expert 0 layers[1] params={per_layer[1]} (expect 2,360,832)")
    check(per_layer[2] == 2_360_832, f"block {i} expert 0 layers[2] params={per_layer[2]} (expect 2,360,832)")
    check(per_layer[3] == 295_104,  f"block {i} expert 0 layers[3] params={per_layer[3]} (expect 295,104)")
    total_exp = sum(per_layer)
    check(total_exp == EXPECTED_PER_EXPERT,
          f"block {i} expert 0 total params={total_exp:,} (expect {EXPECTED_PER_EXPERT:,})")

total = sum(p.numel() for p in model.parameters())
check(total == EXPECTED_TOTAL,
      f"total_params={total:,} (expect {EXPECTED_TOTAL:,})", total)

inactive = 2 * (NUM_EXPERTS - GATE_K) * EXPECTED_PER_EXPERT
check(inactive == EXPECTED_INACTIVE,
      f"inactive_params={inactive:,} (expect {EXPECTED_INACTIVE:,})")

active = total - inactive
check(active == EXPECTED_ACTIVE,
      f"active_params={active:,} (expect {EXPECTED_ACTIVE:,})", active)

# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 7 — Forward + backward")
print("=" * 60)
model.eval()
x = torch.randn(2, 3, 224, 224)
out = model(x)
loss = out.sum()
loss.backward()
check(out.shape == (2, 7), f"output shape={out.shape} (expect (2,7))")
check(True, "backward: no exception")

print()
print("=" * 60)
print("All tests passed.")
print("=" * 60)

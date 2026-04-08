"""
test_expert_init.py — Verify expert width/depth isolation and pretrained weight reuse.

Tests:
  1. Dense blocks (0-7, 9, 11) keep pretrained FFN size regardless of expert_mlp_ratio.
  2. MoE expert layers have correct expert_mlp_ratio-based hidden size.
  3. expert_mlp_ratio=4  → layers[0] and layers[-1] match pretrained fc1/fc2.
  4. expert_mlp_ratio=6  → layers[0][:H_pre] matches pretrained; layers[-1][:, H_pre:] ≈ 0.
  5. Middle layers are identity initialized.
  6. Forward + backward works for all configs.

Run from repo root:
  python test_expert_init.py
"""
import sys, math
sys.path.insert(0, 'domainbed')
import torch
import torch.nn as nn
import vision_transformer as vt

PRETRAINED_H = 768   # DeiT-Ti pretrained fc1 hidden = 192*4
EMBED_DIM    = 192
MLP_RATIO    = 4.0   # dense blocks always use this
NUM_CLASSES  = 7
MOE_LAYERS   = ['F']*8 + ['S','F']*2
MOE_BLOCK_IDX = [8, 10]
DENSE_BLOCK_IDX = [i for i in range(12) if i not in MOE_BLOCK_IDX]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_PRETRAINED_SD = None  # cached pretrained state dict

def _load_pretrained_sd():
    """Load DeiT-Ti pretrained state dict once and cache it."""
    global _PRETRAINED_SD
    if _PRETRAINED_SD is not None:
        return _PRETRAINED_SD
    import timm.models.hub as hub_utils
    # timm caches in ~/.cache/torch/hub/checkpoints/
    url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
    sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    if 'model' in sd:
        sd = sd['model']
    _PRETRAINED_SD = sd
    return sd

def _get_pretrained_fc_weights(block_idx):
    """Return pretrained fc1_w (H, D) and fc2_w (D, H) for given block index."""
    sd = _load_pretrained_sd()
    fc1_w = sd[f'blocks.{block_idx}.mlp.fc1.weight'].clone()  # (768, 192)
    fc2_w = sd[f'blocks.{block_idx}.mlp.fc2.weight'].clone()  # (192, 768)
    return fc1_w, fc2_w

def check(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  {msg}")
        raise AssertionError(msg)

# ---------------------------------------------------------------------------
# Test 1: Dense blocks preserve pretrained FFN shape
# ---------------------------------------------------------------------------
def test_dense_block_shape(model, label):
    print(f"\n[{label}] Test 1 — Dense blocks keep mlp_ratio=4 (hidden={PRETRAINED_H})")
    for i in DENSE_BLOCK_IDX:
        block = model.blocks[i]
        # Dense block has Mlp with fc1 and fc2
        fc1 = getattr(block.mlp, 'fc1', None)
        fc2 = getattr(block.mlp, 'fc2', None)
        if fc1 is None or fc2 is None:
            check(False, f"block {i}: no fc1/fc2 found")
            continue
        check(fc1.weight.shape == (PRETRAINED_H, EMBED_DIM),
              f"block {i} fc1.shape={fc1.weight.shape} (expect ({PRETRAINED_H},{EMBED_DIM}))")
        check(fc2.weight.shape == (EMBED_DIM, PRETRAINED_H),
              f"block {i} fc2.shape={fc2.weight.shape} (expect ({EMBED_DIM},{PRETRAINED_H}))")

# ---------------------------------------------------------------------------
# Test 2: MoE expert hidden size matches expert_mlp_ratio
# ---------------------------------------------------------------------------
def test_moe_expert_shape(model, expert_mlp_ratio, expert_depth, label):
    expected_h = int(EMBED_DIM * expert_mlp_ratio)
    print(f"\n[{label}] Test 2 — MoE expert hidden={expected_h} (expert_mlp_ratio={expert_mlp_ratio})")
    for i in MOE_BLOCK_IDX:
        block = model.blocks[i]
        mlp = block.mlp
        if hasattr(mlp, 'experts') and isinstance(mlp.experts, nn.ModuleList):
            # DeepMoELayer
            for ei, expert in enumerate(mlp.experts):
                h = expert.layers[0].weight.shape[0]
                check(h == expected_h,
                      f"block {i} expert {ei} layers[0].shape={expert.layers[0].weight.shape} (expect hidden={expected_h})")
                check(len(expert.layers) == expert_depth,
                      f"block {i} expert {ei} depth={len(expert.layers)} (expect {expert_depth})")
        elif hasattr(mlp, 'experts') and hasattr(mlp.experts, 'batched_fc1_w'):
            # Tutel
            h = mlp.experts.batched_fc1_w.shape[1]
            check(h == expected_h,
                  f"block {i} Tutel batched_fc1_w hidden={h} (expect {expected_h})")

# ---------------------------------------------------------------------------
# Test 3: expert_mlp_ratio==4 → first/last layers match pretrained fc1/fc2
# ---------------------------------------------------------------------------
def test_pretrained_copy_exact(model, label):
    print(f"\n[{label}] Test 3 — expert_mlp_ratio=4: layers[0] and layers[-1] match pretrained")
    for i in MOE_BLOCK_IDX:
        fc1_pre, fc2_pre = _get_pretrained_fc_weights(i)
        block = model.blocks[i]
        mlp = block.mlp
        if hasattr(mlp, 'experts') and isinstance(mlp.experts, nn.ModuleList):
            for ei, expert in enumerate(mlp.experts):
                w0 = expert.layers[0].weight.data   # (H, D)
                wn = expert.layers[-1].weight.data   # (D, H)
                # fc1: (H, D) == fc1_pre[:H]
                check(torch.allclose(w0, fc1_pre[:w0.shape[0]], atol=1e-5),
                      f"block {i} expert {ei} layers[0] matches pretrained fc1")
                # fc2 stored as (D, H): compare with fc2_pre[:, :H]
                check(torch.allclose(wn, fc2_pre[:, :w0.shape[0]], atol=1e-5),
                      f"block {i} expert {ei} layers[-1] matches pretrained fc2")
        elif hasattr(mlp, 'experts') and hasattr(mlp.experts, 'batched_fc1_w'):
            fc1_w = mlp.experts.batched_fc1_w   # (N, H, D)
            fc2_w = mlp.experts.batched_fc2_w   # (N, H, D) transposed
            H = fc1_w.shape[1]
            for ei in range(fc1_w.shape[0]):
                check(torch.allclose(fc1_w[ei], fc1_pre[:H], atol=1e-5),
                      f"block {i} Tutel expert {ei} batched_fc1_w matches pretrained fc1")

# ---------------------------------------------------------------------------
# Test 4: expert_mlp_ratio>4 → partial pretrained copy, new neurons near-zero
# ---------------------------------------------------------------------------
def test_pretrained_partial_copy(model, expert_mlp_ratio, label):
    expected_h = int(EMBED_DIM * expert_mlp_ratio)
    print(f"\n[{label}] Test 4 — expert_mlp_ratio={expert_mlp_ratio}: partial pretrained copy")
    for i in MOE_BLOCK_IDX:
        fc1_pre, fc2_pre = _get_pretrained_fc_weights(i)
        H_pre = fc1_pre.shape[0]  # 768
        block = model.blocks[i]
        mlp = block.mlp
        if hasattr(mlp, 'experts') and isinstance(mlp.experts, nn.ModuleList):
            for ei, expert in enumerate(mlp.experts):
                w0 = expert.layers[0].weight.data   # (expected_h, D)
                wn = expert.layers[-1].weight.data   # (D, expected_h)
                # First H_pre rows of layers[0] should match pretrained fc1
                check(torch.allclose(w0[:H_pre], fc1_pre, atol=1e-5),
                      f"block {i} expert {ei} layers[0][:H_pre] matches pretrained fc1")
                # Extra rows (H_pre:) should be small (Kaiming, not zero — just not random trunc_normal)
                extra_norm = w0[H_pre:].abs().mean().item()
                check(extra_norm < 0.5,  # Kaiming is ~0.05-0.1 for this size
                      f"block {i} expert {ei} layers[0][H_pre:] extra rows norm={extra_norm:.4f} (small Kaiming expected)")
                # First H_pre cols of layers[-1] should match pretrained fc2 (D, H) -> (D, H_pre)
                check(torch.allclose(wn[:, :H_pre], fc2_pre, atol=1e-5),
                      f"block {i} expert {ei} layers[-1][:, :H_pre] matches pretrained fc2")
                # Extra cols (H_pre:) should be zero (silent start for new neurons)
                extra_zero = wn[:, H_pre:].abs().max().item()
                check(extra_zero < 1e-6,
                      f"block {i} expert {ei} layers[-1][:, H_pre:] extra cols are zero (max={extra_zero:.2e})")
        elif hasattr(mlp, 'experts') and hasattr(mlp.experts, 'batched_fc1_w'):
            fc1_w = mlp.experts.batched_fc1_w   # (N, H, D)
            fc2_w = mlp.experts.batched_fc2_w   # (N, H, D)
            H = fc1_w.shape[1]
            for ei in range(fc1_w.shape[0]):
                check(torch.allclose(fc1_w[ei, :H_pre], fc1_pre, atol=1e-5),
                      f"block {i} Tutel expert {ei} batched_fc1_w[:H_pre] matches pretrained")
                extra_zero_fc2 = fc2_w[ei, H_pre:].abs().max().item()
                check(extra_zero_fc2 < 1e-6,
                      f"block {i} Tutel expert {ei} batched_fc2_w[H_pre:] cols zero (max={extra_zero_fc2:.2e})")

# ---------------------------------------------------------------------------
# Test 5: Middle layers are identity initialized
# ---------------------------------------------------------------------------
def test_middle_layer_identity(model, label):
    print(f"\n[{label}] Test 5 — Middle layers are identity initialized")
    for i in MOE_BLOCK_IDX:
        block = model.blocks[i]
        mlp = block.mlp
        if hasattr(mlp, 'experts') and isinstance(mlp.experts, nn.ModuleList):
            for ei, expert in enumerate(mlp.experts):
                for li, layer in enumerate(expert.layers[1:-1]):
                    w = layer.weight.data
                    H = w.shape[0]
                    eye = torch.eye(H, device=w.device)
                    check(torch.allclose(w, eye, atol=1e-6),
                          f"block {i} expert {ei} layers[{li+1}] is identity (H={H})")
                    check(layer.bias.data.abs().max().item() < 1e-6,
                          f"block {i} expert {ei} layers[{li+1}] bias is zero")

# ---------------------------------------------------------------------------
# Test 6: Forward + backward
# ---------------------------------------------------------------------------
def test_forward_backward(model, label):
    print(f"\n[{label}] Test 6 — Forward + backward")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    try:
        out = model(x)
        loss = out.sum()
        loss.backward()
        check(True, f"forward ok, out.shape={out.shape}")
        check(True, "backward ok, no exception")
    except Exception as e:
        check(False, f"exception: {e}")


# ===========================================================================
# Run all tests
# ===========================================================================
print("=" * 70)
print("Config A: expert_mlp_ratio=4, expert_depth=3 (same width as pretrained)")
print("=" * 70)
m_A = vt.deit_tiny_patch16_224(
    pretrained=True, num_classes=NUM_CLASSES,
    moe_layers=MOE_LAYERS, mlp_ratio=4.0, expert_mlp_ratio=4.0,
    num_experts=6, gate_k=1, prune_ratio=0.0,
    is_tutel=False, drop_path_rate=0.1, router='cosine_top', expert_depth=3)
test_dense_block_shape(m_A, "A")
test_moe_expert_shape(m_A, 4.0, 3, "A")
test_pretrained_copy_exact(m_A, "A")
test_middle_layer_identity(m_A, "A")
test_forward_backward(m_A, "A")

print()
print("=" * 70)
print("Config B: expert_mlp_ratio=6, expert_depth=3 (wider than pretrained)")
print("=" * 70)
m_B = vt.deit_tiny_patch16_224(
    pretrained=True, num_classes=NUM_CLASSES,
    moe_layers=MOE_LAYERS, mlp_ratio=4.0, expert_mlp_ratio=6.0,
    num_experts=6, gate_k=1, prune_ratio=0.0,
    is_tutel=False, drop_path_rate=0.1, router='cosine_top', expert_depth=3)
test_dense_block_shape(m_B, "B")
test_moe_expert_shape(m_B, 6.0, 3, "B")
test_pretrained_partial_copy(m_B, 6.0, "B")
test_middle_layer_identity(m_B, "B")
test_forward_backward(m_B, "B")

print()
print("=" * 70)
print("Config C: expert_mlp_ratio=4, expert_depth=4 (deeper, same width)")
print("=" * 70)
m_C = vt.deit_tiny_patch16_224(
    pretrained=True, num_classes=NUM_CLASSES,
    moe_layers=MOE_LAYERS, mlp_ratio=4.0, expert_mlp_ratio=4.0,
    num_experts=6, gate_k=1, prune_ratio=0.0,
    is_tutel=False, drop_path_rate=0.1, router='cosine_top', expert_depth=4)
test_dense_block_shape(m_C, "C")
test_moe_expert_shape(m_C, 4.0, 4, "C")
test_pretrained_copy_exact(m_C, "C")
test_middle_layer_identity(m_C, "C")
test_forward_backward(m_C, "C")

print()
print("=" * 70)
print("Config D: expert_mlp_ratio=6, expert_depth=4 (wider + deeper)")
print("=" * 70)
m_D = vt.deit_tiny_patch16_224(
    pretrained=True, num_classes=NUM_CLASSES,
    moe_layers=MOE_LAYERS, mlp_ratio=4.0, expert_mlp_ratio=6.0,
    num_experts=6, gate_k=1, prune_ratio=0.0,
    is_tutel=False, drop_path_rate=0.1, router='cosine_top', expert_depth=4)
test_dense_block_shape(m_D, "D")
test_moe_expert_shape(m_D, 6.0, 4, "D")
test_pretrained_partial_copy(m_D, 6.0, "D")
test_middle_layer_identity(m_D, "D")
test_forward_backward(m_D, "D")

print()
print("=" * 70)
print("All tests passed.")
print("=" * 70)

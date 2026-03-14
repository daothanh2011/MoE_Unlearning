"""
Gradient-flow and shape tests for TransparentMoELayer, OrthoLoss, and VarianceLoss.

Run with:
    cd Generalizable-Mixture-of-Experts
    python -m pytest domainbed/test/test_moe_losses.py -v
"""

import math
import torch
import pytest

from domainbed.moe_layer import TransparentMoELayer, CosineGate
from domainbed.losses.moe_specialization_losses import OrthoLoss, VarianceLoss


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def make_layer(top_k=1, n_experts=6, D=64, H=128):
    return TransparentMoELayer(
        model_dim=D,
        n_experts=n_experts,
        hidden_size_per_expert=H,
        top_k=top_k,
        capacity_factor=1.5,
        gate_noise=1.0,
        proj_dim=32,
        init_temperature=0.5,
        fp32_gate=True,
        dropout=0.1,
    )


# -------------------------------------------------------------------------
# TransparentMoELayer: output shapes
# -------------------------------------------------------------------------

class TestTransparentMoELayerShapes:
    def test_output_shape(self):
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=1)
        x = torch.randn(B, T, D)
        y = layer(x)
        assert y.shape == (B, T, D), f"Expected ({B},{T},{D}), got {y.shape}"

    def test_routing_scores_shape_and_sum(self):
        B, T, D = 2, 10, 64
        E = 6
        layer = make_layer(top_k=1)
        layer.eval()
        x = torch.randn(B, T, D)
        _ = layer(x)
        rs = layer.routing_scores
        assert rs.shape == (B * T, E), f"routing_scores shape wrong: {rs.shape}"
        # Softmax rows should sum to ~1
        row_sums = rs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
            "routing_scores rows should sum to 1"

    def test_expert_outputs_shape(self):
        B, T, D = 2, 10, 64
        E = 6
        layer = make_layer(top_k=1)
        x = torch.randn(B, T, D)
        _ = layer(x)
        eo = layer.expert_outputs
        assert eo.shape == (B * T, E, D), f"expert_outputs shape wrong: {eo.shape}"

    def test_expert_outputs_sparse_for_k1(self):
        """With top-1 routing, each token should have exactly 1 non-zero expert row."""
        B, T, D = 2, 10, 64
        E = 6
        layer = make_layer(top_k=1)
        layer.eval()
        x = torch.randn(B, T, D)
        _ = layer(x)
        eo = layer.expert_outputs  # [N, E, D]
        # Count non-zero expert slots per token (using L2 norm per expert)
        norms = eo.norm(dim=-1)   # [N, E]
        nonzero_per_token = (norms > 1e-9).sum(dim=1)  # [N]
        # Should be at most top_k (can be fewer due to capacity)
        assert (nonzero_per_token <= 1).all(), \
            f"With k=1 each token should have ≤1 non-zero expert, got max {nonzero_per_token.max()}"

    def test_l_aux_is_scalar(self):
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=1)
        x = torch.randn(B, T, D)
        _ = layer(x)
        assert layer.l_aux.shape == torch.Size([]), \
            f"l_aux should be scalar, got shape {layer.l_aux.shape}"


# -------------------------------------------------------------------------
# OrthoLoss
# -------------------------------------------------------------------------

class TestOrthoLoss:
    def test_zero_for_k1(self):
        """With top-1 routing expert_outputs has at most one non-zero row per
        token, so Lo should be essentially 0."""
        B, T, D, E = 2, 10, 64, 6
        layer = make_layer(top_k=1)
        layer.eval()
        x = torch.randn(B, T, D)
        _ = layer(x)

        lo_fn = OrthoLoss()
        lo = lo_fn(layer.expert_outputs)
        assert abs(lo.item()) < 1e-6, \
            f"OrthoLoss should be ~0 for k=1, got {lo.item()}"

    def test_nonzero_for_k2(self):
        """With k=2 there are pairs of non-zero experts; Lo should be > 0."""
        B, T, D, E = 2, 10, 64, 6
        layer = make_layer(top_k=2)
        layer.eval()
        x = torch.randn(B, T, D)
        _ = layer(x)

        lo_fn = OrthoLoss()
        lo = lo_fn(layer.expert_outputs)
        assert lo.item() >= 0, "OrthoLoss must be non-negative"
        # Some tokens will have 2 non-zero experts, so Lo > 0
        # (unless all pairs happen to be perfectly orthogonal — extremely unlikely)
        assert lo.item() > 1e-9, \
            f"OrthoLoss should be > 0 for k=2, got {lo.item()}"

    def test_gradient_flows_to_expert_ffn(self):
        """Lo gradients must reach expert FFN weight parameters."""
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=2)
        layer.train()
        x = torch.randn(B, T, D)
        _ = layer(x)

        lo_fn = OrthoLoss()
        lo = lo_fn(layer.expert_outputs)
        lo.backward()

        assert layer.fc1_weight.grad is not None, "fc1_weight.grad should not be None"
        assert layer.fc1_weight.grad.abs().sum() > 0, "fc1_weight.grad should be non-zero"
        assert layer.fc2_weight.grad is not None, "fc2_weight.grad should not be None"
        assert layer.fc2_weight.grad.abs().sum() > 0, "fc2_weight.grad should be non-zero"

    def test_gradient_does_not_flow_to_gate_via_lo(self):
        """Lo does not depend on gate parameters (routing scores not in Lo path)."""
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=2)
        layer.train()
        # Zero any pre-existing grads
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()

        x = torch.randn(B, T, D)
        _ = layer(x)

        lo_fn = OrthoLoss()
        lo = lo_fn(layer.expert_outputs)
        lo.backward()

        # Gate params should have zero or None gradient from Lo alone
        gate_grad = layer.gate.sim_matrix.grad
        assert gate_grad is None or gate_grad.abs().sum().item() == 0, \
            "Lo should not produce gradients on gate sim_matrix"


# -------------------------------------------------------------------------
# VarianceLoss
# -------------------------------------------------------------------------

class TestVarianceLoss:
    def test_output_is_non_positive(self):
        """Lv = -variance ≤ 0 always."""
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=1)
        layer.eval()
        x = torch.randn(B, T, D)
        _ = layer(x)

        lv_fn = VarianceLoss()
        lv = lv_fn(layer.routing_scores)
        assert lv.item() <= 0, f"Lv should be ≤ 0, got {lv.item()}"

    def test_gradient_flows_to_gate(self):
        """Lv gradients must reach the gate (cosine_projector, sim_matrix, temperature)."""
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=1)
        layer.train()
        x = torch.randn(B, T, D)
        _ = layer(x)

        lv_fn = VarianceLoss()
        lv = lv_fn(layer.routing_scores)
        lv.backward()

        assert layer.gate.sim_matrix.grad is not None, "sim_matrix.grad should not be None"
        assert layer.gate.sim_matrix.grad.abs().sum() > 0, "sim_matrix.grad should be non-zero"
        assert layer.gate.cosine_projector.weight.grad is not None, \
            "cosine_projector.weight.grad should not be None"
        assert layer.gate.cosine_projector.weight.grad.abs().sum() > 0

    def test_gradient_does_not_flow_to_expert_ffn_via_lv(self):
        """Lv does not depend on expert FFN parameters (routing_scores not in FFN path)."""
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=1)
        layer.train()
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()

        x = torch.randn(B, T, D)
        _ = layer(x)

        lv_fn = VarianceLoss()
        lv = lv_fn(layer.routing_scores)
        lv.backward()

        assert layer.fc1_weight.grad is None or layer.fc1_weight.grad.abs().sum().item() == 0, \
            "Lv should not produce gradients on fc1_weight"


# -------------------------------------------------------------------------
# Baseline check: l_aux is non-trivially computed
# -------------------------------------------------------------------------

class TestLaux:
    def test_l_aux_positive(self):
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=1)
        layer.train()
        x = torch.randn(B, T, D)
        _ = layer(x)
        assert layer.l_aux.item() >= 0, "l_aux should be non-negative"

    def test_l_aux_gradient_flows_to_gate(self):
        B, T, D = 2, 10, 64
        layer = make_layer(top_k=1)
        layer.train()
        x = torch.randn(B, T, D)
        _ = layer(x)
        layer.l_aux.backward()
        assert layer.gate.sim_matrix.grad is not None
        assert layer.gate.sim_matrix.grad.abs().sum() > 0


# -------------------------------------------------------------------------
# CosineGate: architecture sanity
# -------------------------------------------------------------------------

class TestCosineGate:
    def test_output_shape(self):
        D, E, proj_dim = 64, 6, 32
        gate = CosineGate(D, E, proj_dim)
        x = torch.randn(20, D)
        logits = gate(x)
        assert logits.shape == (20, E), f"Gate output shape wrong: {logits.shape}"

    def test_fp32_output_when_fp32_gate(self):
        D, E = 64, 6
        gate = CosineGate(D, E, fp32_gate=True)
        x = torch.randn(20, D).half()
        logits = gate(x)
        assert logits.dtype == torch.float32, \
            f"fp32_gate=True should produce float32 output, got {logits.dtype}"

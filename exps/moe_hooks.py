"""Hook utilities for capturing intermediate tensors from a GMoEOMoE ViT-S/16.

Two layers of capture:

1. Block-output hooks (forward_hook on `model.blocks[i]`) — gives the full
   (B, 197, 384) output of any transformer block. Used by CKA, Grad-CAM, t-SNE.

2. Monkey-patched DeepMoELayer.forward — re-implements the original forward
   verbatim but stashes `E_raw` (pre-Gram-Schmidt) and `E_orth` (post-GS) into
   a per-block dict. Used by MSO. We monkey-patch instead of editing
   vision_transformer.py to keep eval logic isolated.

Both capture mechanisms write into a shared dict `captured` that the caller
clears between batches.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

# vision_transformer.py reads DOMAINBED_PROJECT_DIR to locate vit_helpers.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DOMAINBED_PROJECT_DIR', _REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from domainbed.vision_transformer import DeepMoELayer, gram_schmidt


# MoE block indices for ViT-S/16 with moe_layers=['F']*8 + ['S','F']*2
MOE_BLOCK_INDICES = (8, 10)


def _patched_moe_forward(captured: Dict[int, dict], block_idx: int):
    """Build a replacement DeepMoELayer.forward bound to a specific block_idx.

    Mirrors domainbed/vision_transformer.py:325-362 line-for-line, but stores
    E_raw (pre-GS) and the post-GS tensor into captured[block_idx].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        T = x_flat.shape[0]

        logits = self.gate_proj(F.normalize(x_flat, dim=-1))
        topk_vals, topk_idx = logits.topk(self.gate_k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)

        E_raw = torch.zeros(T, self.gate_k, D, device=x.device, dtype=x.dtype)
        for ki in range(self.gate_k):
            idx = topk_idx[:, ki]
            for ei in range(self.num_experts):
                mask = (idx == ei)
                if mask.any():
                    E_raw[mask, ki] = self.experts[ei](x_flat[mask])

        # Capture pre-Gram-Schmidt expert outputs
        captured.setdefault(block_idx, {})
        captured[block_idx]['pre_gs'] = E_raw.detach().cpu()

        if self.use_omoe and self.gate_k > 1:
            E_raw = gram_schmidt(E_raw)

        # Capture post-GS (or unchanged if OMoE off / k==1)
        captured[block_idx]['post_gs'] = E_raw.detach().cpu()
        captured[block_idx]['gates'] = gates.detach().cpu()
        captured[block_idx]['topk_idx'] = topk_idx.detach().cpu()

        out = (gates.unsqueeze(-1) * E_raw).sum(dim=1)

        if self.use_balance_loss:
            route_prob = torch.zeros(T, self.num_experts, device=x.device, dtype=gates.dtype)
            route_prob.scatter_(1, topk_idx, gates)
            imp = route_prob.sum(dim=0)
            self.l_aux = imp.var() / (imp.mean() ** 2 + 1e-10)
        else:
            self.l_aux = torch.tensor(0.0, device=x.device)

        return out.reshape(B, S, D)

    return forward


class MoECapture:
    """Context manager that installs hooks + monkey-patches on a ViT model.

    Usage:
        with MoECapture(model, block_indices=(8, 10)) as cap:
            model(x)
            print(cap.captured[8]['pre_gs'].shape)   # [B*197, k, D]
            print(cap.captured[8]['block_out'].shape) # [B, 197, D]
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: tuple = MOE_BLOCK_INDICES,
        also_capture_last_block: bool = False,
    ):
        self.model = model
        self.block_indices = list(block_indices)
        if also_capture_last_block:
            last = len(model.blocks) - 1
            if last not in self.block_indices:
                self.block_indices.append(last)
        self.captured: Dict[int, dict] = {}
        self._original_forwards: Dict[int, callable] = {}
        self._hook_handles: List = []

    def __enter__(self):
        # 1. Monkey-patch DeepMoELayer.forward at MoE blocks. Use duck-typing
        #    rather than isinstance because vision_transformer can be imported
        #    via two distinct paths ("vision_transformer" via DOMAINBED_PROJECT_DIR
        #    and "domainbed.vision_transformer"), giving two distinct class
        #    objects for DeepMoELayer.
        for idx in self.block_indices:
            block = self.model.blocks[idx]
            mlp = block.mlp
            if all(hasattr(mlp, a) for a in ('gate_proj', 'experts', 'gate_k', 'num_experts')):
                self._original_forwards[idx] = mlp.forward
                mlp.forward = types.MethodType(
                    _patched_moe_forward(self.captured, idx), mlp
                )

        # 2. forward_hook on each block to capture full block output
        def make_block_hook(block_idx):
            def hook(module, inp, out):
                self.captured.setdefault(block_idx, {})
                self.captured[block_idx]['block_out'] = out.detach().cpu()
            return hook

        for idx in self.block_indices:
            h = self.model.blocks[idx].register_forward_hook(make_block_hook(idx))
            self._hook_handles.append(h)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original forwards
        for idx, orig in self._original_forwards.items():
            self.model.blocks[idx].mlp.forward = orig
        # Remove hooks
        for h in self._hook_handles:
            h.remove()
        self._original_forwards.clear()
        self._hook_handles.clear()

    def clear(self):
        """Clear cached tensors between batches (call inside the loop)."""
        self.captured.clear()


class GradCAMCapture:
    """Captures forward activation + backward gradient on a single block's mlp.

    For ViT-adapted Grad-CAM at MoE block 8 or 10. Requires gradients enabled.

    Usage:
        cam = GradCAMCapture(model, block_idx=8)
        with cam:
            logits = model(x)              # x.requires_grad not needed
            target = logits[:, pred_class]
            target.sum().backward()
            A = cam.activation             # (B, 197, 384)
            dA = cam.gradient              # (B, 197, 384)
    """

    def __init__(self, model: nn.Module, block_idx: int):
        self.model = model
        self.block_idx = block_idx
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self._handles: List = []

    def __enter__(self):
        target = self.model.blocks[self.block_idx].mlp

        def fwd_hook(module, inp, out):
            self.activation = out
            # register grad hook on the output tensor itself
            out.register_hook(self._grad_hook)

        self._handles.append(target.register_forward_hook(fwd_hook))
        return self

    def _grad_hook(self, grad):
        self.gradient = grad.detach()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

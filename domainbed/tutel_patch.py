"""
Pure-PyTorch fallback for Tutel's custom CUDA kernels.

Apply this patch BEFORE importing any module that imports tutel
(e.g., vision_transformer.py, algorithms.py).

Necessary when Tutel's pre-compiled CUDA extensions don't support
the installed GPU architecture (e.g., compute_120 on RTX 5070 Ti).

The patch replaces three CUDA scatter/gather kernels with equivalent
pure-PyTorch index_add_ / index_select operations:

  func_fwd      : scatter tokens → expert slots (weighted by gates)
  func_bwd_data : gather expert slots → token gradients  (backward data)
  func_bwd_gate : dot-product for gate gradients          (backward gate)

Also patches fast_cumsum_sub_one (used in batch-prioritized routing).

Usage:
    import domainbed.tutel_patch  # apply before importing algorithms
    from domainbed import algorithms
"""

import torch


# ── 1. fast_cumsum_sub_one (used in gating / batch-prioritized routing) ──────
# Original: torch.ops.tutel_ops.cumsum(data)  — requires compiled CUDA op
# Replacement: standard PyTorch cumsum

def _fast_cumsum_sub_one(data: torch.Tensor) -> torch.Tensor:
    """cumsum - 1 along dim 0.  Equivalent to Tutel's CUDA op."""
    return data.long().cumsum(dim=0) - 1


# ── 2. Scatter / gather kernel replacements ──────────────────────────────────
# CUDA kernel semantics (from tutel/jit_kernels/sparse.py):
#
#   func_fwd(gates, indices, locations, src, dst, extra=[N, hidden, capacity])
#       for i in range(N):
#           if locations[i] < capacity and indices[i] >= 0:
#               dst[indices[i] * capacity + locations[i], :] = gates[i] * src[i, :]
#
#   func_bwd_data(gates, indices, locations, grad_src, dispatched, extra)
#       for i in range(N):
#           if locations[i] < capacity and indices[i] >= 0:
#               grad_src[i, :] = gates[i] * dispatched[indices[i]*capacity+locations[i], :]
#           else:
#               grad_src[i, :] = 0
#
#   func_bwd_gate(grad_gates, indices, locations, src, dispatched, extra)
#       for i in range(N):
#           if locations[i] < capacity and indices[i] >= 0:
#               grad_gates[i] = dot(src[i], dispatched[indices[i]*capacity+locations[i]])
#           else:
#               grad_gates[i] = 0
#
# gates is (N,) float32 OR (N,2) float32 (h2 format — both values are equal).
# We always read the first column.

def _get_gate_scalar(gates: torch.Tensor, valid_idx: torch.Tensor) -> torch.Tensor:
    """Extract per-token gate scalar for valid tokens. Handles h2 (N,2) format."""
    if gates.dim() == 2:
        return gates[valid_idx, 0]
    return gates[valid_idx]


def _make_pytorch_fwd(dtype, is_cuda=True):
    def func_fwd(gates, indices, locations, src, dst, extra):
        N, _hidden, capacity = extra
        # src: (N, model_dim), dst: (num_experts * capacity, model_dim) — already 2D
        src2d = src.reshape(-1, src.size(-1))
        dst2d = dst.reshape(-1, dst.size(-1))
        mask = (locations < capacity) & (indices >= 0)
        valid = mask.nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            return
        slots = (indices[valid].long() * capacity + locations[valid].long())
        g = _get_gate_scalar(gates, valid).to(src2d.dtype).unsqueeze(1)  # (V, 1)
        dst2d.index_add_(0, slots, (g * src2d[valid]).to(dst2d.dtype))
    return func_fwd


def _make_pytorch_bwd_data(dtype, is_cuda=True):
    def func_bwd_data(gates, indices, locations, grad_src, dispatched, extra):
        N, _hidden, capacity = extra
        # dispatched may be 3D (num_experts, capacity, model_dim) — flatten to 2D
        dispatched2d = dispatched.reshape(-1, dispatched.size(-1))
        grad2d = grad_src.reshape(-1, grad_src.size(-1))
        mask = (locations < capacity) & (indices >= 0)
        valid = mask.nonzero(as_tuple=True)[0]
        grad2d.zero_()
        if valid.numel() == 0:
            return
        slots = (indices[valid].long() * capacity + locations[valid].long())
        g = _get_gate_scalar(gates, valid).to(dispatched2d.dtype).unsqueeze(1)
        grad2d[valid] = (g * dispatched2d[slots]).to(grad2d.dtype)
    return func_bwd_data


def _make_pytorch_bwd_gate(dtype, is_cuda=True):
    def func_bwd_gate(grad_gates, indices, locations, src, dispatched, extra):
        N, _hidden, capacity = extra
        # dispatched may be 3D — flatten to 2D
        dispatched2d = dispatched.reshape(-1, dispatched.size(-1))
        src2d = src.reshape(-1, src.size(-1))
        mask = (locations < capacity) & (indices >= 0)
        valid = mask.nonzero(as_tuple=True)[0]
        grad_gates.zero_()
        if valid.numel() == 0:
            return
        slots = (indices[valid].long() * capacity + locations[valid].long())
        grad_gates[valid] = (src2d[valid].to(dispatched2d.dtype) * dispatched2d[slots]).sum(dim=1).to(grad_gates.dtype)
    return func_bwd_gate


# ── 3. Apply the patches ─────────────────────────────────────────────────────

def _apply_tutel_patch():
    """
    Monkey-patch Tutel's CUDA-dependent functions with pure-PyTorch equivalents.
    Must be called before any tutel.moe_layer is instantiated.
    """
    # Patch sparse kernel factory functions (used by TutelMoeFastDispatcher.update)
    try:
        from tutel.jit_kernels import sparse as _sparse
        _sparse.create_forward       = _make_pytorch_fwd
        _sparse.create_backward_data = _make_pytorch_bwd_data
        _sparse.create_backward_gate = _make_pytorch_bwd_gate
    except ImportError:
        pass

    # Patch fast_cumsum_sub_one in gating module
    try:
        from tutel.jit_kernels import gating as _gating
        _gating.fast_cumsum_sub_one = _fast_cumsum_sub_one
    except ImportError:
        pass

    # Patch fast_cumsum_sub_one imported directly in fast_dispatch module
    try:
        from tutel.impls import fast_dispatch as _fd
        _fd.fast_cumsum_sub_one = _fast_cumsum_sub_one
    except ImportError:
        pass

    # Clear dispatcher kernel pool so it re-creates with our patched functions
    try:
        from tutel.impls.fast_dispatch import TutelMoeFastDispatcher
        TutelMoeFastDispatcher.kernel_pool.clear()
    except Exception:
        pass


# Apply immediately on import
_apply_tutel_patch()

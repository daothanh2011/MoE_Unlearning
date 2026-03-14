"""
Auxiliary losses from "Advancing Expert Specialization for Better MoE" (NeurIPS 2025).

OrthoLoss (Lo) — encourages the FFN outputs of different selected experts for the
    same token to be orthogonal, improving expert specialization.
    Equation (6) in the paper.

VarianceLoss (Lv) — maximises the variance of per-token routing score distributions
    across experts, improving routing diversity.
    Equation (7) in the paper.

Both classes operate on tensors exposed by TransparentMoELayer (domainbed/moe_layer.py).
"""

import torch
import torch.nn as nn


class OrthoLoss(nn.Module):
    """
    Orthogonality loss between the outputs of selected experts for each token.

    Equation (6):
        Lo = Σ_i Σ_j Σ_{k≠j} [<x̃ij, x̃ik> / (<x̃ik, x̃ik> + ε)] * x̃ik

    where x̃ij is the FFN output of expert j for token i, zero if expert j was
    not selected for token i.  This encourages selected expert outputs to be
    mutually orthogonal — when Lo is small, the projection of x̃ij onto each
    x̃ik (k≠j) is small.

    Note on k=1: with top-1 routing each token has exactly one non-zero expert
    output; all cross-terms are zero and Lo = 0 automatically.  Lo is only
    non-trivial when top_k ≥ 2.

    Args:
        expert_outputs: Tensor [N, E, D]
            Per-expert FFN outputs.  expert_outputs[i, j, :] is x̃ij (zero
            for non-selected experts and capacity-overflow tokens).
            N = batch_size × seq_len, E = n_experts, D = embed_dim.

    Returns:
        Scalar tensor (mean of squared projection norms over tokens and experts).
    """

    def forward(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        N, E, D = expert_outputs.shape

        if E < 2:
            # Trivially zero; preserve computation graph with a differentiable op
            return expert_outputs.sum() * 0.0

        eps = 1e-6

        # All pairwise dot products over the embedding dimension D
        # dot[i, j, k] = <x̃ij, x̃ik>
        dot = torch.einsum('ied,ifd->ief', expert_outputs, expert_outputs)  # [N, E, E]

        # Self-dot products for the denominator: <x̃ik, x̃ik>  →  [N, E]
        self_dot = dot.diagonal(dim1=1, dim2=2)                              # [N, E]

        # Projection ratio: ratio[i, j, k] = <x̃ij, x̃ik> / (<x̃ik, x̃ik> + ε)
        # Broadcast self_dot over the j dimension (dim=1)
        ratio = dot / (self_dot.unsqueeze(1) + eps)                          # [N, E, E]

        # Zero out the diagonal (k == j) to exclude self-projection
        off_diag = (1.0 - torch.eye(E, device=expert_outputs.device,
                                     dtype=expert_outputs.dtype)).unsqueeze(0)  # [1, E, E]
        ratio = ratio * off_diag                                             # [N, E, E]

        # Projection vector: proj[i, j, d] = Σ_{k≠j} ratio[i,j,k] * x̃ik[d]
        proj = torch.einsum('ijk,ikd->ijd', ratio, expert_outputs)          # [N, E, D]

        # Scalar loss: mean of squared L2 norms of projection vectors
        lo = (proj ** 2).sum(dim=-1).mean()                                  # scalar
        return lo


class VarianceLoss(nn.Module):
    """
    Negative routing-score variance loss.

    Equation (7):
        s̄j  = (1/N) Σ_i sij
        Lv  = -(1/n) Σ_i Σ_j (sij - s̄j)²

    where sij is the pre-top-k softmax routing probability for token i to
    expert j.  The negative sign means *minimising* Lv = *maximising* routing
    diversity (routing scores with higher cross-token variance → more selective
    routing behaviour).

    Uses the FULL pre-top-k softmax distribution (not zeroed by top-k), so
    gradients flow to all expert routing parameters.

    Args:
        routing_scores: Tensor [N, E]
            Pre-top-k softmax probabilities.
            N = batch_size × seq_len, E = n_experts.

    Returns:
        Scalar tensor (negative mean squared deviation from per-expert mean).
    """

    def forward(self, routing_scores: torch.Tensor) -> torch.Tensor:
        # Per-expert mean across tokens:  s̄j  →  [E]
        s_bar = routing_scores.mean(dim=0)                       # [E]

        # Squared deviations from the per-expert mean:  (sij - s̄j)²  →  [N, E]
        diff_sq = (routing_scores - s_bar.unsqueeze(0)) ** 2     # [N, E]

        # Negative mean: minimising Lv maximises routing diversity
        lv = -diff_sq.mean()                                      # scalar
        return lv

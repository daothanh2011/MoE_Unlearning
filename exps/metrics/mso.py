"""Mean Squared Overlap (MSO) of expert outputs.

For each token t with k expert output vectors a_{t,1..k} ∈ R^D, MSO is the
mean over all C(k,2) ordered pairs of squared cosine similarities:

    MSO(t) = (2 / (k(k-1))) * Σ_{i<j} (⟨a_{t,i}, a_{t,j}⟩ / (‖a_{t,i}‖ ‖a_{t,j}‖))²

The reported scalar is the mean of MSO(t) over all tokens. MSO ∈ [0, 1]:
0 = mutually orthogonal experts; 1 = collinear experts.

Undefined for k=1 (no pairs); the function returns 0.0 in that case.
"""
from __future__ import annotations

import torch


def mso(E: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute mean squared overlap of expert outputs.

    Args:
        E: tensor of shape [T, k, D] — T tokens, k expert outputs per token, D dim.
        eps: numerical floor for norms.

    Returns:
        scalar float (mean over tokens of pairwise squared cosine sim).
    """
    if E.dim() != 3:
        raise ValueError(f"Expected [T, k, D], got shape {tuple(E.shape)}")
    T, k, D = E.shape
    if k < 2:
        return 0.0

    E = E.float()
    norms = E.norm(dim=-1, keepdim=True).clamp(min=eps)            # [T, k, 1]
    En = E / norms                                                 # unit vectors
    # Gram matrix per token: [T, k, k]
    G = torch.matmul(En, En.transpose(-1, -2))
    G2 = G * G

    # Sum strict-upper triangle, average over k(k-1)/2 pairs
    iu = torch.triu_indices(k, k, offset=1)
    pair_sum = G2[:, iu[0], iu[1]].sum(dim=-1)                     # [T]
    n_pairs = k * (k - 1) // 2
    mso_per_token = pair_sum / n_pairs                              # [T]
    return mso_per_token.mean().item()


class MSOAccumulator:
    """Streaming accumulator: averages MSO across many batches token-weighted."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, E: torch.Tensor):
        if E.dim() != 3 or E.shape[1] < 2:
            return
        val = mso(E)
        n = E.shape[0]
        self.sum += val * n
        self.count += n

    def value(self) -> float:
        return self.sum / self.count if self.count else 0.0

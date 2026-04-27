# domainbed/losses/matchdg_utils.py
"""
MatchDG (Mahajan, Tople, Sharma; ICML 2021) building blocks, ported to
operate on DomainBed-style mini-batches.

Reference paper: "Domain Generalization using Causal Matching"
                 https://arxiv.org/pdf/2006.07500
Reference code:  https://github.com/microsoft/robustdg

Three pieces:

  1.  supervised_contrastive_loss(features, labels, domains, tau, ...)
        Phase-I objective. For each anchor input, positive matches are
        same-class samples from *different* domains; negatives are
        different-class samples (any domain). This realises the paper's
        Eq. (4) on a single minibatch.

  2.  find_cross_domain_matches(features, labels, domains, exclude_self_domain=True)
        Given a batch of representations, returns `(idx_a, idx_b)` pairs
        such that each pair shares a class and comes from different
        domains, and `idx_b` is the *nearest* such partner of `idx_a` in
        representation space. This realises the paper's "iterative
        matching" — at each call, matches are recomputed against the
        current representation. Used both for the Phase-I update of
        positive matches and for Phase-II's matching loss.

  3.  matching_l2_loss(features, idx_a, idx_b)
        The Phase-II regularizer: L2 distance between matched pairs
        (Eq. 3 in the paper).

Notes on simplification
-----------------------
The reference MatchDG implementation maintains an external "p x q" data
matrix that pairs each anchor with one input from every domain so that
positive pairs are sampled per training step. DomainBed's minibatch
already contains one sample-set per source domain; we therefore find
positive matches *within* the concatenated minibatch. This keeps the
algorithm a drop-in for DomainBed's `update(minibatches)` interface
while preserving the paper's *idea*: positives = same class, different
domain, chosen by current-representation similarity (iterative matching).
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Supervised contrastive loss with domain-aware positives (Phase I)
# ---------------------------------------------------------------------------

def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    tau: float = 0.1,
    require_cross_domain_positive: bool = True,
) -> torch.Tensor:
    """
    Paper Eq. (4) on a minibatch.

        l(x_j, x_k) = -log [ exp(sim(j,k)/tau)
                             / (exp(sim(j,k)/tau) + sum_{i: y_i != y_j} exp(sim(j,i)/tau)) ]

    We average over all valid (j, k) positive pairs in the batch.

    Args:
        features: (B, D) — unit-normalisation is applied internally so the
            input does not need to be pre-normalised.
        labels:   (B,)
        domains:  (B,)
        tau: temperature.
        require_cross_domain_positive: if True (paper default), positive
            pairs (j, k) must satisfy y_j == y_k AND d_j != d_k. If False,
            the same-class same-domain pairs are also allowed (closer to
            classic SupCon). The paper insists on cross-domain positives;
            we keep that as the default.

    Returns:
        scalar loss. If the batch contains zero valid positives (very
        small batches / extreme domain imbalance), returns 0 with a
        gradient-preserving shape so that the optimiser does not crash.
    """
    if features.dim() != 2:
        raise ValueError(f'expected (B, D), got {tuple(features.shape)}')

    B = features.size(0)
    if B < 2:
        return features.sum() * 0.0     # no pairs possible

    # Cosine similarity (B, B). Diagonal is masked out below.
    f = F.normalize(features, dim=-1)
    sim = (f @ f.T) / tau                # (B, B)

    # Mask of valid positives — same class, optionally different domain,
    # not the diagonal.
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_domain = domains.unsqueeze(0) != domains.unsqueeze(1)
    eye = torch.eye(B, dtype=torch.bool, device=features.device)

    pos_mask = same_class & ~eye
    if require_cross_domain_positive:
        pos_mask = pos_mask & diff_domain

    # Mask of valid negatives — different class.
    neg_mask = ~same_class               # different class, any domain

    # We use the "mean over positives of the per-positive loss" formulation:
    # for each (j, k) positive, the loss is
    #   -log( exp(sim_jk) / (exp(sim_jk) + sum_{i in negatives_of_j} exp(sim_ji)) )
    # which in code is just the row-wise log-softmax of sim, masked.

    # Numerical stabilisation: subtract per-row max BEFORE exp.
    sim_max = sim.max(dim=1, keepdim=True).values
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)             # (B, B)

    # Denominator for row j = sum over (j, k itself) + (j, all negatives_of_j).
    # We compute it per (j, k) positive: the denominator is exp(sim_jk) +
    # sum over negatives of j. Equivalently:
    #   denom_jk = exp(sim_jk) + (exp_sim[j] * neg_mask[j]).sum()
    neg_sum_per_anchor = (exp_sim * neg_mask).sum(dim=1, keepdim=True)   # (B, 1)
    denom = exp_sim + neg_sum_per_anchor                                  # (B, B)

    # Per-pair loss = -log(exp_sim_jk / denom_jk). Only positive pairs
    # contribute.
    log_prob = sim - torch.log(denom + 1e-12)
    per_pair_loss = -log_prob

    n_pos = pos_mask.float().sum().clamp_min(1.0)
    loss = (per_pair_loss * pos_mask.float()).sum() / n_pos

    # If there were truly zero positives in the batch, return a gradient-
    # bearing zero so the optimiser receives a well-defined tensor.
    if pos_mask.sum() == 0:
        return features.sum() * 0.0
    return loss


# ---------------------------------------------------------------------------
# 2. Cross-domain match-finder (iterative matching, Phase I & II)
# ---------------------------------------------------------------------------

def find_cross_domain_matches(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    exclude_self_domain: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each anchor index i, find the index j != i with
        labels[j] == labels[i]
    and (if exclude_self_domain) domains[j] != domains[i],
    that maximises cosine similarity in `features` space.

    Anchors with no valid partner in the batch are silently dropped
    from the returned pairs.

    Args:
        features: (B, D)
        labels:   (B,)
        domains:  (B,)

    Returns:
        idx_a, idx_b: 1-D LongTensors of equal length `K <= B` with the
        matched-pair indices. `idx_a` is the anchor side and `idx_b`
        is its best cross-domain same-class partner.
    """
    if features.dim() != 2:
        raise ValueError(f'expected (B, D), got {tuple(features.shape)}')

    B = features.size(0)
    if B < 2:
        return (
            torch.empty(0, dtype=torch.long, device=features.device),
            torch.empty(0, dtype=torch.long, device=features.device),
        )

    f = F.normalize(features, dim=-1)
    sim = f @ f.T                                            # (B, B)

    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(B, dtype=torch.bool, device=features.device)
    valid = same_class & ~eye
    if exclude_self_domain:
        valid &= (domains.unsqueeze(0) != domains.unsqueeze(1))

    # Set invalid candidates to -inf so they cannot be argmax'd.
    sim_masked = sim.masked_fill(~valid, float('-inf'))

    # Best partner per anchor.
    best_sim, best_idx = sim_masked.max(dim=1)               # both (B,)
    has_partner = torch.isfinite(best_sim)                   # (B,) bool

    idx_a = torch.arange(B, device=features.device)[has_partner]
    idx_b = best_idx[has_partner]
    return idx_a, idx_b


# ---------------------------------------------------------------------------
# 3. Matching L2 loss (Phase II regularizer)
# ---------------------------------------------------------------------------

def matching_l2_loss(
    features: torch.Tensor,
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
) -> torch.Tensor:
    """
    L2 distance between matched pairs, averaged over pairs.

        L_match = (1 / K) * sum_k || features[idx_a[k]] - features[idx_b[k]] ||_2

    This is the dist(.) term in the paper's Eq. (3) under l2 distance.
    Note the loss is on raw features, NOT normalised features — this
    matches the reference robustdg implementation, which uses the
    last-layer representation directly.
    """
    if idx_a.numel() == 0:
        return features.sum() * 0.0
    diff = features[idx_a] - features[idx_b]
    # Per-pair L2 norm, then mean. (Mean of norms, not norm of mean.)
    return diff.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt().mean()


# ---------------------------------------------------------------------------
# 4. Tiny projection head (Phase I)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Small 2-layer MLP applied on top of the featurizer for Phase-I
    contrastive learning. The contrastive loss is computed on the
    projection-head output; the featurizer's raw output is what gets
    used for Phase-II matching and the eventual classifier — same
    pattern as SimCLR / SupCon.
    """
    def __init__(self, in_dim: int, hidden_dim: int | None = None,
                 out_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
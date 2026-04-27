# domainbed/losses/cirl_utils.py
"""
CIRL (Causality Inspired Representation Learning, CVPR 2022) building blocks,
ported from https://github.com/BIT-DA/CIRL and adapted to operate on
DomainBed-style mini-batches inside DG-OMOE.

Three pieces:

  1.  fourier_amplitude_mix_batched(x)
        Tensor-domain version of CIRL's `colorful_spectrum_mix`. Takes a
        batch of normalized images and produces a same-shape, label-preserving
        augmented batch where the amplitude spectrum has been linearly mixed
        with another (random) sample from the batch while the phase is kept.
        Phase carries object structure (the label cause), amplitude carries
        style — so this is the Fourier-domain "common across domains" prior.

  2.  Masker
        Differentiable top-k mask network (Gumbel-softmax + iterative max).
        Splits a feature vector f into a "superior / causal" subset
        (f * mask) and an "inferior" subset (f * (1 - mask)), both used by
        downstream classifiers. Trained adversarially against those
        classifiers — see `GMOE_CIRL` in algorithms.py.

  3.  factorization_loss(f_a, f_b)
        Barlow-Twins-style cross-correlation objective. Pushes corresponding
        dimensions of (original, augmented) features to be correlated and
        distinct dimensions to be decorrelated.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Fourier amplitude mix (image-level augmentation, on GPU)
# ---------------------------------------------------------------------------

# ImageNet stats — DomainBed normalizes inputs with these, so we de-normalize
# before the FFT (operate in [0, 1] image space) and re-normalize after.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * std + mean


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def fourier_amplitude_mix_batched(
    x: torch.Tensor,
    alpha: float = 1.0,
    ratio: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        x: (B, C, H, W) normalized image tensor (ImageNet stats).
        alpha: upper bound of the per-sample mixing coefficient lam ~ U(0, alpha).
            CIRL's PACS default is 1.0.
        ratio: fraction of the (centered) amplitude spectrum to mix.
            CIRL's default is 1.0 (mix the whole spectrum).

    Returns:
        x_aug: (B, C, H, W) same-shape augmented batch where each sample i has
            had its amplitude spectrum mixed with sample perm[i]'s, while
            keeping its own phase. The label of sample i is preserved.
    """
    if x.dim() != 4:
        raise ValueError(f'expected 4-D (B,C,H,W), got {tuple(x.shape)}')

    B, C, H, W = x.shape
    if B < 2:
        return x.clone()

    # 1. Pick the partner image for each sample (no self-pairing).
    perm = torch.randperm(B, device=x.device)
    same = (perm == torch.arange(B, device=x.device))
    if same.any():
        # rotate any self-mappings by 1 — guarantees a different partner.
        perm[same] = (perm[same] + 1) % B

    # 2. De-normalize to roughly [0, 1] image space and clamp; the FFT is
    #    insensitive to small clipping, but we want non-negative inputs so
    #    that the FFT magnitudes are interpretable as "amplitudes".
    x_img = _denormalize(x).clamp(0.0, 1.0)
    x_partner = x_img[perm]

    # 3. FFT both. We use rfft2 for memory efficiency — the negative-frequency
    #    half is implied by Hermitian symmetry of real input.
    fft_a = torch.fft.fft2(x_img, dim=(-2, -1))
    fft_b = torch.fft.fft2(x_partner, dim=(-2, -1))

    abs_a = fft_a.abs()
    abs_b = fft_b.abs()
    pha_a = torch.angle(fft_a)

    # 4. Center-shift so the low frequencies sit in the middle of the H×W
    #    grid, then mix only the central (h_crop × w_crop) block — same as
    #    CIRL's reference implementation.
    abs_a_c = torch.fft.fftshift(abs_a, dim=(-2, -1))
    abs_b_c = torch.fft.fftshift(abs_b, dim=(-2, -1))

    h_crop = max(1, int(H * math.sqrt(ratio)))
    w_crop = max(1, int(W * math.sqrt(ratio)))
    h0 = H // 2 - h_crop // 2
    w0 = W // 2 - w_crop // 2

    # Per-sample mixing coefficient. Shape (B, 1, 1, 1) so it broadcasts.
    lam = torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) * alpha

    abs_mixed = abs_a_c.clone()
    block_a = abs_a_c[..., h0:h0 + h_crop, w0:w0 + w_crop]
    block_b = abs_b_c[..., h0:h0 + h_crop, w0:w0 + w_crop]
    abs_mixed[..., h0:h0 + h_crop, w0:w0 + w_crop] = (
        lam * block_b + (1.0 - lam) * block_a
    )

    # 5. Un-shift, recombine with the *original phase* of x (this is what
    #    keeps the object/label intact), and inverse-FFT back to image space.
    abs_mixed = torch.fft.ifftshift(abs_mixed, dim=(-2, -1))
    fft_mixed = abs_mixed * torch.exp(1j * pha_a)
    x_aug_img = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real
    x_aug_img = x_aug_img.clamp(0.0, 1.0)

    return _normalize(x_aug_img)


# ---------------------------------------------------------------------------
# 2. Masker — differentiable top-k feature selector
# ---------------------------------------------------------------------------

class Masker(nn.Module):
    """
    Learns a soft mask m in [0, 1]^D over a D-dim feature vector that selects
    approximately k "causal" dimensions. Implementation mirrors the reference
    CIRL repo:

        score = MLP(f)                 # (B, D)
        for _ in range(k):
            soft = gumbel_softmax(score, dim=-1, tau=0.5)   # ~one-hot of D
            m   = max(m, soft)         # accumulate selections, elementwise

    Each gumbel_softmax pass picks (softly) one dimension; doing this k
    times and taking the elementwise max gives an approximately k-hot mask
    in a differentiable way.

    Notes
    -----
    * `in_dim` is the encoder feature size (e.g. 384 for DeiT-S, or
      `expert_dim` for the post-MoE feature in DG-OMOE).
    * `k` must be < in_dim and is typically ~60% of it (CIRL uses 308/512
      for ResNet-18 PACS; we default to round(0.6 * in_dim) for whatever
      backbone is in use).
    * The final BatchNorm is non-affine; it just standardises the logits
      before the Gumbel sampler so that no dimension dominates a priori.
    """

    def __init__(
        self,
        in_dim: int,
        k: int | None = None,
        middle_ratio: int = 4,
        dropout: float = 0.5,
        tau: float = 0.5,
    ):
        super().__init__()
        if k is None:
            k = max(1, int(round(0.6 * in_dim)))
        if not (1 <= k < in_dim):
            raise ValueError(f'k={k} must satisfy 1 <= k < in_dim={in_dim}')

        self.in_dim = in_dim
        self.k = k
        self.tau = tau
        middle = middle_ratio * in_dim

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, in_dim),
        )
        self.bn = nn.BatchNorm1d(in_dim, affine=False)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (B, in_dim) feature vectors.

        Returns:
            mask: (B, in_dim) soft mask in [0, 1], approximately k-hot per
                  row.
        """
        score = self.bn(self.mlp(f))
        mask = torch.zeros_like(score)
        # Iteratively pick k dimensions; each call to gumbel_softmax samples
        # fresh noise so different dimensions are selected each pass.
        cur = score
        for _ in range(self.k):
            soft = F.gumbel_softmax(cur, tau=self.tau, hard=False, dim=-1)
            mask = torch.maximum(mask, soft)
        return mask


# ---------------------------------------------------------------------------
# 3. Factorization (Barlow-Twins) loss between original/augmented features
# ---------------------------------------------------------------------------

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Flatten the off-diagonal entries of a square matrix."""
    n, m = x.shape
    assert n == m, f'expected square matrix, got {(n, m)}'
    # The classic flatten-pop-reshape trick from the Barlow-Twins repo.
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(
    f_a: torch.Tensor,
    f_b: torch.Tensor,
    off_diag_weight: float = 5e-3,
) -> torch.Tensor:
    """
    Barlow-Twins cross-correlation objective:

        L = sum_i (C_ii - 1)^2 / D  +  off_diag_weight * sum_{i!=j} C_ij^2 / (D*(D-1))

    where C is the cross-correlation matrix between batch-normalized f_a and
    f_b. Pushes corresponding dimensions across views to be correlated and
    different dimensions to be decorrelated — a "factorized" representation.

    Args:
        f_a, f_b: (B, D) feature batches from two views (e.g. original /
            Fourier-augmented).
        off_diag_weight: lambda in the Barlow-Twins paper (default 5e-3,
            matching CIRL's reference value).
    """
    if f_a.shape != f_b.shape:
        raise ValueError(f'shape mismatch: {f_a.shape} vs {f_b.shape}')

    # Per-dim z-score over the batch.
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)

    # Cross-correlation: (D, D)
    c = (f_a_norm.T @ f_b_norm) / f_a_norm.size(0)

    on_diag = (c.diagonal() - 1.0).pow(2).mean()
    off_diag = _off_diagonal(c).pow(2).mean()

    return on_diag + off_diag_weight * off_diag


# ---------------------------------------------------------------------------
# Schedule helper (sigmoid ramp-up of the factorization weight)
# ---------------------------------------------------------------------------

def sigmoid_rampup(current: float, rampup_length: float) -> float:
    """
    Exponential ramp-up from 0 -> 1 over `rampup_length` units, following
    Tarvainen & Valpola (2017). Used by CIRL to grow the factorization
    weight from 0 over the first few epochs so the encoder stabilises
    before the Barlow-Twins term kicks in.
    """
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
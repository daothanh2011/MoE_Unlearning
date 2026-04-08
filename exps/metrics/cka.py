"""Debiased linear Centered Kernel Alignment (CKA).

Uses the unbiased HSIC1 estimator from Song et al. (2012), which removes the
bias introduced by finite-sample plug-in HSIC by zeroing out the diagonals of
the centered Gram matrices. This is the recommended estimator for CKA in
high-dimensional / low-sample regimes (Nguyen et al., 2021).

Linear kernel: K = X X^T, L = Y Y^T.
"""
from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def _hsic1(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012, eq. 4).

    K, L: (n, n) Gram matrices. Diagonal is zeroed to remove bias.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"Unbiased HSIC requires n >= 4, got n={n}")
    K = K.astype(np.float64).copy()
    L = L.astype(np.float64).copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)

    KL_trace = np.sum(K * L)                              # tr(KL)  with zero diag
    K_sum = K.sum()
    L_sum = L.sum()
    K_row_L_col = (K.sum(axis=0) * L.sum(axis=0)).sum()   # 1^T K L 1 with zero diag

    factor = 1.0 / (n * (n - 3))
    term1 = KL_trace
    term2 = K_sum * L_sum / ((n - 1) * (n - 2))
    term3 = -2.0 / (n - 2) * K_row_L_col
    return factor * (term1 + term2 + term3)


def linear_cka_debiased(X, Y) -> float:
    """Debiased linear CKA between feature matrices X (n, p) and Y (n, q).

    Same n; columns can differ. Returns scalar in approx [0, 1].

    Requires n >= 4 for the unbiased HSIC estimator. For very small n
    (e.g. n=7 classes), the estimator can be noisy — pair with
    `linear_cka_biased` for cross-checking.
    """
    X = _to_numpy(X)
    Y = _to_numpy(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}")
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic1(K, L)
    hsic_kk = _hsic1(K, K)
    hsic_ll = _hsic1(L, L)
    denom = np.sqrt(max(hsic_kk * hsic_ll, 1e-30))
    return float(hsic_kl / denom)


def linear_cka_biased(X, Y) -> float:
    """Standard (biased) linear CKA — Kornblith et al. 2019.

    Stable for small n (e.g. per-class mean features with n = num_classes).
    Centered Gram matrices, no diagonal-zeroing.
    """
    X = _to_numpy(X)
    Y = _to_numpy(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}")
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ (X @ X.T) @ H
    Lc = H @ (Y @ Y.T) @ H
    hsic_kl = float(np.sum(Kc * Lc))
    hsic_kk = float(np.sum(Kc * Kc))
    hsic_ll = float(np.sum(Lc * Lc))
    denom = float(np.sqrt(max(hsic_kk * hsic_ll, 1e-30)))
    return hsic_kl / denom

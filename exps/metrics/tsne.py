"""PCA → t-SNE wrapper for visualizing CLS-token latent structure."""
from __future__ import annotations

import numpy as np


def pca_then_tsne(
    features: np.ndarray,
    pca_dim: int = 50,
    tsne_perplexity: float = 30.0,
    seed: int = 0,
) -> np.ndarray:
    """Reduce features (N, D) → (N, 2) via PCA-then-tSNE.

    Returns: (N, 2) float32.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    features = np.asarray(features, dtype=np.float32)
    N, D = features.shape
    pca_dim = min(pca_dim, N - 1, D)
    pca = PCA(n_components=pca_dim, random_state=seed)
    reduced = pca.fit_transform(features)
    tsne = TSNE(
        n_components=2,
        perplexity=min(tsne_perplexity, max(5, N // 4)),
        init='pca',
        learning_rate='auto',
        random_state=seed,
    )
    return tsne.fit_transform(reduced).astype(np.float32)

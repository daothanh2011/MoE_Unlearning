"""ViT-adapted Grad-CAM for MoE block outputs.

Standard Grad-CAM was designed for CNN feature maps. For ViT we:
  1. Take activation A ∈ (B, 197, D) and gradient dA ∈ (B, 197, D) at the
     MoE block's mlp output (post-Gram-Schmidt).
  2. Drop the CLS token: A[:, 1:] → (B, 196, D), dA[:, 1:] → (B, 196, D).
  3. Compute channel weights α = mean over patches of dA → (B, D).
  4. CAM = ReLU(Σ_d α_d · A_d) → (B, 196).
  5. Reshape to (B, 14, 14) and bilinearly upsample to (B, 224, 224) for
     overlay on the input image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_cam(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Compute Grad-CAM heatmap from a ViT block's activation + gradient.

    Args:
        activation: (B, 197, D)
        gradient:   (B, 197, D)

    Returns:
        (B, 14, 14) tensor of non-negative CAM values, NOT normalized.
    """
    if activation.shape != gradient.shape:
        raise ValueError(
            f"Activation/gradient shape mismatch: {activation.shape} vs {gradient.shape}"
        )
    if activation.shape[1] != 197:
        raise ValueError(f"Expected 197 tokens (1 CLS + 196 patches), got {activation.shape[1]}")

    A = activation[:, 1:, :]                                  # (B, 196, D)
    dA = gradient[:, 1:, :]                                   # (B, 196, D)
    alpha = dA.mean(dim=1)                                    # (B, D)
    cam = (alpha.unsqueeze(1) * A).sum(dim=-1)                # (B, 196)
    cam = F.relu(cam)
    return cam.reshape(-1, 14, 14)                            # (B, 14, 14)


def upsample_cam(cam: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Bilinearly upsample (B, 14, 14) → (B, size, size) and per-image normalize."""
    cam = cam.unsqueeze(1).float()                            # (B, 1, 14, 14)
    cam = F.interpolate(cam, size=(size, size), mode='bilinear', align_corners=False)
    cam = cam.squeeze(1)                                      # (B, H, W)
    # per-image min-max normalize to [0, 1]
    B = cam.shape[0]
    flat = cam.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    cam = (flat - mn) / (mx - mn + 1e-8)
    return cam.reshape(B, size, size)


def overlay_on_image(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a normalized CAM (H, W) on an RGB image (H, W, 3) using a jet colormap.

    Args:
        img: uint8 HxWx3
        cam: float32 HxW in [0, 1]

    Returns:
        uint8 HxWx3 overlay
    """
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]                               # HxWx3 in [0,1]
    heat = (heat * 255).astype(np.float32)
    img_f = img.astype(np.float32)
    out = (1 - alpha) * img_f + alpha * heat
    return out.clip(0, 255).astype(np.uint8)

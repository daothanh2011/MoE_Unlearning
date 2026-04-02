"""Visualize expert routing assignments for patch tokens in GMOE."""
import argparse
import os
import sys
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domainbed import algorithms


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt['model_hparams']
    algorithm = algorithms.GMOE(
        ckpt['model_input_shape'],
        ckpt['model_num_classes'],
        ckpt['model_num_domains'],
        hparams,
    )
    algorithm.load_state_dict(ckpt['model_dict'], strict=False)
    return algorithm.model.eval().cuda()


def register_gate_hooks(model):
    """Hook the cosine_top gate in each MoE block to capture routing logits."""
    routing_cache = {}

    def make_hook(block_idx):
        def hook(module, inp, out):
            # out: (B*T, num_experts) gate logits
            routing_cache[block_idx] = out.detach().cpu()
        return hook

    for i, blk in enumerate(model.blocks):
        if getattr(blk, 'cur_layer', None) == 'S' and hasattr(blk.mlp, 'gates'):
            blk.mlp.gates[0].register_forward_hook(make_hook(i))

    return routing_cache


def get_sample_images(data_dir, n_images, seed=42):
    """Load n_images random samples from data_dir using ImageFolder."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_images, len(dataset)))

    # Also keep original PIL images for display
    pil_transform = transforms.Compose([transforms.Resize((224, 224))])
    pil_dataset = datasets.ImageFolder(data_dir, transform=pil_transform)

    tensors, pil_imgs = [], []
    for idx in indices:
        tensors.append(dataset[idx][0])
        pil_imgs.append(np.array(pil_dataset[idx][0]))

    return tensors, pil_imgs


def expert_assignments(routing_cache, block_idx, topk=2, grid=14):
    """Extract per-patch top-k expert indices and weights from cached gate logits.

    Returns:
        indices: (grid, grid, k) int64 — top-k expert index per patch
        weights: (grid, grid, k) float — softmax-normalised weights over top-k
    """
    if block_idx not in routing_cache:
        raise KeyError(f"Block {block_idx} has no cached routing data. "
                       f"Available blocks: {list(routing_cache.keys())}")
    logits = routing_cache[block_idx]           # (B*T, num_experts)
    num_tokens = grid * grid + 1                # 197 for 224×224 / patch 16
    if logits.shape[0] >= num_tokens:
        logits = logits[-num_tokens:]
    logits = logits[1:]                         # drop CLS token → (196, num_experts)

    topk = min(topk, logits.shape[1])
    scores, indices = logits.topk(topk, dim=1) # both (196, k)
    weights = torch.softmax(scores.float(), dim=1)  # normalise to sum=1

    return indices.reshape(grid, grid, topk), weights.reshape(grid, grid, topk)


EXPERT_COLORS = np.array([
    [0.122, 0.467, 0.706],  # blue
    [1.000, 0.498, 0.055],  # orange
    [0.173, 0.627, 0.173],  # green
    [0.839, 0.153, 0.157],  # red
    [0.580, 0.404, 0.741],  # purple
    [0.549, 0.337, 0.294],  # brown
])


def overlay_expert_map(img_array, indices, weights, patch_size=16, alpha=0.5):
    """Blend top-k expert colors over each patch, weighted by gate scores.

    Args:
        indices: (grid, grid, k) — top-k expert indices per patch
        weights: (grid, grid, k) — corresponding softmax weights (sum to 1)
    """
    overlay = img_array.astype(np.float32).copy()
    grid = indices.shape[0]
    for r in range(grid):
        for c in range(grid):
            # Weighted mix of k expert colors
            patch_color = np.zeros(3)
            for ki in range(indices.shape[2]):
                expert = indices[r, c, ki].item()
                w = weights[r, c, ki].item()
                patch_color += w * EXPERT_COLORS[expert]
            patch_color *= 255
            r0, c0 = r * patch_size, c * patch_size
            r1, c1 = r0 + patch_size, c0 + patch_size
            overlay[r0:r1, c0:c1] = (
                (1 - alpha) * overlay[r0:r1, c0:c1] + alpha * patch_color
            )
    return overlay.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser('Visualize GMOE expert routing')
    parser.add_argument('--checkpoint', default='train_output/model.pkl')
    parser.add_argument('--data_dir',
                        default='./domainbed/data/terra_incognita/L43')
    parser.add_argument('--n_images', type=int, default=4)
    parser.add_argument('--block', type=int, default=8,
                        help='MoE block index to visualize (8 or 10)')
    parser.add_argument('--topk', type=int, default=2,
                        help='Number of top experts to blend per patch')
    parser.add_argument('--output', default='exps/expert_routing.png')
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.checkpoint)

    routing_cache = register_gate_hooks(model)

    print(f"Loading {args.n_images} images from {args.data_dir}...")
    tensors, pil_imgs = get_sample_images(args.data_dir, args.n_images)

    # Run forward passes one image at a time so the cache holds per-image logits.
    all_indices, all_weights = [], []
    with torch.no_grad():
        for tensor in tensors:
            routing_cache.clear()
            model(tensor.unsqueeze(0).cuda())
            idx, wt = expert_assignments(routing_cache, args.block, topk=args.topk)
            all_indices.append(idx)
            all_weights.append(wt)

    # Build figure
    n = len(tensors)
    num_experts = 6
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, pil_img, idx, wt in zip(axes, pil_imgs, all_indices, all_weights):
        blended = overlay_expert_map(pil_img, idx, wt)
        ax.imshow(blended)
        ax.axis('off')

    legend_handles = [
        mpatches.Patch(color=EXPERT_COLORS[i], label=f'Expert {i}')
        for i in range(num_experts)
    ]
    fig.legend(
        handles=legend_handles,
        loc='center right',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=12,
        frameon=True,
        title='Experts',
    )
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()

"""End-to-end evaluation diagnostics for ViT-S/16 + GMoEOMoE on PACS.

Computes 4 diagnostic metrics on the held-out val splits of selected PACS
domains, capturing tensors at MoE blocks 8 and 10:

  1. Activation MSO (mean squared overlap of expert outputs, pre-Gram-Schmidt)
  2. Debiased linear CKA (pairwise representation similarity across domains)
  3. ViT-adapted Grad-CAM (spatial saliency at MoE block outputs)
  4. PCA → t-SNE (CLS-token latent structure)

Usage:
  python -m exps.eval_moe_diagnostics \\
      --checkpoint train_output/phase1/pacs_omoe_envA/model.pkl \\
      --metric all \\
      --domains P C S \\
      --pairs P-C P-S \\
      --output_dir exps/eval_outputs/pacs_omoe_envA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Repo root on path so `domainbed` imports work when invoked as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# vision_transformer.py expects DOMAINBED_PROJECT_DIR to find vit_helpers.
os.environ.setdefault('DOMAINBED_PROJECT_DIR', _REPO_ROOT)

from domainbed import algorithms, datasets
from domainbed.lib import misc

from exps.moe_hooks import MoECapture, GradCAMCapture, MOE_BLOCK_INDICES
from exps.metrics.mso import MSOAccumulator
from exps.metrics.cka import linear_cka_debiased, linear_cka_biased
from exps.metrics.gradcam_vit import compute_cam, upsample_cam, overlay_on_image
from exps.metrics.tsne import pca_then_tsne


# PACS env name → index
PACS_DOMAINS = {'A': 0, 'C': 1, 'P': 2, 'S': 3}


# ---------------------------------------------------------------------------
# Checkpoint + dataset loading
# ---------------------------------------------------------------------------

def load_algorithm(checkpoint_path: str, device: str = 'cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt['model_hparams']
    # Always instantiate as GMoEOMoE — it gracefully handles use_omoe=False too,
    # and is the only path that wires the use_omoe flag through.
    algorithm = algorithms.GMoEOMoE(
        ckpt['model_input_shape'],
        ckpt['model_num_classes'],
        ckpt['model_num_domains'],
        hparams,
    )
    algorithm.load_state_dict(ckpt['model_dict'], strict=False)
    algorithm.to(device).eval()
    return algorithm, ckpt


def build_pacs_val_splits(
    data_dir: str,
    test_envs: List[int],
    holdout_fraction: float = 0.2,
    trial_seed: int = 0,
    hparams: dict = None,
) -> Dict[str, Subset]:
    """Replicate train.py's per-env out-splits (the held-out 20%) for PACS.

    Returns a dict {domain_letter: Subset} for all 4 PACS domains.
    """
    hparams = hparams or {'data_augmentation': False, 'class_balanced': False}
    dataset = datasets.PACS(data_dir, test_envs, hparams)
    out = {}
    for env_i, env in enumerate(dataset):
        # split_dataset(env, n_first, seed) → (out, in_)
        out_split, _in = misc.split_dataset(
            env,
            int(len(env) * holdout_fraction),
            misc.seed_hash(trial_seed, env_i),
        )
        letter = datasets.PACS.ENVIRONMENTS[env_i]
        out[letter] = out_split
    return out


def make_loader(subset, batch_size: int, num_workers: int = 2):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Metric runners
# ---------------------------------------------------------------------------

def run_mso(model, splits: Dict[str, Subset], device: str, batch_size: int = 64) -> dict:
    """MSO per (block, domain), reporting both pre-GS and post-GS."""
    results = {idx: {} for idx in MOE_BLOCK_INDICES}
    with MoECapture(model, MOE_BLOCK_INDICES) as cap:
        for letter, subset in splits.items():
            accs_pre = {idx: MSOAccumulator() for idx in MOE_BLOCK_INDICES}
            accs_post = {idx: MSOAccumulator() for idx in MOE_BLOCK_INDICES}
            loader = make_loader(subset, batch_size)
            with torch.no_grad():
                for x, _y in loader:
                    cap.clear()
                    model(x.to(device))
                    for idx in MOE_BLOCK_INDICES:
                        accs_pre[idx].update(cap.captured[idx]['pre_gs'])
                        accs_post[idx].update(cap.captured[idx]['post_gs'])
            for idx in MOE_BLOCK_INDICES:
                results[idx][letter] = {
                    'pre_gs': accs_pre[idx].value(),
                    'post_gs': accs_post[idx].value(),
                }
    return results


def extract_block_features(
    model, subset: Subset, device: str, batch_size: int = 32
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], torch.Tensor]:
    """Extract block-output features for all MoE blocks + the last block.

    Returns:
        features: {block_idx: {'cls': (N, D), 'mean_patch': (N, D)}}
        labels:   (N,) int64 — class labels in the same order as features.
    """
    features = {}
    labels_all = []
    with MoECapture(model, MOE_BLOCK_INDICES, also_capture_last_block=True) as cap:
        cls_feats = {idx: [] for idx in cap.block_indices}
        mp_feats = {idx: [] for idx in cap.block_indices}
        loader = make_loader(subset, batch_size)
        with torch.no_grad():
            for x, y in loader:
                cap.clear()
                model(x.to(device))
                for idx in cap.block_indices:
                    bo = cap.captured[idx]['block_out']     # (B, 197, D)
                    cls_feats[idx].append(bo[:, 0, :])
                    mp_feats[idx].append(bo[:, 1:, :].mean(dim=1))
                labels_all.append(y)
        for idx in cap.block_indices:
            features[idx] = {
                'cls': torch.cat(cls_feats[idx], dim=0),
                'mean_patch': torch.cat(mp_feats[idx], dim=0),
            }
    return features, torch.cat(labels_all, dim=0)


def _pair_by_class(
    feats_a: torch.Tensor, labels_a: torch.Tensor,
    feats_b: torch.Tensor, labels_b: torch.Tensor,
    k_per_class: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build paired (X, Y) feature matrices aligned by class label.

    For each class c present in BOTH domains, pick k = min(k_per_class,
    available_a, available_b) samples from each side and stack them in the
    same order. Row i of X and row i of Y are guaranteed to share class c.

    Returns (X, Y, n_classes_used). Both X, Y have shape (k_used * n_classes, D)
    where k_used can vary per class — we always pick min(k_per_class, available).
    """
    rng = np.random.RandomState(seed)
    common = sorted(set(labels_a.tolist()) & set(labels_b.tolist()))

    X_rows = []
    Y_rows = []
    for c in common:
        idx_a = (labels_a == c).nonzero(as_tuple=True)[0].tolist()
        idx_b = (labels_b == c).nonzero(as_tuple=True)[0].tolist()
        k = min(k_per_class, len(idx_a), len(idx_b))
        if k == 0:
            continue
        sel_a = rng.choice(len(idx_a), size=k, replace=False)
        sel_b = rng.choice(len(idx_b), size=k, replace=False)
        for i in range(k):
            X_rows.append(feats_a[idx_a[sel_a[i]]])
            Y_rows.append(feats_b[idx_b[sel_b[i]]])

    X = torch.stack(X_rows, dim=0)
    Y = torch.stack(Y_rows, dim=0)
    return X, Y, len(common)


def run_cka(
    model,
    splits: Dict[str, Subset],
    pairs: List[Tuple[str, str]],
    device: str,
    batch_size: int = 32,
    k_per_class: int = 30,
    seed: int = 0,
) -> dict:
    """Pair-by-class sample CKA between domain pairs at all captured blocks.

    For unpaired cross-domain comparison, raw sample-level CKA is meaningless
    (no per-row correspondence) and per-class-mean CKA is too noisy for the
    PACS class count of 7 (debiased CKA noise floor ≈ ±0.26 at n=7, D=384,
    biased CKA saturates near 1).

    Pair-by-class compromises: take k_per_class samples per class from each
    domain, sorted by class label so row i of X and row i of Y come from the
    same class but DIFFERENT instances. n = k_per_class * num_classes (e.g.
    30 * 7 = 210 for PACS) is large enough that debiased CKA is stable.

    Reports both biased (Kornblith) and debiased CKA. With n=210, both are
    reliable; biased should be slightly higher than debiased.
    """
    needed_domains = set()
    for a, b in pairs:
        needed_domains.add(a)
        needed_domains.add(b)

    feats_by_dom: Dict[str, dict] = {}
    labels_by_dom: Dict[str, torch.Tensor] = {}
    for letter in sorted(needed_domains):
        print(f"  [CKA] extracting features for domain {letter}...")
        feats_by_dom[letter], labels_by_dom[letter] = extract_block_features(
            model, splits[letter], device, batch_size
        )

    block_indices = list(next(iter(feats_by_dom.values())).keys())
    results = {idx: {} for idx in block_indices}

    for a, b in pairs:
        for idx in block_indices:
            res_view = {}
            for view in ('cls', 'mean_patch'):
                X, Y, n_classes = _pair_by_class(
                    feats_by_dom[a][idx][view], labels_by_dom[a],
                    feats_by_dom[b][idx][view], labels_by_dom[b],
                    k_per_class=k_per_class,
                    seed=seed,
                )
                res_view[view] = {
                    'biased': linear_cka_biased(X, Y),
                    'debiased': linear_cka_debiased(X, Y),
                    'n_samples': X.shape[0],
                    'n_classes': n_classes,
                    'k_per_class': k_per_class,
                }
            results[idx][f"{a}-{b}"] = res_view
    return results


def run_gradcam(
    model,
    splits: Dict[str, Subset],
    device: str,
    output_dir: str,
    n_per_domain: int = 8,
):
    """Generate Grad-CAM overlays for a few images per domain at MoE blocks 8, 10."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    inv_norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    inv_norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for letter, subset in splits.items():
        n = min(n_per_domain, len(subset))
        rows = []
        for sample_i in range(n):
            x, y = subset[sample_i]
            x = x.unsqueeze(0).to(device)

            cams_per_block = {}
            for idx in MOE_BLOCK_INDICES:
                with GradCAMCapture(model, idx) as cam_cap:
                    model.zero_grad()
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = (logits[0] + logits[1]) / 2
                    pred = logits.argmax(dim=-1)
                    target = logits[0, pred[0]]
                    target.backward()
                    # Detach activation: it carries requires_grad from the forward
                    # graph; the gradient was already detached inside the hook.
                    cam = compute_cam(cam_cap.activation.detach(), cam_cap.gradient)
                    cam_up = upsample_cam(cam, size=224)[0].cpu().numpy()
                cams_per_block[idx] = cam_up

            img = (x[0].cpu() * inv_norm_std + inv_norm_mean).clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            rows.append((img_np, cams_per_block, int(pred[0])))

        # Plot grid: rows=images, cols=[orig, blk8, blk10]
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None, :]
        for i, (img_np, cams, pred) in enumerate(rows):
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"{letter} #{i} pred={pred}")
            axes[i, 0].axis('off')
            for j, idx in enumerate(MOE_BLOCK_INDICES, start=1):
                overlay = overlay_on_image(img_np, cams[idx])
                axes[i, j].imshow(overlay)
                axes[i, j].set_title(f"block {idx}")
                axes[i, j].axis('off')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"gradcam_{letter}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=120)
        plt.close()
        print(f"  [GradCAM] saved {out_path}")


def run_tsne(
    model,
    splits: Dict[str, Subset],
    device: str,
    output_dir: str,
    batch_size: int = 64,
):
    """PCA→t-SNE on CLS tokens from MoE blocks + last block, color by domain & class."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    feats_by_block = {}
    domain_labels = []
    class_labels = []
    domain_letters = list(splits.keys())

    for letter in domain_letters:
        subset = splits[letter]
        with MoECapture(model, MOE_BLOCK_INDICES, also_capture_last_block=True) as cap:
            cls_per_block = {idx: [] for idx in cap.block_indices}
            loader = make_loader(subset, batch_size)
            ys = []
            with torch.no_grad():
                for x, y in loader:
                    cap.clear()
                    model(x.to(device))
                    for idx in cap.block_indices:
                        cls_per_block[idx].append(cap.captured[idx]['block_out'][:, 0, :])
                    ys.append(y)
            for idx in cap.block_indices:
                feats_by_block.setdefault(idx, []).append(torch.cat(cls_per_block[idx], dim=0))
            ys = torch.cat(ys, dim=0)
            domain_labels.extend([letter] * len(ys))
            class_labels.extend(ys.tolist())

    domain_labels = np.array(domain_labels)
    class_labels = np.array(class_labels)

    for idx, blocks_list in feats_by_block.items():
        X = torch.cat(blocks_list, dim=0).numpy()
        emb = pca_then_tsne(X)
        for color_by, labels in [('domain', domain_labels), ('class', class_labels)]:
            fig, ax = plt.subplots(figsize=(7, 6))
            uniq = sorted(set(labels.tolist()))
            cmap = plt.get_cmap('tab10' if len(uniq) <= 10 else 'tab20')
            for i, u in enumerate(uniq):
                mask = labels == u
                ax.scatter(emb[mask, 0], emb[mask, 1], s=10, label=str(u), color=cmap(i % cmap.N), alpha=0.7)
            ax.legend(fontsize=8, loc='best')
            ax.set_title(f"t-SNE block {idx} (color: {color_by})")
            ax.axis('off')
            out_path = os.path.join(output_dir, f"tsne_block{idx}_by_{color_by}.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=120)
            plt.close()
            print(f"  [t-SNE] saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _stringify(d):
    """Convert int keys (block indices, tuple pair-keys) to str for JSON."""
    if isinstance(d, dict):
        return {str(k): _stringify(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_stringify(v) for v in d]
    return d


def run_one_checkpoint(
    checkpoint_path: str,
    args,
    device: str,
    label: str,
    skip_mso: bool = False,
):
    """Load a checkpoint, run (optionally MSO) + CKA, return a results dict.

    Grad-CAM and t-SNE are NOT run here — they're per-checkpoint outputs that
    main() schedules separately on the primary (OMoE) checkpoint only.

    skip_mso: pass True for baseline checkpoints with gate_k < 2 (MSO is
    undefined for k<2; the metric only measures inter-expert overlap).
    """
    print(f"\n--- {label}: loading {checkpoint_path} ---")
    algorithm, ckpt = load_algorithm(checkpoint_path, device)
    model = algorithm.model

    all_splits = build_pacs_val_splits(
        args.data_dir,
        test_envs=args.test_envs,
        holdout_fraction=args.holdout_fraction,
        trial_seed=args.trial_seed,
        hparams=ckpt['model_hparams'],
    )
    eval_splits = {d: all_splits[d] for d in args.domains}
    for d, sub in eval_splits.items():
        print(f"  domain {d}: {len(sub)} held-out images")
    pairs = [tuple(p.split('-')) for p in args.pairs]

    gate_k = int(ckpt['model_hparams'].get('gate_k', 1))
    out = {
        'checkpoint': checkpoint_path,
        'use_omoe': bool(ckpt['model_hparams'].get('use_omoe', False)),
        'gate_k': gate_k,
        'algorithm': ckpt.get('args', {}).get('algorithm') if isinstance(ckpt.get('args'), dict) else None,
    }

    can_run_mso = (gate_k >= 2) and not skip_mso
    if args.metric in ('mso', 'all') and can_run_mso:
        print(f"  [{label}] MSO ...")
        out['mso'] = run_mso(model, eval_splits, device, batch_size=args.mso_batch)
        for idx, dom_dict in out['mso'].items():
            for d, vals in dom_dict.items():
                print(f"    block {idx} domain {d}: pre={vals['pre_gs']:.4e}  post={vals['post_gs']:.4e}")
    elif args.metric in ('mso', 'all'):
        print(f"  [{label}] skipping MSO (gate_k={gate_k} < 2)")

    if args.metric in ('cka', 'all'):
        print(f"  [{label}] CKA (k={args.cka_per_class} per class)...")
        out['cka'] = run_cka(
            model, all_splits, pairs, device,
            batch_size=args.cka_batch,
            k_per_class=args.cka_per_class,
        )
        for idx, pair_dict in out['cka'].items():
            for pair_name, vals in pair_dict.items():
                cls_b = vals['cls']['biased']
                cls_d = vals['cls']['debiased']
                mp_b = vals['mean_patch']['biased']
                mp_d = vals['mean_patch']['debiased']
                n = vals['cls']['n_samples']
                print(f"    block {idx} {pair_name}: cls(b/d)={cls_b:.4f}/{cls_d:.4f}  mean_patch(b/d)={mp_b:.4f}/{mp_d:.4f}  n={n}")

    return out, model, eval_splits


def make_comparison_plots(omoe: dict, baseline: dict, output_dir: str):
    """Bar charts comparing MSO and CKA between OMoE and baseline."""
    import matplotlib.pyplot as plt
    import numpy as np

    # ----- CKA comparison (per-class biased CKA — stable for small num_classes) -----
    if 'cka' in omoe and 'cka' in baseline:
        block_indices = sorted(omoe['cka'].keys())
        pair_names = sorted({pn for d in omoe['cka'].values() for pn in d.keys()})

        for view in ('cls', 'mean_patch'):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x_labels = []
            omoe_vals = []
            base_vals = []
            for idx in block_indices:
                for pn in pair_names:
                    x_labels.append(f"blk{idx}\n{pn}")
                    omoe_vals.append(omoe['cka'][idx][pn][view]['biased'])
                    base_vals.append(baseline['cka'][idx][pn][view]['biased'])
            x = np.arange(len(x_labels))
            w = 0.38
            ax.bar(x - w/2, base_vals, w, label='baseline (no OMoE)', color='#888888')
            ax.bar(x + w/2, omoe_vals, w, label='OMoE on', color='#1f77b4')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylabel(f'biased linear CKA ({view}, per-class means)')
            ax.set_title(f'Cross-domain CKA: OMoE vs baseline ({view})')
            ax.legend()
            ax.set_ylim(0, 1.05)
            for xi, (a, b) in enumerate(zip(base_vals, omoe_vals)):
                delta = b - a
                ax.text(xi, max(a, b) + 0.02, f'Δ={delta:+.3f}',
                        ha='center', fontsize=8, color='red' if delta > 0 else 'black')
            plt.tight_layout()
            out_path = os.path.join(output_dir, f'cka_comparison_{view}.png')
            plt.savefig(out_path, bbox_inches='tight', dpi=120)
            plt.close()
            print(f"  saved {out_path}")

    # ----- MSO comparison (pre-GS only — post-GS is ~0 for OMoE by design) -----
    # Skipped if baseline has gate_k<2 (MSO undefined there).
    if 'mso' in omoe and 'mso' in baseline:
        block_indices = sorted(omoe['mso'].keys())
        domains = sorted({d for blk in omoe['mso'].values() for d in blk.keys()})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_labels = []
        omoe_pre = []
        base_pre = []
        for idx in block_indices:
            for dom in domains:
                x_labels.append(f"blk{idx}\n{dom}")
                omoe_pre.append(omoe['mso'][idx][dom]['pre_gs'])
                base_pre.append(baseline['mso'][idx][dom]['pre_gs'])
        x = np.arange(len(x_labels))
        w = 0.38
        ax.bar(x - w/2, base_pre, w, label='baseline (no OMoE)', color='#888888')
        ax.bar(x + w/2, omoe_pre, w, label='OMoE on', color='#1f77b4')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('MSO_pre (mean squared overlap)')
        ax.set_title('Expert overlap (pre-Gram-Schmidt): OMoE vs baseline')
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, 'mso_comparison.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=120)
        plt.close()
        print(f"  saved {out_path}")


def compute_deltas(omoe: dict, baseline: dict) -> dict:
    """Per-(block, pair, view) CKA delta, and per-(block, domain) MSO_pre delta."""
    deltas = {'cka': {}, 'mso': {}}
    if 'cka' in omoe and 'cka' in baseline:
        for idx in omoe['cka']:
            deltas['cka'][idx] = {}
            for pn in omoe['cka'][idx]:
                deltas['cka'][idx][pn] = {
                    'cls_biased': omoe['cka'][idx][pn]['cls']['biased'] - baseline['cka'][idx][pn]['cls']['biased'],
                    'mean_patch_biased': omoe['cka'][idx][pn]['mean_patch']['biased'] - baseline['cka'][idx][pn]['mean_patch']['biased'],
                }
    if 'mso' in omoe and 'mso' in baseline:
        for idx in omoe['mso']:
            deltas['mso'][idx] = {}
            for d in omoe['mso'][idx]:
                deltas['mso'][idx][d] = {
                    'pre_gs': omoe['mso'][idx][d]['pre_gs'] - baseline['mso'][idx][d]['pre_gs'],
                }
    return deltas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Primary (OMoE) checkpoint')
    p.add_argument('--baseline_checkpoint', default=None,
                   help='Optional baseline checkpoint (e.g. same setup with use_omoe=false). '
                        'When provided, MSO + CKA are computed for both and a comparison '
                        'JSON + bar charts are written.')
    p.add_argument('--data_dir', default='./domainbed/data')
    p.add_argument('--output_dir', default='exps/eval_outputs/pacs_omoe_envA')
    p.add_argument('--metric', choices=['mso', 'cka', 'gradcam', 'tsne', 'all'], default='all')
    p.add_argument('--domains', nargs='+', default=['P', 'C', 'S'],
                   help='Domain letters to use for MSO / t-SNE / Grad-CAM')
    p.add_argument('--pairs', nargs='+', default=['P-C', 'P-S'],
                   help='Domain pairs for CKA, format LETTER-LETTER')
    p.add_argument('--test_envs', nargs='+', type=int, default=[0],
                   help='Test envs used during training (for replicating splits)')
    p.add_argument('--holdout_fraction', type=float, default=0.2)
    p.add_argument('--trial_seed', type=int, default=0)
    p.add_argument('--mso_batch', type=int, default=64)
    p.add_argument('--cka_batch', type=int, default=32)
    p.add_argument('--cka_per_class', type=int, default=30,
                   help='Samples per class for pair-by-class CKA. n_total = k * num_classes.')
    p.add_argument('--tsne_batch', type=int, default=64)
    p.add_argument('--gradcam_per_domain', type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- Primary (OMoE) checkpoint -----
    omoe_out, omoe_model, omoe_splits = run_one_checkpoint(
        args.checkpoint, args, device, label='OMoE'
    )

    # Grad-CAM and t-SNE are computed on the primary checkpoint only
    if args.metric in ('gradcam', 'all'):
        print("\n=== Grad-CAM (OMoE) ===")
        run_gradcam(omoe_model, omoe_splits, device, args.output_dir,
                    n_per_domain=args.gradcam_per_domain)

    if args.metric in ('tsne', 'all'):
        print("\n=== PCA → t-SNE (OMoE) ===")
        run_tsne(omoe_model, omoe_splits, device, args.output_dir,
                 batch_size=args.tsne_batch)

    # Free GPU before loading the baseline
    del omoe_model, omoe_splits
    torch.cuda.empty_cache()

    # ----- Optional baseline -----
    full_out = {'omoe': omoe_out}
    if args.baseline_checkpoint:
        baseline_out, _baseline_model, _ = run_one_checkpoint(
            args.baseline_checkpoint, args, device, label='baseline'
        )
        del _baseline_model
        torch.cuda.empty_cache()

        full_out['baseline'] = baseline_out
        full_out['delta_omoe_minus_baseline'] = compute_deltas(omoe_out, baseline_out)

        print("\n=== Comparison plots ===")
        make_comparison_plots(omoe_out, baseline_out, args.output_dir)

        print("\n=== Δ summary (OMoE − baseline) ===")
        if 'cka' in full_out['delta_omoe_minus_baseline']:
            for idx, pair_dict in full_out['delta_omoe_minus_baseline']['cka'].items():
                for pn, v in pair_dict.items():
                    print(f"  CKA  block {idx} {pn}: Δcls={v['cls_biased']:+.4f}  Δmean_patch={v['mean_patch_biased']:+.4f}")
        if 'mso' in full_out['delta_omoe_minus_baseline']:
            for idx, dom_dict in full_out['delta_omoe_minus_baseline']['mso'].items():
                for d, v in dom_dict.items():
                    print(f"  MSO  block {idx} {d}: Δpre={v['pre_gs']:+.4e}")

    json_path = os.path.join(args.output_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(_stringify(full_out), f, indent=2)
    print(f"\nWrote {json_path}")


if __name__ == '__main__':
    main()

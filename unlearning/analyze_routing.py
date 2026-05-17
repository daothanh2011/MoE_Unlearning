# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Inspect MoE expert routing per class and per domain.

This is the same diagnostic that ``GMOE_Full_Unlearn.compute_expert_relevance_scores``
runs on the forget set, generalised to:
    - any subset of the data (full / retain / forget / unseen)
    - a per-class breakdown    (rows: classes,  cols: experts)
    - a per-domain breakdown   (rows: domains,  cols: experts)
    - a per-(domain, class) breakdown (optional, can be large)

Outputs are printed as text tables and saved as CSV under ``--out_dir``.
If matplotlib is available, PNG heatmaps are also written.

Example
-------
    python3.11 unlearning/analyze_routing.py \
        --algorithm GMOE_Full_Unlearn \
        --checkpoint_path unlearning/train_output/GMOE_Full_origin_PACS_random_0.4_seed_0/model_final.pt \
        --dataset PACS \
        --hparams '{"model":"deit_small_patch16_224","num_experts":12,"gate_k":3,
                    "mlp_ratio":4,"expert_depth":2,"moe_blocks":[8,10],
                    "lambda_inv":0.001,"lambda_sp":0.01,"lambda_bal":0.01,
                    "lambda_div":0.001,"inv_type":"OT"}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import domainbed.tutel_patch  # noqa: F401
from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib import misc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Analyse GMoE routing.')
    p.add_argument('--algorithm', default='GMOE_Full_Unlearn')
    p.add_argument('--dataset', default='PACS')
    p.add_argument('--data_dir', default='./domainbed/data')
    p.add_argument('--test_envs', type=int, nargs='+', default=[0])
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--hparams', type=str, default=None,
                   help='JSON-serialised hparams dict (matched to training).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_samples', type=int, default=0,
                   help='Optional cap on samples per subset (0 = no cap).')
    p.add_argument('--unlearn_setting', choices=['random', 'class'],
                   default='random')
    p.add_argument('--unlearn_random_ratio', type=float, default=0.4)
    p.add_argument('--unlearn_num_class', type=int, default=1)
    p.add_argument('--subsets', nargs='+',
                   default=['full', 'retain', 'forget'],
                   choices=['full', 'retain', 'forget', 'unseen'])
    p.add_argument('--out_dir', default=None,
                   help='Defaults to <ckpt_dir>/routing_analysis/')
    p.add_argument('--with_class_per_domain', action='store_true',
                   help='Also dump per-(domain, class) tables (large).')
    return p.parse_args()


def _build_hparams(args: argparse.Namespace) -> dict:
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    return hparams


def _resolve_domain_label(dataset, env_idx: int) -> str:
    """Best-effort human-readable name for env ``env_idx``."""
    envs = getattr(dataset, 'ENVIRONMENTS', None)
    if envs and env_idx < len(envs):
        return str(envs[env_idx])
    return f'env{env_idx}'


def _iter_env_indices(dataset) -> List[Tuple[int, int, int]]:
    """Yield (global_idx, env_idx, local_idx) for every sample in dataset.

    Mirrors what ``ConcatDataset`` does internally but exposes env_idx so we
    can group per-domain. Caller can filter / subsample as needed.
    """
    out = []
    g = 0
    for env_idx, env in enumerate(dataset.datasets):
        for local in range(len(env)):
            out.append((g, env_idx, local))
            g += 1
    return out


def _compute_routing_table(
    algorithm,
    indices_with_env: List[Tuple[int, int, int]],
    full_dataset,
    num_classes: int,
    num_domains: int,
    num_experts: int,
    batch_size: int,
    num_workers: int,
    device: str,
    max_samples: int = 0,
):
    """Accumulate per-(class, expert) and per-(domain, expert) routing means.

    Returns (per_class, per_domain, per_class_per_domain, count_class,
             count_domain, count_class_domain).
    """
    if max_samples and max_samples < len(indices_with_env):
        rng = np.random.default_rng(0)
        keep = rng.choice(len(indices_with_env), size=max_samples, replace=False)
        indices_with_env = [indices_with_env[i] for i in keep]

    global_indices = [g for g, _e, _l in indices_with_env]
    env_for_global = {g: e for g, e, _l in indices_with_env}

    sub = Subset(full_dataset, global_indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)

    per_class = torch.zeros(num_classes, num_experts, device=device)
    per_domain = torch.zeros(num_domains, num_experts, device=device)
    per_cd = torch.zeros(num_classes, num_domains, num_experts, device=device)
    count_class = torch.zeros(num_classes, device=device)
    count_domain = torch.zeros(num_domains, device=device)
    count_cd = torch.zeros(num_classes, num_domains, device=device)

    cursor = 0
    algorithm.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _logits, pi, _h = algorithm._forward(x)  # pi: (B, M)

            # Each sample inherits the env of its source row via env_for_global.
            batch_g = global_indices[cursor:cursor + x.size(0)]
            envs = torch.tensor([env_for_global[g] for g in batch_g],
                                device=device, dtype=torch.long)
            cursor += x.size(0)

            per_class.index_add_(0, y, pi)
            count_class.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
            per_domain.index_add_(0, envs, pi)
            count_domain.index_add_(0, envs, torch.ones_like(envs, dtype=torch.float))

            flat_cd = y * num_domains + envs
            per_cd.view(num_classes * num_domains, num_experts).index_add_(
                0, flat_cd, pi)
            count_cd.view(-1).index_add_(0, flat_cd,
                                         torch.ones_like(flat_cd, dtype=torch.float))

    per_class = per_class / count_class.clamp_min(1).unsqueeze(1)
    per_domain = per_domain / count_domain.clamp_min(1).unsqueeze(1)
    per_cd = per_cd / count_cd.clamp_min(1).unsqueeze(-1)

    return {
        'per_class': per_class.cpu().numpy(),
        'per_domain': per_domain.cpu().numpy(),
        'per_class_per_domain': per_cd.cpu().numpy(),
        'count_class': count_class.cpu().numpy().astype(int),
        'count_domain': count_domain.cpu().numpy().astype(int),
        'count_class_domain': count_cd.cpu().numpy().astype(int),
    }


def _format_table(
    mat: np.ndarray, row_labels: List[str], col_labels: List[str],
    fmt: str = '{:>7.3f}',
) -> str:
    head = '{:<16}'.format('') + ''.join(f'{c:>10}' for c in col_labels)
    rows = [head]
    for r, label in enumerate(row_labels):
        cells = ''.join(f'{fmt.format(mat[r, c]):>10}' for c in range(mat.shape[1]))
        rows.append('{:<16}'.format(label) + cells)
    return '\n'.join(rows)


def _entropy(pi: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats. ``pi`` rows should already sum to ~1."""
    p = np.clip(pi, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def _try_save_heatmap(mat: np.ndarray, row_labels, col_labels, title, path):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(0.55 * len(col_labels) + 2,
                                    0.4 * len(row_labels) + 2))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f'{mat[r, c]:.2f}',
                    ha='center', va='center',
                    color='white' if mat[r, c] < 0.6 * mat.max() else 'black',
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _save_csv(mat: np.ndarray, row_labels, col_labels, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + list(col_labels))
        for r, label in enumerate(row_labels):
            w.writerow([label] + [f'{x:.6f}' for x in mat[r]])


def _build_subset_indices(
    dataset, args: argparse.Namespace,
) -> dict:
    """Reproduce the deterministic train/test/unseen + retain/forget split used
    in unlearning/train.py and unlearning/unlearn.py."""
    indices_with_env = _iter_env_indices(dataset)
    full_dataset = ConcatDataset([env for env in dataset])
    total_size = len(full_dataset)
    assert total_size == len(indices_with_env)

    train_size = int(total_size * 0.8)
    test_size = int(total_size * 0.1)

    train_idx = indices_with_env[:train_size]
    test_idx = indices_with_env[train_size: train_size + test_size]
    unseen_idx = indices_with_env[train_size + test_size:]

    if args.unlearn_setting == 'random':
        ratio = float(args.unlearn_random_ratio)
        forget_size = int(train_size * ratio)
        forget_idx = train_idx[:forget_size]
        retain_idx = train_idx[forget_size:]
    else:
        forget_classes = set(range(int(args.unlearn_num_class)))
        retain_idx, forget_idx = [], []
        for g, e, _l in train_idx:
            _x, y = full_dataset[g]
            y = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
            (forget_idx if y in forget_classes else retain_idx).append((g, e, _l))

    return {
        'full_dataset': full_dataset,
        'full': train_idx + test_idx + unseen_idx,
        'retain': retain_idx,
        'forget': forget_idx,
        'unseen': unseen_idx,
    }


def main() -> None:
    args = _parse_args()
    misc.print_separator = lambda: None  # quiet helper; harmless if absent
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hparams = _build_hparams(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset not in vars(datasets):
        raise NotImplementedError(args.dataset)
    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)

    num_classes = dataset.num_classes
    num_domains = len(dataset.datasets)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, num_classes, 1, hparams)
    ckpt = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    algorithm.network.load_state_dict(ckpt)
    algorithm.to(device)

    num_experts = algorithm.moe_head.num_experts
    expert_labels = [f'e{i}' for i in range(num_experts)]

    class_labels = []
    sample_env0 = dataset.datasets[0]
    if hasattr(sample_env0, 'classes') and sample_env0.classes:
        class_labels = [str(c) for c in sample_env0.classes]
    if len(class_labels) != num_classes:
        class_labels = [f'c{i}' for i in range(num_classes)]
    domain_labels = [_resolve_domain_label(dataset, i) for i in range(num_domains)]

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint_path)), 'routing_analysis')
    os.makedirs(out_dir, exist_ok=True)

    splits = _build_subset_indices(dataset, args)
    summary = {}

    for split in args.subsets:
        indices_with_env = splits[split]
        if not indices_with_env:
            print(f'[!] split {split!r} is empty, skipping')
            continue
        print(f'\n=== {split.upper()}  '
              f'(n={len(indices_with_env)}'
              f', max_samples={args.max_samples or "all"}) ===')

        result = _compute_routing_table(
            algorithm,
            indices_with_env,
            splits['full_dataset'],
            num_classes=num_classes,
            num_domains=num_domains,
            num_experts=num_experts,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            max_samples=args.max_samples,
        )

        per_class = result['per_class']
        per_domain = result['per_domain']
        ent_per_class = _entropy(per_class) / np.log(num_experts)  # in [0, 1]
        ent_per_domain = _entropy(per_domain) / np.log(num_experts)

        print('\nMean routing weight per CLASS x EXPERT '
              '(rows sum ~1; entropy=H/log M ∈ [0=hard-routed,1=uniform])')
        print(_format_table(per_class, class_labels, expert_labels))
        print('  normalized_entropy: ' +
              ', '.join(f'{c}={ent_per_class[i]:.3f}' for i, c in enumerate(class_labels)))

        print('\nMean routing weight per DOMAIN x EXPERT')
        print(_format_table(per_domain, domain_labels, expert_labels))
        print('  normalized_entropy: ' +
              ', '.join(f'{d}={ent_per_domain[i]:.3f}' for i, d in enumerate(domain_labels)))

        _save_csv(per_class, class_labels, expert_labels,
                  os.path.join(out_dir, f'{split}_per_class.csv'))
        _save_csv(per_domain, domain_labels, expert_labels,
                  os.path.join(out_dir, f'{split}_per_domain.csv'))

        _try_save_heatmap(per_class, class_labels, expert_labels,
                          f'{split} – routing weight per class',
                          os.path.join(out_dir, f'{split}_per_class.png'))
        _try_save_heatmap(per_domain, domain_labels, expert_labels,
                          f'{split} – routing weight per domain',
                          os.path.join(out_dir, f'{split}_per_domain.png'))

        if args.with_class_per_domain:
            per_cd = result['per_class_per_domain']  # (C, D, M)
            for d, dom in enumerate(domain_labels):
                mat = per_cd[:, d, :]
                title = f'{split} – {dom} – routing per class'
                _save_csv(mat, class_labels, expert_labels,
                          os.path.join(out_dir, f'{split}_{dom}_per_class.csv'))
                _try_save_heatmap(mat, class_labels, expert_labels, title,
                                  os.path.join(out_dir, f'{split}_{dom}_per_class.png'))

        summary[split] = {
            'n_samples': int(len(indices_with_env)),
            'per_class_entropy_mean': float(ent_per_class.mean()),
            'per_domain_entropy_mean': float(ent_per_domain.mean()),
            'top_expert_per_class': {
                class_labels[i]: int(per_class[i].argmax()) for i in range(num_classes)
            },
            'top_expert_per_domain': {
                domain_labels[i]: int(per_domain[i].argmax()) for i in range(num_domains)
            },
        }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n[*] Wrote analysis to {out_dir}')


if __name__ == '__main__':
    main()

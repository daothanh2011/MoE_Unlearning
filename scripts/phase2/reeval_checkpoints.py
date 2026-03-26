"""
Re-evaluate all saved Phase 2 best checkpoints on the correct test split
(80% in-split, same as Phase 1 sweep_acc) and update results.txt.

Usage:
    python scripts/phase2/reeval_checkpoints.py \
        [--train_output_dir train_output] \
        [--data_dir /path/to/terra_incognita_parent] \
        [--output train_output/results.txt]
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'domainbed'))

import domainbed.tutel_patch  # noqa: F401

from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.model_selection import IIDAccuracySelectionMethod

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

K_VALUES = [5, 7, 10, 13, 15]
X_VALUES = [5, 10, 20, 30, 50]
NUM_ENVS = 4
ENV_NAMES = ["location_100", "location_38", "location_43", "location_46"]


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_terra_envs(data_dir):
    terra_dir = os.path.join(data_dir, "terra_incognita")
    envs = sorted([f.name for f in os.scandir(terra_dir) if f.is_dir()])
    return envs, terra_dir


def build_in_split_loader(terra_dir, env_name, trial_seed, env_idx,
                          holdout_fraction=0.2):
    """Return a FastDataLoader for the 80% in-split of one environment.
    Mirrors Phase 1 exactly: out, in_ = split_dataset(env, int(N*0.2), seed)."""
    img_folder = ImageFolder(
        os.path.join(terra_dir, env_name), transform=BASE_TRANSFORM
    )
    n_out = int(len(img_folder) * holdout_fraction)
    _, in_split = misc.split_dataset(
        img_folder, n_out, seed=misc.seed_hash(trial_seed, env_idx)
    )
    return FastDataLoader(dataset=in_split, batch_size=64, num_workers=2)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(algorithm, loader, device='cuda'):
    algorithm.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = algorithm.predict(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Per-env evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_env(env_idx, train_output_dir, terra_dir, terra_envs, device='cuda'):
    """
    For one test environment:
      - Load Phase 1 checkpoint → get architecture + epoch_0 acc on in-split
      - For every saved best_checkpoint_k{k}_x{x}, load Phase 2 weights,
        evaluate on the same in-split
    Returns dict with keys:
        'epoch0_acc': float
        'results': {(k,x): acc}
    """
    phase1_ckpt_path = os.path.join(train_output_dir, "phase1", f"terra_env{env_idx}", "model.pkl")
    addon_dir        = os.path.join(train_output_dir, "phase2", f"terra_addon_D_env{env_idx}")

    print(f"\n{'='*60}")
    print(f" Env {env_idx} ({ENV_NAMES[env_idx]})")
    print(f"{'='*60}")

    # ── Load Phase 1 checkpoint ──────────────────────────────────────────────
    ckpt1 = torch.load(phase1_ckpt_path, map_location='cpu')
    ckpt_args = ckpt1['args']
    trial_seed       = ckpt_args.get('trial_seed', 0)
    holdout_fraction = ckpt_args.get('holdout_fraction', 0.2)

    algorithm = algorithms.GMOE(
        ckpt1['model_input_shape'],
        ckpt1['model_num_classes'],
        ckpt1['model_num_domains'],
        ckpt1['model_hparams'],
    )
    algorithm.load_state_dict(ckpt1['model_dict'], strict=False)
    algorithm = algorithm.to(device)

    # ── Build in-split loader for test env (reused for all checkpoints) ──────
    in_loader = build_in_split_loader(
        terra_dir, terra_envs[env_idx], trial_seed, env_idx, holdout_fraction
    )

    # ── Epoch 0: Phase 1 model on in-split ───────────────────────────────────
    epoch0_acc = evaluate(algorithm, in_loader, device)
    print(f"  Phase 1 (epoch_0) in-split acc = {epoch0_acc:.4f}  "
          f"({int(epoch0_acc*100):.1f}%)")

    # ── Phase 2 best checkpoints ─────────────────────────────────────────────
    results = {}
    for k in K_VALUES:
        for x in X_VALUES:
            ckpt_path = os.path.join(
                addon_dir, f"best_checkpoint_k{k}_x{x}_lr1e-4.pth"
            )
            if not os.path.exists(ckpt_path):
                print(f"  [MISSING] k={k} x={x}")
                continue

            ckpt2 = torch.load(ckpt_path, map_location='cpu')
            # Restore to Phase 1 state first (clean slate), then load Phase 2 weights
            algorithm.load_state_dict(ckpt1['model_dict'], strict=False)
            algorithm.load_state_dict(ckpt2['model_dict'], strict=False)
            algorithm = algorithm.to(device)

            acc = evaluate(algorithm, in_loader, device)
            old_acc = ckpt2.get('test_acc', float('nan'))
            results[(k, x)] = acc
            print(f"  k={k:>2} x={x:>2}:  in-split={acc:.4f} ({acc*100:.1f}%)  "
                  f"[old out-split was {old_acc*100:.1f}%]")

    return {'epoch0_acc': epoch0_acc, 'results': results}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 baseline via collect_results infrastructure
# ──────────────────────────────────────────────────────────────────────────────

def load_phase1_baselines(train_output_dir):
    """Load Phase 1 sweep_acc per env via IIDAccuracySelectionMethod."""
    from domainbed.lib.query import Q
    baselines = {}
    records = Q(reporting.load_records(os.path.join(train_output_dir, "phase1")))
    if not len(records):
        print("  WARNING: no Phase 1 records found")
        return baselines

    gmoe_records = records.filter(
        lambda r: r.get('args', {}).get('algorithm') == 'GMOE'
                  and r.get('args', {}).get('dataset') == 'TerraIncognita'
    )
    for env_idx in range(NUM_ENVS):
        env_records = gmoe_records.filter(
            lambda r, e=env_idx: r.get('args', {}).get('test_envs') == [e]
        )
        if not len(env_records):
            print(f"  WARNING: no GMOE records for env{env_idx}")
            baselines[env_idx] = float('nan')
            continue
        run_result = IIDAccuracySelectionMethod.run_acc(env_records)
        sweep_acc = run_result['test_acc'] if run_result else float('nan')
        baselines[env_idx] = sweep_acc
        print(f"  Phase 1 sweep_acc env{env_idx}: {sweep_acc*100:.1f}%")
    return baselines


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cell(v, best_key, k, x, fmt="{:.1f}%"):
    """Format a grid cell, marking the best with *."""
    if v is None:
        return "  N/A "
    s = fmt.format(v * 100)
    return s + "*" if (k, x) == best_key else s + " "


def _delta_cell(v, best_key, k, x):
    if v is None:
        return "  N/A "
    sign = "+" if v >= 0 else ""
    s = f"{sign}{v*100:.1f}%"
    return s + "*" if (k, x) == best_key else s + " "


def write_results(output_path, eval_data, phase1_baselines, log_dir=None):
    lines = []
    W = 70

    def h(title):
        lines.append("-" * W)
        lines.append(f" {title}")
        lines.append("-" * W)

    lines.append("=" * W)
    lines.append(" GMOE vs GMOE_DHAT — TerraIncognita Re-Evaluation")
    lines.append(f" All accuracies measured on the 80% in-split of the test domain")
    lines.append(f" (same split as Phase 1 sweep_acc — directly comparable)")
    if log_dir:
        lines.append(f" Phase 2 checkpoints: {log_dir}")
    lines.append("=" * W)
    lines.append("")
    lines.append("Metric note:")
    lines.append("  GMOE sweep_acc   = env{test}_in_acc at best val step (Phase 1)")
    lines.append("  GMOE_DHAT best   = env{test}_in_acc of best Phase 2 checkpoint")
    lines.append("  epoch_0 (ref)    = Phase 1 model on same in-split (should ≈ sweep_acc)")
    lines.append("  Δ gain           = best_acc − epoch_0  [finetuning improvement]")
    lines.append("  Best checkpoint  = selected by training-domain val_acc (IID method)")
    lines.append("")

    # ── Main table ──────────────────────────────────────────────────────────
    h("Main Results  (best k/x per env, all on 80% in-split)")
    col_w = 12
    header = f"{'Algorithm':<20}" + "".join(
        f"{n:>{col_w}}" for n in ["L100(0)", "L38(1)", "L43(2)", "L46(3)", "Avg"]
    )
    lines.append(header)
    lines.append("-" * W)

    rows = {}
    best_per_env = {}  # env → (k, x, best_acc, epoch0)
    for env_idx in range(NUM_ENVS):
        d = eval_data[env_idx]
        if not d['results']:
            best_per_env[env_idx] = (None, None, float('nan'), d['epoch0_acc'])
            continue
        bk, bx = max(d['results'], key=lambda kx: d['results'][kx])
        best_acc = d['results'][(bk, bx)]
        best_per_env[env_idx] = (bk, bx, best_acc, d['epoch0_acc'])

    def _row(name, vals):
        return f"{name:<20}" + "".join(f"{v:>{col_w}}" for v in vals)

    p1_vals   = [f"{phase1_baselines.get(i, float('nan'))*100:.1f}%" for i in range(NUM_ENVS)]
    dhat_vals = [f"{best_per_env[i][2]*100:.1f}%" for i in range(NUM_ENVS)]
    ep0_vals  = [f"{best_per_env[i][3]*100:.1f}%" for i in range(NUM_ENVS)]
    delta_vals= []
    delta_nums= []
    for i in range(NUM_ENVS):
        d = best_per_env[i][2] - best_per_env[i][3]
        delta_nums.append(d)
        sign = "+" if d >= 0 else ""
        delta_vals.append(f"{sign}{d*100:.1f}%")

    p1_avg    = np.nanmean([phase1_baselines.get(i, float('nan')) for i in range(NUM_ENVS)])
    dhat_avg  = np.nanmean([best_per_env[i][2] for i in range(NUM_ENVS)])
    ep0_avg   = np.nanmean([best_per_env[i][3] for i in range(NUM_ENVS)])
    delta_avg = np.nanmean(delta_nums)

    lines.append(_row("GMOE (sweep_acc)",  p1_vals  + [f"{p1_avg*100:.1f}%"]))
    lines.append(_row("GMOE_DHAT (best)",  dhat_vals + [f"{dhat_avg*100:.1f}%"]))
    lines.append(_row("epoch_0 (ref)",     ep0_vals  + [f"{ep0_avg*100:.1f}%"]))
    sign = "+" if delta_avg >= 0 else ""
    lines.append(_row("Δ gain",            delta_vals + [f"{sign}{delta_avg*100:.1f}%"]))
    lines.append("")

    # ── Best hparams table ───────────────────────────────────────────────────
    h("Best Hyperparameters per Test Environment")
    hdr = f"  {'Env':>3}  {'Name':<16} {'k':>4} {'x':>4}  {'best_acc':>9}  {'epoch_0':>8}  {'Δ_gain':>8}  {'D-hat':>6}"
    lines.append(hdr)
    lines.append("  " + "-" * (W - 2))
    for env_idx in range(NUM_ENVS):
        bk, bx, bacc, e0 = best_per_env[env_idx]
        if bk is None:
            continue
        # D-hat size from json
        dhat_json = os.path.join(
            os.path.dirname(output_path).replace('results.txt', ''),
            f"terra_addon_D_env{env_idx}", f"dhat_k{bk}_x{bx}.json"
        )
        # Try to get dhat size
        dhat_size = "?"
        for td in [f"train_output/phase2/terra_addon_D_env{env_idx}",
                   f"terra_addon_D_env{env_idx}"]:
            p = os.path.join(td, f"dhat_k{bk}_x{bx}.json")
            if os.path.exists(p):
                with open(p) as f:
                    obj = json.load(f)
                dhat_size = str(obj.get('total', len(obj)))
                break

        delta = bacc - e0
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"  {env_idx:>3}  {ENV_NAMES[env_idx]:<16} {bk:>4} {bx:>4}  "
            f"{bacc*100:>8.1f}%  {e0*100:>7.1f}%  {sign}{delta*100:>6.1f}%  "
            f"{dhat_size:>6}"
        )
    lines.append("")

    # ── Ablation grids ───────────────────────────────────────────────────────
    for env_idx in range(NUM_ENVS):
        d = eval_data[env_idx]
        bk, bx, _, e0 = best_per_env[env_idx]

        h(f"Ablation Grid: env={env_idx} ({ENV_NAMES[env_idx]})  —  Best test_acc (in-split %)")
        col_hdr = f"{'':>6}" + "".join(f"   k={k:<4}" for k in K_VALUES)
        lines.append(col_hdr)
        for x in X_VALUES:
            row = f"x={x:<4}"
            for k in K_VALUES:
                v = d['results'].get((k, x))
                row += "  " + _cell(v, (bk, bx), k, x)
            lines.append(row)
        lines.append("")

        h(f"Δ Gain Grid: env={env_idx} ({ENV_NAMES[env_idx]})  —  (best_acc − epoch_0)")
        lines.append(col_hdr)
        for x in X_VALUES:
            row = f"x={x:<4}"
            for k in K_VALUES:
                v = d['results'].get((k, x))
                delta = (v - e0) if v is not None else None
                row += "  " + _delta_cell(delta, (bk, bx), k, x)
            lines.append(row)
        lines.append("")

    # ── Ranked tables ────────────────────────────────────────────────────────
    for env_idx in range(NUM_ENVS):
        d = eval_data[env_idx]
        e0 = best_per_env[env_idx][3]
        h(f"All Runs Ranked: env={env_idx} ({ENV_NAMES[env_idx]})")
        lines.append(f"  {'Rank':>4}  {'k':>4}  {'x':>4}  {'best_acc':>9}  {'epoch_0':>8}  {'Δ_gain':>8}")
        lines.append("  " + "-" * 50)
        ranked = sorted(d['results'].items(), key=lambda kv: -kv[1])
        for rank, ((k, x), acc) in enumerate(ranked, 1):
            delta = acc - e0
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"  {rank:>4}  {k:>4}  {x:>4}  {acc*100:>8.1f}%  "
                f"{e0*100:>7.1f}%  {sign}{delta*100:>6.1f}%"
            )
        lines.append("")

    lines.append("=" * W)
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"\nResults written to: {output_path}")
    return text, best_per_env


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_output_dir", default="train_output")
    p.add_argument("--data_dir", default=None,
                   help="Parent dir of terra_incognita/. Auto-detected from Phase 1 ckpt if omitted.")
    p.add_argument("--output", default="train_output/phase2/results.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect data_dir from Phase 1 checkpoint if not provided
    data_dir = args.data_dir
    if data_dir is None:
        ckpt0 = torch.load(
            os.path.join(args.train_output_dir, "phase1", "terra_env0", "model.pkl"),
            map_location='cpu'
        )
        data_dir = ckpt0['args']['data_dir']
        print(f"Auto-detected data_dir: {data_dir}")

    terra_envs, terra_dir = get_terra_envs(data_dir)
    print(f"Terra envs: {terra_envs}")

    print("\nLoading Phase 1 baselines (IIDAccuracySelectionMethod)...")
    phase1_baselines = load_phase1_baselines(args.train_output_dir)

    eval_data = {}
    for env_idx in range(NUM_ENVS):
        eval_data[env_idx] = eval_env(
            env_idx, args.train_output_dir, terra_dir, terra_envs, args.device
        )

    # Save raw results as JSON for reference
    json_out = os.path.join(args.train_output_dir, "phase2", "reeval_results.json")
    serializable = {
        str(env_idx): {
            'epoch0_acc': eval_data[env_idx]['epoch0_acc'],
            'results': {f"{k}_{x}": v for (k, x), v in eval_data[env_idx]['results'].items()}
        }
        for env_idx in range(NUM_ENVS)
    }
    with open(json_out, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to: {json_out}")

    write_results(args.output, eval_data, phase1_baselines,
                  log_dir=os.path.join(args.train_output_dir, "phase2"))


if __name__ == "__main__":
    main()

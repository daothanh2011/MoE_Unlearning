"""
Analyze GMOE_DHAT ablation results from finetuning logs.

Parses all log files in --log_dir, loads the GMOE Phase 1 baseline from
--train_output_dir, and writes a combined comparison + ablation report to
--output.

No retraining or checkpoint loading required.

Usage:
    python scripts/analyze_dhat.py
    python scripts/analyze_dhat.py \
        --log_dir logs/ablation_20260322_180826 \
        --train_output_dir train_output \
        --output train_output/results.txt
"""

import argparse
import os
import re
import sys

import numpy as np

# ── Path setup (makes domainbed importable without conda activation) ─────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'domainbed'))

from domainbed.lib import reporting
from domainbed import model_selection

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

K_VALS = [5, 7, 10, 13, 15]
X_VALS = [5, 10, 20, 30, 50]
EVAL_EPOCHS = [0, 5, 10, 15, 20, 25, 30]

DATASET_CONFIGS = {
    'TerraIncognita': {
        'env_names': ['location_100', 'location_38', 'location_43', 'location_46'],
        'num_envs': 4,
    },
    'ColoredMNIST': {
        'env_names': ['+90%', '+80%', '-90%'],
        'num_envs': 3,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(path):
    """
    Parse a single phase2_finetune log file.

    Returns a dict with:
        env, k, x          — experiment identity
        dhat_total         — D-hat dataset size
        initial_acc        — test_acc at epoch 0 (Phase 1 model, before finetuning)
        best_acc           — best test_acc across all epochs
        best_epoch         — epoch at which best_acc was achieved
        epoch_accs         — dict {epoch_int -> test_acc}
    Returns None if parsing fails.
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
    except IOError:
        return None

    # env/k/x from header (handles both old and new format)
    m = re.search(r'Phase 2 Finetuning.*?env=(\d+).*?k=(\d+),\s*x=(\d+)', content)
    if not m:
        return None
    env, k, x = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # D-hat size
    m2 = re.search(r'D-hat total:\s*(\d+)', content)
    dhat_total = int(m2.group(1)) if m2 else None

    # Per-epoch in_acc: "[epoch  0] ... test_acc(env2_in)=X" or "test_acc=X"
    # Per-epoch out_acc: "[epoch  0] ... test_out_acc(env2_out)=X" or "test_out_acc=X"
    epoch_accs = {}      # epoch → in_acc (80%)
    epoch_out_accs = {}  # epoch → out_acc (20%)
    for m3 in re.finditer(r'\[epoch\s+(\d+)\].*?test_acc(?:\(\w+\))?=([\d.]+)', content):
        epoch_accs[int(m3.group(1))] = float(m3.group(2))
    for m3 in re.finditer(r'\[epoch\s+(\d+)\].*?test_out_acc(?:\(\w+\))?=([\d.]+)', content):
        epoch_out_accs[int(m3.group(1))] = float(m3.group(2))

    # Best test_acc (in_acc) — new format: "Best test_acc(in)=X ... at epoch Y"
    m4 = re.search(r'Best test_acc(?:\(\w+\))?=([\d.]+).*?at epoch\s+(\d+)', content)
    best_acc   = float(m4.group(1)) if m4 else (max(epoch_accs.values()) if epoch_accs else None)
    best_epoch = int(m4.group(2))   if m4 else None

    # Best out_acc at same best epoch
    best_out_acc = epoch_out_accs.get(best_epoch) if best_epoch is not None else None

    return {
        'env': env, 'k': k, 'x': x,
        'dhat_total': dhat_total,
        'initial_acc':     epoch_accs.get(0),
        'initial_out_acc': epoch_out_accs.get(0),
        'best_acc':        best_acc,
        'best_out_acc':    best_out_acc,
        'best_epoch':      best_epoch,
        'epoch_accs':      epoch_accs,
        'epoch_out_accs':  epoch_out_accs,
    }


def load_all_logs(log_dir, num_envs=4):
    """
    Load all *.log files from log_dir.

    Returns dict keyed by (env, k, x) → parsed dict.
    Prints a summary of how many logs were found vs expected.
    """
    logs = {}
    for fname in os.listdir(log_dir):
        if not fname.endswith('.log'):
            continue
        path = os.path.join(log_dir, fname)
        result = parse_log(path)
        if result is None:
            print(f"  WARNING: could not parse {fname}")
            continue
        key = (result['env'], result['k'], result['x'])
        logs[key] = result

    expected = set(
        (e, k, x)
        for e in range(num_envs)
        for k in K_VALS
        for x in X_VALS
    )
    missing = sorted(expected - set(logs.keys()))
    if missing:
        print(f"  WARNING: {len(missing)} expected logs missing: {missing[:5]}{'...' if len(missing)>5 else ''}")

    print(f"  Loaded {len(logs)} log files from {log_dir}")
    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 baseline
# ─────────────────────────────────────────────────────────────────────────────

def load_phase1_baseline(train_output_dir, dataset='TerraIncognita', env_names=None, num_envs=4):
    """
    Load GMOE Phase 1 results from train_output_dir and extract per-env metrics.

    Uses IIDAccuracySelectionMethod → returns env{test}_in_acc (80% in-split).

    Returns dict: env_idx → {
        'sweep_acc': env{test}_in_acc at best val step (80%, DomainBed standard),
        'out_acc':   env{test}_out_acc at the same step (20%, for reference),
    }
    """
    print(f"  Loading Phase 1 baseline ({dataset}) from {train_output_dir} ...")
    records = reporting.load_records(train_output_dir)
    gmoe_records = records.filter(
        lambda r: r['args']['algorithm'] == 'GMOE'
                  and r['args']['dataset'] == dataset
    )

    baseline = {}
    for env_idx in range(num_envs):
        group_records = gmoe_records.filter(
            lambda r, e=env_idx: r['args']['test_envs'] == [e]
        )
        if not len(group_records):
            print(f"  WARNING: no GMOE/{dataset} records for test_env={env_idx}")
            baseline[env_idx] = {'sweep_acc': None, 'out_acc': None}
            continue

        run_result = model_selection.IIDAccuracySelectionMethod.run_acc(group_records)
        if run_result is None:
            baseline[env_idx] = {'sweep_acc': None, 'out_acc': None}
            continue

        sweep_acc = run_result['test_acc']  # env{test}_in_acc (80%)

        best_step_records = group_records.filter(
            lambda r, best=run_result: (
                r.get('step') == best.get('step') or
                r.get('epoch') == best.get('epoch')
            )
        )
        out_acc = None
        if len(best_step_records):
            out_acc = best_step_records[0].get(f'env{env_idx}_out_acc')

        baseline[env_idx] = {'sweep_acc': sweep_acc, 'out_acc': out_acc}
        name = env_names[env_idx] if env_names else str(env_idx)
        print(f"    env{env_idx} ({name}): "
              f"in_acc(80%)={100*sweep_acc:.2f}%"
              + (f"  out_acc(20%)={100*out_acc:.2f}%" if out_acc is not None else ""))

    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_best_hparams(logs, env):
    """Return the (k, x) combo with highest best_acc for the given env."""
    best = None
    for k in K_VALS:
        for x in X_VALS:
            entry = logs.get((env, k, x))
            if entry is None or entry['best_acc'] is None:
                continue
            if best is None or entry['best_acc'] > best['best_acc']:
                best = entry
    return best


def build_ablation_grid(logs, env, metric='best_acc'):
    """
    Build a 2D grid of metric values indexed by [x_idx][k_idx].

    metric: 'best_acc' or 'delta' (best_acc - initial_acc)
    Returns (grid, best_cell) where best_cell=(x_idx, k_idx).
    """
    grid = []
    best_val = -1
    best_cell = (0, 0)
    for xi, x in enumerate(X_VALS):
        row = []
        for ki, k in enumerate(K_VALS):
            entry = logs.get((env, k, x))
            if entry is None:
                row.append(None)
                continue
            if metric == 'best_acc':
                val = entry.get('best_acc')
            elif metric == 'delta':
                ba = entry.get('best_acc')
                ia = entry.get('initial_acc')
                val = (ba - ia) if (ba is not None and ia is not None) else None
            else:
                val = entry.get(metric)
            row.append(val)
            if val is not None and val > best_val:
                best_val = val
                best_cell = (xi, ki)
        grid.append(row)
    return grid, best_cell


def build_curve_summary(logs, env, k, x):
    """Return compact epoch-by-epoch accuracy string for eval epochs."""
    entry = logs.get((env, k, x))
    if entry is None:
        return "N/A"
    parts = []
    for ep in EVAL_EPOCHS:
        acc = entry['epoch_accs'].get(ep)
        if acc is not None:
            marker = '*' if ep == entry.get('best_epoch') else ''
            parts.append(f"ep{ep}={100*acc:.1f}{marker}")
    return '  →  '.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, pct=True, plus=False):
    """Format a float as percentage string."""
    if val is None:
        return ' N/A  '
    s = f"{100*val:.1f}"
    if plus and val >= 0:
        s = '+' + s
    if pct:
        s += '%'
    return s


def fmt_grid(grid, best_cell, pct=True, plus=False):
    """Render a k×x grid as a formatted string block."""
    col_w = 8
    header = ' ' * 6 + ''.join(f"{'k='+str(k):>{col_w}}" for k in K_VALS)
    lines = [header]
    for xi, x in enumerate(X_VALS):
        row_str = f"x={x:<4}"
        for ki in range(len(K_VALS)):
            val = grid[xi][ki]
            cell = fmt(val, pct=pct, plus=plus)
            if (xi, ki) == best_cell:
                cell = cell.rstrip('%') + '%*'
            row_str += f"{cell:>{col_w}}"
        lines.append(row_str)
    return '\n'.join(lines)


def write_output(out_path, baseline, logs, log_dir, dataset='TerraIncognita', env_names=None, num_envs=4):
    """Assemble and write the full analysis report."""
    if env_names is None:
        env_names = [str(i) for i in range(num_envs)]
    lines = []
    w = 70

    def sep(char='='):
        lines.append(char * w)

    def section(title):
        lines.append('')
        sep('-')
        lines.append(f' {title}')
        sep('-')

    # ── Header ───────────────────────────────────────────────────────────────
    sep()
    lines.append(f' GMOE vs GMOE_DHAT — {dataset} Ablation Analysis')
    lines.append(f' Logs: {log_dir}  ({len(logs)} runs parsed)')
    sep()
    lines.append('')
    lines.append('Metric note:')
    lines.append('  GMOE baseline (in)  = env{test}_in_acc  (80%) at best val step')
    lines.append('  GMOE baseline (out) = env{test}_out_acc (20%) at same step')
    lines.append('  GMOE_DHAT (in)      = env{test}_in_acc  (80%) from finetuning logs')
    lines.append('  GMOE_DHAT (out)     = env{test}_out_acc (20%) from finetuning logs')
    lines.append('  epoch_0             = Phase 1 model before any finetuning (both splits)')
    lines.append('  Δ gain              = best_acc − epoch_0  [same split, most interpretable]')

    def _avg(vals):
        v = [x for x in vals if x is not None]
        return np.mean(v) if v else None

    def _make_row(label, vals, avg, col_w, plus=False):
        row = f"{label:<22}"
        for v in vals:
            row += f"{fmt(v, plus=plus):<{col_w}}"
        row += f"{fmt(avg, plus=plus):<{col_w}}"
        return row

    # ── Main results table ────────────────────────────────────────────────────
    best_per_env = {e: find_best_hparams(logs, e) for e in range(num_envs)}

    col_w = 12
    env_header = ''.join(f"{n+'('+str(i)+')':<{col_w}}" for i, n in enumerate(env_names))

    for split_label, in_split in [('in_acc (80%)', True), ('out_acc (20%)', False)]:
        section(f'Main Results — {split_label}  (best k/x per env)')
        header = f"{'Algorithm':<22}" + env_header + f"{'Avg':<{col_w}}"
        lines.append(header)
        lines.append('-' * len(header))

        # GMOE baseline
        if in_split:
            gmoe_vals = [baseline[e]['sweep_acc'] for e in range(num_envs)]
        else:
            gmoe_vals = [baseline[e]['out_acc'] for e in range(num_envs)]
        lines.append(_make_row('GMOE (baseline)', gmoe_vals, _avg(gmoe_vals), col_w))

        # GMOE_DHAT best
        if in_split:
            dhat_vals = [best_per_env[e]['best_acc'] if best_per_env[e] else None for e in range(num_envs)]
            ep0_vals  = [best_per_env[e]['initial_acc'] if best_per_env[e] else None for e in range(num_envs)]
        else:
            dhat_vals = [best_per_env[e]['best_out_acc'] if best_per_env[e] else None for e in range(num_envs)]
            ep0_vals  = [best_per_env[e]['initial_out_acc'] if best_per_env[e] else None for e in range(num_envs)]
        lines.append(_make_row('GMOE_DHAT (best)', dhat_vals, _avg(dhat_vals), col_w))
        lines.append(_make_row('epoch_0 (ref)',    ep0_vals,  _avg(ep0_vals),  col_w))

        delta_vals = [
            (dhat_vals[e] - ep0_vals[e]) if (dhat_vals[e] is not None and ep0_vals[e] is not None) else None
            for e in range(num_envs)
        ]
        lines.append(_make_row('Δ gain', delta_vals, _avg(delta_vals), col_w, plus=True))

    # ── Best hyperparameters per env ──────────────────────────────────────────
    section('Best Hyperparameters per Test Environment')

    hdr = (f"  {'Env':>3}  {'Name':<14} {'k':>4} {'x':>4}  {'D-hat':>6}"
           f"  {'best_in':>8}  {'best_out':>9}  {'ep0_in':>7}  {'ep0_out':>8}"
           f"  {'Δ(in)':>7}  {'best_ep':>8}")
    lines.append(hdr)
    lines.append('  ' + '-' * (len(hdr) - 2))
    for e in range(num_envs):
        b = best_per_env[e]
        if b is None:
            lines.append(f"  {e:>3}  {'N/A':<14}")
            continue
        delta_in  = (b['best_acc']     - b['initial_acc'])     if (b['best_acc']     and b['initial_acc'])     else None
        lines.append(
            f"  {e:>3}  {env_names[e]:<14} {b['k']:>4} {b['x']:>4}"
            f"  {str(b['dhat_total'] or 'N/A'):>6}"
            f"  {fmt(b['best_acc']):>8}"
            f"  {fmt(b.get('best_out_acc')):>9}"
            f"  {fmt(b['initial_acc']):>7}"
            f"  {fmt(b.get('initial_out_acc')):>8}"
            f"  {fmt(delta_in, plus=True):>7}"
            f"  {str(b['best_epoch'] or 'N/A'):>8}"
        )

    # ── Ablation grids (best in_acc) ──────────────────────────────────────────
    for e in range(num_envs):
        section(f'Ablation Grid: env={e} ({env_names[e]})  —  best_in_acc (%)')
        grid, best_cell = build_ablation_grid(logs, e, metric='best_acc')
        lines.append(fmt_grid(grid, best_cell))

    # ── Ablation grids (Δ gain, in_acc) ───────────────────────────────────────
    for e in range(num_envs):
        section(f'Δ Gain Grid: env={e} ({env_names[e]})  —  best_in_acc − epoch_0_in (%)')
        grid, best_cell = build_ablation_grid(logs, e, metric='delta')
        lines.append(fmt_grid(grid, best_cell, plus=True))

    # ── Learning curves for best k/x per env ─────────────────────────────────
    section('Learning Curves  (best k/x per env, in_acc,  * = best epoch)')
    for e in range(num_envs):
        b = best_per_env[e]
        if b is None:
            continue
        curve = build_curve_summary(logs, e, b['k'], b['x'])
        lines.append(f"  env={e} ({env_names[e]})  k={b['k']}, x={b['x']}:")
        lines.append(f"    {curve}")
        lines.append('')

    # ── All runs ranked per env ───────────────────────────────────────────────
    for e in range(num_envs):
        section(f'All Runs Ranked: env={e} ({env_names[e]})')
        runs = []
        for k in K_VALS:
            for x in X_VALS:
                entry = logs.get((e, k, x))
                if entry:
                    runs.append(entry)
        runs.sort(key=lambda r: r['best_acc'] if r['best_acc'] is not None else -1, reverse=True)
        lines.append(f"  {'Rank':>4}  {'k':>4}  {'x':>4}  {'D-hat':>6}"
                     f"  {'best_in':>8}  {'best_out':>9}  {'ep0_in':>7}  {'Δ_gain':>8}  {'best_ep':>8}")
        lines.append('  ' + '-' * 68)
        for rank, r in enumerate(runs, 1):
            delta = (r['best_acc'] - r['initial_acc']) if (r['best_acc'] and r['initial_acc']) else None
            lines.append(
                f"  {rank:>4}  {r['k']:>4}  {r['x']:>4}"
                f"  {str(r['dhat_total'] or 'N/A'):>6}"
                f"  {fmt(r['best_acc']):>8}"
                f"  {fmt(r.get('best_out_acc')):>9}"
                f"  {fmt(r['initial_acc']):>7}"
                f"  {fmt(delta, plus=True):>8}"
                f"  {str(r['best_epoch'] or 'N/A'):>8}"
            )

    sep()

    output = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(output)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze GMOE_DHAT ablation logs')
    parser.add_argument('--log_dir', default='logs/ablation_20260322_180826',
                        help='Directory containing env*_k*_x*.log files')
    parser.add_argument('--train_output_dir', default='train_output',
                        help='Directory containing {dataset}_env{i}/ subdirs with Phase 1 results.jsonl')
    parser.add_argument('--dataset', default='TerraIncognita',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name (default: TerraIncognita)')
    parser.add_argument('--output', default='train_output/results.txt',
                        help='Output path for the analysis report')
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    env_names = cfg['env_names']
    num_envs  = cfg['num_envs']

    print('=' * 60)
    print(f'GMOE_DHAT Ablation Analysis — {args.dataset}')
    print('=' * 60)

    print('\n[1/3] Loading log files...')
    logs = load_all_logs(args.log_dir, num_envs=num_envs)

    print('\n[2/3] Loading Phase 1 baseline...')
    baseline = load_phase1_baseline(args.train_output_dir,
                                    dataset=args.dataset,
                                    env_names=env_names,
                                    num_envs=num_envs)

    print(f'\n[3/3] Writing report → {args.output}')
    output = write_output(args.output, baseline, logs, args.log_dir,
                          dataset=args.dataset, env_names=env_names, num_envs=num_envs)

    # Print preview
    preview_lines = output.split('\n')
    print('\n--- Preview (first 60 lines) ---')
    print('\n'.join(preview_lines[:60]))
    if len(preview_lines) > 60:
        print(f'... ({len(preview_lines)} lines total)')

    print(f'\nDone. Full report: {args.output}')


if __name__ == '__main__':
    main()

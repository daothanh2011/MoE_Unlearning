"""
plot_results.py — Visualise the GMoE expert sweep results per dataset.

Plots saved to sweep/figures/{Dataset}/:
  1. heatmap_PR{XX}.png     — num_experts × top_k heatmap per prune_ratio value
  2. acc_vs_prune.png       — best_val_avg_acc vs expert_prune_ratio (per num_experts)
  3. acc_vs_nexpert.png     — best_val_avg_acc vs num_experts (per top_k)
  4. learning_curves.png    — val_avg_acc vs step (filtered by --group_filter)
  5. params_bar.png         — total_params vs active_params per config

Usage:
    python sweep/plot_results.py --dataset PACS
    python sweep/plot_results.py --dataset all
    python sweep/plot_results.py --dataset OfficeHome --group_filter "expert_prune_ratio=0.0"
"""

import argparse
import json
import os

import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — no plots will be generated")


BASELINE = {"num_experts": 6, "gate_k": 2, "expert_prune_ratio": 0.0}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def list_available_datasets(log_dir):
    return [d for d in sorted(os.listdir(log_dir))
            if os.path.isdir(os.path.join(log_dir, d))]


def load_final_summaries(ds_log_dir):
    summaries = {}
    if not os.path.isdir(ds_log_dir):
        return summaries
    for run_id in os.listdir(ds_log_dir):
        path = os.path.join(ds_log_dir, run_id, "final_summary.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    summaries[run_id] = json.load(f)
            except Exception:
                pass
    return summaries


def load_eval_jsonl(ds_log_dir, run_id):
    path = os.path.join(ds_log_dir, run_id, "eval_log.jsonl")
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    return records


def is_baseline(hp):
    return (hp.get("num_experts") == BASELINE["num_experts"] and
            hp.get("gate_k") == BASELINE["gate_k"] and
            abs(hp.get("expert_prune_ratio", 0) - BASELINE["expert_prune_ratio"]) < 1e-9)


def parse_filter(s):
    if not s:
        return {}
    result = {}
    for part in s.split(","):
        k, v = part.strip().split("=")
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Plot 1: Heatmaps (one per prune_ratio value)
# ---------------------------------------------------------------------------

def plot_heatmaps(summaries, cfg, figures_dir, dataset_name):
    if not HAS_MPL:
        return

    grid       = cfg["param_grid"]
    pr_values  = grid["expert_prune_ratio"]
    ne_values  = sorted(set(grid["num_experts"]))
    tk_values  = sorted(set(grid["top_k"]))

    for pr in pr_values:
        data = np.full((len(ne_values), len(tk_values)), np.nan)
        for summary in summaries.values():
            hp = summary.get("hparams", {})
            if abs(hp.get("expert_prune_ratio", -1) - pr) > 1e-9:
                continue
            n = hp.get("num_experts")
            k = hp.get("gate_k")
            acc = (summary.get("best_checkpoint") or {}).get("val_avg_acc")
            if n in ne_values and k in tk_values and acc is not None:
                i = ne_values.index(n)
                j = tk_values.index(k)
                # Average over test envs if multiple runs share same hparams
                if np.isnan(data[i, j]):
                    data[i, j] = acc
                else:
                    data[i, j] = (data[i, j] + acc) / 2.0

        if np.all(np.isnan(data)):
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(data, aspect="auto", cmap="viridis",
                       vmin=np.nanmin(data), vmax=np.nanmax(data))
        ax.set_xticks(range(len(tk_values)))
        ax.set_xticklabels([f"K={k}" for k in tk_values])
        ax.set_yticks(range(len(ne_values)))
        ax.set_yticklabels([f"N={n}" for n in ne_values])
        ax.set_xlabel("top_k (gate_k)")
        ax.set_ylabel("num_experts")
        ax.set_title(f"{dataset_name} — Best val acc (prune_ratio={pr:.1f})")
        plt.colorbar(im, ax=ax, label="val_avg_acc")

        # Mark baseline
        if pr == BASELINE["expert_prune_ratio"]:
            if BASELINE["num_experts"] in ne_values and BASELINE["gate_k"] in tk_values:
                bi = ne_values.index(BASELINE["num_experts"])
                bj = tk_values.index(BASELINE["gate_k"])
                ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                           fill=False, edgecolor="red", linewidth=2.5,
                                           label="baseline"))
                ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        pr_str = f"{int(round(pr * 10)):02d}"
        out    = os.path.join(figures_dir, f"heatmap_PR{pr_str}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: acc vs prune_ratio (per num_experts)
# ---------------------------------------------------------------------------

def plot_acc_vs_prune(summaries, cfg, figures_dir, dataset_name):
    if not HAS_MPL:
        return

    grid      = cfg["param_grid"]
    ne_values = sorted(set(grid["num_experts"]))
    pr_values = sorted(set(grid["expert_prune_ratio"]))
    colors    = cm.tab10(np.linspace(0, 1, len(ne_values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for ne, color in zip(ne_values, colors):
        accs = []
        for pr in pr_values:
            vals = []
            for s in summaries.values():
                hp = s.get("hparams", {})
                if hp.get("num_experts") != ne:
                    continue
                if abs(hp.get("expert_prune_ratio", -1) - pr) > 1e-9:
                    continue
                v = (s.get("best_checkpoint") or {}).get("val_avg_acc")
                if v is not None:
                    vals.append(v)
            accs.append(float(np.mean(vals)) if vals else None)
        valid = [(p, a) for p, a in zip(pr_values, accs) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            lw = 2.5 if ne == BASELINE["num_experts"] else 1.2
            ax.plot(xs, ys, marker="o", label=f"N={ne}", color=color, linewidth=lw)

    # Baseline marker
    bl_pr   = BASELINE["expert_prune_ratio"]
    bl_accs = [
        (s.get("best_checkpoint") or {}).get("val_avg_acc")
        for s in summaries.values()
        if is_baseline(s.get("hparams", {}))
    ]
    bl_accs = [v for v in bl_accs if v is not None]
    if bl_accs:
        bl_acc = float(np.mean(bl_accs))
        ax.axvline(bl_pr, color="red", linestyle="--", linewidth=1, label="baseline PR")
        ax.scatter([bl_pr], [bl_acc], color="red", zorder=5, s=80)

    ax.set_xlabel("expert_prune_ratio")
    ax.set_ylabel("Best val avg acc")
    ax.set_title(f"{dataset_name} — Accuracy vs expert_prune_ratio (best top_k per N)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(figures_dir, "acc_vs_prune.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: acc vs num_experts (per top_k)
# ---------------------------------------------------------------------------

def plot_acc_vs_nexpert(summaries, cfg, figures_dir, dataset_name):
    if not HAS_MPL:
        return

    grid      = cfg["param_grid"]
    ne_values = sorted(set(grid["num_experts"]))
    tk_values = sorted(set(grid["top_k"]))
    colors    = cm.tab10(np.linspace(0, 1, len(tk_values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for tk, color in zip(tk_values, colors):
        accs = []
        for ne in ne_values:
            vals = []
            for s in summaries.values():
                hp = s.get("hparams", {})
                if hp.get("num_experts") != ne or hp.get("gate_k") != tk:
                    continue
                v = (s.get("best_checkpoint") or {}).get("val_avg_acc")
                if v is not None:
                    vals.append(v)
            accs.append(float(np.mean(vals)) if vals else None)
        valid = [(n, a) for n, a in zip(ne_values, accs) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            lw = 2.5 if tk == BASELINE["gate_k"] else 1.2
            ax.plot(xs, ys, marker="s", label=f"K={tk}", color=color, linewidth=lw)

    bl_accs = [
        (s.get("best_checkpoint") or {}).get("val_avg_acc")
        for s in summaries.values()
        if is_baseline(s.get("hparams", {}))
    ]
    bl_accs = [v for v in bl_accs if v is not None]
    if bl_accs:
        bl_acc = float(np.mean(bl_accs))
        ax.axvline(BASELINE["num_experts"], color="red", linestyle="--",
                   linewidth=1, label="baseline N")
        ax.scatter([BASELINE["num_experts"]], [bl_acc], color="red", zorder=5, s=80)

    ax.set_xlabel("num_experts")
    ax.set_ylabel("Best val avg acc")
    ax.set_title(f"{dataset_name} — Accuracy vs num_experts (best prune_ratio per K)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(figures_dir, "acc_vs_nexpert.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4: Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(summaries, ds_log_dir, figures_dir, dataset_name,
                         group_filter):
    if not HAS_MPL:
        return

    filter_kv = parse_filter(group_filter)

    selected = []
    for run_id, s in summaries.items():
        hp = s.get("hparams", {})
        match = all(
            abs(hp.get(k, -999) - v) < 1e-9 if isinstance(v, float)
            else hp.get(k) == v
            for k, v in filter_kv.items()
        )
        if match:
            selected.append((run_id, s))

    if not selected:
        print(f"  [plot_learning_curves] No runs match the filter — skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = cm.tab10(np.linspace(0, 1, max(len(selected), 1)))

    for (run_id, s), color in zip(selected, colors):
        records = load_eval_jsonl(ds_log_dir, run_id)
        if not records:
            continue
        steps = [r["step"] for r in records]
        accs  = [r.get("val_avg_acc") for r in records]
        valid = [(st, ac) for st, ac in zip(steps, accs) if ac is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        hp    = s.get("hparams", {})
        bl    = is_baseline(hp)
        label = f"{run_id}{' [baseline]' if bl else ''}"
        ax.plot(xs, ys, label=label, color=color,
                linewidth=2.5 if bl else 1.2,
                linestyle="-" if bl else "--")

    ax.set_xlabel("step")
    ax.set_ylabel("val avg acc")
    title = f"{dataset_name} — Learning curves"
    if group_filter:
        title += f" [{group_filter}]"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    suffix = group_filter.replace("=", "").replace(",", "_") if group_filter else "all"
    out = os.path.join(figures_dir, f"learning_curves_{suffix}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5: params bar
# ---------------------------------------------------------------------------

def plot_params_bar(summaries, figures_dir, dataset_name):
    if not HAS_MPL:
        return

    run_ids, totals, actives = [], [], []
    for run_id, s in sorted(summaries.items()):
        eff = s.get("efficiency_metrics", {})
        t = eff.get("total_params")
        a = eff.get("active_params")
        if t and a:
            run_ids.append(run_id)
            totals.append(t / 1e6)
            actives.append(a / 1e6)

    if not run_ids:
        return

    x       = np.arange(len(run_ids))
    w       = 0.4
    bl_mask = [is_baseline(summaries[r].get("hparams", {})) for r in run_ids]

    fig, ax = plt.subplots(figsize=(max(8, len(run_ids) * 0.5), 5))
    bars_t = ax.bar(x - w / 2, totals,  w, label="total_params (M)",  color="#aec7e8")
    bars_a = ax.bar(x + w / 2, actives, w, label="active_params (M)", color="#1f77b4")

    for i, bl in enumerate(bl_mask):
        if bl:
            bars_t[i].set_edgecolor("red")
            bars_t[i].set_linewidth(2)
            bars_a[i].set_edgecolor("red")
            bars_a[i].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Parameters (M)")
    ax.set_title(f"{dataset_name} — Total vs active parameters\n(red border = baseline)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(figures_dir, "params_bar.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Per-dataset plotting
# ---------------------------------------------------------------------------

def plot_dataset(dataset_name, log_dir, figures_base_dir, cfg, group_filter):
    ds_log_dir   = os.path.join(log_dir, dataset_name)
    figures_dir  = os.path.join(figures_base_dir, dataset_name)

    summaries = load_final_summaries(ds_log_dir)
    if not summaries:
        print(f"  No final_summary.json files found in {ds_log_dir}.")
        return

    print(f"  Found {len(summaries)} completed runs.")
    os.makedirs(figures_dir, exist_ok=True)

    plot_heatmaps(summaries, cfg, figures_dir, dataset_name)
    plot_acc_vs_prune(summaries, cfg, figures_dir, dataset_name)
    plot_acc_vs_nexpert(summaries, cfg, figures_dir, dataset_name)
    plot_learning_curves(summaries, ds_log_dir, figures_dir, dataset_name, group_filter)
    plot_params_bar(summaries, figures_dir, dataset_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot GMoE sweep results per dataset")
    parser.add_argument("--config",       default="sweep/sweep_config.yaml")
    parser.add_argument("--dataset",      default=None, metavar="NAME_OR_all",
                        help="Dataset name (e.g. PACS) or 'all' / omit for all datasets")
    parser.add_argument("--figures_dir",  default=None,
                        help="Override base figures dir from config")
    parser.add_argument("--group_filter", default=None, metavar="KEY=VAL,...",
                        help="Filter for learning curves, e.g. 'expert_prune_ratio=0.0'")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    outputs     = cfg.get("output", {})
    log_dir     = outputs.get("log_dir",     "sweep/logs")
    figures_dir = args.figures_dir or outputs.get("figures_dir", "sweep/figures")

    if not os.path.isdir(log_dir):
        print(f"Log dir not found: {log_dir}")
        return

    if args.dataset is None or args.dataset.lower() == "all":
        dataset_names = list_available_datasets(log_dir)
        if not dataset_names:
            print(f"No dataset subdirectories found in {log_dir}.")
            return
        print(f"Plotting {len(dataset_names)} dataset(s): {dataset_names}\n")
    else:
        dataset_names = [args.dataset]

    for ds_name in dataset_names:
        print(f"=== {ds_name} ===")
        plot_dataset(ds_name, log_dir, figures_dir, cfg, args.group_filter)
        print()

    print("Done.")


if __name__ == "__main__":
    main()

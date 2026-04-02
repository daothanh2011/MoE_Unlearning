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


def load_jsonl(ds_log_dir, run_id, filename):
    path = os.path.join(ds_log_dir, run_id, filename)
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


def load_eval_jsonl(ds_log_dir, run_id):
    return load_jsonl(ds_log_dir, run_id, "eval_log.jsonl")


def load_dataset_config(dataset_name, configs_dir="sweep/configs"):
    path = os.path.join(configs_dir, f"{dataset_name}.yaml")
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


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


def _get_test_env(run_id, summary):
    """Extract integer test_env from run_id (e.g. 'OfficeHome_N4_K1_PR00_env0' → 0)."""
    te = summary.get("test_env")
    if te is not None and te != "":
        return int(te)
    if "_env" in run_id:
        try:
            return int(run_id.split("_env")[-1])
        except ValueError:
            pass
    return None


def compute_iid_best(eval_records, test_env, n_envs=4):
    """
    DomainBed IID selection:
      - Select step by argmax(mean(out_acc of training envs)) — never touches test domain.
      - Report env{test_env}_in_acc at that step (80% split, more reliable).
    Returns (best_step, test_in_acc, iid_val_acc).
    """
    best_step, best_val, best_test_in = None, -1.0, None
    for rec in eval_records:
        per_domain    = rec.get("val_acc_per_domain", {})
        in_per_domain = rec.get("val_in_acc_per_domain", {})
        train_accs = [
            per_domain.get(f"env{i}_out_acc")
            for i in range(n_envs) if i != test_env
        ]
        train_accs = [a for a in train_accs if a is not None]
        if not train_accs:
            continue
        val = sum(train_accs) / len(train_accs)
        if val > best_val:
            best_val     = val
            best_step    = rec.get("step")
            best_test_in = in_per_domain.get(f"env{test_env}_in_acc")
    return best_step, (best_test_in if best_step is not None else None), (best_val if best_step is not None else None)


def compute_test_domain_best(eval_records, test_env):
    """Oracle reference: argmax(env{test_env}_in_acc). Kept for reference only."""
    key = f"env{test_env}_in_acc"
    best_step, best_acc = None, -1.0
    for rec in eval_records:
        acc = rec.get("val_in_acc_per_domain", {}).get(key)
        if acc is not None and acc > best_acc:
            best_acc, best_step = acc, rec.get("step")
    return best_step, (best_acc if best_acc > -1 else None)


def _precompute_test_accs(summaries, ds_log_dir):
    """
    For every run read eval_log.jsonl and compute IID test accuracy (primary metric):
      step selected by mean(training envs out_acc), report env{test_env}_in_acc.

    Returns dict:
      run_id -> {"test_env": int, "step": int, "acc": float,   ← IID (primary)
                 "oracle_step": int, "oracle_acc": float,       ← Oracle (reference)
                 "eval_records": list[dict]}
    """
    result = {}
    for run_id, s in summaries.items():
        te = _get_test_env(run_id, s)
        if te is None:
            continue
        recs = load_eval_jsonl(ds_log_dir, run_id)
        iid_step, iid_acc, _ = compute_iid_best(recs, te)
        oracle_step, oracle_acc = compute_test_domain_best(recs, te)
        if iid_acc is not None:
            result[run_id] = {
                "test_env":    te,
                "step":        iid_step,    # IID step (used by all plots)
                "acc":         iid_acc,     # test in_acc at IID step (primary metric)
                "oracle_step": oracle_step,
                "oracle_acc":  oracle_acc,
                "eval_records": recs,
            }
    return result


# ---------------------------------------------------------------------------
# Plot 1: Heatmaps (one per prune_ratio value)
# ---------------------------------------------------------------------------

def plot_heatmaps(summaries, cfg, figures_dir, dataset_name, test_accs):
    """Heatmap of OOD test accuracy (env{test_env}_out_acc), averaged over test envs."""
    if not HAS_MPL:
        return

    grid       = cfg["param_grid"]
    pr_values  = grid["expert_prune_ratio"]
    ne_values  = sorted(set(grid["num_experts"]) | {
        s.get("hparams", {}).get("num_experts") for s in summaries.values()
        if s.get("hparams", {}).get("num_experts") is not None})
    tk_values  = sorted(set(grid["top_k"]) | {
        s.get("hparams", {}).get("gate_k") for s in summaries.values()
        if s.get("hparams", {}).get("gate_k") is not None})

    for pr in pr_values:
        data  = np.full((len(ne_values), len(tk_values)), np.nan)
        count = np.zeros((len(ne_values), len(tk_values)), dtype=int)
        for run_id, summary in summaries.items():
            hp = summary.get("hparams", {})
            if abs(hp.get("expert_prune_ratio", -1) - pr) > 1e-9:
                continue
            n   = hp.get("num_experts")
            k   = hp.get("gate_k")
            acc = test_accs.get(run_id, {}).get("acc")   # correct OOD metric
            if n in ne_values and k in tk_values and acc is not None:
                i, j = ne_values.index(n), tk_values.index(k)
                if np.isnan(data[i, j]):
                    data[i, j] = acc
                    count[i, j] = 1
                else:
                    data[i, j] = (data[i, j] * count[i, j] + acc) / (count[i, j] + 1)
                    count[i, j] += 1

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
        ax.set_title(f"{dataset_name} — OOD test acc (prune_ratio={pr:.1f}, avg over envs)")
        plt.colorbar(im, ax=ax, label="test_in_acc (IID selection)")

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

def plot_acc_vs_prune(summaries, cfg, figures_dir, dataset_name, test_accs):
    """OOD test accuracy vs prune_ratio for every (N, K) combination."""
    if not HAS_MPL:
        return

    from matplotlib.lines import Line2D

    grid      = cfg["param_grid"]
    ne_values = sorted(set(grid["num_experts"]) | {
        s.get("hparams", {}).get("num_experts") for s in summaries.values()
        if s.get("hparams", {}).get("num_experts") is not None})
    pr_values = sorted(set(grid["expert_prune_ratio"]))
    tk_values = sorted(set(grid["top_k"]) | {
        s.get("hparams", {}).get("gate_k") for s in summaries.values()
        if s.get("hparams", {}).get("gate_k") is not None})

    ne_colors  = {ne: cm.tab10(i / max(len(ne_values) - 1, 1)) for i, ne in enumerate(ne_values)}
    linestyles = ["-", "--", "-.", ":"]
    markers    = ["o", "s", "^", "D"]
    tk_styles  = {tk: linestyles[i % len(linestyles)] for i, tk in enumerate(tk_values)}
    tk_markers = {tk: markers[i % len(markers)]       for i, tk in enumerate(tk_values)}

    nk_pairs = sorted(set(
        (s.get("hparams", {}).get("num_experts"), s.get("hparams", {}).get("gate_k"))
        for s in summaries.values()
        if s.get("hparams", {}).get("num_experts") and s.get("hparams", {}).get("gate_k")
    ))

    fig, ax = plt.subplots(figsize=(11, 5))
    for (ne, k) in nk_pairs:
        accs = []
        for pr in pr_values:
            vals = [
                test_accs[run_id]["acc"]
                for run_id, s in summaries.items()
                if (run_id in test_accs
                    and s.get("hparams", {}).get("num_experts") == ne
                    and s.get("hparams", {}).get("gate_k") == k
                    and abs(s.get("hparams", {}).get("expert_prune_ratio", -1) - pr) < 1e-9)
            ]
            accs.append(float(np.mean(vals)) if vals else None)
        valid = [(p, a) for p, a in zip(pr_values, accs) if a is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        bl = (ne == BASELINE["num_experts"] and k == BASELINE["gate_k"])
        ax.plot(xs, ys,
                color=ne_colors[ne], linestyle=tk_styles[k], marker=tk_markers[k],
                markersize=5, linewidth=2.5 if bl else 1.4,
                alpha=1.0 if bl else 0.75, zorder=3 if bl else 2)

    legend_elems = (
        [Line2D([0], [0], color=ne_colors[ne], linewidth=2, label=f"N={ne}")
         for ne in ne_values] +
        [Line2D([0], [0], color="gray", linestyle=tk_styles[k],
                marker=tk_markers[k], markersize=5, label=f"K={k}")
         for k in tk_values]
    )
    ax.legend(handles=legend_elems, fontsize=8, ncol=2, title="N=num_experts  K=gate_k")
    ax.set_xlabel("expert_prune_ratio")
    ax.set_ylabel("test_in_acc (IID selection, avg over test envs)")
    ax.set_title(f"{dataset_name} — OOD test acc vs prune_ratio for each (N, K) combination")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(figures_dir, "acc_vs_prune.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: acc vs num_experts (per top_k)
# ---------------------------------------------------------------------------

def plot_acc_vs_nexpert(summaries, cfg, figures_dir, dataset_name, test_accs):
    """OOD test accuracy vs num_experts (per gate_k), averaged over test envs."""
    if not HAS_MPL:
        return

    grid      = cfg["param_grid"]
    ne_values = sorted(set(grid["num_experts"]) | {
        s.get("hparams", {}).get("num_experts") for s in summaries.values()
        if s.get("hparams", {}).get("num_experts") is not None})
    tk_values = sorted(set(grid["top_k"]) | {
        s.get("hparams", {}).get("gate_k") for s in summaries.values()
        if s.get("hparams", {}).get("gate_k") is not None})
    colors    = cm.tab10(np.linspace(0, 1, len(tk_values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for tk, color in zip(tk_values, colors):
        accs = []
        for ne in ne_values:
            vals = [
                test_accs[run_id]["acc"]
                for run_id, s in summaries.items()
                if (run_id in test_accs
                    and s.get("hparams", {}).get("num_experts") == ne
                    and s.get("hparams", {}).get("gate_k") == tk)
            ]
            accs.append(float(np.mean(vals)) if vals else None)
        valid = [(n, a) for n, a in zip(ne_values, accs) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker="s", label=f"K={tk}", color=color,
                    linewidth=2.5 if tk == BASELINE["gate_k"] else 1.2)

    bl_accs = [test_accs[r]["acc"] for r, s in summaries.items()
               if r in test_accs and is_baseline(s.get("hparams", {}))]
    if bl_accs:
        ax.axvline(BASELINE["num_experts"], color="red", linestyle="--",
                   linewidth=1, label="baseline N")
        ax.scatter([BASELINE["num_experts"]], [float(np.mean(bl_accs))],
                   color="red", zorder=5, s=80)

    ax.set_xlabel("num_experts")
    ax.set_ylabel("test_in_acc (IID selection, avg over test envs)")
    ax.set_title(f"{dataset_name} — OOD test acc vs num_experts (avg over prune_ratio per K)")
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
        te = _get_test_env(run_id, s)
        key = f"env{te}_in_acc" if te is not None else None
        steps, accs = [], []
        for r in records:
            acc = r.get("val_in_acc_per_domain", {}).get(key) if key else None
            if acc is not None:
                steps.append(r["step"])
                accs.append(acc)
        if not steps:
            continue
        hp    = s.get("hparams", {})
        bl    = is_baseline(hp)
        label = f"{run_id}{' [baseline]' if bl else ''}"
        ax.plot(steps, accs, label=label, color=color,
                linewidth=2.5 if bl else 1.2,
                linestyle="-" if bl else "--")

    ax.set_xlabel("step")
    ax.set_ylabel("test_in_acc (IID selection)")
    title = f"{dataset_name} — Learning curves (OOD test domain only)"
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
    """Total vs active params per unique (N, K, PR) config — one bar pair per config."""
    if not HAS_MPL:
        return

    # Group by (N, K, PR) — params are identical across test envs for the same config
    seen = {}
    for run_id, s in sorted(summaries.items()):
        hp  = s.get("hparams", {})
        eff = s.get("efficiency_metrics", {})
        n   = hp.get("num_experts")
        k   = hp.get("gate_k")
        pr  = hp.get("expert_prune_ratio", 0.0)
        t   = eff.get("total_params")
        a   = eff.get("active_params")
        if None in (n, k, t, a):
            continue
        key = (n, k, pr)
        if key not in seen:
            seen[key] = (t, a)

    if not seen:
        return

    configs = sorted(seen.keys())
    totals  = [seen[c][0] / 1e6 for c in configs]
    actives = [seen[c][1] / 1e6 for c in configs]
    labels  = [f"N{n}_K{k}\nPR{pr:.1f}" for (n, k, pr) in configs]
    bl_mask = [
        (n == BASELINE["num_experts"] and k == BASELINE["gate_k"] and
         abs(pr - BASELINE["expert_prune_ratio"]) < 1e-9)
        for (n, k, pr) in configs
    ]

    x  = np.arange(len(configs))
    w  = 0.4

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 0.6), 5))
    bars_t = ax.bar(x - w / 2, totals,  w, label="total_params (M)",  color="#aec7e8")
    bars_a = ax.bar(x + w / 2, actives, w, label="active_params (M)", color="#1f77b4")

    for i, bl in enumerate(bl_mask):
        if bl:
            bars_t[i].set_edgecolor("red")
            bars_t[i].set_linewidth(2)
            bars_a[i].set_edgecolor("red")
            bars_a[i].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Parameters (M)")
    ax.set_title(f"{dataset_name} — Total vs active parameters per config\n(red border = baseline)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(figures_dir, "params_bar.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 6: Per-domain accuracy breakdown (avg across test envs)
# ---------------------------------------------------------------------------

def plot_per_domain_accuracy(summaries, ds_log_dir, figures_dir, dataset_name, domain_names):
    """Bar chart: per-domain accuracy for each unique (N, K, PR) config."""
    if not HAS_MPL:
        return

    n_domains = len(domain_names)
    # Build per-config data: group by (num_experts, gate_k, expert_prune_ratio, test_env)
    # and for each run, record per-domain out-acc at best checkpoint step
    config_domain_acc = {}  # (N, K, PR) -> list of (test_env, [d0,d1,d2,d3])

    for run_id, s in summaries.items():
        hp       = s.get("hparams", {})
        n        = hp.get("num_experts")
        k        = hp.get("gate_k")
        pr       = hp.get("expert_prune_ratio", 0.0)
        best_step = (s.get("best_checkpoint") or {}).get("step")
        if n is None or k is None:
            continue

        eval_recs = load_eval_jsonl(ds_log_dir, run_id)
        if not eval_recs:
            continue
        best_rec = next((r for r in eval_recs if r.get("step") == best_step), eval_recs[-1])
        in_per_domain = best_rec.get("val_in_acc_per_domain", {})
        accs = [in_per_domain.get(f"env{d}_in_acc") for d in range(n_domains)]
        if any(a is None for a in accs):
            continue

        key = (n, k, pr)
        config_domain_acc.setdefault(key, []).append(accs)

    if not config_domain_acc:
        return

    # Average over test envs for each config
    configs = sorted(config_domain_acc.keys())
    avg_accs = []
    labels   = []
    for (n, k, pr) in configs:
        mat = config_domain_acc[(n, k, pr)]
        avg = [sum(col) / len(col) for col in zip(*mat)]
        avg_accs.append(avg)
        labels.append(f"N{n}K{k}\nPR{pr:.1f}")

    x     = np.arange(len(configs))
    width = 0.8 / n_domains
    colors = cm.Set2(np.linspace(0, 1, n_domains))

    # Auto y-axis: start near the data minimum (clamped at 0.6 if all values > 0.6)
    all_vals = [v for row in avg_accs for v in row]
    y_min = max(0.0, (np.floor(min(all_vals) * 20) / 20) - 0.01)  # floor to 0.05 steps
    if min(all_vals) > 0.6:
        y_min = max(y_min, 0.6)

    fig, ax = plt.subplots(figsize=(max(12, len(configs) * 0.7), 5))
    for d in range(n_domains):
        vals = [avg_accs[i][d] for i in range(len(configs))]
        ax.plot(x, vals, "-o", color=colors[d], label=domain_names[d],
                linewidth=1.5, markersize=4, zorder=3)
        # Value annotations above each marker
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=5, rotation=90, color="black")

    ax.set_ylim(y_min, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("val out-acc at best checkpoint")
    ax.set_title(f"{dataset_name} — Per-domain accuracy (avg over test envs)")
    ax.legend(fontsize=8, title="Domain")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(figures_dir, "per_domain_acc.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 7: Worst-domain vs Avg accuracy scatter (fairness tradeoff)
# ---------------------------------------------------------------------------

def plot_worst_vs_avg(summaries, ds_log_dir, figures_dir, dataset_name, test_accs):
    """Scatter: x=avg accuracy, y=worst-domain accuracy, colored by num_experts."""
    if not HAS_MPL:
        return

    ne_all = sorted(set(
        s.get("hparams", {}).get("num_experts") for s in summaries.values()
        if s.get("hparams", {}).get("num_experts") is not None
    ))
    colors = {ne: cm.tab10(i / max(len(ne_all) - 1, 1)) for i, ne in enumerate(ne_all)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for run_id, s in summaries.items():
        if run_id not in test_accs:
            continue
        hp       = s.get("hparams", {})
        ne       = hp.get("num_experts")
        te       = test_accs[run_id]["test_env"]
        ood_acc  = test_accs[run_id]["acc"]          # correct OOD metric (x-axis)
        td_step  = test_accs[run_id]["step"]

        # Worst in_acc over ALL domains at the IID best checkpoint
        eval_recs = test_accs[run_id].get("eval_records") or load_eval_jsonl(ds_log_dir, run_id)
        best_rec  = next((r for r in eval_recs if r.get("step") == td_step), eval_recs[-1])
        per_dom   = best_rec.get("val_in_acc_per_domain", {})
        all_ood   = [v for v in per_dom.values() if v is not None]
        worst_acc = min(all_ood) if all_ood else None
        if worst_acc is None:
            continue

        bl = is_baseline(hp)
        ax.scatter(ood_acc, worst_acc,
                   color=colors.get(ne, "gray"),
                   marker="*" if bl else "o",
                   s=120 if bl else 40,
                   zorder=3 if bl else 2,
                   alpha=0.8)

    for ne in ne_all:
        ax.scatter([], [], color=colors[ne], label=f"N={ne}", s=40)
    ax.scatter([], [], color="black", marker="*", s=120, label="baseline")

    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="y=x")

    ax.set_xlabel("test_in_acc (IID selection)")
    ax.set_ylabel("Worst out_acc across all domains at IID best checkpoint")
    ax.set_title(f"{dataset_name} — test_in_acc vs Worst-domain acc (fairness tradeoff)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(figures_dir, "worst_vs_avg.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 8: Pareto frontier — accuracy vs active parameters
# ---------------------------------------------------------------------------

def plot_pareto(summaries, figures_dir, dataset_name, test_accs):
    """Scatter: x=active_params (M), y=avg OOD acc across test envs per config.

    Each point = one unique (N, K, PR) config, averaged over all test environments.
    """
    if not HAS_MPL:
        return

    # Group by (N, K, PR): collect acc values and take active_params from first run
    groups = {}  # (N, K, PR) -> {"accs": [], "active": float, "ne": int, "bl": bool}
    for run_id, s in summaries.items():
        if run_id not in test_accs:
            continue
        hp     = s.get("hparams", {})
        eff    = s.get("efficiency_metrics", {})
        acc    = test_accs[run_id]["acc"]
        active = eff.get("active_params")
        ne     = hp.get("num_experts")
        k      = hp.get("gate_k")
        pr     = hp.get("expert_prune_ratio", 0.0)
        if acc is None or active is None or ne is None or k is None:
            continue
        key = (ne, k, pr)
        if key not in groups:
            groups[key] = {"accs": [], "active": active, "ne": ne, "bl": is_baseline(hp)}
        groups[key]["accs"].append(acc)

    if not groups:
        return

    # Build averaged points
    points = []
    for (ne, k, pr), g in groups.items():
        avg_acc = float(np.mean(g["accs"]))
        label   = f"N{ne}_K{k}_PR{int(round(pr * 10)):02d}"
        points.append((g["active"] / 1e6, avg_acc, ne, g["bl"], label))

    # Pareto frontier (minimise params, maximise acc)
    points.sort(key=lambda p: p[0])
    pareto, best_acc = [], -1.0
    for pt in points:
        if pt[1] > best_acc:
            best_acc = pt[1]
            pareto.append(pt)

    ne_all = sorted(set(p[2] for p in points if p[2] is not None))
    colors = {ne: cm.tab10(i / max(len(ne_all) - 1, 1)) for i, ne in enumerate(ne_all)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for (active, acc, ne, bl, label) in points:
        ax.scatter(active, acc,
                   color=colors.get(ne, "gray"),
                   marker="*" if bl else "o",
                   s=140 if bl else 50,
                   alpha=0.85,
                   zorder=3 if bl else 2)
        ax.annotate(label, (active, acc),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=5, alpha=0.75)

    if pareto:
        px, py = zip(*[(p[0], p[1]) for p in pareto])
        ax.step(px, py, where="post", color="red", linewidth=1.5,
                linestyle="--", label="Pareto frontier", zorder=4)

    for ne in ne_all:
        ax.scatter([], [], color=colors[ne], label=f"N={ne}", s=40)
    ax.scatter([], [], color="black", marker="*", s=120, label="baseline")

    ax.set_xlabel("Active parameters (M)")
    ax.set_ylabel("test_in_acc (IID selection, avg over 4 test envs)")
    ax.set_title(f"{dataset_name} — Pareto frontier: OOD acc vs Active params\n"
                 f"(each point = 1 config averaged over all test environments)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(figures_dir, "pareto_frontier.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 9: Convergence speed heatmap (step reaching 90% of best acc)
# ---------------------------------------------------------------------------

def plot_convergence_heatmap(summaries, ds_log_dir, cfg, figures_dir, dataset_name,
                             test_accs):
    """Heatmap: num_experts × gate_k showing steps to 90% of OOD test acc (PR=0 only)."""
    if not HAS_MPL:
        return

    grid      = cfg["param_grid"]
    ne_values = sorted(set(grid["num_experts"]) | {
        s.get("hparams", {}).get("num_experts") for s in summaries.values()
        if s.get("hparams", {}).get("num_experts") is not None})
    tk_values = sorted(set(grid["top_k"]) | {
        s.get("hparams", {}).get("gate_k") for s in summaries.values()
        if s.get("hparams", {}).get("gate_k") is not None})
    data      = np.full((len(ne_values), len(tk_values)), np.nan)
    count     = np.zeros_like(data, dtype=int)

    for run_id, s in summaries.items():
        if run_id not in test_accs:
            continue
        hp = s.get("hparams", {})
        pr = hp.get("expert_prune_ratio", -1)
        if abs(pr) > 1e-9:
            continue
        n        = hp.get("num_experts")
        k        = hp.get("gate_k")
        te       = test_accs[run_id]["test_env"]
        best_acc = test_accs[run_id]["acc"]        # correct OOD metric
        if n not in ne_values or k not in tk_values:
            continue

        dom_key   = f"env{te}_in_acc"
        eval_recs = test_accs[run_id].get("eval_records") or load_eval_jsonl(ds_log_dir, run_id)
        threshold = 0.9 * float(best_acc)
        conv_step = None
        for rec in eval_recs:
            acc = rec.get("val_in_acc_per_domain", {}).get(dom_key)
            if acc is not None and acc >= threshold:
                conv_step = rec.get("step")
                break
        if conv_step is None:
            continue

        i, j = ne_values.index(n), tk_values.index(k)
        if np.isnan(data[i, j]):
            data[i, j] = conv_step
            count[i, j] = 1
        else:
            data[i, j] = (data[i, j] * count[i, j] + conv_step) / (count[i, j] + 1)
            count[i, j] += 1

    if np.all(np.isnan(data)):
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(tk_values)))
    ax.set_xticklabels([f"K={k}" for k in tk_values])
    ax.set_yticks(range(len(ne_values)))
    ax.set_yticklabels([f"N={n}" for n in ne_values])
    ax.set_xlabel("top_k (gate_k)")
    ax.set_ylabel("num_experts")
    ax.set_title(f"{dataset_name} — Steps to reach 90% best acc (PR=0, lower=faster)")
    for i in range(len(ne_values)):
        for j in range(len(tk_values)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{int(data[i,j])}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="convergence step")
    plt.tight_layout()
    out = os.path.join(figures_dir, "convergence_heatmap.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Learning curves — grouped by (N, K), one figure per pair
# ---------------------------------------------------------------------------

def plot_learning_curves_grouped(summaries, ds_log_dir, figures_dir, dataset_name):
    """One figure per (N, K) pair showing all PR curves averaged over test envs."""
    if not HAS_MPL:
        return

    # Group runs by (N, K, PR)
    groups = {}
    for run_id, s in summaries.items():
        hp = s.get("hparams", {})
        n  = hp.get("num_experts")
        k  = hp.get("gate_k")
        pr = hp.get("expert_prune_ratio", 0.0)
        if n is None or k is None:
            continue
        groups.setdefault((n, k, pr), []).append((run_id, s))

    nk_pairs = sorted(set((n, k) for (n, k, _) in groups.keys()))
    pr_all   = sorted(set(pr for (_, _, pr) in groups.keys()))
    colors   = {pr: cm.tab10(i / max(len(pr_all) - 1, 1)) for i, pr in enumerate(pr_all)}

    for (ne, k) in nk_pairs:
        fig, ax = plt.subplots(figsize=(9, 5))
        has_any = False

        for pr in pr_all:
            key = (ne, k, pr)
            if key not in groups:
                continue

            # Collect OOD curves: each run uses only its own test domain acc
            all_curves = []
            for run_id, s in groups[key]:
                te = _get_test_env(run_id, s)
                if te is None:
                    continue
                recs = load_eval_jsonl(ds_log_dir, run_id)
                if not recs:
                    continue
                dom_key = f"env{te}_in_acc"
                steps = [r["step"] for r in recs]
                accs  = [r.get("val_in_acc_per_domain", {}).get(dom_key) for r in recs]
                all_curves.append((steps, accs))

            if not all_curves:
                continue

            # Average across test_envs (shared step schedule)
            steps = all_curves[0][0]
            avg_accs = []
            for i in range(len(steps)):
                vals = [c[1][i] for c in all_curves
                        if i < len(c[1]) and c[1][i] is not None]
                avg_accs.append(float(np.mean(vals)) if vals else None)

            valid = [(s, a) for s, a in zip(steps, avg_accs) if a is not None]
            if not valid:
                continue

            xs, ys = zip(*valid)
            bl = (ne == BASELINE["num_experts"] and k == BASELINE["gate_k"]
                  and abs(pr - BASELINE["expert_prune_ratio"]) < 1e-9)
            ax.plot(xs, ys, color=colors[pr], label=f"PR={pr:.1f}",
                    linewidth=2.5 if bl else 1.5,
                    linestyle="-" if bl else "--",
                    marker="o" if bl else None, markersize=4)
            has_any = True

        if not has_any:
            plt.close(fig)
            continue

        ax.set_xlabel("step")
        ax.set_ylabel("test_in_acc (IID selection, avg over test envs)")
        ax.set_title(f"{dataset_name} — OOD learning curves: N={ne}, K={k}")
        ax.legend(fontsize=8, title="prune_ratio")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(figures_dir, f"learning_curves_N{ne}_K{k}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 10: Aux loss decay curves
# ---------------------------------------------------------------------------

def plot_aux_loss_curves(summaries, ds_log_dir, figures_dir, dataset_name, group_filter):
    """Line plot: total aux loss over training steps, grouped by (N, K, PR)."""
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
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = cm.tab10(np.linspace(0, 1, max(len(selected), 1)))

    for (run_id, s), color in zip(selected, colors):
        expert_recs = load_jsonl(ds_log_dir, run_id, "expert_stats.jsonl")
        if not expert_recs:
            continue
        steps     = [r["step"] for r in expert_recs]
        aux_total = [r.get("total_aux_loss") for r in expert_recs]
        valid     = [(st, al) for st, al in zip(steps, aux_total) if al is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        hp     = s.get("hparams", {})
        bl     = is_baseline(hp)
        label  = f"N{hp.get('num_experts')}K{hp.get('gate_k')}PR{hp.get('expert_prune_ratio',0):.1f}"
        ax.plot(xs, ys, label=label, color=color,
                linewidth=2.5 if bl else 1.2,
                linestyle="-" if bl else "--")

    ax.set_xlabel("step")
    ax.set_ylabel("total aux loss (load balancing)")
    title = f"{dataset_name} — Expert load balancing (aux loss)"
    if group_filter:
        title += f" [{group_filter}]"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    suffix = group_filter.replace("=", "").replace(",", "_") if group_filter else "all"
    out = os.path.join(figures_dir, f"aux_loss_curves_{suffix}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 11: Cross-env aggregated bar (mean ± std across 4 test envs)
# ---------------------------------------------------------------------------

def plot_cross_env_bar(summaries, cfg, figures_dir, dataset_name, test_accs):
    """OOD test acc mean±std across test envs, per (N, K, PR) config."""
    if not HAS_MPL:
        return

    # Group by (num_experts, gate_k, expert_prune_ratio), using correct OOD metric
    groups = {}
    for run_id, s in summaries.items():
        if run_id not in test_accs:
            continue
        hp  = s.get("hparams", {})
        acc = test_accs[run_id]["acc"]              # env{test_env}_out_acc
        n   = hp.get("num_experts")
        k   = hp.get("gate_k")
        pr  = hp.get("expert_prune_ratio", 0.0)
        if acc is None or n is None or k is None:
            continue
        key = (n, k, pr)
        groups.setdefault(key, []).append(acc)

    if not groups:
        return

    # Sort by mean accuracy descending
    keys  = sorted(groups.keys(), key=lambda c: -float(np.mean(groups[c])))
    means = [float(np.mean(groups[k])) for k in keys]
    stds  = [float(np.std(groups[k]))  for k in keys]
    labels = [f"N{n}\nK{k}\nPR{pr:.1f}" for (n, k, pr) in keys]
    bl_mask = [
        (n == BASELINE["num_experts"] and k == BASELINE["gate_k"] and
         abs(pr - BASELINE["expert_prune_ratio"]) < 1e-9)
        for (n, k, pr) in keys
    ]

    x      = np.arange(len(keys))
    colors = ["tomato" if bl else "steelblue" for bl in bl_mask]

    # Auto y-axis: start near the data minimum
    y_min = max(0.0, (np.floor(min(means) * 20) / 20) - 0.01)
    if min(means) > 0.6:
        y_min = max(y_min, 0.6)

    # Point colors: baseline = tomato, others = steelblue
    pt_colors = ["tomato" if bl else "steelblue" for bl in bl_mask]

    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.55), 5))

    # Main line + error bars
    ax.plot(x, means, "-", color="steelblue", linewidth=1.2, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", capsize=3,
                ecolor="gray", elinewidth=1.0, zorder=3)
    # Colored markers per config
    for xi, (m, c) in enumerate(zip(means, pt_colors)):
        ax.scatter(xi, m, color=c, s=40, zorder=4)

    # Value annotations above each point
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.003, f"{m:.3f}", ha="center", va="bottom",
                fontsize=6, rotation=90, color="black")

    ax.set_ylim(y_min, max(means) + max(stds) + 0.06)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("test_in_acc (IID selection)")
    ax.set_title(f"{dataset_name} — Config ranking by OOD test acc (mean ± std across test envs)")
    ax.scatter([], [], color="tomato",    s=40, label="baseline")
    ax.scatter([], [], color="steelblue", s=40, label="other configs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(figures_dir, "cross_env_bar.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 12: Config ranking per test domain (one figure per domain)
# ---------------------------------------------------------------------------

def plot_cross_env_per_domain(summaries, cfg, figures_dir, dataset_name, domain_names,
                              test_accs):
    """One ranking figure per test domain, using env{test_env}_out_acc (OOD metric)."""
    if not HAS_MPL:
        return

    n_domains = len(domain_names)

    # Bucket each run by its test_env, storing the correct OOD accuracy
    by_env = {}  # test_env -> {(N, K, PR): acc}
    for run_id, s in summaries.items():
        if run_id not in test_accs:
            continue
        te  = test_accs[run_id]["test_env"]
        acc = test_accs[run_id]["acc"]              # env{te}_out_acc
        hp  = s.get("hparams", {})
        n   = hp.get("num_experts")
        k   = hp.get("gate_k")
        pr  = hp.get("expert_prune_ratio", 0.0)
        if acc is None or n is None or k is None:
            continue
        by_env.setdefault(te, {})[(n, k, pr)] = acc

    if not by_env:
        return

    for env_id in sorted(by_env.keys()):
        config_acc = by_env[env_id]
        domain_label = domain_names[env_id] if env_id < n_domains else f"env{env_id}"

        # Sort configs by accuracy descending
        keys   = sorted(config_acc.keys(), key=lambda c: -config_acc[c])
        accs   = [config_acc[k] for k in keys]
        labels = [f"N{n}\nK{k}\nPR{pr:.1f}" for (n, k, pr) in keys]
        bl_mask = [
            (n == BASELINE["num_experts"] and k == BASELINE["gate_k"] and
             abs(pr - BASELINE["expert_prune_ratio"]) < 1e-9)
            for (n, k, pr) in keys
        ]

        x      = np.arange(len(keys))
        y_min  = max(0.0, (np.floor(min(accs) * 20) / 20) - 0.01)
        if min(accs) > 0.6:
            y_min = max(y_min, 0.6)

        fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.55), 5))

        ax.plot(x, accs, "-", color="steelblue", linewidth=1.2, zorder=2)
        for xi, (acc, bl) in enumerate(zip(accs, bl_mask)):
            ax.scatter(xi, acc, color="tomato" if bl else "steelblue", s=40, zorder=4)
        for xi, acc in enumerate(accs):
            ax.text(xi, acc + 0.003, f"{acc:.3f}", ha="center", va="bottom",
                    fontsize=6, rotation=90, color="black")

        ax.set_ylim(y_min, max(accs) + 0.06)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel(f"in_acc — env{env_id}_in_acc")
        ax.set_title(f"{dataset_name} — Config ranking | test domain: {domain_label} (env{env_id})")
        ax.scatter([], [], color="tomato",    s=40, label="baseline")
        ax.scatter([], [], color="steelblue", s=40, label="other configs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        out = os.path.join(figures_dir, f"cross_env_domain{env_id}_{domain_label}.png")
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

    ds_cfg       = load_dataset_config(dataset_name)
    domain_names = ds_cfg.get("domain_names") or [f"env{i}" for i in range(4)]

    # Precompute IID metrics (step by training-domain val, report test in_acc)
    print("  Precomputing IID test accuracies from eval_log.jsonl ...")
    test_accs = _precompute_test_accs(summaries, ds_log_dir)
    print(f"  IID metrics ready for {len(test_accs)}/{len(summaries)} runs.")

    # Original plots (updated to use test_accs)
    plot_heatmaps(summaries, cfg, figures_dir, dataset_name, test_accs)
    plot_acc_vs_prune(summaries, cfg, figures_dir, dataset_name, test_accs)
    plot_acc_vs_nexpert(summaries, cfg, figures_dir, dataset_name, test_accs)
    # Learning curves: per-(N,K) grouped figures + optional filtered view
    plot_learning_curves_grouped(summaries, ds_log_dir, figures_dir, dataset_name)
    if group_filter:
        plot_learning_curves(summaries, ds_log_dir, figures_dir, dataset_name, group_filter)
    plot_params_bar(summaries, figures_dir, dataset_name)

    # Extended plots (updated to use test_accs)
    plot_per_domain_accuracy(summaries, ds_log_dir, figures_dir, dataset_name, domain_names)
    plot_worst_vs_avg(summaries, ds_log_dir, figures_dir, dataset_name, test_accs)
    plot_pareto(summaries, figures_dir, dataset_name, test_accs)
    plot_convergence_heatmap(summaries, ds_log_dir, cfg, figures_dir, dataset_name, test_accs)
    plot_aux_loss_curves(summaries, ds_log_dir, figures_dir, dataset_name, group_filter)
    plot_cross_env_bar(summaries, cfg, figures_dir, dataset_name, test_accs)
    plot_cross_env_per_domain(summaries, cfg, figures_dir, dataset_name, domain_names, test_accs)


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

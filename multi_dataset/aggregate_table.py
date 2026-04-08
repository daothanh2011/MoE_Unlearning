"""
aggregate_table.py — Build the final DG benchmark table from per-run logs.

For each (algo, dataset, test_env, seed):
  1. Read multi_dataset/{algo}/{dataset}/env{E}/seed{N}/log/eval_log.jsonl
  2. IID model selection: argmax(mean(training envs out_acc)) → record env{E}_in_acc
Then aggregate:
  - per_domain.csv  : (algo, dataset, test_env, seed) → test_in_acc
  - per_dataset.csv : (algo, dataset)                 → mean ± std (over envs × seeds)
  - final_table.csv : algos × datasets matrix         → mean ± std
  - final_table.md  : same matrix in pretty markdown form
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sweep.aggregate_results import compute_iid_best, load_jsonl  # noqa: E402
from sweep.run_sweep import load_dataset_config  # noqa: E402


def load_benchmark_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def filter_names(items, allowed):
    if not allowed:
        return items
    allow_set = set(allowed)
    return [it for it in items if (it["name"] if isinstance(it, dict) else it) in allow_set]


def collect_per_run(cfg, algos_filter, datasets_filter, seeds_filter):
    """Yield dicts: {algo, dataset, test_env, seed, test_in_acc, n_envs}."""
    output_root = cfg["output_root"]
    algorithms  = filter_names(cfg["algorithms"], algos_filter)
    datasets    = filter_names(cfg["datasets"],   datasets_filter)
    seeds       = [s for s in cfg["seeds"] if (not seeds_filter or s in seeds_filter)]

    for algo_entry in algorithms:
        algo_name = algo_entry["name"]
        for ds_name in datasets:
            try:
                ds_cfg = load_dataset_config(ds_name)
            except FileNotFoundError:
                continue
            test_envs = ds_cfg["test_envs"]
            n_envs = len(test_envs)  # assumes test_envs lists every domain
            for test_env in test_envs:
                for seed in seeds:
                    log_path = os.path.join(
                        output_root, algo_name, ds_name,
                        f"env{test_env}", f"seed{seed}",
                        "log", "eval_log.jsonl",
                    )
                    if not os.path.exists(log_path):
                        yield {
                            "algo": algo_name, "dataset": ds_name,
                            "test_env": test_env, "seed": seed,
                            "test_in_acc": None, "best_step": None,
                            "n_envs": n_envs, "status": "missing",
                        }
                        continue
                    records = load_jsonl(log_path)
                    best_step, test_in_acc = compute_iid_best(
                        records, test_env, n_envs=n_envs)
                    yield {
                        "algo": algo_name, "dataset": ds_name,
                        "test_env": test_env, "seed": seed,
                        "test_in_acc": test_in_acc, "best_step": best_step,
                        "n_envs": n_envs,
                        "status": "ok" if test_in_acc is not None else "no_iid_step",
                    }


def write_per_domain(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "dataset", "test_env", "seed",
                    "test_in_acc", "best_step", "status"])
        for r in rows:
            w.writerow([r["algo"], r["dataset"], r["test_env"], r["seed"],
                        f"{r['test_in_acc']:.4f}" if r["test_in_acc"] is not None else "",
                        r["best_step"] if r["best_step"] is not None else "",
                        r["status"]])
    print(f"wrote {path}")


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def aggregate_per_dataset(rows):
    """Group by (algo, dataset, test_env), then by (algo, dataset).
    Returns nested dict: out[algo][dataset] = (mean, std, n)."""
    by_cell = {}  # (algo, dataset, test_env) -> [seed accs]
    for r in rows:
        if r["test_in_acc"] is None:
            continue
        key = (r["algo"], r["dataset"], r["test_env"])
        by_cell.setdefault(key, []).append(r["test_in_acc"])

    # First reduce seeds → per-(algo, dataset, test_env) mean
    per_env_mean = {}  # (algo, dataset, test_env) -> mean over seeds
    for key, accs in by_cell.items():
        per_env_mean[key] = statistics.mean(accs)

    # Then average across test_envs per (algo, dataset)
    by_ad = {}  # (algo, dataset) -> [env means]
    for (algo, ds, env), m in per_env_mean.items():
        by_ad.setdefault((algo, ds), []).append(m)

    # Std across seeds: pool seed-level stds → mean of stds, OR std across env means.
    # We report mean ± std where std = stdev across (test_env × seed) cells (paper convention).
    by_ad_all_seeds = {}  # (algo, ds) -> all seed accs
    for (algo, ds, env), accs in by_cell.items():
        by_ad_all_seeds.setdefault((algo, ds), []).extend(accs)

    out = {}
    for (algo, ds), accs in by_ad_all_seeds.items():
        m, s, n = _mean_std(accs)
        out.setdefault(algo, {})[ds] = (m, s, n)
    return out, per_env_mean


def write_per_dataset_csv(table, datasets, algorithms, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "dataset", "mean_acc", "std_acc", "n_runs"])
        for algo in algorithms:
            for ds in datasets:
                cell = table.get(algo, {}).get(ds)
                if cell:
                    m, s, n = cell
                    w.writerow([algo, ds,
                                f"{m:.4f}" if m is not None else "",
                                f"{s:.4f}" if s is not None else "",
                                n])
                else:
                    w.writerow([algo, ds, "", "", 0])
    print(f"wrote {path}")


def write_final_table(table, datasets, algorithms, csv_path, md_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Algorithm"] + datasets)
        for algo in algorithms:
            row = [algo]
            for ds in datasets:
                cell = table.get(algo, {}).get(ds)
                if cell and cell[0] is not None:
                    m, s, _ = cell
                    row.append(f"{m*100:.1f} ± {s*100:.1f}")
                else:
                    row.append("—")
            w.writerow(row)
    print(f"wrote {csv_path}")

    # Markdown table
    col_widths = [max(len("Algorithm"), max(len(a) for a in algorithms))]
    rendered_rows = []
    for algo in algorithms:
        cells = []
        for ds in datasets:
            cell = table.get(algo, {}).get(ds)
            if cell and cell[0] is not None:
                m, s, _ = cell
                cells.append(f"{m*100:.1f} ± {s*100:.1f}")
            else:
                cells.append("—")
        rendered_rows.append((algo, cells))

    for i, ds in enumerate(datasets, start=1):
        w_col = max(len(ds), max(len(r[1][i-1]) for r in rendered_rows))
        col_widths.append(w_col)

    def fmt_row(label, cells):
        parts = [label.ljust(col_widths[0])]
        for i, c in enumerate(cells, start=1):
            parts.append(c.rjust(col_widths[i]))
        return "| " + " | ".join(parts) + " |"

    header  = fmt_row("Algorithm", datasets)
    divider = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines   = [header, divider] + [fmt_row(a, c) for a, c in rendered_rows]
    with open(md_path, "w") as f:
        f.write("# DG Benchmark — Per-Dataset Accuracy (mean ± std, %)\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def main():
    p = argparse.ArgumentParser(description="Aggregate multi_dataset benchmark logs into final table")
    p.add_argument("--config",   default="multi_dataset/benchmark_config.yaml")
    p.add_argument("--algos",    nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--seeds",    nargs="+", type=int, default=None)
    p.add_argument("--out_dir",  default="multi_dataset/results")
    args = p.parse_args()

    cfg = load_benchmark_config(args.config)

    rows = list(collect_per_run(cfg, args.algos, args.datasets, args.seeds))

    write_per_domain(rows, os.path.join(args.out_dir, "per_domain.csv"))

    table, _ = aggregate_per_dataset(rows)
    algos    = [a["name"] for a in filter_names(cfg["algorithms"], args.algos)]
    datasets = filter_names(cfg["datasets"], args.datasets)

    write_per_dataset_csv(
        table, datasets, algos,
        os.path.join(args.out_dir, "per_dataset.csv"),
    )
    write_final_table(
        table, datasets, algos,
        os.path.join(args.out_dir, "final_table.csv"),
        os.path.join(args.out_dir, "final_table.md"),
    )

    # Summary
    n_total   = len(rows)
    n_ok      = sum(1 for r in rows if r["status"] == "ok")
    n_missing = sum(1 for r in rows if r["status"] == "missing")
    print(f"\nSummary: {n_ok}/{n_total} runs aggregated ({n_missing} missing)")


if __name__ == "__main__":
    main()

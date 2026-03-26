"""
aggregate_results.py — Compile per-run final_summary.json files into sweep_summary.csv.

Usage:
    python sweep/aggregate_results.py --dataset PACS
    python sweep/aggregate_results.py --dataset all    # aggregate every dataset
    python sweep/aggregate_results.py                  # same as --dataset all
    python sweep/aggregate_results.py --log_dir sweep/logs --dataset OfficeHome
"""

import argparse
import csv
import glob
import json
import os

import yaml


COLUMNS = [
    "run_id", "dataset", "test_env",
    "num_experts", "gate_k", "expert_prune_ratio", "hidden_size_per_expert",
    "total_params", "active_params", "params_utilization_ratio", "peak_mem_gb",
    "best_val_avg_acc", "best_val_step",
    "final_val_avg_acc",
    "val_acc_std_last_10pct",
    "total_train_time_sec", "avg_sec_per_step",
    "collapse_detected", "status",
]


def load_sweep_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def list_available_datasets(log_dir):
    """Find all dataset subdirectories in log_dir."""
    return [d for d in sorted(os.listdir(log_dir))
            if os.path.isdir(os.path.join(log_dir, d))]


def parse_summary(path):
    with open(path) as f:
        return json.load(f)


def extract_row(summary):
    hp  = summary.get("hparams", {})
    eff = summary.get("efficiency_metrics", {})
    stb = summary.get("stability_metrics", {})
    bst = summary.get("best_checkpoint", {}) or {}
    fin = summary.get("final_checkpoint", {}) or {}

    # Parse test_env from run_id if not in summary
    run_id   = summary.get("run_id", "")
    test_env = summary.get("test_env", "")
    if not test_env and "_env" in run_id:
        try:
            test_env = int(run_id.split("_env")[-1])
        except ValueError:
            pass

    # Parse dataset from run_id if not in summary
    dataset = summary.get("dataset", "")
    if not dataset and run_id:
        dataset = run_id.split("_N")[0] if "_N" in run_id else ""

    return {
        "run_id":                   run_id,
        "dataset":                  dataset,
        "test_env":                 test_env,
        "num_experts":              hp.get("num_experts", ""),
        "gate_k":                   hp.get("gate_k", ""),
        "expert_prune_ratio":       hp.get("expert_prune_ratio", ""),
        "hidden_size_per_expert":   eff.get("hidden_size_per_expert", ""),
        "total_params":             eff.get("total_params", ""),
        "active_params":            eff.get("active_params", ""),
        "params_utilization_ratio": eff.get("params_utilization_ratio", ""),
        "peak_mem_gb":              eff.get("peak_mem_gb", ""),
        "best_val_avg_acc":         bst.get("val_avg_acc", ""),
        "best_val_step":            bst.get("step", ""),
        "final_val_avg_acc":        fin.get("val_avg_acc", ""),
        "val_acc_std_last_10pct":   stb.get("val_acc_std_last_10pct_steps", ""),
        "total_train_time_sec":     eff.get("total_train_time_sec", ""),
        "avg_sec_per_step":         eff.get("avg_sec_per_step", ""),
        "collapse_detected":        stb.get("collapse_detected", ""),
        "status":                   summary.get("status", ""),
    }


def sort_key(r):
    v = r.get("best_val_avg_acc", "")
    try:
        return -float(v)
    except (TypeError, ValueError):
        return 0.0


def print_table(rows):
    if not rows:
        print("No results to display.")
        return
    key_cols = ["run_id", "num_experts", "gate_k", "expert_prune_ratio",
                "best_val_avg_acc", "best_val_step", "active_params",
                "peak_mem_gb", "status"]
    col_w = {}
    for h in key_cols:
        col_w[h] = max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0))
    sep = "  ".join("-" * col_w[h] for h in key_cols)
    print("  ".join(h.ljust(col_w[h]) for h in key_cols))
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(col_w[h]) for h in key_cols))


def aggregate_dataset(dataset_name, log_dir, results_dir):
    ds_log_dir = os.path.join(log_dir, dataset_name)
    if not os.path.isdir(ds_log_dir):
        print(f"  [WARN] Log dir not found: {ds_log_dir}")
        return [], [], []

    rows    = []
    missing = []
    failed  = []

    for run_id in sorted(os.listdir(ds_log_dir)):
        summary_path = os.path.join(ds_log_dir, run_id, "final_summary.json")
        if not os.path.exists(summary_path):
            missing.append(run_id)
            continue
        try:
            summary = parse_summary(summary_path)
            row = extract_row(summary)
            if row["status"] == "failed":
                failed.append(run_id)
            rows.append(row)
        except Exception as e:
            print(f"    [WARN] {run_id}: failed to parse — {e}")
            missing.append(run_id)

    if not rows:
        return rows, missing, failed

    rows.sort(key=sort_key)

    ds_results_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(ds_results_dir, exist_ok=True)
    csv_out = os.path.join(ds_results_dir, "sweep_summary.csv")

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows → {csv_out}")
    return rows, missing, failed


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate sweep final_summary.json → per-dataset CSV")
    parser.add_argument("--config",      default="sweep/sweep_config.yaml")
    parser.add_argument("--dataset",     default=None, metavar="NAME_OR_all",
                        help="Dataset name (e.g. PACS) or 'all' / omit for all datasets")
    parser.add_argument("--log_dir",     default=None,
                        help="Override log dir from config")
    parser.add_argument("--results_dir", default=None,
                        help="Override results dir from config")
    args = parser.parse_args()

    cfg         = load_sweep_config(args.config)
    outputs     = cfg.get("output", {})
    log_dir     = args.log_dir     or outputs.get("log_dir",     "sweep/logs")
    results_dir = args.results_dir or outputs.get("results_dir", "sweep/results")

    if not os.path.isdir(log_dir):
        print(f"Log dir not found: {log_dir}")
        return

    if args.dataset is None or args.dataset.lower() == "all":
        dataset_names = list_available_datasets(log_dir)
        if not dataset_names:
            print(f"No dataset subdirectories found in {log_dir}.")
            return
        print(f"Aggregating {len(dataset_names)} dataset(s): {dataset_names}\n")
    else:
        dataset_names = [args.dataset]

    all_rows = []
    for ds_name in dataset_names:
        print(f"=== {ds_name} ===")
        rows, missing, failed = aggregate_dataset(ds_name, log_dir, results_dir)
        all_rows.extend(rows)

        if missing:
            n_show = min(len(missing), 5)
            print(f"  Missing/unreadable ({len(missing)}): "
                  f"{missing[:n_show]}{'...' if len(missing) > n_show else ''}")
        if failed:
            print(f"  Failed runs ({len(failed)}): {failed}")
        collapse = [r["run_id"] for r in rows if r.get("collapse_detected") is True]
        if collapse:
            print(f"  Collapse detected: {collapse}")

        if rows:
            print(f"\n  === Ranked by best_val_avg_acc ===")
            print_table(rows)
        print()

    if len(dataset_names) > 1 and all_rows:
        # Write combined summary across all datasets
        combined_csv = os.path.join(results_dir, "sweep_summary_all.csv")
        os.makedirs(results_dir, exist_ok=True)
        all_rows.sort(key=sort_key)
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Combined summary ({len(all_rows)} rows) → {combined_csv}")


if __name__ == "__main__":
    main()

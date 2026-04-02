"""
aggregate_results.py — Compile per-run final_summary.json files into sweep_summary.csv.

Usage:
    python sweep/aggregate_results.py --dataset PACS
    python sweep/aggregate_results.py --dataset all    # aggregate every dataset
    python sweep/aggregate_results.py                  # same as --dataset all
    python sweep/aggregate_results.py --log_dir sweep/logs --dataset OfficeHome

Key metric: test_domain_acc
    For a run with test_env=i, the correct OOD test accuracy is ONLY env{i}_out_acc,
    NOT val_avg_acc (which averages over all 4 domains including training domains).
    test_domain_acc is computed from eval_log.jsonl, not from final_summary.json.
"""

import argparse
import csv
import json
import os

import yaml


COLUMNS = [
    "run_id", "dataset", "test_env",
    "num_experts", "gate_k", "expert_prune_ratio", "hidden_size_per_expert",
    "total_params", "active_params", "params_utilization_ratio", "peak_mem_gb",
    # PRIMARY metric (DomainBed IID): best step by mean(training envs out_acc), report test in_acc
    "test_domain_in_acc", "test_domain_iid_step",
    # Oracle reference: best step by test out_acc (inflated — do not use for ranking)
    "test_domain_acc", "test_domain_best_step",
    # Reference only — avg over ALL domains incl. training ones (do not use for ranking)
    "best_val_avg_acc", "best_val_step",
    "final_val_avg_acc",
    "val_acc_std_last_10pct",
    "total_train_time_sec", "avg_sec_per_step",
    "collapse_detected", "status",
    # Per-domain accuracy at IID best checkpoint
    "best_d0_acc", "best_d1_acc", "best_d2_acc", "best_d3_acc",
    "best_worst_acc", "best_std_acc",
    # Convergence: first step where mean(training envs out_acc) >= 90% of final IID val_acc
    "convergence_step",
    # MoE load-balancing: total auxiliary loss at final checkpoint
    "final_aux_loss",
]


def load_sweep_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
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


def _parse_test_env(run_id):
    """Extract integer test_env from run_id (e.g. 'OfficeHome_N4_K1_PR00_env0' → 0)."""
    if "_env" in run_id:
        try:
            return int(run_id.split("_env")[-1])
        except ValueError:
            pass
    return None


def compute_iid_best(eval_records, test_env, n_envs=4):
    """
    DomainBed IID selection method:
      - Select step by argmax(mean(out_acc of training envs)) — never looks at test domain.
      - Report env{test_env}_in_acc at that step (80% split, larger and more reliable).
    Returns (best_step, test_in_acc).
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
    return best_step, (best_test_in if best_step is not None else None)


def _iid_val_acc(eval_records, test_env, iid_best_step, n_envs=4):
    """Return the mean(training envs out_acc) at the IID best step (for convergence threshold)."""
    for rec in eval_records:
        if rec.get("step") != iid_best_step:
            continue
        per_domain = rec.get("val_acc_per_domain", {})
        train_accs = [
            per_domain.get(f"env{i}_out_acc")
            for i in range(n_envs) if i != test_env
        ]
        train_accs = [a for a in train_accs if a is not None]
        return sum(train_accs) / len(train_accs) if train_accs else None
    return None


def compute_test_domain_best(eval_records, test_env):
    """
    Oracle reference: checkpoint with highest env{test_env}_in_acc.
    Kept for reference only — inflated because it peeks at the test domain.
    Returns (best_step, best_in_acc).
    """
    key = f"env{test_env}_in_acc"
    best_step, best_acc = None, -1.0
    for rec in eval_records:
        acc = rec.get("val_in_acc_per_domain", {}).get(key)
        if acc is not None and acc > best_acc:
            best_acc, best_step = acc, rec.get("step")
    return best_step, (best_acc if best_acc > -1 else None)


def extract_extra_metrics(eval_records, expert_records, test_env, iid_best_step,
                          iid_val_acc, n_envs=4):
    """Extract per-domain accuracy, convergence speed, and aux loss from JSONL logs.

    Uses IID best step (selected by training-domain validation) for per-domain breakdown.
    Convergence = first step where mean(training envs out_acc) >= 90% of iid_val_acc.
    """
    extra = {
        "best_d0_acc": "", "best_d1_acc": "", "best_d2_acc": "", "best_d3_acc": "",
        "best_worst_acc": "", "best_std_acc": "",
        "convergence_step": "",
        "final_aux_loss": "",
    }

    if eval_records:
        best_rec = next(
            (r for r in eval_records if r.get("step") == iid_best_step), None
        )
        if best_rec is None:
            best_rec = eval_records[-1]

        in_per_domain = best_rec.get("val_in_acc_per_domain", {})
        extra["best_d0_acc"]    = in_per_domain.get("env0_in_acc", "")
        extra["best_d1_acc"]    = in_per_domain.get("env1_in_acc", "")
        extra["best_d2_acc"]    = in_per_domain.get("env2_in_acc", "")
        extra["best_d3_acc"]    = in_per_domain.get("env3_in_acc", "")
        extra["best_worst_acc"] = best_rec.get("val_worst_domain_acc", "")
        extra["best_std_acc"]   = best_rec.get("val_std_acc", "")

        # Convergence: first step where mean(training envs out_acc) >= 90% of iid_val_acc
        if iid_val_acc and test_env is not None:
            threshold = 0.9 * float(iid_val_acc)
            for rec in eval_records:
                train_accs = [
                    rec.get("val_acc_per_domain", {}).get(f"env{i}_out_acc")
                    for i in range(n_envs) if i != test_env
                ]
                train_accs = [a for a in train_accs if a is not None]
                if train_accs and (sum(train_accs) / len(train_accs)) >= threshold:
                    extra["convergence_step"] = rec.get("step", "")
                    break

    if expert_records:
        extra["final_aux_loss"] = expert_records[-1].get("total_aux_loss", "")

    return extra


def list_available_datasets(log_dir):
    """Find all dataset subdirectories in log_dir."""
    return [d for d in sorted(os.listdir(log_dir))
            if os.path.isdir(os.path.join(log_dir, d))]


def parse_summary(path):
    with open(path) as f:
        return json.load(f)


def extract_row(summary, test_env, test_domain_in_acc, iid_best_step,
                test_domain_acc, test_domain_best_step, extra=None):
    hp  = summary.get("hparams", {})
    eff = summary.get("efficiency_metrics", {})
    stb = summary.get("stability_metrics", {})
    bst = summary.get("best_checkpoint", {}) or {}
    fin = summary.get("final_checkpoint", {}) or {}

    run_id  = summary.get("run_id", "")
    dataset = summary.get("dataset", "")
    if not dataset and run_id:
        dataset = run_id.split("_N")[0] if "_N" in run_id else ""

    row = {
        "run_id":                   run_id,
        "dataset":                  dataset,
        "test_env":                 test_env if test_env is not None else "",
        "num_experts":              hp.get("num_experts", ""),
        "gate_k":                   hp.get("gate_k", ""),
        "expert_prune_ratio":       hp.get("expert_prune_ratio", ""),
        "hidden_size_per_expert":   eff.get("hidden_size_per_expert", ""),
        "total_params":             eff.get("total_params", ""),
        "active_params":            eff.get("active_params", ""),
        "params_utilization_ratio": eff.get("params_utilization_ratio", ""),
        "peak_mem_gb":              eff.get("peak_mem_gb", ""),
        # PRIMARY metric (DomainBed IID): step selected by training-domain val, report test in_acc
        "test_domain_in_acc":       test_domain_in_acc if test_domain_in_acc is not None else "",
        "test_domain_iid_step":     iid_best_step if iid_best_step is not None else "",
        # Oracle reference: best step by test out_acc (inflated — do not use for ranking)
        "test_domain_acc":          test_domain_acc if test_domain_acc is not None else "",
        "test_domain_best_step":    test_domain_best_step if test_domain_best_step is not None else "",
        # Reference (avg over all domains incl. training) — NOT the ranking metric
        "best_val_avg_acc":         bst.get("val_avg_acc", ""),
        "best_val_step":            bst.get("step", ""),
        "final_val_avg_acc":        fin.get("val_avg_acc", ""),
        "val_acc_std_last_10pct":   stb.get("val_acc_std_last_10pct_steps", ""),
        "total_train_time_sec":     eff.get("total_train_time_sec", ""),
        "avg_sec_per_step":         eff.get("avg_sec_per_step", ""),
        "collapse_detected":        stb.get("collapse_detected", ""),
        "status":                   summary.get("status", ""),
    }
    if extra:
        row.update(extra)
    else:
        row.update({
            "best_d0_acc": "", "best_d1_acc": "", "best_d2_acc": "", "best_d3_acc": "",
            "best_worst_acc": "", "best_std_acc": "",
            "convergence_step": "",
            "final_aux_loss": "",
        })
    return row


def sort_key(r):
    """Sort by test_domain_in_acc (IID metric, primary), fall back to best_val_avg_acc."""
    for col in ("test_domain_in_acc", "best_val_avg_acc"):
        v = r.get(col, "")
        try:
            return -float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def print_table(rows):
    if not rows:
        print("No results to display.")
        return
    key_cols = ["run_id", "num_experts", "gate_k", "expert_prune_ratio",
                "test_domain_in_acc", "test_domain_acc", "test_domain_iid_step",
                "active_params", "peak_mem_gb", "status"]
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
        run_dir      = os.path.join(ds_log_dir, run_id)
        summary_path = os.path.join(run_dir, "final_summary.json")
        if not os.path.exists(summary_path):
            missing.append(run_id)
            continue
        try:
            summary        = parse_summary(summary_path)
            eval_records   = load_jsonl(os.path.join(run_dir, "eval_log.jsonl"))
            expert_records = load_jsonl(os.path.join(run_dir, "expert_stats.jsonl"))

            test_env = _parse_test_env(run_id)
            if test_env is not None and eval_records:
                # IID (primary): step by training-domain val, report test in_acc
                iid_best_step, test_domain_in_acc = compute_iid_best(
                    eval_records, test_env)
                # Oracle (reference): step by test out_acc
                test_domain_best_step, test_domain_acc = compute_test_domain_best(
                    eval_records, test_env)
                # IID val_acc for convergence threshold
                iid_val_acc = _iid_val_acc(eval_records, test_env, iid_best_step)
            else:
                iid_best_step = test_domain_in_acc = None
                test_domain_best_step = test_domain_acc = None
                iid_val_acc = None

            extra = extract_extra_metrics(
                eval_records, expert_records,
                test_env, iid_best_step, iid_val_acc)
            row = extract_row(summary, test_env,
                              test_domain_in_acc, iid_best_step,
                              test_domain_acc, test_domain_best_step,
                              extra)
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
            print(f"\n  === Ranked by test_domain_acc (OOD, correct metric) ===")
            print_table(rows)
        print()

    if len(dataset_names) > 1 and all_rows:
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

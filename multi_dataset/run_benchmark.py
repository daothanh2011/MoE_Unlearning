"""
run_benchmark.py — Sequential, resumable orchestrator that runs N algorithms ×
M datasets × K test_envs × S seeds, calling domainbed.scripts.train per cell.

Each cell writes its SweepLogger output to:
    multi_dataset/{algo_name}/{dataset}/env{E}/seed{N}/log/

Usage:
    python multi_dataset/run_benchmark.py --dry-run
    python multi_dataset/run_benchmark.py --algos ERM_ViTS16 --datasets PACS --seeds 0
    python multi_dataset/run_benchmark.py --resume
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Allow `from sweep.run_sweep import ...` when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sweep.run_sweep import (  # noqa: E402  reuse helpers
    load_dataset_config,
    load_status,
    save_status,
    update_status,
)


def load_benchmark_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def filter_list(items, allowed):
    if not allowed:
        return items
    allow_set = set(allowed)
    return [it for it in items if (it["name"] if isinstance(it, dict) else it) in allow_set]


def build_run_id(algo_name, dataset, test_env, seed):
    return f"{algo_name}__{dataset}__env{test_env}__seed{seed}"


def build_command(algo_entry, dataset_cfg, test_env, seed, cfg):
    algo_name = algo_entry["name"]
    algorithm = algo_entry["algorithm"]
    model = algo_entry["model"]
    extra_hparams = dict(algo_entry.get("hparams") or {})

    dataset_name = dataset_cfg["dataset"]
    output_root = cfg["output_root"]

    run_id = build_run_id(algo_name, dataset_name, test_env, seed)
    cell_dir = os.path.join(output_root, algo_name, dataset_name, f"env{test_env}", f"seed{seed}")
    log_dir = os.path.join(cell_dir, "log")
    output_dir = os.path.join(cell_dir, "train_output")

    hparams = {"model": model}
    hparams.update(extra_hparams)

    cmd = [
        sys.executable, "-m", "domainbed.scripts.train",
        "--algorithm",     algorithm,
        "--dataset",       dataset_name,
        "--data_dir",      os.path.abspath(dataset_cfg["data_dir"]),
        "--output_dir",    output_dir,
        "--sweep_log_dir", log_dir,
        "--sweep_run_id",  run_id,
        "--test_envs",     str(test_env),
        "--seed",          str(seed),
        "--hparams",       json.dumps(hparams),
        "--steps",         str(cfg["steps"]),
        "--checkpoint_freq", str(cfg["checkpoint_freq"]),
    ]
    return run_id, cmd, cell_dir, log_dir


def run_one(run_id, cmd, cell_dir, log_dir, status_path, dry_run):
    if dry_run:
        print(" ".join(cmd))
        return run_id, True

    os.makedirs(cell_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = os.path.join(cell_dir, "stdout.log")
    update_status(status_path, run_id, "running")
    print(f"[{run_id}] starting → {stdout_log}")

    try:
        with open(stdout_log, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
            proc.wait()
        success = proc.returncode == 0
    except Exception as e:
        print(f"[{run_id}] ERROR: {e}")
        update_status(status_path, run_id, "failed")
        return run_id, False

    state = "completed" if success else "failed"
    update_status(status_path, run_id, state)
    print(f"[{run_id}] {state} (exit {proc.returncode})")
    return run_id, success


def build_all_runs(cfg, algos_filter, datasets_filter, seeds_filter, envs_filter):
    algorithms = filter_list(cfg["algorithms"], algos_filter)
    datasets   = filter_list(cfg["datasets"],   datasets_filter)
    seeds      = [s for s in cfg["seeds"] if (not seeds_filter or s in seeds_filter)]

    runs = []
    for algo_entry in algorithms:
        for ds_name in datasets:
            try:
                ds_cfg = load_dataset_config(ds_name)
            except FileNotFoundError:
                print(f"[WARN] dataset config not found: {ds_name} — skipping")
                continue
            test_envs = ds_cfg["test_envs"]
            if envs_filter:
                test_envs = [e for e in test_envs if e in envs_filter]
            for test_env in test_envs:
                for seed in seeds:
                    run_id, cmd, cell_dir, log_dir = build_command(
                        algo_entry, ds_cfg, test_env, seed, cfg)
                    runs.append((run_id, cmd, cell_dir, log_dir))
    return runs


def main():
    p = argparse.ArgumentParser(description="Multi-algorithm × multi-dataset DG benchmark orchestrator")
    p.add_argument("--config",   default="multi_dataset/benchmark_config.yaml")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--resume",   action="store_true",
                   help="Skip runs already marked completed in status file")
    p.add_argument("--algos",    nargs="+", default=None, help="Subset of algorithm names")
    p.add_argument("--datasets", nargs="+", default=None, help="Subset of datasets")
    p.add_argument("--seeds",    nargs="+", type=int, default=None, help="Subset of seeds")
    p.add_argument("--envs",     nargs="+", type=int, default=None, help="Subset of test_envs")
    args = p.parse_args()

    cfg = load_benchmark_config(args.config)
    status_path = cfg["status_file"]

    runs = build_all_runs(cfg, args.algos, args.datasets, args.seeds, args.envs)
    if not runs:
        print("No runs to execute.")
        return

    if args.resume:
        status = load_status(status_path)
        before = len(runs)
        runs = [r for r in runs if status.get(r[0]) != "completed"]
        print(f"--resume: skipping {before - len(runs)} completed runs; {len(runs)} remain")

    if args.dry_run:
        print(f"\n=== DRY RUN — {len(runs)} runs ===")
        for run_id, cmd, _, _ in runs:
            print(f"[{run_id}]")
            print("  " + " ".join(cmd))
        return

    # Mark all as pending
    status = load_status(status_path)
    for run_id, _, _, _ in runs:
        if status.get(run_id) != "completed":
            status[run_id] = "pending"
    save_status(status_path, status)

    print(f"\nLaunching {len(runs)} runs sequentially")
    for run_id, cmd, cell_dir, log_dir in runs:
        run_one(run_id, cmd, cell_dir, log_dir, status_path, dry_run=False)

    print("\n=== Benchmark complete. Final status ===")
    final = load_status(status_path)
    counts = {"completed": 0, "failed": 0, "pending": 0, "running": 0}
    for state in final.values():
        counts[state] = counts.get(state, 0) + 1
    print(counts)


if __name__ == "__main__":
    main()

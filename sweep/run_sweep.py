"""
run_sweep.py — Launch the GMoE expert sweep defined in sweep_config.yaml.

Usage:
    # Run sweep for a specific dataset (all test environments):
    python sweep/run_sweep.py --dataset PACS
    python sweep/run_sweep.py --dataset OfficeHome --dry-run
    python sweep/run_sweep.py --dataset PACS --parallel 2
    python sweep/run_sweep.py --dataset PACS --filter "num_experts=6"
    python sweep/run_sweep.py --dataset PACS --resume   # skip already-completed runs

    # Run sweep for all datasets:
    python sweep/run_sweep.py --dataset all
"""

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path="sweep/sweep_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset_config(dataset_name, configs_dir="sweep/configs"):
    """Load per-dataset config. Accepts a dataset name (e.g. 'PACS') or a path."""
    if os.path.isfile(dataset_name):
        path = dataset_name
    else:
        path = os.path.join(configs_dir, f"{dataset_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def list_available_datasets(configs_dir="sweep/configs"):
    paths = glob.glob(os.path.join(configs_dir, "*.yaml"))
    return [os.path.splitext(os.path.basename(p))[0] for p in sorted(paths)]


_MODEL_TAG = {
    "deit_tiny_patch16_224":  "Ti",
    "deit_small_patch16_224": "S",
    "deit_base_patch16_224":  "B",
}


def make_run_id(dataset, num_experts, top_k, expert_prune_ratio, mlp_ratio, expert_depth, test_env, model="deit_small_patch16_224"):
    pr_int   = int(round(expert_prune_ratio * 10))
    mlp_str  = str(mlp_ratio).replace('.', 'p')   # e.g. 5.2 → "5p2"
    model_tag = _MODEL_TAG.get(model, model)       # fallback to full name if unknown
    return f"{dataset}_{model_tag}_N{num_experts}_K{top_k}_PR{pr_int:02d}_MLP{mlp_str}_D{expert_depth}_env{test_env}"


def check_constraint(num_experts, top_k, expert_prune_ratio, mlp_ratio, expert_depth, constraints):
    for expr in constraints:
        local = {"top_k": top_k, "num_experts": num_experts,
                 "expert_prune_ratio": expert_prune_ratio,
                 "mlp_ratio": mlp_ratio, "expert_depth": expert_depth}
        try:
            if not eval(expr, {"__builtins__": {}}, local):
                return False
        except Exception:
            pass
    return True


def parse_filter(filter_str):
    """Parse 'num_experts=6,top_k=2' → {'num_experts': 6, 'top_k': 2}"""
    if not filter_str:
        return {}
    result = {}
    for part in filter_str.split(","):
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
# Command builder
# ---------------------------------------------------------------------------

def build_command(run_id, dataset_cfg, num_experts, top_k, expert_prune_ratio,
                  mlp_ratio, expert_depth, test_env, cfg):
    fixed   = cfg["fixed_params"]
    outputs = cfg["output"]

    hparams = {
        "num_experts":        num_experts,
        "gate_k":             top_k,
        "mlp_ratio":          mlp_ratio,
        "expert_prune_ratio": expert_prune_ratio,
        "expert_depth":       expert_depth,
        "model":              fixed.get("model", "deit_small_patch16_224"),
    }

    dataset_name = dataset_cfg["dataset"]
    log_dir      = os.path.join(outputs["log_dir"], dataset_name, run_id)
    output_dir   = os.path.join("train_output", "phase1", dataset_name, run_id)

    cmd = [
        sys.executable, "-m", "domainbed.scripts.train",
        "--algorithm",      fixed["algorithm"],
        "--dataset",        dataset_name,
        "--data_dir",       os.path.abspath(dataset_cfg["data_dir"]),
        "--output_dir",     output_dir,
        "--sweep_log_dir",  log_dir,
        "--sweep_run_id",   run_id,
        "--test_envs",      str(test_env),
        "--seed",           str(fixed.get("seed", 0)),
        "--hparams",        json.dumps(hparams),
    ]

    if fixed.get("steps"):
        cmd += ["--steps", str(fixed["steps"])]
    if fixed.get("checkpoint_freq"):
        cmd += ["--checkpoint_freq", str(fixed["checkpoint_freq"])]

    return cmd, log_dir, output_dir


# ---------------------------------------------------------------------------
# Status helpers (atomic write)
# ---------------------------------------------------------------------------

def load_status(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_status(path, status):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def update_status(path, run_id, state):
    status = load_status(path)
    status[run_id] = state
    save_status(path, status)


# ---------------------------------------------------------------------------
# Run one job
# ---------------------------------------------------------------------------

def run_one(run_id, cmd, log_dir, status_path, dry_run):
    if dry_run:
        print(" ".join(cmd))
        return run_id, True

    os.makedirs(log_dir, exist_ok=True)
    stdout_log = os.path.join(log_dir, "stdout.log")
    update_status(status_path, run_id, "running")
    print(f"[{run_id}] starting  → {stdout_log}")

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
    print(f"[{run_id}] {state}  (exit {proc.returncode})")
    return run_id, success


# ---------------------------------------------------------------------------
# Build run list for one dataset
# ---------------------------------------------------------------------------

def build_runs_for_dataset(dataset_cfg, cfg, filter_kv):
    grid        = cfg["param_grid"]
    constraints = cfg.get("constraints", [])

    keys   = list(grid.keys())  # num_experts, top_k, expert_prune_ratio
    combos = list(itertools.product(*[grid[k] for k in keys]))

    test_envs    = dataset_cfg["test_envs"]
    dataset_name = dataset_cfg["dataset"]
    model        = cfg["fixed_params"].get("model", "deit_small_patch16_224")

    runs = []
    for combo in combos:
        vals = dict(zip(keys, combo))
        n  = vals["num_experts"]
        k  = vals["top_k"]
        pr = vals["expert_prune_ratio"]
        mr = vals["mlp_ratio"]
        ed = vals["expert_depth"]

        if not check_constraint(n, k, pr, mr, ed, constraints):
            continue

        # Apply --filter
        skip = False
        for fk, fv in filter_kv.items():
            actual = {"num_experts": n, "top_k": k, "expert_prune_ratio": pr,
                      "mlp_ratio": mr, "expert_depth": ed}.get(fk)
            if actual is None:
                continue
            if isinstance(fv, float):
                if abs(actual - fv) > 1e-9:
                    skip = True
            else:
                if actual != fv:
                    skip = True
        if skip:
            continue

        for env in test_envs:
            run_id = make_run_id(dataset_name, n, k, pr, mr, ed, env, model)
            cmd, log_dir, output_dir = build_command(
                run_id, dataset_cfg, n, k, pr, mr, ed, env, cfg)
            runs.append((run_id, cmd, log_dir))

    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Launch the GMoE expert sweep")
    parser.add_argument("--config",    default="sweep/sweep_config.yaml")
    parser.add_argument("--dataset",   default=None, metavar="NAME_OR_all",
                        help="Dataset name (e.g. PACS) or 'all' for every config in sweep/configs/")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--parallel",  type=int, default=1, metavar="N",
                        help="Run N jobs concurrently")
    parser.add_argument("--filter",    default=None, metavar="KEY=VAL,...",
                        help="Only run configs matching all key=val pairs, e.g. 'num_experts=6'")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip runs already marked completed in sweep_status.json")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    outputs     = cfg["output"]
    status_path = outputs["sweep_status"]
    filter_kv   = parse_filter(args.filter)

    # Determine which datasets to process
    if args.dataset is None or args.dataset.lower() == "all":
        dataset_names = list_available_datasets()
        if not dataset_names:
            print("No dataset configs found in sweep/configs/. "
                  "Create sweep/configs/{Dataset}.yaml files first.")
            return
        print(f"Running sweep for all datasets: {dataset_names}")
    else:
        dataset_names = [args.dataset]

    # Collect all runs across datasets
    all_runs = []
    for ds_name in dataset_names:
        try:
            dataset_cfg = load_dataset_config(ds_name)
        except FileNotFoundError:
            print(f"[WARN] Dataset config not found for '{ds_name}' — skipping.")
            continue
        runs = build_runs_for_dataset(dataset_cfg, cfg, filter_kv)
        all_runs.extend(runs)
        print(f"  {ds_name}: {len(runs)} runs "
              f"({len(dataset_cfg['test_envs'])} envs × {len(runs)//len(dataset_cfg['test_envs'])} combos)")

    if not all_runs:
        print("No valid runs to execute.")
        return

    if args.dry_run:
        print(f"\n=== DRY RUN — {len(all_runs)} total runs ===")
        for run_id, cmd, _ in all_runs:
            print(" ".join(cmd))
        return

    # --resume: filter out completed
    if args.resume:
        status   = load_status(status_path)
        all_runs = [(rid, cmd, ld) for rid, cmd, ld in all_runs
                    if status.get(rid) != "completed"]
        print(f"Resuming: {len(all_runs)} runs remaining after skipping completed.")

    # Initialise all as pending
    status = load_status(status_path)
    for run_id, _, _ in all_runs:
        if status.get(run_id) != "completed":
            status[run_id] = "pending"
    save_status(status_path, status)

    print(f"\nLaunching {len(all_runs)} runs (parallel={args.parallel})")

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_one, rid, cmd, ld, status_path, False): rid
                for rid, cmd, ld in all_runs
            }
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[{rid}] unhandled exception: {e}")
    else:
        for run_id, cmd, log_dir in all_runs:
            run_one(run_id, cmd, log_dir, status_path, False)

    print("\n=== Sweep complete. Final status ===")
    final = load_status(status_path)
    for name, state in sorted(final.items()):
        print(f"  {name}: {state}")


if __name__ == "__main__":
    main()

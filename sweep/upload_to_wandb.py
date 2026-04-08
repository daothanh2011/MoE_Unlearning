"""
Upload sweep/logs/PACS env3 training curves to W&B.

Each run gets its own W&B run under the same project, grouped by config.
Logs train_loss, moe_aux_loss, val_avg_acc, val_worst_domain_acc,
val_best_domain_acc, per-domain out/in accuracies, and step_time.

Usage:
    python sweep/upload_to_wandb.py \
        --logs_dir sweep/logs/PACS \
        --env 3 \
        --project PACS_sweep \
        [--entity YOUR_WANDB_ENTITY]
"""

import argparse
import json
import os
from pathlib import Path

import wandb


def load_jsonl(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def upload_run(run_dir: Path, project: str, entity):
    hparams_path = run_dir / "hparams.json"
    run_meta_path = run_dir / "run_meta.json"
    train_log_path = run_dir / "train_log.jsonl"
    eval_log_path = run_dir / "eval_log.jsonl"

    if not hparams_path.exists():
        print(f"  [skip] no hparams.json in {run_dir.name}")
        return

    with open(hparams_path) as f:
        hparams = json.load(f)

    run_meta = {}
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)

    run_id = hparams.get("run_id", run_dir.name)

    config = {**hparams}
    if "git" in run_meta:
        config["git_commit"] = run_meta["git"].get("commit_hash", "")
    if "environment" in run_meta:
        config["hostname"] = run_meta["environment"].get("hostname", "")
        config["gpu_model"] = run_meta["environment"].get("gpu_model", "")

    # Build step-indexed dicts from both logs
    train_by_step = {r["step"]: r for r in load_jsonl(train_log_path)} if train_log_path.exists() else {}
    eval_by_step  = {r["step"]: r for r in load_jsonl(eval_log_path)}  if eval_log_path.exists() else {}

    all_steps = sorted(set(train_by_step) | set(eval_by_step))

    print(f"  Uploading {run_id}  ({len(all_steps)} steps)...")

    wandb_run = wandb.init(
        project=project,
        entity=entity,
        name=run_id,
        id=run_id.replace("/", "_"),  # wandb run id must be URL-safe
        config=config,
        reinit=True,
        resume="allow",
    )

    for step in all_steps:
        log_dict = {"step": step}

        tr = train_by_step.get(step, {})
        if "train_loss" in tr:
            log_dict["train/loss"] = tr["train_loss"]
        if "moe_aux_loss" in tr:
            log_dict["train/moe_aux_loss"] = tr["moe_aux_loss"]
        if "step_time" in tr:
            log_dict["train/step_time"] = tr["step_time"]
        if "mem_gb" in tr:
            log_dict["train/mem_gb"] = tr["mem_gb"]

        ev = eval_by_step.get(step, {})
        if "val_avg_acc" in ev:
            log_dict["eval/val_avg_acc"] = ev["val_avg_acc"]
        if "val_worst_domain_acc" in ev:
            log_dict["eval/val_worst_domain_acc"] = ev["val_worst_domain_acc"]
        if "val_best_domain_acc" in ev:
            log_dict["eval/val_best_domain_acc"] = ev["val_best_domain_acc"]
        if "val_std_acc" in ev:
            log_dict["eval/val_std_acc"] = ev["val_std_acc"]
        # per-domain out accuracies
        for k, v in ev.get("val_acc_per_domain", {}).items():
            log_dict[f"eval/out/{k}"] = v
        # per-domain in accuracies
        for k, v in ev.get("val_in_acc_per_domain", {}).items():
            log_dict[f"eval/in/{k}"] = v

        wandb.log(log_dict, step=step)

    # Log best-val summary if available
    final_summary_path = run_dir / "final_summary.json"
    if final_summary_path.exists():
        with open(final_summary_path) as f:
            summary = json.load(f)
        for k, v in summary.items():
            wandb.run.summary[k] = v

    wandb_run.finish()
    print(f"  Done: {run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="sweep/logs/PACS",
                        help="Root directory containing per-run log folders")
    parser.add_argument("--env", type=int, default=3,
                        help="Which test environment index to upload (default: 3)")
    parser.add_argument("--project", default="PACS_sweep",
                        help="W&B project name")
    parser.add_argument("--entity", default=None,
                        help="W&B entity (team/user). Leave blank to use default.")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    suffix = f"_env{args.env}"

    run_dirs = sorted(p for p in logs_dir.iterdir() if p.is_dir() and p.name.endswith(suffix))

    if not run_dirs:
        print(f"No runs found ending with '{suffix}' in {logs_dir}")
        return

    print(f"Found {len(run_dirs)} env{args.env} runs in {logs_dir}")

    for run_dir in run_dirs:
        upload_run(run_dir, project=args.project, entity=args.entity)

    print(f"\nAll done. View at: https://wandb.ai/{args.entity or '<your-entity>'}/{args.project}")


if __name__ == "__main__":
    main()

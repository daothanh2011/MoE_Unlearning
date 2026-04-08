"""
sweep_logger.py — Per-run structured logging for the GMoE expert sweep.

Written to {log_dir}/ :
  hparams.json        — hyperparams + compute-cost fields (written at run start)
  run_meta.json       — environment / timing metadata (written at start; finalised at end)
  train_log.jsonl     — per-step training metrics (appended every log_freq steps)
  eval_log.jsonl      — per-checkpoint eval metrics (appended every checkpoint_freq steps)
  expert_stats.jsonl  — MoE-specific diagnostics per checkpoint
  final_summary.json  — aggregated results (written at run end)
"""

import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Atomic JSON write helpers
# ---------------------------------------------------------------------------

def _atomic_write(path, obj):
    """Write JSON atomically: write to .tmp then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_env_info():
    import sys
    info = {
        "python_version": sys.version.split()[0],
        "hostname": platform.node(),
    }
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["gpu_model"] = torch.cuda.get_device_name(0)
            info["num_gpus"] = torch.cuda.device_count()
        else:
            info["cuda_version"] = None
            info["gpu_model"] = None
            info["num_gpus"] = 0
    except ImportError:
        pass
    return info


def _get_git_info():
    git = {}
    try:
        git["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git["dirty"] = len(dirty_out) > 0
    except Exception:
        git = {"commit_hash": None, "branch": None, "dirty": None}
    return git


# ---------------------------------------------------------------------------
# Compute-cost helpers
# ---------------------------------------------------------------------------

_BACKBONE_LABEL = {192: "DeiT-Ti/16", 384: "DeiT-S/16", 768: "DeiT-B/16"}


def _compute_params(model, hparams):
    """
    Returns (total_params, active_params, hidden_size_per_expert, embed_dim).

    Architecture: moe_layers = ['F']*8 + ['S','F']*2  →  2 MoE ('S') blocks.
    Inactive expert params = num_moe_layers * (num_experts - gate_k) * params_per_expert
    active_params = total_params - inactive_expert_params

    params_per_expert counts weights AND biases for all layers:
      fc_first  : D*H + H
      fc_middle : (depth-2) * (H*H + H)
      fc_last   : H*D + D
    Verified against actual model param counts (see verify_params.py).
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Read embed_dim from the model to support DeiT-Ti (192), S (384), B (768)
    embed_dim = getattr(model, 'embed_dim', None)
    if embed_dim is None:
        pe = getattr(model, 'patch_embed', None)
        embed_dim = pe.proj.weight.shape[0] if pe is not None else 384
    num_moe_layers      = 2            # blocks 8 and 10 in the moe_layers config
    num_experts         = hparams.get("num_experts",       6)
    gate_k              = hparams.get("gate_k",            1)
    mlp_ratio           = hparams.get("mlp_ratio",         4.0)
    expert_prune_ratio  = hparams.get("expert_prune_ratio", 0.0)
    expert_depth        = hparams.get("expert_depth",      2)

    hidden_size = max(1, int(embed_dim * mlp_ratio * (1 - expert_prune_ratio)))
    # Params per expert (weights + biases):
    #   fc_first  : embed_dim × hidden_size  +  hidden_size (bias)
    #   fc_middle : hidden_size × hidden_size  +  hidden_size (bias)  [×(depth-2)]
    #   fc_last   : hidden_size × embed_dim  +  embed_dim (bias)
    params_per_expert = (embed_dim * hidden_size + hidden_size
                         + (expert_depth - 2) * (hidden_size * hidden_size + hidden_size)
                         + hidden_size * embed_dim + embed_dim)
    inactive = num_moe_layers * (num_experts - gate_k) * params_per_expert
    active   = total_params - inactive

    return total_params, active, hidden_size, embed_dim


# ---------------------------------------------------------------------------
# SweepLogger
# ---------------------------------------------------------------------------

class SweepLogger:
    """Writes structured log files for one sweep run."""

    def __init__(self, log_dir, run_id, hparams, args, model=None):
        """
        Parameters
        ----------
        log_dir : str
            Directory to write all log files for this run.
        run_id : str
            Human-readable run identifier (e.g. 'gmoe_N6_K2_PR00').
        hparams : dict
            Full hyperparameter dict from the training script.
        args : argparse.Namespace
            Parsed CLI arguments from the training script.
        model : nn.Module or None
            Instantiated model (used to compute total/active params).
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir  = log_dir
        self.run_id   = run_id
        self.hparams  = hparams
        self.args     = args
        self._algorithm_name = getattr(args, 'algorithm', None)
        self.start_ts = datetime.now(timezone.utc)
        self.start_t  = time.time()

        # paths
        self.hparams_path      = os.path.join(log_dir, "hparams.json")
        self.run_meta_path     = os.path.join(log_dir, "run_meta.json")
        self.train_log_path    = os.path.join(log_dir, "train_log.jsonl")
        self.eval_log_path     = os.path.join(log_dir, "eval_log.jsonl")
        self.expert_stats_path = os.path.join(log_dir, "expert_stats.jsonl")
        self.final_path        = os.path.join(log_dir, "final_summary.json")

        # compute-cost fields
        if model is not None:
            self.total_params, self.active_params, self.hidden_size, self.embed_dim = \
                _compute_params(model, hparams)
        else:
            self.total_params = self.active_params = self.hidden_size = self.embed_dim = None

        # best-val tracker
        self._best_val_acc   = -1.0
        self._best_val_step  = None
        self._peak_mem_gb    = 0.0
        self._train_records  = []   # (step, elapsed, vals) for final_summary

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def write_hparams_json(self):
        """Write hparams.json at run start."""
        doc = dict(self.hparams)
        doc["run_id"]                 = self.run_id
        if hasattr(self, '_algorithm_name') and self._algorithm_name:
            doc["algorithm"]          = self._algorithm_name
        embed_dim = self.embed_dim if self.embed_dim is not None else 384
        doc["backbone"]               = _BACKBONE_LABEL.get(embed_dim, f"DeiT-{embed_dim}d/16")
        doc["model_dim"]              = embed_dim
        doc["hidden_size_per_expert"] = self.hidden_size
        if self.total_params is not None:
            doc["total_params"]            = self.total_params
            doc["active_params"]           = self.active_params
            doc["params_utilization_ratio"] = (
                round(self.active_params / self.total_params, 4)
                if self.total_params else None
            )
        _atomic_write(self.hparams_path, doc)

    def start_run_meta(self):
        """Write initial run_meta.json (status=running)."""
        doc = {
            "run_id":      self.run_id,
            "status":      "running",
            "start_time":  self.start_ts.isoformat(),
            "end_time":    None,
            "duration_sec": None,
            "exit_code":   None,
            "error_message": None,
            "environment": _get_env_info(),
            "git":         _get_git_info(),
        }
        _atomic_write(self.run_meta_path, doc)

    # ------------------------------------------------------------------
    # Per-step
    # ------------------------------------------------------------------

    def append_train_log(self, step, step_vals, mem_gb=None):
        """Append one record to train_log.jsonl."""
        elapsed = round(time.time() - self.start_t, 1)
        record = {
            "step":         step,
            "elapsed_sec":  elapsed,
            "train_loss":   round(step_vals.get("loss",     0.0), 6),
            "moe_aux_loss": round(step_vals.get("loss_aux", 0.0), 6),
            "step_time":    round(step_vals.get("step_time", 0.0), 4),
        }
        if mem_gb is not None:
            record["mem_gb"] = round(mem_gb, 3)
        self._train_records.append(record)
        _append_jsonl(self.train_log_path, record)

    # ------------------------------------------------------------------
    # Per-checkpoint
    # ------------------------------------------------------------------

    def append_eval_log(self, step, results):
        """
        Append one record to eval_log.jsonl.
        `results` is the dict already built by train.py (has env*_out_acc keys).
        """
        elapsed = round(time.time() - self.start_t, 1)

        out_acc = {
            k: round(v, 4)
            for k, v in results.items()
            if re.match(r"env\d+_out_acc", k)
        }
        in_acc = {
            k: round(v, 4)
            for k, v in results.items()
            if re.match(r"env\d+_in_acc", k)
        }

        vals = list(out_acc.values())
        avg  = round(sum(vals) / len(vals), 4) if vals else None
        worst = round(min(vals), 4) if vals else None
        best  = round(max(vals), 4) if vals else None
        std   = round(
            (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5, 4
        ) if vals and len(vals) > 1 else 0.0

        is_best = avg is not None and avg > self._best_val_acc
        if is_best:
            self._best_val_acc  = avg
            self._best_val_step = step

        mem_gb = results.get("mem_gb", 0.0)
        self._peak_mem_gb = max(self._peak_mem_gb, mem_gb or 0.0)

        record = {
            "step":                  step,
            "elapsed_sec":           elapsed,
            "val_acc_per_domain":    out_acc,
            "val_in_acc_per_domain": in_acc,
            "val_avg_acc":           avg,
            "val_worst_domain_acc":  worst,
            "val_best_domain_acc":   best,
            "val_std_acc":           std,
            "mem_gb":                round(mem_gb, 3),
            "is_best_val":           is_best,
            "best_val_acc_so_far":   self._best_val_acc,
            "best_step_so_far":      self._best_val_step,
        }
        _append_jsonl(self.eval_log_path, record)

    def append_expert_stats(self, step, algorithm):
        """
        Append MoE diagnostics to expert_stats.jsonl.
        Collects aux_loss per MoE block (only l_aux is accessible via Tutel).
        Non-MoE algorithms (no `model.blocks`) get an empty record so the
        sweep logger stays compatible across algorithms.
        """
        layer_aux = []
        moe_model = getattr(algorithm, "model", None)
        blocks = getattr(moe_model, "blocks", None) if moe_model is not None else None
        if blocks is not None:
            for block in blocks:
                al = getattr(block, "aux_loss", None)
                if al is not None:
                    try:
                        layer_aux.append(round(float(al), 6))
                    except Exception:
                        pass

        record = {
            "step":                  step,
            "moe_layers_aux_loss":   layer_aux,
            "total_aux_loss":        round(sum(layer_aux), 6) if layer_aux else 0.0,
            "num_experts":           self.hparams.get("num_experts",       6),
            "gate_k":                self.hparams.get("gate_k",            1),
            "mlp_ratio":             self.hparams.get("mlp_ratio",         4.0),
            "expert_depth":          self.hparams.get("expert_depth",      2),
            "expert_prune_ratio":    self.hparams.get("expert_prune_ratio", 0.0),
        }
        _append_jsonl(self.expert_stats_path, record)

    # ------------------------------------------------------------------
    # Run end
    # ------------------------------------------------------------------

    def write_final_summary(self):
        """Write final_summary.json at run end."""
        elapsed = round(time.time() - self.start_t, 1)

        # stability: std of val_avg_acc over last 10% of logged eval records
        eval_accs = []
        if os.path.exists(self.eval_log_path):
            with open(self.eval_log_path) as f:
                eval_records = [json.loads(l) for l in f if l.strip()]
            n_tail = max(1, len(eval_records) // 10)
            tail = eval_records[-n_tail:]
            eval_accs = [r["val_avg_acc"] for r in tail if r.get("val_avg_acc") is not None]

        val_std_tail = 0.0
        val_mean_tail = None
        if eval_accs:
            val_mean_tail = round(sum(eval_accs) / len(eval_accs), 4)
            val_std_tail  = round(
                (sum((v - val_mean_tail) ** 2 for v in eval_accs) / len(eval_accs)) ** 0.5, 4
            )

        # last eval record for "final" metrics
        final_val_avg = None
        if eval_accs:
            final_val_avg = eval_accs[-1]

        total_params = self.total_params
        active_params = self.active_params
        util_ratio = round(active_params / total_params, 4) if (total_params and active_params) else None

        doc = {
            "run_id": self.run_id,
            "status": "completed",
            "best_checkpoint": {
                "step":         self._best_val_step,
                "val_avg_acc":  self._best_val_acc if self._best_val_acc > -1 else None,
            },
            "final_checkpoint": {
                "step":         self._train_records[-1]["step"] if self._train_records else None,
                "val_avg_acc":  final_val_avg,
            },
            "peak_metrics": {
                "peak_val_avg_acc":  self._best_val_acc if self._best_val_acc > -1 else None,
                "peak_val_step":     self._best_val_step,
            },
            "stability_metrics": {
                "val_acc_std_last_10pct_steps":  val_std_tail,
                "val_acc_mean_last_10pct_steps": val_mean_tail,
                "training_converged":            val_std_tail < 0.01 if eval_accs else None,
                "collapse_detected":             False,
                "collapse_step":                 None,
            },
            "efficiency_metrics": {
                "total_train_time_sec":    elapsed,
                "avg_sec_per_step":        round(elapsed / max(1, len(self._train_records)), 4),
                "total_params":            total_params,
                "active_params":           active_params,
                "hidden_size_per_expert":  self.hidden_size,
                "params_utilization_ratio": util_ratio,
                "peak_mem_gb":             round(self._peak_mem_gb, 3),
            },
            "hparams": dict(self.hparams),
        }
        _atomic_write(self.final_path, doc)

    def close_run_meta(self, exit_code=0, error_message=None):
        """Finalise run_meta.json with end time and status."""
        end_ts  = datetime.now(timezone.utc)
        elapsed = round(time.time() - self.start_t, 1)
        status  = "completed" if exit_code == 0 else "failed"

        doc = {
            "run_id":        self.run_id,
            "status":        status,
            "start_time":    self.start_ts.isoformat(),
            "end_time":      end_ts.isoformat(),
            "duration_sec":  elapsed,
            "exit_code":     exit_code,
            "error_message": error_message,
            "environment":   _get_env_info(),
            "git":           _get_git_info(),
        }
        _atomic_write(self.run_meta_path, doc)

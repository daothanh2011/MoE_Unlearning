"""
Compute FLOPs for each sweep run and write back into hparams.json.

Formula:
    FLOPs = 2 * L * active_params          # all linear layers (QKV, proj, MLP/experts)
          + 4 * L^2 * d * num_layers       # attention softmax (QK^T and Attn*V)

active_params already accounts for:
  - top-K routing (only K/N experts active)
  - expert_prune_ratio (pruned weights removed)

So FLOPs naturally reflects the actual per-forward-pass computation.
"""

import json
from pathlib import Path

# Architecture constants per backbone
BACKBONE_CFG = {
    "DeiT-S/16": {"num_layers": 12, "image_size": 224, "patch_size": 16},
    "DeiT-B/16": {"num_layers": 12, "image_size": 224, "patch_size": 16},
    "ViT-S/16":  {"num_layers": 12, "image_size": 224, "patch_size": 16},
}


def compute_flops(hparams: dict) -> int:
    backbone = hparams.get("backbone", "DeiT-S/16")
    cfg = BACKBONE_CFG.get(backbone)
    if cfg is None:
        raise ValueError(f"Unknown backbone: {backbone}")

    d = hparams["model_dim"]
    active_params = hparams["active_params"]
    num_layers = cfg["num_layers"]
    L = (cfg["image_size"] // cfg["patch_size"]) ** 2 + 1  # patches + cls token

    linear_flops    = 2 * L * active_params
    attention_flops = 4 * (L ** 2) * d * num_layers

    return linear_flops + attention_flops


def main():
    logs_dir = Path("sweep/logs")
    files = sorted(logs_dir.glob("**/hparams.json"))
    print(f"Found {len(files)} hparams.json files\n")

    for path in files:
        with open(path) as f:
            hparams = json.load(f)

        flops = compute_flops(hparams)
        hparams["FLOPs"] = flops

        with open(path, "w") as f:
            json.dump(hparams, f, indent=2)

        run_id = hparams.get("run_id", path.parent.name)
        print(f"{run_id:50s}  {flops/1e9:.3f} GFLOPs")

    print(f"\nDone — updated {len(files)} files.")


if __name__ == "__main__":
    main()

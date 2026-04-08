"""
Phase 2 finetuning pipeline for GMOE on Terra Incognita.

Pipeline:
  Step 0: Load Phase 1 checkpoint and reconstruct model.
  Step 1: Extract penultimate-layer embeddings from training data.
  Step 2: Cluster embeddings per class (K-Means, optional PCA).
  Step 3: Sample D-hat from clusters for balanced diversity.
  Step 4: Finetune only MOE expert + router layers on D-hat.
  Step 5: Save Phase 2 checkpoint.

Usage:
  python -m domainbed.scripts.phase2_finetune --config configs/phase2_config.yaml
  python -m domainbed.scripts.phase2_finetune --config configs/phase2_config.yaml --env 1 --k 10 --x 30
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Allow running as `python -m domainbed.scripts.phase2_finetune`
# mirrors sys.path setup in domainbed/scripts/train.py
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, _REPO_ROOT)
# algorithms.py does bare `import vision_transformer`, so domainbed/ must be on path
sys.path.insert(0, os.path.join(_REPO_ROOT, 'domainbed'))

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml is required: pip install pyyaml")

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn")

# Patch Tutel CUDA kernels with pure-PyTorch fallbacks BEFORE importing
# vision_transformer / algorithms (which import tutel at module level).
# Required when Tutel's compiled extensions don't support the current GPU
# architecture (e.g., compute_120 on RTX 5070 Ti).
import domainbed.tutel_patch  # noqa: F401 — side-effect import

from domainbed import algorithms, datasets
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Transforms (must match domainbed/datasets.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

def get_terra_envs(data_dir):
    """Scan terra_incognita directory and return sorted env subdirectory names.
    Mirrors MultipleEnvironmentImageFolder which uses os.scandir + sorted()."""
    terra_dir = os.path.join(data_dir, "terra_incognita")
    envs = sorted([f.name for f in os.scandir(terra_dir) if f.is_dir()])
    return envs, terra_dir


# Module-level storage for CMNIST tensor data (set in main() when dataset=ColoredMNIST).
# Maps integer indices (used as "paths" throughout the pipeline) to (image_tensor, label).
_CMNIST_TENSORS = None  # list of (image_tensor, label)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path, cli_overrides):
    """Load YAML config and apply CLI overrides. Returns a flat-ish nested dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides (non-None values only)
    if cli_overrides.get('env') is not None:
        cfg['experiment']['env'] = cli_overrides['env']
    if cli_overrides.get('k') is not None:
        cfg['clustering']['k'] = cli_overrides['k']
    if cli_overrides.get('x') is not None:
        cfg['sampling']['x'] = cli_overrides['x']
    if cli_overrides.get('lr') is not None:
        cfg['finetuning']['lr'] = cli_overrides['lr']
    if cli_overrides.get('epochs') is not None:
        cfg['finetuning']['epochs'] = cli_overrides['epochs']
    if cli_overrides.get('skip_extract'):
        cfg['skip']['extract'] = True
    if cli_overrides.get('skip_cluster'):
        cfg['skip']['cluster'] = True

    return cfg


def resolve_paths(cfg):
    """Resolve all path templates from the config and add them as cfg['resolved']."""
    env = cfg['experiment']['env']
    k = cfg['clustering']['k']
    x = cfg['sampling']['x']
    lr = cfg['finetuning']['lr']

    env_prefix = cfg['phase1'].get('env_prefix', 'terra')
    phase1_dir = os.path.join(cfg['phase1']['ckpt_base'], f"{env_prefix}_env{env}")
    phase1_ckpt = os.path.join(phase1_dir, cfg['phase1']['ckpt_filename'])

    embed_dir = cfg['embedding']['save_dir']
    if embed_dir is None:
        embed_dir = os.path.join(phase1_dir, "embeddings")

    output_dir = os.path.join(cfg['output']['base_dir'], f"{env_prefix}_addon_D_env{env}")
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    ckpt_name = f"checkpoint_k{k}_x{x}_lr{lr_str}.pth"

    cfg['resolved'] = {
        'phase1_dir': phase1_dir,
        'phase1_ckpt': phase1_ckpt,
        'embed_dir': embed_dir,
        'output_dir': output_dir,
        'ckpt_name': ckpt_name,
        'dhat_json': os.path.join(output_dir, f"dhat_k{k}_x{x}.json"),
        'embed_npy': os.path.join(embed_dir, "embeddings.npy"),
        'labels_npy': os.path.join(embed_dir, "labels.npy"),
        'paths_json': os.path.join(embed_dir, "paths.json"),
        'clusters_npy': os.path.join(embed_dir, f"clusters_k{k}.npy"),
    }
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Load Phase 1 model
# ─────────────────────────────────────────────────────────────────────────────

def load_phase1_model(ckpt_path):
    """
    Load Phase 1 model.pkl and reconstruct the GMOE algorithm.

    Returns:
        algorithm: GMOE instance with loaded weights, on CUDA
        ckpt_args: dict of training args saved in the checkpoint
        ckpt: full checkpoint dict
    """
    print(f"\n[Step 0] Loading Phase 1 checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_args = ckpt['args']
    hparams = ckpt['model_hparams']
    num_classes = ckpt['model_num_classes']
    input_shape = ckpt['model_input_shape']
    num_domains = ckpt['model_num_domains']

    print(f"  dataset      : {ckpt_args.get('dataset', 'TerraIncognita')}")
    print(f"  test_envs    : {ckpt_args.get('test_envs')}")
    print(f"  num_classes  : {num_classes}")
    print(f"  num_domains  : {num_domains}")
    print(f"  hparams      : lr={hparams.get('lr')}, wd={hparams.get('weight_decay')}")

    algorithm = algorithms.GMOE(input_shape, num_classes, num_domains, hparams)
    incompatible = algorithm.load_state_dict(ckpt['model_dict'], strict=False)
    if incompatible.missing_keys:
        print(f"  WARNING missing keys : {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"  WARNING unexpected   : {incompatible.unexpected_keys}")
    algorithm = algorithm.cuda()
    algorithm.eval()

    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"  total params : {total_params / 1e6:.2f}M")

    # Sanity check: forward pass shapes
    with torch.no_grad():
        dummy = torch.randn(2, *input_shape).cuda()
        feat = algorithm.predict(dummy, forward_feature=True)
        logit = algorithm.predict(dummy)
    assert feat.shape == (2, 384), f"Expected (2,384), got {feat.shape}"
    assert logit.shape == (2, num_classes), f"Expected (2,{num_classes}), got {logit.shape}"
    print(f"  forward check: features {feat.shape}, logits {logit.shape} ✓")

    return algorithm, ckpt_args, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Train split reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def build_train_split(ckpt_args, data_dir, env_idx, dataset_name='TerraIncognita'):
    """
    Reproduce the Phase 1 training split exactly using the same trial_seed
    and holdout_fraction stored in the checkpoint args.

    For TerraIncognita: returns list of (filepath, class_idx).
    For ColoredMNIST:   returns list of (int_idx, class_idx) where int_idx
                        is an index into the global _CMNIST_TENSORS list.

    Returns:
        all_samples: list of (path_or_idx, class_idx)
        env_boundaries: list of (start, end) indices per env in all_samples
    """
    global _CMNIST_TENSORS

    trial_seed = ckpt_args.get('trial_seed', 0)
    holdout_fraction = ckpt_args.get('holdout_fraction', 0.2)
    test_envs = ckpt_args.get('test_envs', [env_idx])

    # ── ColoredMNIST branch ──────────────────────────────────────────────────
    if dataset_name == 'ColoredMNIST':
        hparams = {'data_augmentation': False}
        full_dataset = datasets.ColoredMNIST(data_dir, list(test_envs), hparams)
        env_names = datasets.ColoredMNIST.ENVIRONMENTS

        all_samples = []   # (int_idx, class_idx)
        all_tensors = []   # (image_tensor, label) — built in order
        env_boundaries = []

        for i, env_data in enumerate(full_dataset):
            if i in test_envs:
                continue
            n_out = int(len(env_data) * holdout_fraction)
            _, in_split = misc.split_dataset(
                env_data, n_out, seed=misc.seed_hash(trial_seed, i)
            )
            start = len(all_samples)
            for j in range(len(in_split)):
                x, y = in_split[j]
                idx = len(all_tensors)
                all_tensors.append((x, int(y)))
                all_samples.append((idx, int(y)))
            end = len(all_samples)
            env_boundaries.append((start, end))
            print(f"  env {i} ({env_names[i]}): {end - start} training samples")

        _CMNIST_TENSORS = all_tensors  # make available to dataset wrappers
        print(f"  total training samples: {len(all_samples)}")
        return all_samples, env_boundaries

    # ── TerraIncognita branch (default) ─────────────────────────────────────
    terra_envs, terra_dir = get_terra_envs(data_dir)

    all_samples = []   # (filepath, class_idx)
    env_boundaries = []

    for i, env_name in enumerate(terra_envs):
        if i in test_envs:
            continue  # skip test env
        env_path = os.path.join(terra_dir, env_name)
        if not os.path.isdir(env_path):
            raise FileNotFoundError(f"Terra env directory not found: {env_path}")

        # Load with base transform (transform does not affect split)
        img_folder = ImageFolder(env_path, transform=BASE_TRANSFORM)

        # Reproduce the same split as Phase 1.
        # train.py: out, in_ = split_dataset(env, int(N * holdout_fraction), seed)
        # → first return (size n_out) is out-split; second is in-split.
        n_out = int(len(img_folder) * holdout_fraction)
        _, in_split = misc.split_dataset(
            img_folder, n_out,
            seed=misc.seed_hash(trial_seed, i)
        )

        start = len(all_samples)
        for j in range(len(in_split)):
            orig_idx = in_split.keys[j]
            path, class_idx = img_folder.samples[orig_idx]
            all_samples.append((path, class_idx))
        end = len(all_samples)
        env_boundaries.append((start, end))

        print(f"  env {i} ({env_name}): {end - start} training samples")

    print(f"  total training samples: {len(all_samples)}")
    return all_samples, env_boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract embeddings
# ─────────────────────────────────────────────────────────────────────────────

class _PathDataset(torch.utils.data.Dataset):
    """Lightweight dataset that loads images from (path, label) pairs.
    path can be a file path (str) or an integer index into _CMNIST_TENSORS."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if isinstance(path, int):
            # CMNIST: return pre-loaded tensor directly (transform not applied —
            # tensors are already normalized by ColoredMNIST.color_dataset)
            img_tensor, _ = _CMNIST_TENSORS[path]
            return img_tensor, label
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def extract_embeddings(algorithm, train_samples, embed_dir, batch_size):
    """
    Step 1: Run inference on training data and save penultimate-layer embeddings.

    Uses BASE_TRANSFORM (no augmentation) for deterministic embeddings.
    Saves: embeddings.npy (N,384), labels.npy (N,), paths.json (N paths)
    """
    print(f"\n[Step 1] Extracting embeddings → {embed_dir}")
    os.makedirs(embed_dir, exist_ok=True)

    dataset = _PathDataset(train_samples, BASE_TRANSFORM)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    algorithm.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.cuda()
            feats = algorithm.predict(images, forward_feature=True)  # (B, 384)
            all_embeddings.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            if (batch_idx + 1) % 20 == 0:
                print(f"  {(batch_idx + 1) * batch_size}/{len(dataset)} samples processed")

    embeddings = np.concatenate(all_embeddings, axis=0)   # (N, D)
    labels = np.concatenate(all_labels, axis=0)            # (N,)
    paths = [s[0] for s in train_samples]  # str file paths or int indices

    embed_path = os.path.join(embed_dir, "embeddings.npy")
    labels_path = os.path.join(embed_dir, "labels.npy")
    paths_path = os.path.join(embed_dir, "paths.json")

    np.save(embed_path, embeddings)
    np.save(labels_path, labels)
    with open(paths_path, 'w') as f:
        json.dump(paths, f)

    print(f"  saved embeddings: {embeddings.shape}")
    print(f"  label distribution: { {int(c): int((labels==c).sum()) for c in np.unique(labels)} }")
    return embeddings, labels, paths


def load_embeddings(embed_dir):
    """Load previously extracted embeddings from disk."""
    print(f"\n[Step 1] Loading embeddings from disk: {embed_dir}")
    embeddings = np.load(os.path.join(embed_dir, "embeddings.npy"))
    labels = np.load(os.path.join(embed_dir, "labels.npy"))
    with open(os.path.join(embed_dir, "paths.json"), 'r') as f:
        paths = json.load(f)
    print(f"  loaded embeddings: {embeddings.shape}")
    return embeddings, labels, paths


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Cluster embeddings per class
# ─────────────────────────────────────────────────────────────────────────────

def cluster_embeddings(embeddings, labels, k, pca_dims, seed, embed_dir):
    """
    Step 2: K-Means clustering within each class.

    If pca_dims > 0, applies PCA per class before clustering.
    Prints silhouette score per class.
    Saves: clusters_k{k}.npy with cluster id per sample.

    Returns:
        cluster_assignments: np.ndarray (N,) with cluster id per sample
    """
    print(f"\n[Step 2] Clustering — k={k}, pca_dims={pca_dims}")
    num_classes = int(labels.max()) + 1
    cluster_assignments = np.full(len(labels), -1, dtype=np.int32)

    print(f"  {'Class':>6}  {'Samples':>8}  {'Silhouette':>12}")
    print(f"  {'-----':>6}  {'-------':>8}  {'----------':>12}")

    for c in range(num_classes):
        mask = labels == c
        emb_c = embeddings[mask]
        idxs = np.where(mask)[0]

        # PCA reduction
        if pca_dims > 0 and emb_c.shape[1] > pca_dims and emb_c.shape[0] > pca_dims:
            pca = PCA(n_components=pca_dims, random_state=seed)
            emb_c = pca.fit_transform(emb_c)

        # Clip k to number of samples
        k_eff = min(k, len(emb_c))
        if k_eff < k:
            print(f"  WARNING: class {c} has only {len(emb_c)} samples, using k={k_eff}")

        kmeans = KMeans(n_clusters=k_eff, random_state=seed, n_init=10)
        cluster_ids = kmeans.fit_predict(emb_c)
        cluster_assignments[idxs] = cluster_ids

        sil = "N/A"
        if k_eff > 1 and len(emb_c) > k_eff:
            try:
                sil = f"{silhouette_score(emb_c, cluster_ids):.4f}"
            except Exception:
                pass
        print(f"  {c:>6}  {len(emb_c):>8}  {sil:>12}")

    assert (cluster_assignments >= 0).all(), "Some samples were not assigned a cluster"

    clusters_path = os.path.join(embed_dir, f"clusters_k{k}.npy")
    np.save(clusters_path, cluster_assignments)
    print(f"  saved cluster assignments: {clusters_path}")
    return cluster_assignments


def load_clusters(embed_dir, k):
    """Load previously computed cluster assignments from disk."""
    path = os.path.join(embed_dir, f"clusters_k{k}.npy")
    print(f"\n[Step 2] Loading clusters from disk: {path}")
    return np.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Sample D-hat
# ─────────────────────────────────────────────────────────────────────────────

def sample_dhat(paths, labels, cluster_assignments, k, x, seed, env_idx, output_dir, save=True):
    """
    Step 3: Sample X images per (class, cluster) group to form D-hat.

    Returns:
        dhat_samples: list of (filepath, label) for all D-hat images
    """
    print(f"\n[Step 3] Sampling D-hat — k={k}, x={x} samples/cluster")
    os.makedirs(output_dir, exist_ok=True)

    num_classes = int(labels.max()) + 1
    rng = np.random.RandomState(seed)

    dhat_records = []
    dhat_samples = []  # (path, label)

    cluster_size_warn = False
    for c in range(num_classes):
        for cluster_id in range(k):
            mask = (labels == c) & (cluster_assignments == cluster_id)
            group_idxs = np.where(mask)[0]

            if len(group_idxs) == 0:
                print(f"  WARNING: class={c}, cluster={cluster_id} is empty — skipping")
                continue

            if len(group_idxs) < x:
                cluster_size_warn = True

            n_sample = min(x, len(group_idxs))
            chosen = rng.choice(group_idxs, size=n_sample, replace=False)

            for idx in chosen:
                dhat_records.append({
                    'path': paths[idx],
                    'label': int(labels[idx]),
                    'cluster': int(cluster_assignments[idx]),
                })
                dhat_samples.append((paths[idx], int(labels[idx])))

    if cluster_size_warn:
        print(f"  WARNING: some clusters had fewer than x={x} samples (sampled all available)")

    # Print summary
    total = len(dhat_samples)
    class_counts = {}
    for _, lbl in dhat_samples:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    print(f"  D-hat total: {total} images")
    print(f"  class distribution: {class_counts}")

    if save:
        dhat_path = os.path.join(output_dir, f"dhat_k{k}_x{x}.json")
        dhat_meta = {
            'env': env_idx, 'k': k, 'x': x, 'seed': seed,
            'total': total, 'class_counts': class_counts,
            'samples': dhat_records,
        }
        with open(dhat_path, 'w') as f:
            json.dump(dhat_meta, f, indent=2)
        print(f"  saved D-hat JSON: {dhat_path}")

    return dhat_samples


def load_dhat(dhat_json_path):
    """Load a previously sampled D-hat from JSON."""
    print(f"\n[Step 3] Loading D-hat from disk: {dhat_json_path}")
    with open(dhat_json_path, 'r') as f:
        meta = json.load(f)
    dhat_samples = [(r['path'], r['label']) for r in meta['samples']]
    print(f"  loaded D-hat: {len(dhat_samples)} images")
    return dhat_samples


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Finetune MOE layers
# ─────────────────────────────────────────────────────────────────────────────

class FilteredDataset(torch.utils.data.Dataset):
    """
    Dataset that loads images from a list of (filepath, label) pairs.

    Supports any torchvision transform. Used to load D-hat samples.

    Test T7:
        - len() == number of pairs
        - __getitem__ returns (Tensor[3,224,224], int)
        - augmented transform yields different tensors on two calls (stochastic)
        - base transform yields identical tensors on two calls (deterministic)
    """
    def __init__(self, path_label_pairs, transform):
        self.samples = path_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if isinstance(path, int):
            # CMNIST: return pre-loaded tensor directly
            img_tensor, _ = _CMNIST_TENSORS[path]
            return img_tensor, label
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def freeze_non_moe(algorithm):
    """
    Freeze all parameters except the MOE expert and router layers.
    Unfrozen: model.blocks.8.mlp.* and model.blocks.10.mlp.*

    Test T8:
        - trainable param count == sum of blocks.8.mlp.* + blocks.10.mlp.* params
        - blocks.0.attn.qkv.weight.requires_grad == False
        - frozen param norms unchanged after one optimizer step
    """
    for name, param in algorithm.model.named_parameters():
        unfreeze = (
            name.startswith('blocks.8.mlp.') or
            name.startswith('blocks.10.mlp.')
        )
        param.requires_grad = unfreeze

    trainable = [p for p in algorithm.model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in algorithm.model.parameters())
    print(f"  trainable params: {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.2f}%)")
    return trainable


def evaluate(algorithm, loader, device='cuda'):
    """Evaluate accuracy on a FastDataLoader. Returns (acc, loss)."""
    algorithm.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
            total_loss += loss.item() * len(y)
    return correct / total, total_loss / total


def evaluate_all_envs(algorithm, all_env_loaders):
    """
    Evaluate the model on pre-built in/out loaders for ALL environments.

    Args:
        algorithm: GMOE model
        all_env_loaders: dict {env_idx: (in_loader, out_loader)}

    Returns:
        dict with keys env{i}_in_acc and env{i}_out_acc for every env
    """
    accs = {}
    for i, (in_loader, out_loader) in all_env_loaders.items():
        in_acc, _ = evaluate(algorithm, in_loader)
        out_acc, _ = evaluate(algorithm, out_loader)
        accs[f"env{i}_in_acc"] = in_acc
        accs[f"env{i}_out_acc"] = out_acc
    return accs


def write_results_record(output_dir, record):
    """Append one JSON line to {output_dir}/results.jsonl."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.jsonl"), 'a') as f:
        f.write(json.dumps(record) + '\n')


def finetune_moe(algorithm, dhat_samples, test_env_idx, data_dir, cfg):
    """
    Step 4: Finetune MOE expert + router layers on D-hat.

    Freezes all parameters except blocks.8.mlp.* and blocks.10.mlp.*.
    Trains with CE loss using the augmented (or base) transform.
    Evaluates on the held-out test env every eval_every epochs.

    Returns:
        best_acc: best test accuracy achieved during finetuning
    """
    ft_cfg = cfg['finetuning']
    lr = ft_cfg['lr']
    epochs = ft_cfg['epochs']
    batch_size = ft_cfg['batch_size']
    augment = ft_cfg.get('augment', True)
    eval_every = ft_cfg.get('eval_every', 5)
    save_best = ft_cfg.get('save_best', True)

    output_dir = cfg['resolved']['output_dir']
    ckpt_name = cfg['resolved']['ckpt_name']
    k = cfg['clustering']['k']
    x = cfg['sampling']['x']
    env_idx = cfg['experiment']['env']

    ckpt_args = cfg['_ckpt_args']
    trial_seed = ckpt_args.get('trial_seed', 0)
    holdout_fraction = ckpt_args.get('holdout_fraction', 0.2)
    test_envs_set = set(ckpt_args.get('test_envs', [env_idx]))

    # ── hparams_seed: unique int per (k, x, lr) for collect_results.py ──────
    import hashlib as _hashlib
    hparams_seed = int(
        _hashlib.md5(f"{k}_{x}_{lr:.2e}".encode()).hexdigest(), 16
    ) % (2 ** 31 - 1)

    print(f"\n[Step 4] Finetuning MOE layers")
    print(f"  D-hat size   : {len(dhat_samples)}")
    print(f"  lr           : {lr}  (hparams_seed={hparams_seed})")
    print(f"  epochs       : {epochs}")
    print(f"  augment D-hat: {augment}")

    # ── Freeze ──────────────────────────────────────────────────────────────
    trainable = freeze_non_moe(algorithm)
    optimizer = torch.optim.Adam(trainable, lr=lr)

    # ── D-hat DataLoader ────────────────────────────────────────────────────
    dhat_transform = AUGMENT_TRANSFORM if augment else BASE_TRANSFORM
    dhat_dataset = FilteredDataset(dhat_samples, dhat_transform)
    dhat_loader = torch.utils.data.DataLoader(
        dhat_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # ── Build eval loaders for ALL environments (once, reused each epoch) ─
    # Mirrors Phase 1 split exactly: same trial_seed + holdout_fraction.
    # in_loader → env{i}_in_acc; out_loader → env{i}_out_acc.
    dataset_name = cfg['data'].get('dataset', 'TerraIncognita')
    all_env_loaders = {}

    if dataset_name == 'ColoredMNIST':
        hparams = {'data_augmentation': False}
        full_dataset = datasets.ColoredMNIST(data_dir, list(test_envs_set), hparams)
        num_envs = len(full_dataset)
        for i, env_data in enumerate(full_dataset):
            n_out = int(len(env_data) * holdout_fraction)
            out_split, in_split = misc.split_dataset(
                env_data, n_out, seed=misc.seed_hash(trial_seed, i)
            )
            all_env_loaders[i] = (
                FastDataLoader(dataset=in_split,  batch_size=64, num_workers=2),
                FastDataLoader(dataset=out_split, batch_size=64, num_workers=2),
            )
    else:
        terra_envs, terra_dir = get_terra_envs(data_dir)
        num_envs = len(terra_envs)
        for i, env_name in enumerate(terra_envs):
            img_folder = ImageFolder(
                os.path.join(terra_dir, env_name), transform=BASE_TRANSFORM
            )
            # Match Phase 1: out, in_ = split_dataset(env, int(N*holdout_fraction), seed)
            n_out = int(len(img_folder) * holdout_fraction)
            out_split, in_split = misc.split_dataset(
                img_folder, n_out, seed=misc.seed_hash(trial_seed, i)
            )
            all_env_loaders[i] = (
                FastDataLoader(dataset=in_split,  batch_size=64, num_workers=2),
                FastDataLoader(dataset=out_split, batch_size=64, num_workers=2),
            )

    def _make_record(epoch, env_accs):
        return {
            "args": {
                "algorithm": "GMOE_Phase2",
                "dataset": ckpt_args.get('dataset', dataset_name),
                "test_envs": [env_idx],
                "trial_seed": trial_seed,
                "hparams_seed": hparams_seed,
            },
            "step": epoch,
            **env_accs,
        }

    def _val_acc(env_accs):
        """Mean out_acc over training envs — matches IIDAccuracySelectionMethod."""
        return sum(
            env_accs[f"env{i}_out_acc"]
            for i in range(num_envs) if i not in test_envs_set
        ) / (num_envs - len(test_envs_set))

    # ── Initial evaluation (epoch 0) ────────────────────────────────────────
    env_accs0 = evaluate_all_envs(algorithm, all_env_loaders)
    write_results_record(output_dir, _make_record(0, env_accs0))
    init_test_acc     = env_accs0[f"env{env_idx}_in_acc"]
    init_test_out_acc = env_accs0[f"env{env_idx}_out_acc"]
    init_val_acc      = _val_acc(env_accs0)
    print(f"\n  [epoch  0] val_acc={init_val_acc:.4f}  "
          f"test_acc(env{env_idx}_in)={init_test_acc:.4f}  "
          f"test_out_acc(env{env_idx}_out)={init_test_out_acc:.4f}  (before finetuning)")

    best_val  = init_val_acc
    best_acc  = init_test_acc
    best_state = copy.deepcopy(algorithm.state_dict())
    best_epoch = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        algorithm.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for x_batch, y_batch in dhat_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            logits = algorithm.predict(x_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            epoch_correct += (logits.argmax(1) == y_batch).sum().item()
            epoch_total += len(y_batch)

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total
        elapsed    = time.time() - t0

        if epoch % eval_every == 0 or epoch == epochs:
            algorithm.eval()
            env_accs = evaluate_all_envs(algorithm, all_env_loaders)
            write_results_record(output_dir, _make_record(epoch, env_accs))

            test_acc     = env_accs[f"env{env_idx}_in_acc"]
            test_out_acc = env_accs[f"env{env_idx}_out_acc"]
            val_acc      = _val_acc(env_accs)
            print(f"  [epoch {epoch:>3}] "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
                  f"test_out_acc={test_out_acc:.4f}  ({elapsed:.1f}s)")

            # Best checkpoint selected by val_acc (mirrors IIDAccuracySelectionMethod)
            if val_acc > best_val:
                best_val      = val_acc
                best_acc      = test_acc
                best_out_acc  = test_out_acc
                best_epoch    = epoch
                best_state    = copy.deepcopy(algorithm.state_dict())
                if save_best:
                    _save_checkpoint(
                        algorithm, best_state, output_dir,
                        f"best_{ckpt_name}", cfg, epoch, test_acc
                    )
                    print(f"  → new best val_acc={best_val:.4f}  "
                          f"test_acc(in)={best_acc:.4f}  test_out_acc(out)={best_out_acc:.4f}  (saved)")
        else:
            print(f"  [epoch {epoch:>3}] "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"({elapsed:.1f}s)")

    print(f"\n  Best test_acc(in)={best_acc:.4f}  test_out_acc(out)={best_out_acc:.4f}  "
          f"val_acc={best_val:.4f}  at epoch {best_epoch}")
    algorithm.load_state_dict(best_state)
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(algorithm, state_dict, output_dir, filename, cfg, epoch, test_acc):
    """Internal helper to write a checkpoint file."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torch.save({
        'model_dict': state_dict,
        'epoch': epoch,
        'test_acc': test_acc,
        'config': cfg,
        'dhat_json': cfg['resolved']['dhat_json'],
    }, save_path)
    return save_path


def save_final_checkpoint(algorithm, output_dir, cfg, epoch, test_acc):
    """Step 5: Save the final Phase 2 checkpoint."""
    print(f"\n[Step 5] Saving final checkpoint")
    ckpt_name = cfg['resolved']['ckpt_name']
    path = _save_checkpoint(
        algorithm, algorithm.state_dict(),
        output_dir, ckpt_name, cfg, epoch, test_acc
    )
    print(f"  saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 2 GMOE finetuning on D-hat (cluster-sampled reweighting dataset)'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (e.g. configs/phase2_config.yaml)')
    parser.add_argument('--env', type=int, default=None,
                        help='Override config: test env index (0-3)')
    parser.add_argument('--k', type=int, default=None,
                        help='Override config: clusters per class')
    parser.add_argument('--x', type=int, default=None,
                        help='Override config: samples per cluster')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override config: finetuning learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override config: finetuning epochs')
    parser.add_argument('--skip_extract', action='store_true',
                        help='Override config: skip embedding extraction (reload from disk)')
    parser.add_argument('--skip_cluster', action='store_true',
                        help='Override config: skip clustering + D-hat sampling (reload JSON)')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load and resolve config ──────────────────────────────────────────────
    cfg = load_config(args.config, vars(args))
    cfg = resolve_paths(cfg)

    env_idx = cfg['experiment']['env']
    seed = cfg['experiment']['seed']
    k = cfg['clustering']['k']
    x = cfg['sampling']['x']
    pca_dims = cfg['clustering']['pca_dims']
    embed_dir = cfg['resolved']['embed_dir']
    output_dir = cfg['resolved']['output_dir']
    data_dir = cfg['data']['data_dir']

    dataset_name = cfg['data'].get('dataset', 'TerraIncognita')
    print("=" * 60)
    if dataset_name == 'TerraIncognita':
        terra_envs, _ = get_terra_envs(data_dir)
        env_display = terra_envs[env_idx]
    else:
        env_display = datasets.ColoredMNIST.ENVIRONMENTS[env_idx] if dataset_name == 'ColoredMNIST' else str(env_idx)
    print(f"Phase 2 Finetuning — dataset={dataset_name}, env={env_idx} ({env_display}), k={k}, x={x}")
    print(f"  phase1_ckpt  : {cfg['resolved']['phase1_ckpt']}")
    print(f"  output_dir   : {output_dir}")
    print(f"  D-hat JSON   : {cfg['resolved']['dhat_json']}")
    print("=" * 60)

    # ── Step 0: Load Phase 1 model ───────────────────────────────────────────
    algorithm, ckpt_args, ckpt = load_phase1_model(cfg['resolved']['phase1_ckpt'])
    cfg['_ckpt_args'] = ckpt_args  # stash for use in finetune step

    # ── CMNIST pre-load: populate _CMNIST_TENSORS even when skipping extraction
    # so that FilteredDataset / _PathDataset can look up tensors by index.
    if dataset_name == 'ColoredMNIST' and cfg['skip']['extract']:
        build_train_split(ckpt_args, data_dir, env_idx, dataset_name=dataset_name)
        # _CMNIST_TENSORS is now set as a side-effect of build_train_split

    # ── Step 1: Extract or load embeddings ──────────────────────────────────
    if cfg['skip']['extract'] and os.path.exists(cfg['resolved']['embed_npy']):
        embeddings, labels, paths = load_embeddings(embed_dir)
    else:
        train_samples, _ = build_train_split(ckpt_args, data_dir, env_idx, dataset_name=dataset_name)
        embeddings, labels, paths = extract_embeddings(
            algorithm, train_samples, embed_dir,
            batch_size=cfg['embedding']['batch_size']
        )

    # ── Steps 2-3: Cluster and sample D-hat (or reload) ─────────────────────
    if cfg['skip']['cluster'] and os.path.exists(cfg['resolved']['dhat_json']):
        dhat_samples = load_dhat(cfg['resolved']['dhat_json'])
    else:
        # Step 2: Cluster
        clusters_path = cfg['resolved']['clusters_npy']
        if cfg['skip']['cluster'] and os.path.exists(clusters_path):
            cluster_assignments = load_clusters(embed_dir, k)
        else:
            cluster_assignments = cluster_embeddings(
                embeddings, labels, k, pca_dims, seed, embed_dir
            )

        # Step 3: Sample D-hat
        dhat_samples = sample_dhat(
            paths, labels, cluster_assignments,
            k=k, x=x, seed=seed,
            env_idx=env_idx,
            output_dir=output_dir,
            save=cfg['sampling']['save_dhat']
        )

    # ── Step 4: Finetune ─────────────────────────────────────────────────────
    best_acc = finetune_moe(algorithm, dhat_samples, env_idx, data_dir, cfg)

    # ── Step 5: Save final checkpoint ────────────────────────────────────────
    save_final_checkpoint(
        algorithm, output_dir, cfg,
        epoch=cfg['finetuning']['epochs'],
        test_acc=best_acc
    )

    print("\n" + "=" * 60)
    print(f"Done. Best test_acc (env{env_idx}): {best_acc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

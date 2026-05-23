"""Shared forget/retain splits for unlearning scripts.

CIFAR-10 / CIFAR-100 follow the LoTUS protocol:
  - official train split for training; official test split halved (stratified) into
    test + unseen (val) for MIA
  - random unlearning: first ``frac_per_class`` fraction **per class** → forget

Other datasets (PACS, etc.) keep the legacy 80/10/10 sequential split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

LOTUS_CIFAR_DATASETS = frozenset({"CIFAR10", "CIFAR100"})

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            if isinstance(x, torch.Tensor):
                x = to_pil_image(x)
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


@dataclass
class UnlearningSplits:
    train_subset: Subset
    test_subset: Subset
    unseen_subset: Subset
    retain_subset: Subset
    forget_subset: Subset
    protocol: str
    full_dataset: ConcatDataset
    routing_indices: Dict[str, List[Tuple[int, int, int]]] = field(default_factory=dict)


def is_lotus_cifar_dataset(dataset_name: str) -> bool:
    return dataset_name in LOTUS_CIFAR_DATASETS


def _label_value(y) -> int:
    return int(y.item()) if isinstance(y, torch.Tensor) else int(y)


def _get_targets(data) -> torch.Tensor:
    if hasattr(data, "targets"):
        targets = data.targets
        if isinstance(targets, torch.Tensor):
            return targets.detach().clone()
        return torch.tensor(targets)
    raise ValueError("Dataset has no targets attribute")


def _split_forget_retain_per_class(data_train, frac_per_class: float):
    """LoTUS-style per-class forget split (first fraction of each class)."""
    forget_parts = []
    retain_parts = []
    targets = _get_targets(data_train)
    classes = data_train.classes
    for class_name in classes:
        class_idx = data_train.class_to_idx[class_name]
        indices = torch.where(targets == class_idx)[0]
        n_forget = int(len(indices) * frac_per_class)
        forget_parts.append(Subset(data_train, indices[:n_forget].tolist()))
        retain_parts.append(Subset(data_train, indices[n_forget:].tolist()))
    forget = ConcatDataset(forget_parts)
    retain = ConcatDataset(retain_parts)
    return forget, retain


def _subset_indices_list(subset) -> List[int]:
    if isinstance(subset, ConcatDataset):
        out = []
        for ds in subset.datasets:
            out.extend(_subset_indices_list(ds))
        return out
    return list(subset.indices)


def _routing_from_subset(
    subset,
    env_idx: int,
    train_len: int,
    base_is_train: bool,
) -> List[Tuple[int, int, int]]:
    indices = _subset_indices_list(subset)
    out = []
    for local in indices:
        g = local if base_is_train else train_len + local
        out.append((g, env_idx, local))
    return out


def _dg_routing_indices(full_dataset, train_size, test_size, retain_idx, forget_idx):
    train_end = train_size
    test_end = train_size + test_size

    def _slice_routing(start, end):
        return [(g, 0, g) for g in range(start, end)]

    return {
        "full": [(g, 0, g) for g in range(len(full_dataset))],
        "retain": [(g, 0, g) for g in retain_idx],
        "forget": [(g, 0, g) for g in forget_idx],
        "unseen": _slice_routing(test_end, len(full_dataset)),
        "test": _slice_routing(train_end, test_end),
    }


def _build_cifar_lotus_splits(dataset, args) -> UnlearningSplits:
    train_ds = dataset.train_dataset
    held_out = dataset.test_dataset
    train_len = len(train_ds)

    labels = _get_targets(held_out).numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    val_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

    test_subset = Subset(held_out, test_idx.tolist())
    unseen_subset = Subset(held_out, val_idx.tolist())
    train_subset = Subset(train_ds, list(range(len(train_ds))))

    if args.unlearn_setting == "random":
        frac = float(args.unlearn_random_ratio) if args.unlearn_random_ratio else 0.1
        forget_raw, retain_raw = _split_forget_retain_per_class(train_ds, frac)
        print(
            f"[*] Unlearn Setting: LoTUS per-class RANDOM | "
            f"frac_per_class={frac} | Retain: {len(retain_raw)} | Forget: {len(forget_raw)}"
        )
    elif args.unlearn_setting == "class":
        num_class_forget = int(args.unlearn_num_class) if args.unlearn_num_class else 1
        forget_classes = set(range(num_class_forget))
        targets = _get_targets(train_ds)
        forget_indices, retain_indices = [], []
        for idx in range(len(train_ds)):
            y = _label_value(targets[idx])
            (forget_indices if y in forget_classes else retain_indices).append(idx)
        forget_raw = Subset(train_ds, forget_indices)
        retain_raw = Subset(train_ds, retain_indices)
        print(
            f"[*] Unlearn Setting: LoTUS CLASS | Classes to forget: {sorted(forget_classes)} | "
            f"Retain: {len(retain_raw)} | Forget: {len(forget_raw)}"
        )
    else:
        raise ValueError("unlearn_setting must be 'random' or 'class'")

    full_dataset = ConcatDataset([train_ds, held_out])
    routing = {
        "full": _routing_from_subset(train_subset, 0, train_len, True)
        + _routing_from_subset(Subset(held_out, list(range(len(held_out)))), 0, train_len, False),
        "retain": _routing_from_subset(retain_raw, 0, train_len, True),
        "forget": _routing_from_subset(forget_raw, 0, train_len, True),
        "unseen": _routing_from_subset(unseen_subset, 0, train_len, False),
        "test": _routing_from_subset(test_subset, 0, train_len, False),
    }

    return UnlearningSplits(
        train_subset=train_subset,
        test_subset=test_subset,
        unseen_subset=unseen_subset,
        retain_subset=retain_raw,
        forget_subset=forget_raw,
        protocol="lotus_cifar",
        full_dataset=full_dataset,
        routing_indices=routing,
    )


def _build_dg_splits(dataset, args) -> UnlearningSplits:
    full_dataset = ConcatDataset([env for env in dataset])
    total_size = len(full_dataset)

    train_size = int(total_size * 0.8)
    test_size = int(total_size * 0.1)

    train_indices = list(range(train_size))
    test_indices = list(range(train_size, train_size + test_size))
    unseen_indices = list(range(train_size + test_size, total_size))

    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)
    unseen_subset = Subset(full_dataset, unseen_indices)

    if args.unlearn_setting == "random":
        forget_ratio = float(args.unlearn_random_ratio) if args.unlearn_random_ratio else 0.1
        forget_size = int(train_size * forget_ratio)
        forget_indices = train_indices[:forget_size]
        retain_indices = train_indices[forget_size:]
        print(
            f"[*] Unlearn Setting: SEQUENTIAL 'RANDOM' | "
            f"Retain: {len(retain_indices)} | Forget: {len(forget_indices)}"
        )
    elif args.unlearn_setting == "class":
        num_class_forget = int(args.unlearn_num_class) if args.unlearn_num_class else 1
        forget_classes = set(range(num_class_forget))
        retain_indices, forget_indices = [], []
        for idx in train_indices:
            _, y = full_dataset[idx]
            y_val = _label_value(y)
            (forget_indices if y_val in forget_classes else retain_indices).append(idx)
        print(
            f"[*] Unlearn Setting: SEQUENTIAL CLASS | Classes to forget: {sorted(forget_classes)} | "
            f"Retain: {len(retain_indices)} | Forget: {len(forget_indices)}"
        )
    else:
        raise ValueError("unlearn_setting must be 'random' or 'class'")

    retain_subset = Subset(full_dataset, retain_indices)
    forget_subset = Subset(full_dataset, forget_indices)

    routing = _dg_routing_indices(
        full_dataset, train_size, test_size, retain_indices, forget_indices
    )

    return UnlearningSplits(
        train_subset=train_subset,
        test_subset=test_subset,
        unseen_subset=unseen_subset,
        retain_subset=retain_subset,
        forget_subset=forget_subset,
        protocol="dg_sequential",
        full_dataset=full_dataset,
        routing_indices=routing,
    )


def build_unlearning_splits(dataset, args) -> UnlearningSplits:
    if is_lotus_cifar_dataset(args.dataset):
        return _build_cifar_lotus_splits(dataset, args)
    return _build_dg_splits(dataset, args)


def get_transforms(dataset_name: str, protocol: str):
    """Return (train_transform, eval_transform) for ApplyTransform wrappers."""
    if protocol == "lotus_cifar":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        return train_transform, eval_transform

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor()])
    return train_transform, eval_transform


def output_ul_param(args) -> str:
    if args.unlearn_setting == "random":
        return str(args.unlearn_random_ratio if args.unlearn_random_ratio else "0.1")
    val = args.unlearn_num_class if args.unlearn_num_class else 1
    return str(val)

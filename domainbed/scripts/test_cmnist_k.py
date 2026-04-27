"""Smoke test for the ColoredMNIST_K dataset (K source + 1 fixed test).

Usage:
    python -m domainbed.scripts.test_colored_mnist_k \
        --data_dir ./domainbed/data/MNIST \
        --out_dir /tmp/cmnist_k_viz
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from domainbed import datasets, hparams_registry


def build(data_dir, K, test_size=10000, seed=0):
    torch.manual_seed(seed)
    hparams = hparams_registry.default_hparams('ERM', 'ColoredMNIST_K')
    hparams['num_source_domains'] = K
    hparams['test_size'] = test_size
    return datasets.get_dataset_class('ColoredMNIST_K')(data_dir, [K], hparams)


def parse_p(env_name):
    return float(env_name.split('=')[1])


def to_rgb(x):
    blue = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.cat([x[:, :1], x[:, 1:2], blue], dim=1)


def check_K(ds, K, test_size=10000, total=70000):
    n_envs = len(ds)
    assert n_envs == K + 1, f"K={K}: expected {K+1} envs, got {n_envs}"

    # test env: last index, fixed size, p=0.5
    test_x = ds[K].tensors[0]
    assert len(test_x) == test_size, \
        f"K={K}: test env size {len(test_x)} != {test_size}"
    assert abs(parse_p(ds.ENVIRONMENTS[-1]) - 0.5) < 1e-9, \
        f"K={K}: last env p must be 0.5"

    # source envs: equal split of (total - test_size)
    per_env = (total - test_size) // K
    for i in range(K):
        n = len(ds[i].tensors[0])
        assert n == per_env, \
            f"K={K}: source env {i} size {n} != {per_env}"

    # source p-values evenly spaced, exclude 0.5
    train_p = sorted(parse_p(n) for n in ds.ENVIRONMENTS[:-1])
    assert all(abs(p - 0.5) > 1e-6 for p in train_p), \
        f"K={K}: source envs must not include p=0.5"
    assert len(set(train_p)) == K, \
        f"K={K}: source p-values must be unique"

    # per-image structure
    for i in range(n_envs):
        x = ds[i].tensors[0]
        y = ds[i].tensors[1]
        assert x.shape[1:] == (2, 28, 28)
        assert x.dtype == torch.float32
        assert x.min() >= 0 and x.max() <= 1
        assert set(y.unique().tolist()) <= {0, 1}
        ch0_zero = (x[:, 0].abs().sum(dim=(1, 2)) == 0)
        ch1_zero = (x[:, 1].abs().sum(dim=(1, 2)) == 0)
        assert torch.logical_xor(ch0_zero, ch1_zero).all(), \
            f"K={K}, env {i}: each sample must have exactly one zeroed channel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', default='/tmp/cmnist_k_viz')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for K in [2, 5, 10]:
        ds = build(args.data_dir, K)
        print(f"K={K}: envs={len(ds)}  names={ds.ENVIRONMENTS}")
        for i, name in enumerate(ds.ENVIRONMENTS):
            print(f"  env {i}: {name}  size={len(ds[i])}")
        check_K(ds, K)

    # save grid for K=10 case (full curve endpoint)
    ds = build(args.data_dir, 10)
    for i in range(len(ds)):
        x = ds[i].tensors[0][:16]
        save_image(to_rgb(x), os.path.join(args.out_dir, f'k10_env_{i}.png'),
                   nrow=4, padding=2)
    print(f"saved K=10 grids to {args.out_dir}/k10_env_*.png")

    # determinism (K=5)
    a = build(args.data_dir, 5)[0].tensors[0]
    b = build(args.data_dir, 5)[0].tensors[0]
    assert torch.equal(a, b), "rebuild with same seed should be byte-identical"
    print("determinism check OK")

    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
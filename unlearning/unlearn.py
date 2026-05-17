# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import PIL
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
import torchvision
from torch.utils.data import ConcatDataset, random_split, dataset
from torchvision import transforms

import metrics

# Patch Tutel CUDA kernels with pure-PyTorch fallbacks BEFORE importing
# vision_transformer / algorithms (which import tutel at module level).
# Required when Tutel's compiled extensions don't support the current GPU.
import domainbed.tutel_patch  # noqa: F401

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.sweep_logger import SweepLogger

from torchvision.transforms.functional import to_pil_image

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


def _mia_in_stop_band(mia_score, low, high):
    """True if loss-based MIA attack accuracy is in [low, high] (chance = 0.5)."""
    return low <= mia_score <= high


if __name__ == "__main__":
    WANDB_PROJECT = "sparse_moe_unlearn"

    parser = argparse.ArgumentParser(description='Unlearning')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--debug', default="True")
    
    parser.add_argument('--unlearn_algo', type=str, default='finetune',
                        choices=['finetune', 'ga', 'rl', 'boundary_shrink', 'wfisher', 'l1_sparse',
                                 'modular'])
    parser.add_argument('--modular_unlearn_topk', type=int, default=None,
                        help='Top-k experts by mean routing on forget set (modular unlearn).')
    parser.add_argument('--modular_unlearn_tau', type=float, default=None,
                        help='If set, select experts with s_m > tau instead of top-k.')
    parser.add_argument('--modular_unlearn_beta', type=float, default=None,
                        help='Weight on retain CE in modular unlearn (default 1.0).')
    parser.add_argument('--modular_unlearn_gamma', type=float, default=None,
                        help='Weight on KL distillation in modular unlearn (default 1.0).')
    parser.add_argument('--modular_score_max_batches', type=int, default=None,
                        help='Max batches when averaging routing over forget data (default 200).')
    parser.add_argument('--modular_unlearn_lr', type=float, default=None,
                        help='LR for modular phase (default: hparams lr).')
    parser.add_argument('--modular_unlearn_use_modular_reg', action='store_true',
                        help='Add optional L_div on selected experts during unlearn (default off).')
    parser.add_argument('--modular_unlearn_lambda_div', type=float, default=None,
                        help='λ_div for optional unlearn decorrelation (default 0; try 0.01 if --modular_unlearn_use_modular_reg).')

    parser.add_argument('--no_mia_early_stop', action='store_true',
                        help='Disable stopping when MIA is in the target band.')
    parser.add_argument('--mia_stop_low', type=float, default=0.48,
                        help='Lower bound of MIA attack accuracy for early stop (default 0.48).')
    parser.add_argument('--mia_stop_high', type=float, default=0.52,
                        help='Upper bound of MIA attack accuracy for early stop (default 0.52).')
    
    parser.add_argument('--unlearn_setting', default='random', choices=['random', 'class'])
    parser.add_argument('--unlearn_random_ratio', default=None) # 0.1
    parser.add_argument('--unlearn_num_class', default=None) # 1

    parser.add_argument('--data_dir', type=str, default='./domainbed/data')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM") 
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0)
    parser.add_argument('--trial_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--drop_out', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--steps', type=int, default=10**6)
    parser.add_argument('--checkpoint_freq', type=int, default=None)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--sweep_log_dir', type=str, default=None)
    parser.add_argument('--sweep_run_id', type=str, default=None)

    parser.add_argument('--num_step_per_evaluate', type=int, default=None, 
                        help='Number of steps to run before triggering evaluation')

    args = parser.parse_args()

    ul_param = args.unlearn_random_ratio if args.unlearn_setting == 'random' else args.unlearn_num_class
    if ul_param is None: ul_param = "default"
    args.output_dir = f"unlearning/train_output/unlearn_{args.unlearn_algo}_{args.algorithm}_{args.dataset}_{args.unlearn_setting}_{ul_param}_seed_{args.seed}"

    start_step = 0

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.unlearn_algo == 'modular':
        if args.modular_unlearn_topk is not None:
            hparams['modular_unlearn_topk'] = args.modular_unlearn_topk
        if args.modular_unlearn_tau is not None:
            hparams['modular_unlearn_tau'] = args.modular_unlearn_tau
        if args.modular_unlearn_beta is not None:
            hparams['modular_unlearn_beta'] = args.modular_unlearn_beta
        if args.modular_unlearn_gamma is not None:
            hparams['modular_unlearn_gamma'] = args.modular_unlearn_gamma
        if args.modular_score_max_batches is not None:
            hparams['modular_score_max_batches'] = args.modular_score_max_batches
        if args.modular_unlearn_lr is not None:
            hparams['modular_unlearn_lr'] = args.modular_unlearn_lr
        if args.modular_unlearn_use_modular_reg:
            hparams['modular_unlearn_use_modular_reg'] = True
        if args.modular_unlearn_lambda_div is not None:
            hparams['modular_unlearn_lambda_div'] = args.modular_unlearn_lambda_div

    if args.batch_size is not None:
        hparams['batch_size'] = args.batch_size
    if args.drop_out is not None:
        hparams['drop_out'] = args.drop_out
    if args.lr is not None:
        hparams['lr'] = args.lr
    if args.weight_decay is not None:
        hparams['weight_decay'] = args.weight_decay

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    if not (args.debug == "True"):
        # wandb.login(key="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")

        _NEVER_SHOW = {
            'data_augmentation', 'resnet18', 'resnet_dropout',
            'nonlinear_classifier', 'class_balanced',
            'val_augment', 'freeze_bn', 'pretrained', 'optimizer',
        }
        relevant_keys = {k for k in hparams if k not in _NEVER_SHOW}

        hparam_str = '_'.join(f'{k}={hparams[k]}' for k in sorted(relevant_keys))
        run_name = f'unlearn_{args.unlearn_algo}_{args.algorithm}_{args.dataset}_{args.unlearn_setting}_{ul_param}_seed_{args.seed}'
        
        if len(run_name) > 128:
            run_name = run_name[:125] + '...'

        # wandb.init(
        #     project=WANDB_PROJECT,
        #     name=run_name,
        #     config={
        #         'dataset': args.dataset,
        #         'algorithm': args.algorithm,
        #         'unlearn_algo': args.unlearn_algo,
        #         'seed': args.seed,
        #         'trial_seed': args.trial_seed,
        #         'hparams_seed': args.hparams_seed,
        #         **{f'hp/{k}': hparams[k] for k in sorted(relevant_keys)},
        #     },
        #     settings=wandb.Settings(start_method='thread'),
        # )

    # deterministic split
    full_dataset = ConcatDataset([env for env in dataset])
    total_size = len(full_dataset)

    # 80% train, 10% test, 10% unseen
    train_size = int(total_size * 0.8)
    test_size = int(total_size * 0.1)
    unseen_size = total_size - train_size - test_size

    all_indices = list(range(total_size))

    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size : train_size + test_size]
    unseen_indices = all_indices[train_size + test_size :]

    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)
    unseen_subset = Subset(full_dataset, unseen_indices)

    if args.unlearn_setting == 'random':
        forget_ratio = float(args.unlearn_random_ratio) if args.unlearn_random_ratio else 0.1
        forget_size = int(train_size * forget_ratio)
        retain_size = train_size - forget_size

        forget_indices = train_indices[:forget_size]
        retain_indices = train_indices[forget_size:]

        retain_subset = Subset(full_dataset, retain_indices)
        forget_subset = Subset(full_dataset, forget_indices)
        
        print(f"[*] Unlearn Setting: SEQUENTIAL 'RANDOM' | Retain: {retain_size} | Forget: {forget_size}")

    elif args.unlearn_setting == 'class':
        num_class_forget = int(args.unlearn_num_class) if args.unlearn_num_class else 1
        
        all_classes = list(range(dataset.num_classes))
        forget_classes = all_classes[:num_class_forget]

        print(f"[*] Unlearn Setting: SEQUENTIAL CLASS | Classes to forget: {forget_classes}")

        retain_indices = []
        forget_indices = []

        for idx in train_indices:
            _, y = full_dataset[idx] 
            y_val = y.item() if isinstance(y, torch.Tensor) else y
            
            if y_val in forget_classes:
                forget_indices.append(idx)
            else:
                retain_indices.append(idx)

        retain_subset = Subset(full_dataset, retain_indices)
        forget_subset = Subset(full_dataset, forget_indices)
        
        print(f"[*] Retain size: {len(retain_subset)} | Forget size: {len(forget_subset)}")

    else:
        raise ValueError("unlearn_setting must be 'random' or 'class'")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    unseen_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = ApplyTransform(train_subset, transform=train_transform)
    test_set = ApplyTransform(test_subset, transform=test_transform)

    forget_set_train = ApplyTransform(forget_subset, transform=train_transform)
    forget_set_test = ApplyTransform(forget_subset, transform=test_transform)
    retain_set_train = ApplyTransform(retain_subset, transform=train_transform)
    retain_set_test = ApplyTransform(retain_subset, transform=test_transform)
    unseen_set = ApplyTransform(unseen_subset, transform=unseen_transform)

    retain_train_loader = InfiniteDataLoader(dataset=retain_set_train, weights=None, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)
    forget_train_loader = InfiniteDataLoader(dataset=forget_set_train, weights=None, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)

    retain_test_loader = FastDataLoader(dataset=retain_set_test, batch_size=64, num_workers=dataset.N_WORKERS)
    forget_test_loader = FastDataLoader(dataset=forget_set_test, batch_size=64, num_workers=dataset.N_WORKERS)
    unseen_loader = FastDataLoader(dataset=unseen_set, batch_size=64, num_workers=dataset.N_WORKERS)
    test_loader = FastDataLoader(dataset=test_set, batch_size=64, num_workers=dataset.N_WORKERS)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                1, hparams) 

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        algorithm.network.load_state_dict(checkpoint)
    else:
        print("[WARNING] No --checkpoint_path provided for unlearning! Training from scratch?")

    algorithm.to(device)

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_vals = collections.defaultdict(lambda: [])

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        network_state_dict = algorithm.network.state_dict()
        torch.save(network_state_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None

    print(f"[*] Starting Unlearning Process with Algorithm: {args.unlearn_algo.upper()}")
    print(f"[*] Output directory set to: {args.output_dir}")
    
    if args.unlearn_algo == 'wfisher':
        step_start_time = time.time()
        step_vals = algorithm.update_wfisher(retain_train_loader, forget_train_loader)
        
        results = {'step': 1, 'epoch': 1}
        for key, val in step_vals.items():
            results[key] = val

        results['test_acc'] = metrics.test_acc(algorithm, test_loader, device)
        results['retain_acc'] = metrics.retain_acc(algorithm, retain_test_loader, device)
        results['forget_acc'] = metrics.forget_acc(algorithm, forget_test_loader, device)
        results['mia_score'] = metrics.mia(algorithm, forget_test_loader, unseen_loader, device)

        results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)
        results['step_time'] = time.time() - step_start_time

        results_keys = sorted(results.keys())
        misc.print_row(results_keys, colwidth=12)
        misc.print_row([results[key] for key in results_keys], colwidth=12)

        # if wandb.run: wandb.log(results)
        with open(os.path.join(args.output_dir, 'results.jsonl'), 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")
            
        save_checkpoint('model_unlearned.pt')

    elif args.unlearn_algo == 'modular':
        if not hasattr(algorithm, 'begin_modular_unlearn'):
            raise NotImplementedError(
                'modular unlearning requires GMOE_Full_Unlearn (explicit MoE head + routing).')

        steps_per_epoch = max(1, int(len(retain_set_train) / hparams['batch_size']))
        eval_interval = (args.num_step_per_evaluate
                         if args.num_step_per_evaluate is not None
                         else steps_per_epoch)

        sel_info = algorithm.begin_modular_unlearn(forget_train_loader, device)
        print(f"[*] Modular unlearn — selected experts {sel_info['expert_indices']}")
        print(f"[*] Mean routing scores on forget (per expert): {sel_info['scores']}")
        if not args.no_mia_early_stop:
            print(f"[*] MIA early stop band: [{args.mia_stop_low}, {args.mia_stop_high}] "
                  f"(attack accuracy ~0.5 = indistinguishable forget vs unseen loss)")

        forget_iterator = iter(forget_train_loader)
        retain_iterator = iter(retain_train_loader)

        try:
            for step in range(start_step, n_steps):
                step_start_time = time.time()

                xf, yf = next(forget_iterator)
                xr, yr = next(retain_iterator)
                forget_mb = [(xf.to(device), yf.to(device))]
                retain_mb = [(xr.to(device), yr.to(device))]

                step_vals = algorithm.update_modular_unlearn(forget_mb, retain_mb)

                checkpoint_vals['step_time'].append(time.time() - step_start_time)
                for key, val in step_vals.items():
                    checkpoint_vals[key].append(val)

                if (step > 0 and step % eval_interval == 0) or (step == n_steps - 1):
                    current_epoch = step // steps_per_epoch
                    results = {'step': step, 'epoch': current_epoch}

                    for key, val in checkpoint_vals.items():
                        results[key] = np.mean(val)

                    results['test_acc'] = metrics.test_acc(algorithm, test_loader, device)
                    results['retain_acc'] = metrics.retain_acc(algorithm, retain_test_loader, device)
                    results['forget_acc'] = metrics.forget_acc(algorithm, forget_test_loader, device)
                    results['mia_score'] = metrics.mia(algorithm, forget_test_loader, unseen_loader, device)

                    results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)
                    results['step_time'] = time.time() - step_start_time

                    results_keys = sorted(results.keys())
                    if results_keys != last_results_keys:
                        misc.print_row(results_keys, colwidth=12)
                        last_results_keys = results_keys
                    misc.print_row([results[key] for key in results_keys], colwidth=12)

                    # if wandb.run:
                    #     wandb.log(results)

                    epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                    with open(epochs_path, 'a') as f:
                        f.write(json.dumps(results, sort_keys=True) + "\n")

                    checkpoint_vals = collections.defaultdict(lambda: [])

                    if args.save_model_every_checkpoint:
                        save_checkpoint(f'model_epoch{current_epoch}.pt')

                    current_mia = results['mia_score']

                    if (not args.no_mia_early_stop
                            and _mia_in_stop_band(
                                current_mia, args.mia_stop_low, args.mia_stop_high)):
                        print(f"\n[!] Early stopping triggered at step {step}!")
                        print(f"[*] Target MIA in [{args.mia_stop_low}, {args.mia_stop_high}]: {current_mia}")
                        break
        finally:
            algorithm.end_modular_unlearn()

        save_checkpoint('model_unlearned_final.pt')

    else:
        if args.unlearn_algo in ['finetune', 'l1_sparse']:
            active_iterator = iter(retain_train_loader)
            active_train_set = retain_set_train
        elif args.unlearn_algo in ['ga', 'rl', 'boundary_shrink']:
            active_iterator = iter(forget_train_loader)
            active_train_set = forget_set_train
        steps_per_epoch = max(1, int(len(active_train_set) / hparams['batch_size']))

        eval_interval = (args.num_step_per_evaluate
                         if args.num_step_per_evaluate is not None
                         else steps_per_epoch)

        for step in range(start_step, n_steps):
            step_start_time = time.time()

            x, y = next(active_iterator)
            minibatches_device = [(x.to(device), y.to(device))]

            if args.unlearn_algo == 'finetune':
                step_vals = algorithm.update_finetune(minibatches_device)
            elif args.unlearn_algo == 'ga':
                step_vals = algorithm.update_ga(minibatches_device)
            elif args.unlearn_algo == 'rl':
                step_vals = algorithm.update_rl(minibatches_device)
            elif args.unlearn_algo == 'boundary_shrink':
                step_vals = algorithm.update_boundary_shrink(minibatches_device)
            elif args.unlearn_algo == 'l1_sparse':
                step_vals = algorithm.update_l1_sparse(minibatches_device)

            checkpoint_vals['step_time'].append(time.time() - step_start_time)
            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

            if (step > 0 and step % eval_interval == 0) or (step == n_steps - 1):
                current_epoch = step // steps_per_epoch
                results = {'step': step, 'epoch': current_epoch}

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                results['test_acc'] = metrics.test_acc(algorithm, test_loader, device)
                results['retain_acc'] = metrics.retain_acc(algorithm, retain_test_loader, device)
                results['forget_acc'] = metrics.forget_acc(algorithm, forget_test_loader, device)
                results['mia_score'] = metrics.mia(algorithm, forget_test_loader, unseen_loader, device)

                results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)
                results['step_time'] = time.time() - step_start_time

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys], colwidth=12)

                # if wandb.run:
                #     wandb.log(results)

                epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                checkpoint_vals = collections.defaultdict(lambda: [])

                if args.save_model_every_checkpoint:
                    save_checkpoint(f'model_epoch{current_epoch}.pt')

                current_mia = results['mia_score']

                if (not args.no_mia_early_stop
                        and _mia_in_stop_band(
                            current_mia, args.mia_stop_low, args.mia_stop_high)):
                    print(f"\n[!] Early stopping triggered at step {step}!")
                    print(f"[*] Target MIA in [{args.mia_stop_low}, {args.mia_stop_high}]: {current_mia}")
                    break

        save_checkpoint('model_unlearned_final.pt')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
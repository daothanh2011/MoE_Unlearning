# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import copy
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
import splits

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


def _mia_in_stop_band(mia_score, low, high):
    """True if loss-based MIA attack accuracy is in [low, high] (chance = 0.5)."""
    return low <= mia_score <= high


# Checkpoints from train.py use these names; unlearn needs *Unlearn algorithm classes.
_UNLEARN_ALGORITHM_ALIASES = {
    'ERM': 'ERM_Unlearn',
    'GMOE_Full': 'GMOE_Full_Unlearn',
    'GMOE_ModularLearn': 'GMOE_Full_Unlearn',
}


def resolve_unlearn_algorithm_name(algorithm_name, unlearn_algo):
    """Map training algorithm to one that implements update_finetune / update_ga / …"""
    if unlearn_algo == 'modular':
        if algorithm_name not in ('GMOE_Full_Unlearn',):
            raise ValueError(
                f"modular unlearning requires --algorithm GMOE_Full_Unlearn "
                f"(got {algorithm_name!r}).")
        return algorithm_name
    alias = _UNLEARN_ALGORITHM_ALIASES.get(algorithm_name)
    return alias if alias is not None else algorithm_name


def _infer_gold_checkpoint_path(checkpoint_path):
    """Guess retrained gold checkpoint from origin train output path."""
    if checkpoint_path is None:
        return None
    for origin_tag in ("_origin_", "/origin_"):
        if origin_tag in checkpoint_path:
            candidate = checkpoint_path.replace(origin_tag, "_retrained_", 1)
            if os.path.isfile(candidate):
                return candidate
    return None


def _load_algorithm_network(algorithm_class, dataset, hparams, checkpoint_path, device):
    algo = algorithm_class(dataset.input_shape, dataset.num_classes, 1, hparams)
    state = torch.load(checkpoint_path, map_location="cpu")
    algo.network.load_state_dict(state)
    algo.to(device)
    algo.eval()
    return algo


def _apply_lotus_metrics(results, unlearned, original, gold, loaders, device, num_classes):
    """Merge LoTUS-style JSD / RF-JSD / Avg Gap into results dict."""
    forget_loader, retain_loader, test_loader, unseen_loader = loaders
    lotus = metrics.lotus_evaluation(
        unlearned,
        original,
        gold,
        forget_loader,
        retain_loader,
        test_loader,
        unseen_loader,
        device,
        num_classes=num_classes,
    )
    # Keep existing keys; add LoTUS metrics (overwrite acc/mia with same values).
    for key in (
        "forget_acc", "retain_acc", "test_acc", "mia_score",
        "js_forget", "rf_jsd",
        "mia_gap", "forget_acc_gap", "retain_acc_gap", "test_acc_gap", "avg_gap",
        "gold_forget_acc", "gold_retain_acc", "gold_test_acc", "gold_mia_score",
    ):
        if key in lotus and not (isinstance(lotus[key], float) and np.isnan(lotus[key])):
            results[key] = lotus[key]
    return results


if __name__ == "__main__":
    WANDB_PROJECT = "sparse_moe_unlearn"

    parser = argparse.ArgumentParser(description='Unlearning')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument(
        '--gold_checkpoint_path', type=str, default=None,
        help='Retrained gold-standard checkpoint (train.py --train_setting retrained). '
             'If omitted, tries to infer from --checkpoint_path by replacing origin→retrained.')
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

    ul_param = splits.output_ul_param(args)
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

    unlearn_algorithm = resolve_unlearn_algorithm_name(
        args.algorithm, args.unlearn_algo)
    if unlearn_algorithm != args.algorithm:
        print(f"[*] Unlearn algorithm: {args.algorithm} -> {unlearn_algorithm} "
              f"(checkpoint-compatible; adds {args.unlearn_algo} update methods)")

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(unlearn_algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            unlearn_algorithm, args.dataset,
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

    split_bundle = splits.build_unlearning_splits(dataset, args)
    train_transform, eval_transform = splits.get_transforms(
        args.dataset, split_bundle.protocol)

    retain_subset = split_bundle.retain_subset
    forget_subset = split_bundle.forget_subset

    retain_set_train = splits.ApplyTransform(retain_subset, transform=train_transform)
    forget_set_train = splits.ApplyTransform(forget_subset, transform=train_transform)
    retain_set_test = splits.ApplyTransform(retain_subset, transform=eval_transform)
    forget_set_test = splits.ApplyTransform(forget_subset, transform=eval_transform)
    test_set = splits.ApplyTransform(split_bundle.test_subset, transform=eval_transform)
    unseen_set = splits.ApplyTransform(split_bundle.unseen_subset, transform=eval_transform)

    retain_train_loader = InfiniteDataLoader(dataset=retain_set_train, weights=None, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)
    forget_train_loader = InfiniteDataLoader(dataset=forget_set_train, weights=None, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)

    retain_test_loader = FastDataLoader(dataset=retain_set_test, batch_size=64, num_workers=dataset.N_WORKERS)
    forget_test_loader = FastDataLoader(dataset=forget_set_test, batch_size=64, num_workers=dataset.N_WORKERS)
    unseen_loader = FastDataLoader(dataset=unseen_set, batch_size=64, num_workers=dataset.N_WORKERS)
    test_loader = FastDataLoader(dataset=test_set, batch_size=64, num_workers=dataset.N_WORKERS)

    algorithm_class = algorithms.get_algorithm_class(unlearn_algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                1, hparams) 

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        algorithm.network.load_state_dict(checkpoint)
    else:
        print("[WARNING] No --checkpoint_path provided for unlearning! Training from scratch?")

    algorithm.to(device)

    gold_checkpoint = args.gold_checkpoint_path or _infer_gold_checkpoint_path(args.checkpoint_path)
    original_algorithm = _load_algorithm_network(
        algorithm_class, dataset, hparams, args.checkpoint_path, device
    ) if args.checkpoint_path else None
    gold_algorithm = None
    if gold_checkpoint is not None:
        print(f"[*] Gold-standard (retrained) checkpoint: {gold_checkpoint}")
        gold_algorithm = _load_algorithm_network(
            algorithm_class, dataset, hparams, gold_checkpoint, device
        )
    else:
        print("[WARNING] No gold checkpoint — JSD vs gold and Avg Gap will be skipped. "
              "Train with: train.py --train_setting retrained ... then pass --gold_checkpoint_path")

    eval_loaders = (forget_test_loader, retain_test_loader, test_loader, unseen_loader)

    def _evaluate(results):
        if original_algorithm is None:
            results['test_acc'] = metrics.test_acc(algorithm, test_loader, device)
            results['retain_acc'] = metrics.retain_acc(algorithm, retain_test_loader, device)
            results['forget_acc'] = metrics.forget_acc(algorithm, forget_test_loader, device)
            results['mia_score'] = metrics.mia(algorithm, forget_test_loader, unseen_loader, device)
            return results
        return _apply_lotus_metrics(
            results, algorithm, original_algorithm, gold_algorithm,
            eval_loaders, device, dataset.num_classes,
        )

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

        results = _evaluate(results)

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

                    results = _evaluate(results)

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

                results = _evaluate(results)

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
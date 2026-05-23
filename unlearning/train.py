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

if __name__ == "__main__":
    WANDB_PROJECT = "sparse_moe_train"

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--debug', default="True")
    parser.add_argument('--train_setting', default='origin', choices=['origin', 'retrained'])
    
    parser.add_argument('--unlearn_setting', default='random', choices=['random', 'class'])
    parser.add_argument('--unlearn_random_ratio', default=None) # 0.1
    parser.add_argument('--unlearn_num_class', default=None) # 1

    parser.add_argument('--data_dir', type=str, default='./domainbed/data')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM") 
    parser.add_argument('--checkpoint_path', type=str, default=None)
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
                        help='Number of steps to run before triggering evaluation. Only applies to retrained mode.')

    parser.add_argument('--retrain_from_scratch', action='store_true',
                        help='Retrained mode: do not load --checkpoint_path (LoTUS gold trains from scratch).')

    args = parser.parse_args()

    ul_param = splits.output_ul_param(args)
    args.output_dir = f"unlearning/train_output/{args.algorithm}_{args.train_setting}_{args.dataset}_{args.unlearn_setting}_{ul_param}_seed_{args.seed}"

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
        run_name = f'{args.algorithm}_{args.train_setting}_{args.dataset}_{args.unlearn_setting}_{ul_param}_seed_{args.seed}'
        
        if len(run_name) > 128:
            run_name = run_name[:125] + '...'

        # wandb.init(
        #     project=WANDB_PROJECT,
        #     name=run_name,
        #     config={
        #         'dataset': args.dataset,
        #         'algorithm': args.algorithm,
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

    train_subset = split_bundle.train_subset
    test_subset = split_bundle.test_subset
    unseen_subset = split_bundle.unseen_subset
    retain_subset = split_bundle.retain_subset
    forget_subset = split_bundle.forget_subset

    train_set = splits.ApplyTransform(train_subset, transform=train_transform)
    test_set = splits.ApplyTransform(test_subset, transform=eval_transform)

    forget_set_train = splits.ApplyTransform(forget_subset, transform=train_transform)
    forget_set_test = splits.ApplyTransform(forget_subset, transform=eval_transform)
    retain_set_train = splits.ApplyTransform(retain_subset, transform=train_transform)
    retain_set_test = splits.ApplyTransform(retain_subset, transform=eval_transform)
    unseen_set = splits.ApplyTransform(unseen_subset, transform=eval_transform)

    if args.train_setting == 'retrained':
        active_train_set = retain_set_train
        train_loader = InfiniteDataLoader(dataset=retain_set_train, weights=None, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)
        test_loader = FastDataLoader(dataset=test_set, batch_size=64, num_workers=dataset.N_WORKERS)
        print("[*] Train Setting: RETRAINED -> Using retain_set for train and test.")
    else:
        active_train_set = train_set
        train_loader = InfiniteDataLoader(dataset=train_set, weights=None, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)
        test_loader = FastDataLoader(dataset=test_set, batch_size=64, num_workers=dataset.N_WORKERS)
        print("[*] Train Setting: ORIGIN -> Using original train and test sets.")
        
    retain_test_loader = FastDataLoader(dataset=retain_set_test, batch_size=64, num_workers=dataset.N_WORKERS)
    forget_test_loader = FastDataLoader(dataset=forget_set_test, batch_size=64, num_workers=dataset.N_WORKERS)
    unseen_loader = FastDataLoader(dataset=unseen_set, batch_size=64, num_workers=dataset.N_WORKERS)

    steps_per_epoch = max(1, int(len(active_train_set) / hparams['batch_size']))
    
    if args.train_setting == 'retrained' and args.num_step_per_evaluate is not None:
        eval_interval = args.num_step_per_evaluate
    else:
        eval_interval = steps_per_epoch

    train_minibatches_iterator = iter(train_loader)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                1, hparams) 

    if args.checkpoint_path is not None and not (
            args.train_setting == 'retrained' and args.retrain_from_scratch):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        algorithm.network.load_state_dict(checkpoint)
    elif args.train_setting == 'retrained' and args.retrain_from_scratch:
        print("[*] Retrained mode: training from scratch (LoTUS-style, no warm-start).")

    algorithm.to(device)

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_vals = collections.defaultdict(lambda: [])

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        network_state_dict = algorithm.network.state_dict()
        torch.save(network_state_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None

    print(f"[*] Starting Standard Training Loop (Evaluating every {eval_interval} steps)...")
    print(f"[*] Output directory set to: {args.output_dir}")
    
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        
        x, y = next(train_minibatches_iterator)
        minibatches_device = [(x.to(device), y.to(device))]
        
        step_vals = algorithm.update(minibatches_device)

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

            if args.train_setting == 'retrained':
                current_mia = results['mia_score']
                if (0.50 <= current_mia <= 0.51) or (50.0 <= current_mia <= 51.0):
                    print(f"\n[!] Early stopping triggered at step {step}!")
                    print(f"[*] Target MIA achieved: {current_mia}")
                    break

    save_checkpoint('model_final.pt')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
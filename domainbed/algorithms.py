# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
from itertools import chain

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch.autograd as autograd
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)
from copy import deepcopy
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_transformer
from collections import defaultdict, OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
# from domainbed import resnet_variants
import torchvision.models as models
from domainbed.losses.moe_specialization_losses import OrthoLoss, VarianceLoss
from domainbed.losses.matchdg_utils import *
from domainbed.losses.gmoe_utils import *
from domainbed.deit_transformer import *


ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'MatchDG',
    'GMoEOMoE',
    'GMOE_OMOE',
    'GMOE_InvA',
    'GMOE_InvB',
    'GMOE_Full',
    'GMOE_InvMMD',
    'GMOE_InvOT',
    'GMOE_InvAdv',
    'GMOE_InvED'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class MovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 100
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = (param_k.data * self.sma_count + param_q.data) / (1. + self.sma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = param_q.data


class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier).cuda()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ERM_CIRL(ERM):
    """
    ERM + CIRL causal-feature objectives.

    Architecture is identical to standard ERM:
        featurizer (ResNet / ViT / MLP)  ->  Linear classifier
    plus three CIRL-specific train-time additions:

        * a parallel `classifier_ad` head trained on the masked-out
          (non-causal) feature subset
        * a `Masker` that learns a soft top-k mask over feature dims
        * a Fourier amplitude mix done on-the-fly on the GPU

    Pipeline (per minibatch):

        1. Build a doubled batch
               x_full = [x_orig, x_aug]
           where x_aug is a Fourier amplitude mix of x_orig with random
           partners — same labels, perturbed style.

        2. Forward both halves through the featurizer:
               f = featurizer(x_full)        # (2B, D)

        3. The Masker M predicts a soft k-hot mask over f's D dims:
               f_sup = f * mask              # causal subset
               f_inf = f * (1 - mask)        # non-causal subset
           and feeds them through two parallel linear classifiers:
               classifier      -> L_cls_sup
               classifier_ad   -> L_cls_inf

        4. **Step 1** updates featurizer + both classifiers with
                L = 0.5 * (L_cls_sup + L_cls_inf)
                  + lambda_const * factorization_loss(f_orig, f_aug)
           (mask is .detach()'d here.)

        5. **Step 2** updates the masker only, with
                L_mask = 0.5 * L_cls_sup - 0.5 * L_cls_inf
           This pushes the masker to pick the dimensions that are most
           predictive (low L_cls_sup) while making the complementary
           subset *un*predictive (high L_cls_inf).

    For the first `cirl_warmup_epoch` epochs the mask is held at all-ones
    (no split), letting the encoder reach a sane init before the
    adversarial game begins. The factorization weight is also ramped up
    over the same window via sigmoid_rampup.

    Hyperparameters (registered in hparams_registry.py):
        cirl_alpha          (float, default 1.0)  -- Fourier mix strength
        cirl_ratio          (float, default 1.0)  -- Fourier mix spectrum crop
        cirl_k              (int,   default None) -- masker top-k; default 60% of feature dim
        cirl_lam_const      (float, default 5.0)  -- factorization weight (post-warmup)
        cirl_off_diag       (float, default 5e-3) -- Barlow-Twins lambda
        cirl_warmup_epoch   (int,   default 5)    -- warmup epochs (mask held at 1)
        cirl_warmup_total   (int,   default 5)    -- sigmoid ramp-up length
        cirl_masker_lr      (float, default None) -- separate LR for the masker; falls back to lr
        cirl_steps_per_epoch(int,   default 100)  -- approx. steps/epoch (drives warmup)

    At eval time only the featurizer + main `classifier` are used -- no
    mask, no Fourier mix. `predict()` is overridden to call the
    featurizer + classifier directly (skipping the original ERM
    `nn.Sequential` wrapper that would route through both pieces).
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Bypass ERM.__init__'s optimizer build -- we need our own optimizers
        # that exclude the masker. Call Algorithm.__init__ directly.
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'],
        )
        # The original ERM keeps a sequential reference around as `self.network`.
        # We keep one too for back-compat (e.g. checkpoint loading scripts that
        # look up `algorithm.network`), but it is not used in update().
        self.network = nn.Sequential(self.featurizer, self.classifier).cuda()

        # ---- CIRL hparams ----
        self.cirl_alpha = hparams.get('cirl_alpha', 1.0)
        self.cirl_ratio = hparams.get('cirl_ratio', 1.0)
        self.cirl_lam_const = hparams.get('cirl_lam_const', 5.0)
        self.cirl_off_diag = hparams.get('cirl_off_diag', 5e-3)
        self.cirl_warmup_epoch = int(hparams.get('cirl_warmup_epoch', 5))
        self.cirl_warmup_total = int(hparams.get('cirl_warmup_total', 5))

        feat_dim = self.featurizer.n_outputs
        cirl_k = hparams.get('cirl_k', None)

        self.masker = Masker(in_dim=feat_dim, k=cirl_k).cuda()

        # Adversarial classifier -- twin of `self.classifier`, trained on
        # the inferior (1 - mask) features. Always linear regardless of
        # `nonlinear_classifier`, matching CIRL's reference (which uses a
        # plain Linear for both heads).
        self.classifier_ad = nn.Linear(feat_dim, num_classes).cuda()

        # Two optimizers: one for everything except the masker, one for
        # the masker alone. Adversarial gradients should not leak into
        # the encoder/classifiers, and the encoder/classifier gradients
        # should not flow back into the masker either.
        main_params = (
                list(self.featurizer.parameters())
                + list(self.classifier.parameters())
                + list(self.classifier_ad.parameters())
        )
        self.optimizer = torch.optim.Adam(
            main_params,
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )

        masker_lr = hparams.get('cirl_masker_lr', None) or hparams['lr']
        self.masker_optim = torch.optim.Adam(
            self.masker.parameters(),
            lr=masker_lr,
            weight_decay=hparams['weight_decay'],
        )

        # Used by the warm-up / ramp-up schedules. We tick this in
        # update() based on a steps-per-epoch estimate so the algorithm
        # stays compatible with DomainBed's step-based train loop.
        self._step = 0
        self._steps_per_epoch = int(hparams.get('cirl_steps_per_epoch', 100))

    # -- helpers --------------------------------------------------------------

    @property
    def _current_epoch(self) -> float:
        return self._step / max(1, self._steps_per_epoch)

    # -- the two-step CIRL update --------------------------------------------

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # ---- 1. Build the doubled batch with Fourier-mixed augmentation ----
        x_aug = fourier_amplitude_mix_batched(
            all_x, alpha=self.cirl_alpha, ratio=self.cirl_ratio
        )
        x_full = torch.cat([all_x, x_aug], dim=0)
        y_full = torch.cat([all_y, all_y], dim=0)
        B = all_x.size(0)

        # =========================================================
        # Step 1: update featurizer + both classifiers
        # =========================================================
        self.optimizer.zero_grad()

        f = self.featurizer(x_full)  # (2B, D)

        # Mask. For the first warmup epochs we hold the mask at all-ones --
        # both classifiers see the full feature, no adversarial split yet.
        in_warmup = self._current_epoch < self.cirl_warmup_epoch
        if in_warmup:
            mask_sup = torch.ones_like(f)
        else:
            mask_sup = self.masker(f.detach())  # detach: encoder owns step 1
        mask_inf = 1.0 - mask_sup

        f_sup = f * mask_sup
        f_inf = f * mask_inf

        scores_sup = self.classifier(f_sup)
        scores_inf = self.classifier_ad(f_inf)

        loss_cls_sup = F.cross_entropy(scores_sup, y_full)
        loss_cls_inf = F.cross_entropy(scores_inf, y_full)

        # Factorization between original and augmented feature views.
        f_orig, f_aug = f[:B], f[B:]
        loss_fac = factorization_loss(
            f_orig, f_aug, off_diag_weight=self.cirl_off_diag
        )
        const_w = self.cirl_lam_const * sigmoid_rampup(
            self._current_epoch, self.cirl_warmup_total
        )

        loss = (
                0.5 * loss_cls_sup
                + 0.5 * loss_cls_inf
                + const_w * loss_fac
        )
        loss.backward()
        self.optimizer.step()

        # =========================================================
        # Step 2: update the masker adversarially (only after warmup)
        # =========================================================
        if not in_warmup:
            self.masker_optim.zero_grad()

            # Re-forward with no encoder grads -- the masker only ever sees
            # detached features. The classifiers' forward is differentiable
            # but we won't .step() the main optimizer in this block, so
            # their grads simply accumulate and are zeroed at the top of
            # the next update() call.
            with torch.no_grad():
                f2 = self.featurizer(x_full)

            mask_sup2 = self.masker(f2)  # masker IS differentiable
            mask_inf2 = 1.0 - mask_sup2
            scores_sup2 = self.classifier(f2 * mask_sup2)
            scores_inf2 = self.classifier_ad(f2 * mask_inf2)

            adv_loss = (
                    0.5 * F.cross_entropy(scores_sup2, y_full)
                    - 0.5 * F.cross_entropy(scores_inf2, y_full)
            )
            adv_loss.backward()
            self.masker_optim.step()

            adv_val = adv_loss.item()
        else:
            adv_val = 0.0

        self._step += 1

        return {
            'loss': loss.item(),
            'loss_cls_sup': loss_cls_sup.item(),
            'loss_cls_inf': loss_cls_inf.item(),
            'loss_fac': loss_fac.item(),
            'loss_adv': adv_val,
            'cirl_const_w': const_w,
            'in_warmup': float(in_warmup),
        }

    def predict(self, x):
        # Override ERM.predict() so eval skips the masker / classifier_ad
        # entirely and just runs featurizer -> classifier on the unmasked,
        # unaugmented input.
        return self.classifier(self.featurizer(x))


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MatchDG(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Bypass ERM.__init__'s optimizer build so we can set up two
        # parameter groups (Phase I and Phase II use different submodules).
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'],
        )
        # Kept around for back-compat with checkpoint scripts that look
        # up `algorithm.network`. Not used by update().
        self.network = nn.Sequential(self.featurizer, self.classifier).cuda()

        # ---- MatchDG hparams ----
        self.phase1_steps = int(hparams.get('matchdg_phase1_steps', 1500))
        self.lambda_match = float(hparams.get('matchdg_lambda_match', 1.0))
        self.tau = float(hparams.get('matchdg_tau', 0.1))
        self.match_update_freq = int(hparams.get('matchdg_match_update_freq', 100))

        # Phase-I projection head: featurizer.n_outputs -> proj_dim.
        proj_dim = hparams.get('matchdg_proj_dim', None) or self.featurizer.n_outputs
        self.projection_head = ProjectionHead(
            in_dim=self.featurizer.n_outputs,
            out_dim=proj_dim,
        ).cuda()

        # Two optimizers. Phase I trains featurizer + projection head;
        # the classifier is idle. Phase II trains featurizer + classifier;
        # the projection head is idle (and we don't actually call its
        # forward at all, so it just sits in the state dict).
        phase1_lr = hparams.get('matchdg_phase1_lr', None) or hparams['lr']
        self.optimizer_phase1 = torch.optim.Adam(
            list(self.featurizer.parameters())
            + list(self.projection_head.parameters()),
            lr=phase1_lr,
            weight_decay=hparams['weight_decay'],
        )
        self.optimizer_phase2 = torch.optim.Adam(
            list(self.featurizer.parameters())
            + list(self.classifier.parameters()),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )

        # Match table is built lazily on first use and refreshed every
        # match_update_freq steps. We don't precompute one before training
        # starts -- the very first update()  step will build one.
        self._step = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_ids(minibatches) -> torch.Tensor:
        """Build a (B,) domain-index tensor aligned with the concatenated batch."""
        ids = [
            torch.full((x.size(0),), d, dtype=torch.long)
            for d, (x, _) in enumerate(minibatches)
        ]
        return torch.cat(ids).to(minibatches[0][0].device)

    def _in_phase1(self) -> bool:
        return self._step < self.phase1_steps

    # ------------------------------------------------------------------
    # The MatchDG update
    # ------------------------------------------------------------------

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = self._domain_ids(minibatches)

        # ===========================================================
        # Phase I: contrastive learning of the representation
        # ===========================================================
        if self._in_phase1():
            self.optimizer_phase1.zero_grad()
            f = self.featurizer(all_x)                      # (B, D)
            z = self.projection_head(f)                     # (B, proj_dim)
            loss_con = supervised_contrastive_loss(
                z, all_y, all_d, tau=self.tau,
                require_cross_domain_positive=True,
            )
            loss_con.backward()
            self.optimizer_phase1.step()

            self._step += 1
            return {
                'loss':         loss_con.item(),
                'loss_con':     loss_con.item(),
                'phase':        1.0,
                'lambda_match': 0.0,
            }

        # ===========================================================
        # Phase II: ERM + matching L2 regularizer
        # ===========================================================
        # First Phase-II step (or any periodic re-match step): rebuild
        # the in-batch match table using *current* featurizer features.
        # Always rebuild on every step here -- the match table is
        # batch-local in our implementation, so it has to be recomputed
        # every step anyway. (`match_update_freq` is preserved as a hparam
        # for parity with the reference implementation, but in practice
        # affects nothing because batches change every step.)
        self.optimizer_phase2.zero_grad()
        f = self.featurizer(all_x)                          # (B, D)

        # Re-match in detached feature space (the matching itself is
        # non-differentiable; we only differentiate THROUGH the matched
        # features, not THROUGH the matching decision).
        with torch.no_grad():
            idx_a, idx_b = find_cross_domain_matches(
                f.detach(), all_y, all_d, exclude_self_domain=True
            )

        scores = self.classifier(f)
        loss_cls = F.cross_entropy(scores, all_y)
        loss_match = matching_l2_loss(f, idx_a, idx_b)
        loss = loss_cls + self.lambda_match * loss_match

        loss.backward()
        self.optimizer_phase2.step()

        self._step += 1
        return {
            'loss':         loss.item(),
            'loss_cls':     loss_cls.item(),
            'loss_match':   loss_match.item(),
            'phase':        2.0,
            'lambda_match': self.lambda_match,
            'n_matches':    int(idx_a.numel()),
        }

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def predict(self, x):
        """At eval time, skip the projection head and matching machinery."""
        return self.classifier(self.featurizer(x))


class GMOE(Algorithm):
    """
    SFMOE
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GMOE, self).__init__(input_shape, num_classes, num_domains, hparams)
        num_experts  = hparams.get('num_experts',       6)
        gate_k       = hparams.get('gate_k',            1)
        mlp_ratio    = hparams.get('mlp_ratio',         4.0)
        prune_ratio  = hparams.get('expert_prune_ratio', 0.0)
        expert_depth = hparams.get('expert_depth',      2)
        model_name   = hparams.get('model',             'deit_small_patch16_224')
        model_factory = getattr(vision_transformer, model_name)
        self.model = model_factory(pretrained=True, num_classes=num_classes, moe_layers=['F'] * 8 + ['S', 'F'] * 2, mlp_ratio=4.0, expert_mlp_ratio=mlp_ratio, num_experts=num_experts, gate_k=gate_k, prune_ratio=prune_ratio, is_tutel=True, drop_path_rate=0.1, router='cosine_top', expert_depth=expert_depth).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.ortho_loss_fn = OrthoLoss()
        self.variance_loss_fn = VarianceLoss()

    def _preprocess(self, x):
        """Resize to 224×224 and expand to 3 channels if needed (e.g. CMNIST 2×28×28)."""
        if x.shape[1] != 3:
            ch_mean = x.mean(dim=1, keepdim=True)
            x = torch.cat([x, ch_mean], dim=1)  # (B,2,H,W) → (B,3,H,W)
            if x.shape[1] != 3:  # fallback for other channel counts
                x = x[:, :3, :, :]
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        device = all_x.device
        loss = F.cross_entropy(self.predict(all_x), all_y)

        loss_aux      = torch.tensor(0., device=device)
        ortho_loss    = torch.tensor(0., device=device)
        variance_loss = torch.tensor(0., device=device)

        for block in self.model.blocks:
            if getattr(block, 'aux_loss', None) is not None:
                loss_aux = loss_aux + block.aux_loss

                if (block.expert_outputs is not None
                        and self.hparams.get('ortho_loss_weight', 0.0) > 0):
                    ortho_loss = ortho_loss + self.ortho_loss_fn(block.expert_outputs)

                if (block.routing_scores is not None
                        and self.hparams.get('variance_loss_weight', 0.0) > 0):
                    variance_loss = variance_loss + self.variance_loss_fn(block.routing_scores)

        loss = (loss
                + loss_aux
                + self.hparams.get('ortho_loss_weight', 0.0) * ortho_loss
                + self.hparams.get('variance_loss_weight', 0.0) * variance_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss':           loss.item(),
            'loss_aux':       loss_aux.item(),
            'loss_ortho':     ortho_loss.item(),
            'loss_variance':  variance_loss.item(),
        }

    def predict(self, x, forward_feature=False):
        x = self._preprocess(x)
        if forward_feature:
            return self.model.forward_features(x)
        else:
            prediction = self.model(x)
            if type(prediction) is tuple:
                return (prediction[0] + prediction[1]) / 2
            else:
                return prediction


class GMoEOMoE(GMOE):
    """GMOE + OMoE Gram-Schmidt orthogonalization.

    Always uses the custom PyTorch MoE path (force_custom_moe=True) so that
    OMoE can be applied for all expert_depth values including depth=2 (bypasses Tutel).

    Two new hparams gate the extra mechanisms:
      use_omoe         (bool, default False): enable Gram-Schmidt orthogonalization
      use_balance_loss (bool, default False): enable importance CV² auxiliary loss

    With both False this is a pure-PyTorch GMOE baseline — useful for isolating
    the effect of each mechanism in ablation comparisons.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Bypass GMOE.__init__ and call Algorithm.__init__ directly so we can
        # build the model with the extra flags.
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        num_experts       = hparams.get('num_experts',        6)
        gate_k            = hparams.get('gate_k',             1)
        mlp_ratio         = hparams.get('mlp_ratio',          4.0)
        prune_ratio       = hparams.get('expert_prune_ratio', 0.0)
        expert_depth      = hparams.get('expert_depth',       2)
        use_omoe          = hparams.get('use_omoe',           False)
        use_balance_loss  = hparams.get('use_balance_loss',   False)
        model_name        = hparams.get('model',              'deit_small_patch16_224')
        model_factory     = getattr(vision_transformer, model_name)
        self.model = model_factory(
            pretrained=True, num_classes=num_classes,
            moe_layers=['F'] * 8 + ['S', 'F'] * 2,
            mlp_ratio=4.0, expert_mlp_ratio=mlp_ratio,
            num_experts=num_experts, gate_k=gate_k,
            prune_ratio=prune_ratio, is_tutel=True,
            drop_path_rate=0.1, router='cosine_top',
            expert_depth=expert_depth,
            force_custom_moe=True,
            use_omoe=use_omoe,
            use_balance_loss=use_balance_loss,
        ).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
    # update() and predict() inherited from GMOE unchanged


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
                                          num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
                                             self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
             list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
             list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                   hparams, conditional=False, class_balance=False)


#
#
# class CDANN(AbstractDANN):
#     """Conditional DANN"""
#
#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(CDANN, self).__init__(input_shape, num_classes, num_domains,
#                                     hparams, conditional=True, class_balance=True)
#
#
class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                                                        >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Base class shared by all variants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GMoEVariantBase(nn.Module):
    """
    Shared backbone + MoE head used by all variants.

    Architecture (from the paper):
        z      = DeiTFeaturizer(x)          CLS token, 384-dim
        h_m    = Expert_m(z)               M small expert MLPs → r-dim each
        pi(x)  = softmax(Router(z))        soft routing weights (B, M)
        h(x)   = sum_m pi_m * h_m          weighted aggregation
        y_hat  = Classifier(h(x))

    Subclasses implement update() with their specific loss combination.
    """
    NUM_EXPERTS = 6
    EXPERT_DIM = 256  # r in the paper

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        self.num_domains = num_domains

        # ---- Architecture hparams (matches original GMOE / GMoEOMoE names) ----
        num_experts  = hparams.get('num_experts',        self.NUM_EXPERTS)
        gate_k       = hparams.get('gate_k',             1)
        mlp_ratio    = hparams.get('mlp_ratio',          4.0)
        prune_ratio  = hparams.get('expert_prune_ratio', 0.0)
        expert_depth = hparams.get('expert_depth',       2)
        model_name   = hparams.get('model',              'deit_small_patch16_224')
        freeze_backbone = hparams.get('freeze_backbone', False)

        # gate_k is accepted for CLI/sweep compatibility with the original GMOE
        # but does not apply to soft routing (our router is a weighted sum
        # over all experts).  Warn if someone asks for hard top-k.
        if gate_k != 1:
            print(f'[GMoEVariantBase] gate_k={gate_k} requested but soft routing '
                  f'uses all experts — gate_k is ignored.')

        # ---- Backbone ----
        self.featurizer = DeiTFeaturizer(
            model_name=model_name,
            pretrained=True,
        ).cuda()

        # Optional freeze — disables gradients on backbone params and forces
        # eval mode so BatchNorm / Dropout stats don't drift.  Params with
        # requires_grad=False are filtered out of the optimizer below so
        # Adam doesn't allocate moments for them (saves memory).
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.featurizer.parameters():
                p.requires_grad_(False)
            self.featurizer.eval()
            print(f'[{type(self).__name__}] backbone FROZEN — '
                  f'only moe_head (+ discriminators if any) will be trained')

        # ---- Explicit MoE head ----
        # expert_dim defaults to the ViT embed_dim so each expert outputs at
        # the same scale as the backbone features (matches the paper's r = D).
        expert_dim = hparams.get('expert_dim', self.featurizer.n_outputs)

        self.moe_head = ExplicitMoEHead(
            in_dim=self.featurizer.n_outputs,
            expert_dim=expert_dim,
            num_experts=num_experts,
            num_classes=num_classes,
            mlp_ratio=mlp_ratio,
            prune_ratio=prune_ratio,
            expert_depth=expert_depth,
        ).cuda()

        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.moe_head.parameters()),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )

        self._print_param_counts()

    def train(self, mode=True):
        """
        Override train() so a frozen backbone stays in eval mode even when
        the outer module is set to train.  nn.Module.train() normally flips
        everything, which would re-enable Dropout / BN updates in the
        frozen backbone.
        """
        super().train(mode)
        if getattr(self, 'freeze_backbone', False):
            self.featurizer.eval()
        return self

    def _count_params(self, module):
        """Return (total, trainable) parameter counts for a module."""
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _print_param_counts(self):
        """Log parameter counts for the full model + each component."""
        model_name = self.featurizer.model_name
        feat_t, feat_tr = self._count_params(self.featurizer)
        head_t, head_tr = self._count_params(self.moe_head)

        # Per-expert breakdown inside the MoE head
        expert_t = sum(p.numel() for p in self.moe_head.experts.parameters())
        router_t = sum(p.numel() for p in self.moe_head.router.parameters())
        clf_t = sum(p.numel() for p in self.moe_head.classifier.parameters())

        extra_t = extra_tr = 0
        if getattr(self, 'discriminators', None) is not None:
            extra_t, extra_tr = self._count_params(self.discriminators)

        total_t = feat_t + head_t + extra_t
        total_tr = feat_tr + head_tr + extra_tr

        def fmt(n):
            return f'{n:>13,}  ({n / 1e6:6.2f}M)'

        print(f'\n[{type(self).__name__}] param counts  —  backbone={model_name}')
        print(f'  num_experts={self.moe_head.num_experts}  '
              f'expert_dim={self.moe_head.expert_dim}  '
              f'expert_depth={self.moe_head.expert_depth}')
        frozen_tag = '  [FROZEN]' if getattr(self, 'freeze_backbone', False) else ''
        print(f'  featurizer (ViT){frozen_tag}: {fmt(feat_t)}  trainable={fmt(feat_tr)}')
        print(f'  moe_head total         : {fmt(head_t)}  trainable={fmt(head_tr)}')
        print(f'    ├─ experts ({self.moe_head.num_experts}×)         : {fmt(expert_t)}')
        print(f'    ├─ router              : {fmt(router_t)}')
        print(f'    └─ classifier          : {fmt(clf_t)}')
        if extra_t > 0:
            print(f'  discriminators         : {fmt(extra_t)}  trainable={fmt(extra_tr)}')
        print(f'  ─────────────────────────────────────────────')
        print(f'  TOTAL                  : {fmt(total_t)}  trainable={fmt(total_tr)}\n')

    def _forward(self, x):
        """Returns (logits, pi, h_stack)."""
        z = self.featurizer(x)
        return self.moe_head(z)

    def predict(self, x):
        logits, _, _ = self._forward(x)
        return logits

    def _get_domain_ids(self, minibatches):
        """Build a (B,) domain-index tensor aligned with the concatenated batch."""
        ids = [
            torch.full((x.size(0),), d, dtype=torch.long)
            for d, (x, _) in enumerate(minibatches)
        ]
        return torch.cat(ids).to(minibatches[0][0].device)

    def update(self, minibatches, unlabeled=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Variant A: expert-wise mean alignment (first-order invariance)
# ---------------------------------------------------------------------------

class GMOE_InvA(GMoEVariantBase):
    """
    L = L_cls + lambda_inv * L_inv^A

    L_inv^A = sum_m sum_c sum_{d != d'} || mu_{m,d,c} - mu_{m,d',c} ||^2

    Hparams:
        lambda_inv (float, default 0.1): weight on the invariance loss
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.lambda_inv = hparams.get('lambda_inv', 0.0)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        dom_id = self._get_domain_ids(minibatches)

        logits, pi, h_stack = self._forward(all_x)

        l_cls = F.cross_entropy(logits, all_y)
        l_inv = loss_inv_A(h_stack, pi, all_y, self.num_classes,
                           dom_id, self.num_domains)
        loss = l_cls + self.lambda_inv * l_inv

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'loss_cls': l_cls.item(),
            'loss_inv': l_inv.item(),
        }


# ---------------------------------------------------------------------------
# Variant B: expert-wise mean + covariance alignment (second-order invariance)
# ---------------------------------------------------------------------------

class GMOE_InvB(GMoEVariantBase):
    """
    L = L_cls + lambda_inv * L_inv^B

    L_inv^B = sum_{m,c} sum_{d != d'} (
        || mu_{m,d,c} - mu_{m,d',c} ||^2
      + alpha * || Sigma_{m,d,c} - Sigma_{m,d',c} ||_F^2
    )

    Hparams:
        lambda_inv (float, default 0.1): weight on the invariance loss
        alpha_cov  (float, default 0.1): weight on the covariance term inside L_inv^B
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.lambda_inv = hparams.get('lambda_inv', 0.1)
        self.alpha_cov = hparams.get('alpha_cov', 0.1)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        dom_id = self._get_domain_ids(minibatches)

        logits, pi, h_stack = self._forward(all_x)

        l_cls = F.cross_entropy(logits, all_y)
        l_inv = loss_inv_B(h_stack, pi, all_y, self.num_classes,
                           dom_id, self.num_domains, alpha=self.alpha_cov)
        loss = l_cls + self.lambda_inv * l_inv

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'loss_cls': l_cls.item(),
            'loss_inv': l_inv.item(),
        }


# ---------------------------------------------------------------------------
# Variant Full: all six loss terms
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Variant Full: unified objective over multiple invariance types
# ---------------------------------------------------------------------------

class GMOE_Full(GMoEVariantBase):
    """
    Unified full-objective variant.  Dispatches the invariance term across
    five variants via the `inv_type` hparam:

        inv_type = 'A'     → Option A  (first-order mean alignment)
                   'B'     → Option B  (mean + covariance alignment)
                   'MMD'   → conditional MMD²  (multi-bandwidth RBF)
                   'OT'    → conditional entropic Wasserstein (Sinkhorn)
                   'ED'    → conditional energy distance

    Total objective:
        L = L_cls
          + lambda_inv * L_inv_<inv_type>
          + lambda_sp  * L_sp    (sparsity / routing entropy)
          + lambda_bal * L_bal   (load balancing)
          + lambda_div * L_div   (expert diversity)

    (Conditional-independence term L_cind is intentionally removed.)

    Common hparams:
        inv_type    (str,   default 'A')
        lambda_inv  (float, default 0.1)
        lambda_sp   (float, default 0.01)
        lambda_bal  (float, default 0.01)
        lambda_div  (float, default 0.01)

    Variant-specific hparams (only those matching inv_type are used):
        B    : alpha_cov       (float, default 0.1)
        MMD  : alpha           (float, default 4.0)
               mmd_sigmas      (tuple, default (1,2,4,8,16))
        OT   : alpha           (float, default 4.0)
               ot_epsilon      (float, default 0.1)
               sinkhorn_iters  (int,   default 50)
        ED   : alpha           (float, default 4.0)

    Legacy compatibility:
        Setting `use_inv_b=True` is still accepted and overrides inv_type
        to 'B' for backwards compatibility with earlier experiments.
    """

    _VALID_INV_TYPES = {'A', 'B', 'MMD', 'OT', 'ED'}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        # --- common weights ---
        self.lambda_inv = hparams.get('lambda_inv', 0.01)
        self.lambda_sp  = hparams.get('lambda_sp',  0.01)
        self.lambda_bal = hparams.get('lambda_bal', 0.01)
        self.lambda_div = hparams.get('lambda_div', 0.01)

        # --- resolve inv_type (with legacy use_inv_b shim) ---
        inv_type = hparams.get('inv_type', 'A')
        if inv_type not in self._VALID_INV_TYPES:
            raise ValueError(
                f'inv_type={inv_type!r} not in {sorted(self._VALID_INV_TYPES)}')
        self.inv_type = inv_type

        # --- variant-specific hparams (only the relevant ones are read) ---
        self.alpha_cov      = hparams.get('alpha_cov',      0.1)
        self.alpha          = hparams.get('alpha',          4.0)
        self.sigmas         = tuple(hparams.get('mmd_sigmas', (1., 2., 4., 8., 16.)))
        self.epsilon        = hparams.get('ot_epsilon',     0.1)
        self.sinkhorn_iters = hparams.get('sinkhorn_iters', 50)

    def _compute_invariance(self, h_stack, pi, all_y, dom_id):
        """Dispatch to the selected invariance loss."""
        C, D = self.num_classes, self.num_domains
        if self.inv_type == 'A':
            return loss_inv_A(h_stack, pi, all_y, C, dom_id, D)
        if self.inv_type == 'B':
            return loss_inv_B(h_stack, pi, all_y, C, dom_id, D,
                              alpha=self.alpha_cov)
        if self.inv_type == 'MMD':
            return loss_inv_MMD(h_stack, pi, all_y, C, dom_id, D,
                                alpha=self.alpha, sigmas=self.sigmas)
        if self.inv_type == 'OT':
            return loss_inv_OT(h_stack, pi, all_y, C, dom_id, D,
                               alpha=self.alpha,
                               epsilon=self.epsilon,
                               sinkhorn_iters=self.sinkhorn_iters)
        if self.inv_type == 'ED':
            return loss_inv_ED(h_stack, pi, all_y, C, dom_id, D,
                               alpha=self.alpha)
        raise RuntimeError(f'unreachable: inv_type={self.inv_type}')

    def update(self, minibatches, unlabeled=None):
        all_x  = torch.cat([x for x, y in minibatches])
        all_y  = torch.cat([y for x, y in minibatches])
        dom_id = self._get_domain_ids(minibatches)

        logits, pi, h_stack = self._forward(all_x)

        l_cls = F.cross_entropy(logits, all_y)
        l_inv = self._compute_invariance(h_stack, pi, all_y, dom_id)
        l_sp  = loss_sparse(pi)
        l_bal = loss_balance(pi)
        l_div = loss_diversity(h_stack)

        loss = (l_cls
                + self.lambda_inv * l_inv
                + self.lambda_sp  * l_sp
                + self.lambda_bal * l_bal
                + self.lambda_div * l_div)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss':     loss.item(),
            'loss_cls': l_cls.item(),
            'loss_inv': l_inv.item(),
            'loss_sp':  l_sp.item(),
            'loss_bal': l_bal.item(),
            'loss_div': l_div.item(),
        }


# ===========================================================================
# Subset-aware invariance variants
# ===========================================================================
#
# All four variants replace the first-order mean alignment (Option A) of
# GMOE_InvA with a stronger distributional-matching loss.  They share the
# same skeleton — classification + invariance — and differ only in the
# invariance objective.

# from domainbed.gmoe_utils import (
#     loss_inv_MMD,
#     loss_inv_OT,
#     loss_inv_Adv,
#     loss_inv_ED,
#     ConditionalDomainDiscriminators,
# )


# ---------------------------------------------------------------------------
# GMOE_InvMMD — conditional MMD
# ---------------------------------------------------------------------------

class GMOE_InvMMD(GMoEVariantBase):
    """
    L = L_cls + lambda_inv * L_inv^MMD

    Hparams:
        lambda_inv (float, default 0.1)
        alpha      (float, default 4.0)   routing-weight temperature
        mmd_sigmas (tuple,  default (1,2,4,8,16))
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.lambda_inv = hparams.get('lambda_inv', 0.1)
        self.alpha = hparams.get('alpha', 4.0)
        self.sigmas = tuple(hparams.get('mmd_sigmas', (1., 2., 4., 8., 16.)))

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        dom_id = self._get_domain_ids(minibatches)

        logits, pi, h_stack = self._forward(all_x)

        l_cls = F.cross_entropy(logits, all_y)
        l_inv = loss_inv_MMD(h_stack, pi, all_y, self.num_classes,
                             dom_id, self.num_domains,
                             alpha=self.alpha, sigmas=self.sigmas)
        loss = l_cls + self.lambda_inv * l_inv

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_cls': l_cls.item(), 'loss_inv': l_inv.item()}


# ---------------------------------------------------------------------------
# GMOE_InvOT — entropic optimal transport
# ---------------------------------------------------------------------------

class GMOE_InvOT(GMoEVariantBase):
    """
    L = L_cls + lambda_inv * L_inv^OT

    Hparams:
        lambda_inv     (float, default 0.1)
        alpha          (float, default 4.0)
        ot_epsilon     (float, default 0.1)   Sinkhorn entropy reg
        sinkhorn_iters (int,   default 50)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.lambda_inv = hparams.get('lambda_inv', 0.1)
        self.alpha = hparams.get('alpha', 4.0)
        self.epsilon = hparams.get('ot_epsilon', 0.1)
        self.sinkhorn_iters = hparams.get('sinkhorn_iters', 50)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        dom_id = self._get_domain_ids(minibatches)

        logits, pi, h_stack = self._forward(all_x)

        l_cls = F.cross_entropy(logits, all_y)
        l_inv = loss_inv_OT(h_stack, pi, all_y, self.num_classes,
                            dom_id, self.num_domains,
                            alpha=self.alpha,
                            epsilon=self.epsilon,
                            sinkhorn_iters=self.sinkhorn_iters)
        loss = l_cls + self.lambda_inv * l_inv

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_cls': l_cls.item(), 'loss_inv': l_inv.item()}


# ---------------------------------------------------------------------------
# GMOE_InvAdv — conditional adversarial alignment
# ---------------------------------------------------------------------------

class GMOE_InvAdv(GMoEVariantBase):
    """
    L = L_cls + lambda_inv * L_inv^Adv

    The discriminators are trained jointly with the rest of the network via
    a gradient-reversal layer, so a single optimiser step updates both.

    Hparams:
        lambda_inv (float, default 0.1)
        alpha      (float, default 4.0)
        grl_lambda (float, default 1.0)   gradient-reversal scale
        disc_hidden(int,   default 128)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.lambda_inv = hparams.get('lambda_inv', 0.1)
        self.alpha = hparams.get('alpha', 4.0)
        self.grl_lambda = hparams.get('grl_lambda', 1.0)

        self.discriminators = ConditionalDomainDiscriminators(
            num_experts=self.NUM_EXPERTS,
            num_classes=num_classes,
            feat_dim=self.EXPERT_DIM,
            num_domains=num_domains,
            hidden=hparams.get('disc_hidden', 128),
        ).cuda()

        # Re-create the optimiser to include discriminator params
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters())
            + list(self.moe_head.parameters())
            + list(self.discriminators.parameters()),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        dom_id = self._get_domain_ids(minibatches)

        logits, pi, h_stack = self._forward(all_x)

        l_cls = F.cross_entropy(logits, all_y)
        l_inv = loss_inv_Adv(h_stack, pi, all_y, self.num_classes,
                             dom_id, self.num_domains,
                             self.discriminators,
                             alpha=self.alpha,
                             grl_lambda=self.grl_lambda)
        loss = l_cls + self.lambda_inv * l_inv

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_cls': l_cls.item(), 'loss_inv': l_inv.item()}


# ---------------------------------------------------------------------------
# GMOE_InvED — energy distance
# ---------------------------------------------------------------------------

class GMOE_InvED(GMoEVariantBase):
    """
    L = L_cls + lambda_inv * L_inv^ED

    Hparams:
        lambda_inv (float, default 0.1)
        alpha      (float, default 4.0)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.lambda_inv = hparams.get('lambda_inv', 0.1)
        self.alpha = hparams.get('alpha', 4.0)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        dom_id = self._get_domain_ids(minibatches)

        logits, pi, h_stack = self._forward(all_x)

        l_cls = F.cross_entropy(logits, all_y)
        l_inv = loss_inv_ED(h_stack, pi, all_y, self.num_classes,
                            dom_id, self.num_domains,
                            alpha=self.alpha)
        loss = l_cls + self.lambda_inv * l_inv

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_cls': l_cls.item(), 'loss_inv': l_inv.item()}
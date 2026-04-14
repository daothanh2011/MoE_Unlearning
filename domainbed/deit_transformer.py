# gmoe_utils.py
# Shared building blocks for GMoE variants:
#   - DeiTFeaturizer   — pretrained DeiT-small backbone (CLS token output)
#   - ExplicitMoEHead  — M expert MLPs + soft router + classifier
#   - loss functions   — inv_A, inv_B, sparse, balance, diversity, cond_independence

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import vision_transformer as vit_module


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class DeiTFeaturizer(nn.Module):
    """
    Wraps the repo's VisionTransformer (DeiT-small config) to expose the
    CLS-token feature vector.  All layers are dense ('F') — no MoE, no Tutel.
    Instantiates VisionTransformer directly, bypassing timm's model registry
    so there is no conflict with the repo's @register_model decorators.
    embed_dim = 384 for deit_small.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = vit_module.VisionTransformer(
            img_size=224, patch_size=16, in_chans=3,
            num_classes=0,           # no head — returns raw features
            embed_dim=384, depth=12, num_heads=6,
            mlp_ratio=4., qkv_bias=True,
            distilled=True,          # DeiT-small checkpoint has distillation token
            drop_path_rate=0.1,
            moe_layers=['F'] * 12,   # all-dense, no Tutel
            num_experts=1,
            router='cosine_top',
        )
        self.n_outputs = 384

        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
                map_location='cpu', check_hash=True,
            )
            state = checkpoint.get('model', checkpoint)
            # Drop classifier head keys — not present when num_classes=0
            state = {k: v for k, v in state.items()
                     if not k.startswith('head')}
            missing, _ = self.vit.load_state_dict(state, strict=False)
            if missing:
                print(f'[DeiTFeaturizer] missing keys (expected if head removed): {missing}')

    def forward(self, x):
        # forward_features returns (cls, dist) tuple for distilled DeiT
        out = self.vit.forward_features(x)
        if isinstance(out, tuple):
            return out[0]   # CLS token → (B, 384)
        return out


# ---------------------------------------------------------------------------
# Explicit MoE head
# ---------------------------------------------------------------------------

class ExplicitMoEHead(nn.Module):
    """
    Implements:
        h_m = E_m(z)          M expert MLPs  (z → r-dim)
        pi(x) = softmax(G(z)) soft routing weights
        h(x)  = sum_m pi_m * h_m
        y_hat = C(h(x))       linear classifier

    Returns per-expert outputs h_m and routing weights pi so that the
    variant-specific loss functions can operate on them.
    """
    def __init__(self, in_dim, expert_dim, num_experts, num_classes):
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim  = expert_dim

        # M expert MLPs: each z → r
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim),
            )
            for _ in range(num_experts)
        ])

        # Soft routing network: z → (M,)
        self.router = nn.Linear(in_dim, num_experts, bias=True)

        # Final classifier: r → num_classes
        self.classifier = nn.Linear(expert_dim, num_classes)

    def forward(self, z):
        """
        Args:
            z: (B, in_dim)
        Returns:
            logits:  (B, num_classes)
            pi:      (B, M)     routing weights
            h_stack: (B, M, r)  per-expert representations
        """
        h_list  = [E(z) for E in self.experts]     # M × (B, r)
        h_stack = torch.stack(h_list, dim=1)        # (B, M, r)

        pi = F.softmax(self.router(z), dim=-1)      # (B, M)
        h  = (pi.unsqueeze(-1) * h_stack).sum(dim=1)  # (B, r)

        logits = self.classifier(h)
        return logits, pi, h_stack


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _routed_class_mean(h_m, pi_m, y, num_classes):
    """
    Routing-weighted class-conditional mean for a single expert.

    Args:
        h_m:  (B, r)  expert m output
        pi_m: (B,)    routing weight for expert m
        y:    (B,)    class labels
    Returns:
        mu:   (C, r)  — zero rows where class has no samples in this batch
    """
    r  = h_m.size(1)
    mu = h_m.new_zeros(num_classes, r)
    for c in range(num_classes):
        mask  = (y == c).float()
        w     = pi_m * mask
        denom = w.sum()
        if denom > 1e-8:
            mu[c] = (w.unsqueeze(-1) * h_m).sum(0) / denom
    return mu


def _routed_class_cov(h_m, pi_m, mu_m, y, num_classes):
    """
    Routing-weighted class-conditional covariance for a single expert.

    Args:
        h_m:  (B, r)
        pi_m: (B,)
        mu_m: (C, r)  class means from _routed_class_mean
        y:    (B,)
    Returns:
        Sigma: (C, r, r)
    """
    r     = h_m.size(1)
    Sigma = h_m.new_zeros(num_classes, r, r)
    for c in range(num_classes):
        mask  = (y == c).float()
        w     = pi_m * mask
        denom = w.sum()
        if denom > 1e-8:
            diff     = h_m - mu_m[c]                          # (B, r)
            Sigma[c] = (w.unsqueeze(-1).unsqueeze(-1)
                        * diff.unsqueeze(-1) * diff.unsqueeze(-2)
                       ).sum(0) / denom
    return Sigma


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_inv_A(h_stack, pi, y, num_classes, domain_ids, num_domains):
    """
    Option A — Expert-wise first-order (mean) alignment.

    L_inv^A = sum_m sum_c sum_{d != d'} || mu_{m,d,c} - mu_{m,d',c} ||^2

    Args:
        h_stack:    (B, M, r)
        pi:         (B, M)
        y:          (B,)   class labels
        domain_ids: (B,)   domain indices in [0, num_domains)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        mu_per_domain = []
        for d in range(num_domains):
            mask_d = (domain_ids == d)
            if mask_d.sum() == 0:
                mu_per_domain.append(None)
                continue
            mu_per_domain.append(
                _routed_class_mean(h_m[mask_d], pi_m[mask_d], y[mask_d], num_classes)
            )

        for d in range(num_domains):
            if mu_per_domain[d] is None:
                continue
            for dp in range(d + 1, num_domains):
                if mu_per_domain[dp] is None:
                    continue
                diff = mu_per_domain[d] - mu_per_domain[dp]   # (C, r)
                loss = loss + (diff ** 2).sum()

    return loss


def loss_inv_B(h_stack, pi, y, num_classes, domain_ids, num_domains, alpha=1.0):
    """
    Option B — Expert-wise second-order (mean + covariance) alignment.

    L_inv^B = sum_{m,c} sum_{d != d'} (
        || mu_{m,d,c} - mu_{m,d',c} ||^2
      + alpha * || Sigma_{m,d,c} - Sigma_{m,d',c} ||_F^2
    )

    Args:
        alpha: weight on the covariance term (default 1.0)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        mu_per_domain    = []
        sigma_per_domain = []
        for d in range(num_domains):
            mask_d = (domain_ids == d)
            if mask_d.sum() == 0:
                mu_per_domain.append(None)
                sigma_per_domain.append(None)
                continue
            mu_d    = _routed_class_mean(h_m[mask_d], pi_m[mask_d], y[mask_d], num_classes)
            sigma_d = _routed_class_cov(h_m[mask_d], pi_m[mask_d], mu_d, y[mask_d], num_classes)
            mu_per_domain.append(mu_d)
            sigma_per_domain.append(sigma_d)

        for d in range(num_domains):
            if mu_per_domain[d] is None:
                continue
            for dp in range(d + 1, num_domains):
                if mu_per_domain[dp] is None:
                    continue
                mu_diff    = mu_per_domain[d] - mu_per_domain[dp]
                sigma_diff = sigma_per_domain[d] - sigma_per_domain[dp]
                loss = loss + (mu_diff ** 2).sum()
                loss = loss + alpha * (sigma_diff ** 2).sum()

    return loss


def loss_sparse(pi):
    """
    Sparsity penalty — minimises routing entropy to encourage peaked routing.

    L_sp = E_x [ -sum_m pi_m log pi_m ]
    (minimising this minimises entropy → more concentrated routing)

    Args:
        pi: (B, M)
    """
    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)   # (B,)
    return entropy.mean()


def loss_balance(pi):
    """
    Load balancing — penalises deviation of mean routing from uniform 1/M.

    L_bal = sum_m ( E[pi_m] - 1/M )^2

    Args:
        pi: (B, M)
    """
    M    = pi.size(1)
    mean = pi.mean(dim=0)                     # (M,)
    return ((mean - 1.0 / M) ** 2).sum()


def loss_diversity(h_stack):
    """
    Expert diversity — minimises cross-expert batch correlation.

    L_div = sum_{m != n} || (1/B) H_m^T H_n ||_F^2

    Args:
        h_stack: (B, M, r)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()
    for m in range(M):
        for n in range(M):
            if m == n:
                continue
            C = (h_stack[:, m, :].T @ h_stack[:, n, :]) / B   # (r, r)
            loss = loss + (C ** 2).sum()
    return loss


def loss_cond_independence(h_stack, y, num_classes):
    """
    Conditional independence — minimises class-conditional cross-expert correlation.

    L_cind = sum_c sum_{m != n} || C_mn^(c) ||_F^2

    where C_mn^(c) is the class-conditional cross-correlation matrix between
    experts m and n.

    Args:
        h_stack: (B, M, r)
        y:       (B,)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()

    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        H_c = h_stack[mask]           # (B_c, M, r)
        B_c = H_c.size(0)
        for m in range(M):
            for n in range(M):
                if m == n:
                    continue
                Hm = H_c[:, m, :] - H_c[:, m, :].mean(0)
                Hn = H_c[:, n, :] - H_c[:, n, :].mean(0)
                C_mn = (Hm.T @ Hn) / max(B_c - 1, 1)   # (r, r)
                loss = loss + (C_mn ** 2).sum()

    return loss
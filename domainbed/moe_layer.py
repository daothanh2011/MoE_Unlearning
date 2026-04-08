"""
TransparentMoELayer — a drop-in replacement for Tutel's moe_layer that exposes the
intermediate routing scores and per-expert output tensors needed to compute Lo and Lv.

Replicates Tutel's CosineTopKGate + load_importance_loss exactly so that
pre-trained weights (gate parameters, expert FFN weights) remain compatible.

Architecture mirrors Tutel's setup used by GMoE:
  gate:     CosineTopKGate (proj_dim=256, fp32_gate=True, init_t=0.5)
  experts:  n_experts x FFN(D -> H -> D, GELU + Dropout(0.1))
  routing:  top-k with capacity_factor=1.5, batch-prioritized, gate_noise=1.0
  l_aux:    load_importance_loss (is_gshard_loss=False)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


# ---------------------------------------------------------------------------
# CosineGate — exact replica of tutel/gates/cosine_top.py::CosineTopKGate
# ---------------------------------------------------------------------------

class CosineGate(nn.Module):
    """
    Computes routing logits via cosine similarity with a learnable temperature.

    Parameters match Tutel's CosineTopKGate so that state-dict keys can be
    mapped when loading a Tutel-trained checkpoint.
    """

    def __init__(self, model_dim: int, num_experts: int,
                 proj_dim: int = 256, init_t: float = 0.5,
                 fp32_gate: bool = True):
        super().__init__()
        self.fp32_gate = fp32_gate
        self.clamp_max = math.log(1.0 / 0.01)          # log(100) ≈ 4.605

        # Learnable temperature (same init as Tutel: log(1/init_t) = log(2))
        self.temperature = nn.Parameter(
            torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True
        )
        # Linear projection into the cosine-similarity latent space
        self.cosine_projector = nn.Linear(model_dim, proj_dim)  # with bias (Tutel default)
        # Expert similarity matrix — init N(0, 0.01) exactly as Tutel does
        self.sim_matrix = nn.Parameter(torch.randn(proj_dim, num_experts))
        nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        returns logits: [N, E]  (fp32)
        """
        if self.fp32_gate:
            x = x.float()
            cp = self.cosine_projector.float()
            sm = self.sim_matrix.float()
        else:
            cp = self.cosine_projector
            sm = self.sim_matrix

        logits = torch.matmul(
            F.normalize(cp(x), dim=1),   # [N, proj_dim]
            F.normalize(sm, dim=0),       # [proj_dim, E]
        )                                  # [N, E]
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        return logits * logit_scale        # [N, E]


# ---------------------------------------------------------------------------
# Auxiliary loss — exact replica of tutel/impls/losses.py::load_importance_loss
# (GMoE uses is_gshard_loss=False and gate_noise=1.0)
# ---------------------------------------------------------------------------

def _load_importance_loss(scores_no_noise: torch.Tensor,
                           topk_logits_noisy: torch.Tensor,
                           num_experts: int,
                           gate_noise: float) -> torch.Tensor:
    """
    Replica of Tutel's load_importance_loss.

    Args:
        scores_no_noise:   softmax(logits_clean),    shape [N, E]
        topk_logits_noisy: logits_noisy at top-k positions, shape [N, k]
        num_experts:       E
        gate_noise:        gate noise std scale (1.0 in GMoE)
    Returns:
        scalar loss tensor
    """
    # ---- importance loss: variance of per-expert importance sums ----
    Impi = scores_no_noise.float().sum(dim=0)            # [E]
    l_imp = Impi.var() / (Impi.mean() ** 2 + 1e-10)

    # ---- load loss: variance of expected per-expert token load ----
    normal = Normal(
        torch.tensor([0.0], device=scores_no_noise.device),
        torch.tensor([gate_noise / num_experts], device=scores_no_noise.device),
    )
    threshold = topk_logits_noisy[:, -1].view(-1, 1).float()  # [N, 1] — min top-k noisy logit
    diff = scores_no_noise.float() - threshold                  # [N, E]
    prob = normal.cdf(diff)                                     # [N, E]
    Load = prob.sum(dim=0)                                      # [E]
    l_load = Load.var() / (Load.mean() ** 2 + 1e-10)

    return (l_imp + l_load) / 2.0


# ---------------------------------------------------------------------------
# TransparentMoELayer
# ---------------------------------------------------------------------------

class TransparentMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts FFN layer that exposes intermediate tensors for
    computing the orthogonality loss (Lo) and variance loss (Lv).

    After each forward() call the following attributes are set:
        self.l_aux          — scalar Tensor: load-importance auxiliary loss
        self.routing_scores — Tensor [N, E]: pre-top-k clean softmax probabilities
        self.expert_outputs — Tensor [N, E, D]: per-expert FFN outputs;
                               zero for experts not selected or capacity-overflowed

    Args:
        model_dim             input/output feature dimension (D)
        n_experts             number of experts (E)
        hidden_size_per_expert  inner FFN dimension (H)
        top_k                 number of experts per token (default 1)
        capacity_factor       max tokens per expert relative to average (default 1.5)
        gate_noise            noise std scale added during training (default 1.0)
        proj_dim              projection dimension inside CosineGate (default 256)
        init_temperature      initial temperature for CosineGate (default 0.5)
        fp32_gate             run gate in fp32 (default True, matching Tutel)
        dropout               dropout probability inside expert FFN (default 0.1)
    """

    def __init__(self,
                 model_dim: int,
                 n_experts: int,
                 hidden_size_per_expert: int,
                 top_k: int = 1,
                 capacity_factor: float = 1.5,
                 gate_noise: float = 1.0,
                 proj_dim: int = 256,
                 init_temperature: float = 0.5,
                 fp32_gate: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate_noise = gate_noise
        self.model_dim = model_dim
        self.hidden_size = hidden_size_per_expert

        E, H, D = n_experts, hidden_size_per_expert, model_dim

        # CosineTopKGate
        self.gate = CosineGate(D, E, proj_dim, init_temperature, fp32_gate)

        # Expert FFNs stored as batched weight matrices: [E, H, D] and [E, D, H]
        # Initialised identically to nn.Linear (kaiming uniform + uniform bias)
        self.fc1_weight = nn.Parameter(torch.empty(E, H, D))
        self.fc1_bias   = nn.Parameter(torch.zeros(E, H))
        self.fc2_weight = nn.Parameter(torch.empty(E, D, H))
        self.fc2_bias   = nn.Parameter(torch.zeros(E, D))

        for e in range(E):
            nn.init.kaiming_uniform_(self.fc1_weight[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.fc2_weight[e], a=math.sqrt(5))
            bound1 = 1.0 / math.sqrt(D)
            nn.init.uniform_(self.fc1_bias[e], -bound1, bound1)
            bound2 = 1.0 / math.sqrt(H)
            nn.init.uniform_(self.fc2_bias[e], -bound2, bound2)

        self.act_dropout = nn.Dropout(dropout)

        # Output attributes (set during forward)
        self.l_aux: torch.Tensor = None
        self.routing_scores: torch.Tensor = None
        self.expert_outputs: torch.Tensor = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]  (ViT sequence of patch embeddings)
        returns: [B, T, D]
        """
        orig_dtype = x.dtype
        B, T, D = x.shape
        N = B * T
        E = self.n_experts

        x_flat = x.reshape(N, D)   # [N, D]

        # ----------------------------------------------------------
        # 1. Routing
        # ----------------------------------------------------------
        logits = self.gate(x_flat)   # [N, E], fp32

        # Clean softmax (no noise) — used for l_aux importance term and for Lv
        scores_no_noise = F.softmax(logits, dim=1)   # [N, E], fp32
        self.routing_scores = scores_no_noise         # keep in graph for Lv gradient

        # Noisy logits for top-k selection during training
        if self.training and self.gate_noise > 0:
            noise = self.gate_noise * torch.randn_like(logits) / E
            logits_noisy = logits + noise
        else:
            logits_noisy = logits
        scores_noisy = F.softmax(logits_noisy, dim=1)   # [N, E], fp32

        # ----------------------------------------------------------
        # 2. Top-k selection
        # ----------------------------------------------------------
        top_vals, top_indices = torch.topk(scores_noisy, self.top_k, dim=1)
        # [N, k] each

        # Noisy logits at selected positions — for load_importance_loss threshold
        top_logits_noisy = logits_noisy.gather(1, top_indices)   # [N, k]

        # ----------------------------------------------------------
        # 3. Auxiliary load-importance loss (replica of Tutel)
        # ----------------------------------------------------------
        self.l_aux = _load_importance_loss(
            scores_no_noise, top_logits_noisy, E, self.gate_noise
        )

        # ----------------------------------------------------------
        # 4. Build gate-weight matrix [N, E] with capacity enforcement
        # ----------------------------------------------------------
        gate_weights = torch.zeros(N, E, dtype=scores_noisy.dtype,
                                   device=x.device)
        gate_weights.scatter_(1, top_indices, top_vals)   # [N, E]

        # Batch-prioritized capacity: per expert, keep only the top-`capacity`
        # tokens (by routing score); overflow tokens get zeroed out.
        capacity = max(1, int(math.ceil(self.capacity_factor * N / E)))
        for j in range(E):
            col = gate_weights[:, j]              # [N]
            nonzero_count = int((col > 0).sum().item())
            if nonzero_count > capacity:
                # Sort descending; zero out tokens beyond capacity
                sorted_idx = torch.argsort(col, descending=True)
                overflow_idx = sorted_idx[capacity:]
                gate_weights[overflow_idx, j] = 0.0

        # ----------------------------------------------------------
        # 5. Run all E expert FFNs on all N tokens (vectorized)
        #    fc1: [N, D]  -> [N, E, H]
        #    act: GELU + Dropout
        #    fc2: [N, E, H] -> [N, E, D]
        # ----------------------------------------------------------
        x_f = x_flat.float()   # ensure fp32 for expert computation

        # hidden[n, e, h] = sum_d x_f[n, d] * fc1_weight[e, h, d] + fc1_bias[e, h]
        hidden = torch.einsum('nd,ehd->neh', x_f, self.fc1_weight.float()) \
                 + self.fc1_bias.float().unsqueeze(0)          # [N, E, H]
        hidden = self.act_dropout(F.gelu(hidden))

        # all_expert_out[n, e, d] = sum_h hidden[n,e,h]*fc2_weight[e,d,h] + fc2_bias[e,d]
        all_expert_out = torch.einsum('neh,edh->ned', hidden, self.fc2_weight.float()) \
                         + self.fc2_bias.float().unsqueeze(0)  # [N, E, D]

        # ----------------------------------------------------------
        # 6. Separate binary selection mask from gate-score weights:
        #
        #   expert_outputs  (for Lo): raw FFN output * BINARY indicator I{sij>0}
        #       — gradient flows only to FFN params, NOT to gate params,
        #         matching paper equations (6) and (8).
        #
        #   For model output: weighted combination using actual gate scores.
        # ----------------------------------------------------------
        # Binary indicator: 1 where token i selected expert j, else 0.
        # The comparison gate_weights>0 is non-differentiable, so gradients
        # from Lo cannot flow through selection_mask to gate parameters.
        selection_mask = (gate_weights > 0).float()               # [N, E], no grad path to gate

        # expert_outputs[i, j, :] = FFN_j(x_i) if selected, else 0
        self.expert_outputs = all_expert_out * selection_mask.unsqueeze(-1)  # [N, E, D]

        # ----------------------------------------------------------
        # 7. Aggregate: gate-score–weighted sum over experts (model output)
        # ----------------------------------------------------------
        # Use gate_weights (actual routing scores, not binary mask) for model output
        y = (all_expert_out * gate_weights.unsqueeze(-1)).sum(dim=1)   # [N, D]
        y = y.to(orig_dtype).reshape(B, T, D)
        return y

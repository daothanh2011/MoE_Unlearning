import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import linear_model, model_selection

def retain_acc(network, retain_loader, device):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y in retain_loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)

            correct += (p.argmax(1) == y).sum().item()
            total += x.size(0)
            
    network.train()
    return correct / total if total > 0 else 0.0

def test_acc(network, test_loader, device):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)

            correct += (p.argmax(1) == y).sum().item()
            total += x.size(0)
            
    network.train()
    return correct / total if total > 0 else 0.0

def forget_acc(network, forget_loader, device):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y in forget_loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)

            correct += (p.argmax(1) == y).sum().item()
            total += x.size(0)
            
    network.train()
    return correct / total if total > 0 else 0.0

def mia(network, forget_loader, unseen_loader, device):
    criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_losses(loader):
        all_losses = []
        network.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                
                logits = network.predict(x)
                
                losses = criterion(logits, y).cpu().detach().numpy()
                all_losses.extend(losses)
        return np.array(all_losses)

    forget_losses = compute_losses(forget_loader)
    unseen_losses = compute_losses(unseen_loader)

    min_len = min(len(forget_losses), len(unseen_losses))
    forget_losses = forget_losses[:min_len]
    unseen_losses = unseen_losses[:min_len]

    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    
    labels_mia = [0] * min_len + [1] * min_len

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=10, random_state=42)
    
    mia_scores = model_selection.cross_val_score(
        attack_model, samples_mia, labels_mia, cv=cv, scoring="accuracy"
    )
    
    mia_mean = mia_scores.mean()
    network.train()

    return mia_mean


def js_divergence(p, q, epsilon=1e-8):
    """Jensen-Shannon divergence between probability tensors (LoTUS / BadT formulation)."""
    p = torch.clamp(p, min=epsilon, max=1.0)
    q = torch.clamp(q, min=epsilon, max=1.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(p.log(), m, reduction="batchmean")
        + F.kl_div(q.log(), m, reduction="batchmean")
    )


def _collect_softmax_probs(network, loader, device):
    """Stack softmax outputs for all samples in loader."""
    probs = []
    network.eval()
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            logits = network.predict(x)
            probs.append(F.softmax(logits, dim=1).detach().cpu())
    network.train()
    if not probs:
        return torch.empty(0, 0)
    return torch.cat(probs, dim=0)


def js_forget(unlearned, gold, forget_loader, device):
    """
    JSD on forget set: mean distributional gap vs gold-standard model (LoTUS ``log_js``).
    JSD = (1/|D_f|) sum_x JS(f_un(x), f_gold(x)) - implemented via batch JS on stacked probs.
    """
    un_probs = _collect_softmax_probs(unlearned, forget_loader, device)
    gold_probs = _collect_softmax_probs(gold, forget_loader, device)
    if un_probs.numel() == 0 or gold_probs.numel() == 0:
        return float("nan")
    return float(js_divergence(un_probs, gold_probs).item())


def _class_mean_probs(network, loader, device):
    """Per-class mean softmax vectors, L1-normalized (for RF-JSD)."""
    class_probs = {}
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = network.predict(x)
            batch_probs = F.softmax(logits, dim=1).detach().cpu()
            for i, label in enumerate(y):
                label = int(label.item() if isinstance(label, torch.Tensor) else label)
                class_probs.setdefault(label, []).append(batch_probs[i])
    network.train()

    mean_probs = {}
    for label, vectors in class_probs.items():
        mean = torch.stack(vectors).mean(dim=0)
        mean_probs[label] = mean / mean.sum().clamp(min=1e-8)
    return mean_probs


def rf_jsd(unlearned, original, forget_loader, unseen_loader, device, num_classes=None):
    """
    Retrain-Free JSD (RF-JSD): class-wise mean probs on forget (unlearned) vs unseen (original).
    Does not require a retrained gold model.
    """
    forget_means = _class_mean_probs(unlearned, forget_loader, device)
    unseen_means = _class_mean_probs(original, unseen_loader, device)

    if num_classes is not None:
        for c in range(num_classes):
            if c not in forget_means:
                unseen_means.pop(c, None)

    common = sorted(set(forget_means.keys()) & set(unseen_means.keys()))
    if not common:
        return float("nan")

    p = torch.stack([forget_means[c] for c in common])
    q = torch.stack([unseen_means[c] for c in common])
    return float(js_divergence(p, q).item())


def classification_bundle(network, forget_loader, retain_loader, test_loader, unseen_loader, device):
    """Standard accuracy + MIA metrics used in LoTUS / SSD-style evaluation."""
    return {
        "forget_acc": forget_acc(network, forget_loader, device),
        "retain_acc": retain_acc(network, retain_loader, device),
        "test_acc": test_acc(network, test_loader, device),
        "mia_score": mia(network, forget_loader, unseen_loader, device),
    }


def avg_gap(unlearned_metrics, gold_metrics):
    """
    Average Gap vs gold-standard (retrained) model [Chundawat et al. / LoTUS paper]:
      Avg Gap = (|d_MIA| + |d_f| + |d_r| + |d_t|) / 4
    Lower is closer to gold on all four criteria.
    """
    deltas = {
        "mia_gap": abs(unlearned_metrics["mia_score"] - gold_metrics["mia_score"]),
        "forget_acc_gap": abs(unlearned_metrics["forget_acc"] - gold_metrics["forget_acc"]),
        "retain_acc_gap": abs(unlearned_metrics["retain_acc"] - gold_metrics["retain_acc"]),
        "test_acc_gap": abs(unlearned_metrics["test_acc"] - gold_metrics["test_acc"]),
    }
    deltas["avg_gap"] = (
        deltas["mia_gap"]
        + deltas["forget_acc_gap"]
        + deltas["retain_acc_gap"]
        + deltas["test_acc_gap"]
    ) / 4.0
    return deltas


def lotus_evaluation(
    unlearned,
    original,
    gold,
    forget_loader,
    retain_loader,
    test_loader,
    unseen_loader,
    device,
    num_classes=None,
):
    """
    Full LoTUS-style metric dict for an unlearned model.

    Gold-standard model f_gold: retrained from scratch on retain set only (train.py --train_setting retrained).
    Original f_pre: full model before unlearning (for RF-JSD; same weights as unlearn start checkpoint).
    """
    out = classification_bundle(
        unlearned, forget_loader, retain_loader, test_loader, unseen_loader, device
    )
    out["rf_jsd"] = rf_jsd(
        unlearned, original, forget_loader, unseen_loader, device, num_classes=num_classes
    )
    if gold is not None:
        gold_m = classification_bundle(
            gold, forget_loader, retain_loader, test_loader, unseen_loader, device
        )
        out["js_forget"] = js_forget(unlearned, gold, forget_loader, device)
        out.update({f"gold_{k}": v for k, v in gold_m.items()})
        out.update(avg_gap(out, gold_m))
    else:
        out["js_forget"] = float("nan")
    return out
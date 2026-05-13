import hashlib
import operator
import sys
from collections import Counter
from collections import OrderedDict
from datetime import datetime
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
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
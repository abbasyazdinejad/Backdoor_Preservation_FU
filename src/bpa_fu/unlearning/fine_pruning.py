"""Fine-Pruning (Liu, Dolan-Gavitt and Garg, RAID 2018) adapted to the federated setting.

Post-hoc backdoor mitigation, not an unlearning method: it is included to answer whether an
ordinary mitigation with the same data access as retained-data fine-tuning removes the backdoor.

1. Pruning. The mean post-ReLU activation of every channel of the last convolutional stage is
   computed on the retained clients' data (each client can compute this statistic locally and the
   server aggregates it; the simulation computes it on the union). Channels are pruned in increasing
   order of activation until the accuracy on the retained clients' data drops by more than
   `max_acc_drop` relative to the unpruned model (the 4% rule of the paper).
   * SimpleCNN: channels of conv2 (weights and bias set to zero) -- unchanged from the main study.
   * ResNet-18 (CIFAR variant): channels of the final feature map produced by the last residual stage
     (layer4 output, 512 channels). Because of the residual connection a channel cannot be removed by
     zeroing one convolution; the channel is masked at the stage output during pruning and recovery and,
     for a self-contained checkpoint, the corresponding classifier input columns are set to zero, which is
     exactly equivalent on the forward path (global average pooling and the classifier are linear).
2. Fine-tuning. The pruned model is fine-tuned with `recovery_rounds` FedAvg rounds on the retained
   clients, identical to retained-data fine-tuning. Pruned channels are masked during fine-tuning
   (their output is multiplied by zero, so their weights receive no gradient and stay at zero); the
   saved checkpoint therefore needs no mask.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from ..federated_training import LocalTrainConfig, run_fedavg


def _pruning_target(model: nn.Module):
    """Return ("conv2", module) for SimpleCNN or ("layer4", module) for ResNet-18."""
    if hasattr(model, "conv2") and hasattr(model, "fc1"):
        return "conv2", model.conv2
    if hasattr(model, "layer4") and hasattr(model, "fc"):
        return "layer4", model.layer4
    raise ValueError("fine_pruning supports SimpleCNN (conv2) and ResNet18CIFAR (layer4 feature channels)")


@torch.no_grad()
def _channel_activation_and_accuracy(model: nn.Module, loader: DataLoader, device) -> tuple[torch.Tensor, float]:
    """Mean post-ReLU channel activation of the pruning target and clean accuracy on `loader`."""
    model.eval()
    target, module = _pruning_target(model)
    acts = None
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if target == "conv2":
            h = model.pool(F.relu(model.conv1(x)))
            a = F.relu(model.conv2(h))
            out = model.fc2(F.relu(model.fc1(model.pool(a).flatten(1))))
        else:
            a = model.features(x)                      # layer4 output, already post-ReLU
            out = model.fc(F.adaptive_avg_pool2d(a, 1).flatten(1))
        s = a.mean(dim=(2, 3)).sum(dim=0)
        acts = s if acts is None else acts + s
        correct += int((out.argmax(1) == y).sum()); n += int(y.numel())
    return acts / n, correct / n


@torch.no_grad()
def _pooled_features(model: nn.Module, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval(); fs, ys = [], []
    for x, y in loader:
        fs.append(F.adaptive_avg_pool2d(model.features(x.to(device)), 1).flatten(1)); ys.append(y.to(device))
    return torch.cat(fs), torch.cat(ys)


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum()); n += int(y.numel())
    return correct / n


def fine_pruning(model: nn.Module, client_datasets, retained_clients: list[int], deleted_clients: list[int], cfg: LocalTrainConfig, device, seed: int,
                 max_acc_drop: float = 0.04, recovery_rounds: int = 5, log=None, **kwargs) -> nn.Module:
    target, module = _pruning_target(model)
    model.to(device)
    data = ConcatDataset([client_datasets[c] for c in retained_clients])
    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=0)
    acts, base_acc = _channel_activation_and_accuracy(model, loader, device)
    order = torch.argsort(acts).tolist()  # least active first
    n_channels = int(acts.numel())
    pruned: list[int] = []
    mask = torch.ones(n_channels, device=device)
    if target == "conv2":
        w, b = model.conv2.weight, model.conv2.bias
        for ch in order:
            w_bak, b_bak = w[ch].clone(), b[ch].clone()
            with torch.no_grad():
                w[ch].zero_(); b[ch].zero_()
            acc = _accuracy(model, loader, device)
            if base_acc - acc > max_acc_drop:
                with torch.no_grad():
                    w[ch].copy_(w_bak); b[ch].copy_(b_bak)
                break
            pruned.append(ch)
        mask[pruned] = 0.0
        handle = module.register_forward_hook(lambda m, i, o: o * mask.view(1, -1, 1, 1))
    else:
        # ResNet path: the mask acts on the pooled final-stage features, so pruning accuracy can be evaluated exactly
        # from cached pooled features and the classifier (global average pooling and the classifier are linear).
        feats, labels = _pooled_features(model, loader, device)
        W, bvec = model.fc.weight.detach(), model.fc.bias.detach()
        def masked_acc(mask_):
            return float((((feats * mask_) @ W.T + bvec).argmax(1) == labels).float().mean())
        for ch in order:
            mask[ch] = 0.0
            if base_acc - masked_acc(mask) > max_acc_drop:
                mask[ch] = 1.0
                break
            pruned.append(ch)
        handle = module.register_forward_hook(lambda m, i, o: o * mask.view(1, -1, 1, 1))
    if log is not None:
        log(f"fine-pruning: pruned {len(pruned)}/{n_channels} channels of {target} (retained-data accuracy {base_acc:.4f} -> {_accuracy(model, loader, device):.4f})")
    try:
        if recovery_rounds > 0:
            model, _ = run_fedavg(model, client_datasets, retained_clients, recovery_rounds, cfg, device, seed=seed + 19, log=log)
    finally:
        handle.remove()
    with torch.no_grad():  # pruned channels received no gradient; make the checkpoint self-contained
        if target == "conv2":
            model.conv2.weight[pruned] = 0.0; model.conv2.bias[pruned] = 0.0
        else:
            model.fc.weight[:, pruned] = 0.0   # equivalent to masking the pooled feature channels
    model.n_pruned_channels = len(pruned)  # type: ignore[attr-defined]
    model.pruned_layer = target  # type: ignore[attr-defined]
    return model

"""Gradient negation: bounded gradient ascent on the identified data, then federated recovery.

Following the projected-gradient-ascent formulation of Halimi et al. (2022), the
unlearning party maximises the training loss on the data of the deleted clients while
keeping the parameters inside an L2 ball of radius `radius` around the compromised model
theta* (projection after every step). Ascent stops as soon as the mean loss on the deleted
data exceeds `loss_threshold` (default: loss_threshold_mult * ln(K), where ln(K) is the loss
of a uniform predictor over K classes; the multiplier 4 was selected on a pilot run, see
logs/pilot/negation_sweep.csv) or after `ascent_epochs` epochs, whichever comes first. The ascent is followed by
`recovery_rounds` FedAvg rounds with the retained clients to recover utility.

Unlike short retraining and FedEraser, this method requires access to the identified
(deleted) samples; in our threat model this corresponds to oracle identification of the
malicious contribution after a security alert.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from ..federated_training import LocalTrainConfig, make_optimizer, run_fedavg


def _project(model: nn.Module, ref: dict[str, torch.Tensor], radius: float) -> float:
    with torch.no_grad():
        params = dict(model.named_parameters())
        dist = float(sum(((params[k] - ref[k]) ** 2).sum() for k in params).sqrt())
        if dist > radius:
            scale = radius / dist
            for k in params:
                params[k].copy_(ref[k] + scale * (params[k] - ref[k]))
            return radius
        return dist


@torch.no_grad()
def _mean_loss(model: nn.Module, loader: DataLoader, device) -> float:
    crit = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot += float(crit(model(x), y))
        n += int(y.numel())
    model.train()
    return tot / n


def gradient_negation(model: nn.Module, client_datasets, retained_clients: list[int], deleted_clients: list[int], cfg: LocalTrainConfig, device, seed: int,
                      ascent_epochs: int = 5, ascent_lr: float = 0.002, radius: float = 1.0, loss_threshold: float | None = None,
                      recovery_rounds: int = 5, check_every: int = 1, num_classes: int | None = None, loss_threshold_mult: float = 4.0, log=None, **kwargs) -> nn.Module:
    del_data = ConcatDataset([client_datasets[c] for c in deleted_clients])
    model.to(device)
    ref = {k: v.detach().clone() for k, v in model.named_parameters()}
    gen = torch.Generator().manual_seed(seed + 271)
    loader = DataLoader(del_data, batch_size=cfg.batch_size, shuffle=True, generator=gen, num_workers=0)
    eval_loader = DataLoader(del_data, batch_size=512, shuffle=False, num_workers=0)
    if loss_threshold is None:
        if num_classes is None:
            raise ValueError("num_classes or loss_threshold is required")
        loss_threshold = loss_threshold_mult * math.log(num_classes)
    asc_cfg = LocalTrainConfig(local_epochs=ascent_epochs, batch_size=cfg.batch_size, optimizer="sgd", lr=ascent_lr, momentum=0.0)
    opt = make_optimizer(model, asc_cfg)
    crit = nn.CrossEntropyLoss()
    model.train()
    steps = 0
    stopped = False
    for ep in range(ascent_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = -crit(model(x), y)
            loss.backward()
            opt.step()
            dist = _project(model, ref, radius)
            steps += 1
            if steps % check_every == 0:
                cur = _mean_loss(model, eval_loader, device)
                if cur >= loss_threshold:
                    stopped = True
                    break
        cur = _mean_loss(model, eval_loader, device)
        if log is not None:
            log(f"negation ascent epoch {ep + 1}/{ascent_epochs}: steps={steps} loss_del={cur:.3f} dist={dist:.3f} (threshold {loss_threshold:.3f})")
        if stopped or cur >= loss_threshold:
            break
    if recovery_rounds > 0:
        model, _ = run_fedavg(model, client_datasets, retained_clients, recovery_rounds, cfg, device, seed=seed + 13, log=log)
    return model

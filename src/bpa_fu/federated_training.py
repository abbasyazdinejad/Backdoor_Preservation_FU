"""FedAvg simulation with optional recording of per-client update history.

The server holds a global model theta^(t). In every round every selected client copies the
global model, runs `local_epochs` epochs of SGD on its local dataset and returns the
update delta_i = theta_i - theta^(t). The server aggregates with sample-size weights
(McMahan et al., 2017):  theta^(t+1) = theta^(t) + sum_i (n_i / n) * delta_i.

When `history_every` is set, the update of every participating client is stored every
`history_every` rounds (FedEraser's retaining interval Delta_t), together with the global
model at that round, so that FedEraser can later be executed faithfully.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import TensorImageDataset
from .models import clone_state


@dataclass
class LocalTrainConfig:
    local_epochs: int = 2
    batch_size: int = 64
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0


@dataclass
class UpdateHistory:
    """Stored client updates (deltas) and global states at retained rounds."""

    rounds: list[int] = field(default_factory=list)
    global_states: list[dict[str, torch.Tensor]] = field(default_factory=list)
    client_updates: list[dict[int, dict[str, torch.Tensor]]] = field(default_factory=list)
    client_sizes: list[dict[int, int]] = field(default_factory=list)

    def to_cpu(self) -> "UpdateHistory":
        return UpdateHistory(
            rounds=list(self.rounds),
            global_states=[{k: v.cpu() for k, v in s.items()} for s in self.global_states],
            client_updates=[{c: {k: v.cpu() for k, v in u.items()} for c, u in r.items()} for r in self.client_updates],
            client_sizes=[dict(d) for d in self.client_sizes],
        )


def make_optimizer(model: nn.Module, cfg: LocalTrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def local_train(
    model: nn.Module,
    dataset: TensorImageDataset,
    cfg: LocalTrainConfig,
    device: torch.device,
    epochs: int | None = None,
    generator: torch.Generator | None = None,
    maximize: bool = False,
) -> nn.Module:
    """Run local SGD epochs in place. `maximize=True` performs gradient ascent (used by negation)."""
    epochs = cfg.local_epochs if epochs is None else epochs
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, generator=generator, num_workers=0)
    opt = make_optimizer(model, cfg)
    crit = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            if maximize:
                loss = -loss
            loss.backward()
            opt.step()
    return model


def state_delta(new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: (new[k] - old[k]) for k in old}


def fedavg_aggregate(global_state: dict[str, torch.Tensor], updates: dict[int, dict[str, torch.Tensor]], sizes: dict[int, int]) -> dict[str, torch.Tensor]:
    """theta + sum_i w_i * delta_i with w_i = n_i / sum n."""
    total = float(sum(sizes[c] for c in updates))
    new = {k: v.clone() for k, v in global_state.items()}
    for c, upd in updates.items():
        w = sizes[c] / total
        for k in new:
            if new[k].dtype.is_floating_point:
                new[k] += w * upd[k].to(new[k].device)
    return new


def run_fedavg(
    model: nn.Module,
    client_datasets: list[TensorImageDataset],
    participating: list[int],
    rounds: int,
    cfg: LocalTrainConfig,
    device: torch.device,
    seed: int,
    clients_per_round: float = 1.0,
    history_every: int | None = None,
    round_callback: Callable[[int, nn.Module], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[nn.Module, UpdateHistory | None]:
    """Train `model` in place with FedAvg over the `participating` client ids."""
    rng = np.random.RandomState(seed + 104729)
    gen = torch.Generator().manual_seed(seed + 15485863)
    hist = UpdateHistory() if history_every else None
    sizes = {c: len(client_datasets[c]) for c in participating}
    model.to(device)
    for t in range(rounds):
        t0 = time.time()
        global_state = clone_state(model)
        m = max(1, int(round(clients_per_round * len(participating))))
        selected = sorted(rng.choice(participating, size=m, replace=False).tolist()) if m < len(participating) else list(participating)
        updates: dict[int, dict[str, torch.Tensor]] = {}
        for c in selected:
            model.load_state_dict(global_state)
            local_train(model, client_datasets[c], cfg, device, generator=gen)
            updates[c] = state_delta(clone_state(model), global_state)
        new_state = fedavg_aggregate(global_state, updates, sizes)
        model.load_state_dict(new_state)
        if hist is not None and (t % history_every == 0):
            hist.rounds.append(t)
            hist.global_states.append({k: v.cpu() for k, v in global_state.items()})
            hist.client_updates.append({c: {k: v.cpu() for k, v in u.items()} for c, u in updates.items()})
            hist.client_sizes.append({c: sizes[c] for c in updates})
        if round_callback is not None:
            round_callback(t, model)
        if log is not None:
            log(f"round {t + 1}/{rounds} clients={len(selected)} time={time.time() - t0:.1f}s")
    return model, hist

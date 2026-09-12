"""FedEraser (Liu et al., IWQoS 2021), client-level unlearning; see audit/federaser_protocol_audit.md.

Requirements: during the original training the server stored, every Delta_t rounds, the global model and the update
(delta) of every participating client (UpdateHistory, rounds t_1 = 0, t_2 = Delta_t, ...).

Reconstruction (Algorithm 1 of the paper, with the tensor-wise norm of the public reproduction):
  theta_u <- stored global model of the first retained round (uncontaminated initial model)
  first retained round: theta_u <- theta_u + sum_{i in R} w_i U_i^{t_1}   (no calibration, Sec. III-B-1 of the paper)
  every later retained round t_j:
      (1) calibration training: each retained client i trains theta_u for E_cal = ceil(r * E) epochs and returns U^_i
      (2) calibration, per state-dict tensor:  U~_i = ||U_i^{t_j}|| * U^_i / ||U^_i||      (Eq. 1)
      (3) aggregation with w_i = n_i / sum_{k in R} n_k:  theta_u <- theta_u + sum_i w_i U~_i   (Eqs. 2-3)
The deleted clients' stored updates are discarded and their data is never accessed.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from ..federated_training import LocalTrainConfig, UpdateHistory, local_train, state_delta
from ..models import clone_state


def calibrate_update(stored: dict[str, torch.Tensor], calib: dict[str, torch.Tensor], eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Eq. (1) applied per state-dict tensor: magnitude of the stored update, direction of the calibration update."""
    out = {}
    for k in stored:
        s, c = stored[k].to(calib[k].device), calib[k]
        if not s.dtype.is_floating_point:
            out[k] = c.clone()
            continue
        out[k] = s.norm() * c / (c.norm() + eps)
    return out


def federaser_reconstruct(history: UpdateHistory, retained: list[int], calibrate_fn: Callable[[dict[str, torch.Tensor], int], dict[str, torch.Tensor]],
                          first_round_uncalibrated: bool = True, log=None) -> dict[str, torch.Tensor]:
    """Core of Algorithm 1. `calibrate_fn(theta_u, client)` returns the client's calibration update U^ from theta_u.
    Returns the reconstructed state dict. Pure function of the history and calibrate_fn (used by the unit test)."""
    if len(history.rounds) == 0:
        raise ValueError("FedEraser requires the stored update history of the original training")
    retained_set = set(retained)
    theta_u = {k: v.clone() for k, v in history.global_states[0].items()}
    for j, r in enumerate(history.rounds):
        stored, sizes = history.client_updates[j], history.client_sizes[j]
        parts = [c for c in stored if c in retained_set]
        if not parts:
            raise ValueError(f"no retained client update stored at round {r}")
        total = float(sum(sizes[c] for c in parts))
        new_state = {k: v.clone() for k, v in theta_u.items()}
        for c in parts:
            w = sizes[c] / total
            if j == 0 and first_round_uncalibrated:
                upd = {k: stored[c][k].to(theta_u[k].device) for k in theta_u}
            else:
                upd = calibrate_update(stored[c], calibrate_fn(theta_u, c))
            for k in new_state:
                if new_state[k].dtype.is_floating_point:
                    new_state[k] += w * upd[k]
        theta_u = new_state
        if log is not None:
            log(f"federaser round {r} ({j + 1}/{len(history.rounds)}): {len(parts)} retained clients, {'uncalibrated' if (j == 0 and first_round_uncalibrated) else 'calibrated'}")
    return theta_u


def federaser(model: nn.Module, client_datasets, retained_clients: list[int], deleted_clients: list[int], cfg: LocalTrainConfig, device, seed: int,
              history: UpdateHistory | None = None, calibration_ratio: float = 0.5, first_round_uncalibrated: bool = True, log=None, **kwargs) -> nn.Module:
    if history is None or len(history.rounds) == 0:
        raise ValueError("FedEraser requires the stored update history of the original training")
    e_cal = max(1, int(math.ceil(calibration_ratio * cfg.local_epochs)))
    gen = torch.Generator().manual_seed(seed + 1009)
    model.to(device)

    def calibrate_fn(theta_u: dict[str, torch.Tensor], c: int) -> dict[str, torch.Tensor]:
        model.load_state_dict(theta_u)
        local_train(model, client_datasets[c], cfg, device, epochs=e_cal, generator=gen)
        return state_delta(clone_state(model), theta_u)

    hist_dev = UpdateHistory(rounds=list(history.rounds), global_states=[{k: v.to(device) for k, v in history.global_states[0].items()}] + history.global_states[1:],
                             client_updates=history.client_updates, client_sizes=history.client_sizes)
    theta_u = federaser_reconstruct(hist_dev, retained_clients, calibrate_fn, first_round_uncalibrated=first_round_uncalibrated, log=log)
    model.load_state_dict(theta_u)
    return model

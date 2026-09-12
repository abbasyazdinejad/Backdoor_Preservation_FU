"""Full retraining from scratch on the retained clients (counterfactual reference).

The model is re-initialised with the same seed-derived initialisation procedure as the
original training and trained with the identical FedAvg protocol (same rounds, local
epochs, optimiser) using only the retained clients. This is the model that would have been
obtained had the deleted clients never participated; we use its ASR as the measured
counterfactual security reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..federated_training import LocalTrainConfig, run_fedavg


def full_retraining(model: nn.Module, client_datasets, retained_clients: list[int], deleted_clients: list[int], cfg: LocalTrainConfig, device, seed: int, rounds: int, model_factory=None, log=None, **kwargs) -> nn.Module:
    if model_factory is None:
        raise ValueError("full_retraining needs a model_factory to re-initialise the model")
    torch.manual_seed(seed + 2003)
    fresh = model_factory().to(device)
    fresh, _ = run_fedavg(fresh, client_datasets, retained_clients, rounds, cfg, device, seed=seed + 17, log=log)
    return fresh


def cf_sameinit(model: nn.Module, client_datasets, retained_clients: list[int], deleted_clients: list[int], cfg: LocalTrainConfig, device, seed: int, rounds: int, history=None, model_factory=None, log=None, **kwargs) -> nn.Module:
    """Same-initialisation counterfactual: FedAvg on the retained clients with the identical protocol, starting from the
    stored round-0 global model of the compromised training (history.global_states[0]). Identical procedure to
    scripts/train_sameinit_counterfactuals.py (data-order generator seed + 17)."""
    if history is None or history.rounds[0] != 0:
        raise ValueError("cf_sameinit needs the stored history whose first entry is the round-0 global model")
    fresh = model_factory().to(device)
    fresh.load_state_dict({k: v.to(device) for k, v in history.global_states[0].items()})
    fresh, _ = run_fedavg(fresh, client_datasets, retained_clients, rounds, cfg, device, seed=seed + 17, log=log)
    return fresh

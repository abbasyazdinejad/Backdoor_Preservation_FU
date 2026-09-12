"""Short retraining: federated fine-tuning of the compromised model on retained clients.

This is the simplest retained-data-only approximate procedure: theta* is used as the
initial model and `rounds` FedAvg rounds are executed with the retained clients only.
The deleted clients' data is never accessed.
"""
from __future__ import annotations

import torch.nn as nn

from ..federated_training import LocalTrainConfig, run_fedavg


def short_retraining(model: nn.Module, client_datasets, retained_clients: list[int], deleted_clients: list[int], cfg: LocalTrainConfig, device, seed: int, rounds: int, log=None, **kwargs) -> nn.Module:
    model, _ = run_fedavg(model, client_datasets, retained_clients, rounds, cfg, device, seed=seed + 11, log=log)
    return model

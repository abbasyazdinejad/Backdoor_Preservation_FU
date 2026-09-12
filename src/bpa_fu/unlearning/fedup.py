"""FedUP (Romandini, Borcea, Montanari, Foschini, arXiv:2508.13853, 2025): pruning-based federated unlearning for model
poisoning attacks. Implementation from Algorithms 1-2 of the paper (no official code exists); see
audit/v8_fedup_baseline_protocol_audit.md for the documented interpretations.

Inputs: the local models of every client from the last training round (reconstructed exactly by
experiment.reconstruct_last_round) and the global model of the previous round.
  1. avgMal, avgBen = (sample-size-weighted) averages of the malicious / benign last-round local models
  2. difference = (avgMal - avgBen)^2 ; rank = difference * |globalModel^{t-1}|   (magnitude weighting)
  3. per Conv2d/Linear weight tensor: mask the top `prune_fraction` entries of rank
  4. pruned model = avgBen with the masked weights set to zero (soft pruning: no mask during recovery)
  5. recovery: `recovery_rounds` FedAvg rounds on the retained clients with the protocol's local optimizer
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..federated_training import LocalTrainConfig, run_fedavg


def prunable_keys(model: nn.Module) -> list[str]:
    """State-dict keys of the weight tensors of dense and convolutional layers (biases and BatchNorm excluded)."""
    keys = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            keys.append(f"{name}.weight" if name else "weight")
    return keys


def weighted_average(states: dict[int, dict[str, torch.Tensor]], sizes: dict[int, int]) -> dict[str, torch.Tensor]:
    total = float(sum(sizes[c] for c in states))
    out = None
    for c, sd in states.items():
        w = sizes[c] / total
        if out is None:
            out = {k: (w * v.float()) if v.dtype.is_floating_point else v.clone() for k, v in sd.items()}
        else:
            for k in out:
                if out[k].dtype.is_floating_point:
                    out[k] += w * sd[k].float()
    return out


def fedup_mask(avg_mal: dict, avg_ben: dict, global_prev: dict, keys: list[str], prune_fraction: float) -> dict[str, torch.Tensor]:
    """Algorithm 1: per-layer top-P mask of rank = (avgMal - avgBen)^2 * |globalModel^{t-1}| (True = prune)."""
    masks = {}
    for k in keys:
        diff = (avg_mal[k].float() - avg_ben[k].float()) ** 2
        rank = diff * global_prev[k].float().abs()
        n = rank.numel(); n_prune = int(math.ceil(prune_fraction * n))
        m = torch.zeros(n, dtype=torch.bool, device=rank.device)
        if n_prune > 0:
            idx = torch.topk(rank.flatten(), n_prune, largest=True).indices
            m[idx] = True
        masks[k] = m.view_as(rank)
    return masks


def fedup(model: nn.Module, client_datasets, retained_clients: list[int], deleted_clients: list[int], cfg: LocalTrainConfig, device, seed: int,
          last_round_models: dict[int, dict[str, torch.Tensor]] | None = None, global_prev: dict[str, torch.Tensor] | None = None,
          prune_fraction: float = 0.10, recovery_rounds: int = 5, log=None, **kwargs) -> nn.Module:
    if last_round_models is None or global_prev is None:
        raise ValueError("fedup requires the last-round local models of all clients and the previous global model (experiment.reconstruct_last_round)")
    sizes = {c: len(client_datasets[c]) for c in last_round_models}
    mal = {c: last_round_models[c] for c in deleted_clients}
    ben = {c: last_round_models[c] for c in retained_clients}
    if not mal or not ben:
        raise ValueError("fedup needs at least one malicious and one benign last-round model")
    avg_mal, avg_ben = weighted_average(mal, sizes), weighted_average(ben, sizes)
    keys = prunable_keys(model)
    masks = fedup_mask(avg_mal, avg_ben, {k: global_prev[k] for k in keys}, keys, prune_fraction)
    pruned = {k: v.clone() for k, v in avg_ben.items()}
    n_pruned = 0; n_total = 0
    for k in keys:
        pruned[k][masks[k]] = 0.0; n_pruned += int(masks[k].sum()); n_total += masks[k].numel()
    model.to(device)
    model.load_state_dict({k: v.to(device) if v.dtype.is_floating_point else v.to(device) for k, v in pruned.items()})
    if log is not None:
        log(f"fedup: pruned {n_pruned}/{n_total} weights ({100 * n_pruned / n_total:.2f}%) in {len(keys)} dense/conv layers of the benign average (P={prune_fraction})")
    if recovery_rounds > 0:
        model, _ = run_fedavg(model, client_datasets, retained_clients, recovery_rounds, cfg, device, seed=seed + 23, log=log)
    model.fedup_n_pruned = n_pruned  # type: ignore[attr-defined]
    model.fedup_n_prunable = n_total  # type: ignore[attr-defined]
    return model

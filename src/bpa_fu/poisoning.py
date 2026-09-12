"""Malicious-client data poisoning with fixed (seeded) poisoned index sets.

Each malicious client replaces a fraction `poison_rate` of its local samples with triggered
copies relabelled to the target class (BadNets-style dirty-label poisoning). The poisoned
indices are drawn once from the client's non-target samples with a seeded RNG, so the
poisoned set is fixed across epochs and rounds and can be recorded for the deletion target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .datasets import TensorImageDataset
from .triggers import PatchTrigger


@dataclass
class PoisonRecord:
    client_id: int
    poisoned_local_positions: np.ndarray  # positions within the client's index array
    poisoned_global_indices: np.ndarray  # indices into the global training set


def build_client_datasets(
    train: TensorImageDataset,
    parts: list[np.ndarray],
    malicious_clients: list[int],
    poison_rate: float,
    target_label: int,
    trigger: PatchTrigger,
    seed: int,
) -> tuple[list[TensorImageDataset], list[PoisonRecord]]:
    """Return per-client datasets (poisoned for malicious clients) and poison records."""
    rng = np.random.RandomState(seed + 7919)
    client_ds: list[TensorImageDataset] = []
    records: list[PoisonRecord] = []
    for cid, idx in enumerate(parts):
        imgs = train.images[idx].clone()
        labs = train.labels[idx].clone()
        if cid in malicious_clients and poison_rate > 0:
            eligible = np.where(labs.numpy() != target_label)[0]
            n_poison = int(round(poison_rate * len(idx)))
            n_poison = min(n_poison, len(eligible))
            pos = np.sort(rng.choice(eligible, size=n_poison, replace=False))
            imgs[pos] = trigger.apply_uint8(imgs[pos])
            labs[pos] = target_label
            records.append(PoisonRecord(cid, pos, idx[pos]))
        client_ds.append(TensorImageDataset(imgs, labs))
    return client_ds, records


def poison_fraction(records: list[PoisonRecord], parts: list[np.ndarray]) -> dict[int, float]:
    return {r.client_id: len(r.poisoned_local_positions) / len(parts[r.client_id]) for r in records}

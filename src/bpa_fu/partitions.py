"""Deterministic client partitioning (IID and Dirichlet non-IID)."""
from __future__ import annotations

import numpy as np


def iid_partition(labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    """Shuffle indices with `seed` and split into `num_clients` near-equal shards."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(labels))
    return [np.sort(s) for s in np.array_split(idx, num_clients)]


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, beta: float, seed: int, min_size: int = 10, max_tries: int = 1000
) -> list[np.ndarray]:
    """Label-skew partition following Hsu et al. (2019) / Li et al. (2022, NIID-Bench).

    For every class c a proportion vector p_c ~ Dir(beta) over clients is sampled and the
    class indices are split accordingly. Resampled until every client has >= min_size samples.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    for _ in range(max_tries):
        client_idx: list[list[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            p = rng.dirichlet(np.repeat(beta, num_clients))
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                client_idx[k].extend(part.tolist())
        sizes = [len(ci) for ci in client_idx]
        if min(sizes) >= min_size:
            return [np.sort(np.array(ci, dtype=np.int64)) for ci in client_idx]
    raise RuntimeError("Dirichlet partition failed to satisfy min_size; lower min_size or raise beta")


def make_partition(labels: np.ndarray, num_clients: int, scheme: str, seed: int, beta: float | None = None) -> list[np.ndarray]:
    if scheme == "iid":
        return iid_partition(labels, num_clients, seed)
    if scheme == "dirichlet":
        if beta is None:
            raise ValueError("beta is required for the dirichlet partition")
        return dirichlet_partition(labels, num_clients, beta, seed)
    raise ValueError(f"unknown partition scheme {scheme!r}")


def check_partition(parts: list[np.ndarray], n: int) -> None:
    """Assert the partition is a disjoint cover of range(n)."""
    allidx = np.concatenate(parts)
    if len(allidx) != n or len(np.unique(allidx)) != n:
        raise AssertionError("partition is not a disjoint cover of the dataset")
    if allidx.min() != 0 or allidx.max() != n - 1:
        raise AssertionError("partition indices out of range")


def partition_statistics(parts: list[np.ndarray], labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return a (num_clients, num_classes) count matrix."""
    m = np.zeros((len(parts), num_classes), dtype=np.int64)
    for k, p in enumerate(parts):
        m[k] = np.bincount(labels[p], minlength=num_classes)
    return m

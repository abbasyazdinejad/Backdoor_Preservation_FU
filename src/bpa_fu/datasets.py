"""Dataset loading.

All three datasets are loaded as 32x32 tensors in [0, 1] with no augmentation and no
normalisation, matching the preprocessing of the original implementation (ToTensor only,
Resize(32,32) for MNIST and GTSRB). Images are cached as uint8 tensors in memory so that
client partitions and poisoned copies are cheap and deterministic.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATASET_SPECS = {
    "cifar10": {"in_channels": 3, "num_classes": 10, "n_train": 50000, "n_test": 10000},
    "mnist": {"in_channels": 1, "num_classes": 10, "n_train": 60000, "n_test": 10000},
    "gtsrb": {"in_channels": 3, "num_classes": 43, "n_train": 26640, "n_test": 12630},
}


@dataclass
class TensorImageDataset(Dataset):
    """In-memory dataset of uint8 images (N, C, H, W) and int64 labels."""

    images: torch.Tensor  # uint8
    labels: torch.Tensor  # int64

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx].float().div_(255.0), int(self.labels[idx])

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.images.numpy().tobytes())
        h.update(self.labels.numpy().tobytes())
        return h.hexdigest()


def _to_uint8_tensor(ds) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append((x * 255.0).round().to(torch.uint8))
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.int64)


def _cache_path(root: str, name: str, split: str) -> Path:
    return Path(root) / "_cache" / f"{name}_{split}_32x32_uint8.pt"


def load_dataset(name: str, root: str = "data", download: bool = True) -> tuple[TensorImageDataset, TensorImageDataset, dict]:
    """Return (train, test, spec). Cached uint8 tensors are written to <root>/_cache."""
    name = name.lower()
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(DATASET_SPECS)}")
    spec = dict(DATASET_SPECS[name])
    out = []
    for split in ("train", "test"):
        cp = _cache_path(root, name, split)
        if cp.exists():
            d = torch.load(cp)
            ds = TensorImageDataset(d["images"], d["labels"])
        else:
            tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            if name == "cifar10":
                raw = datasets.CIFAR10(root=root, train=(split == "train"), download=download, transform=tf)
            elif name == "mnist":
                raw = datasets.MNIST(root=root, train=(split == "train"), download=download, transform=tf)
            else:
                raw = datasets.GTSRB(root=root, split=split, download=download, transform=tf)
            images, labels = _to_uint8_tensor(raw)
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"images": images, "labels": labels}, cp)
            ds = TensorImageDataset(images, labels)
        expected = spec["n_train"] if split == "train" else spec["n_test"]
        if len(ds) != expected:
            raise RuntimeError(f"{name} {split}: expected {expected} samples, found {len(ds)}")
        if ds.images.shape[1] != spec["in_channels"] or ds.images.shape[2:] != (32, 32):
            raise RuntimeError(f"{name} {split}: unexpected image shape {tuple(ds.images.shape)}")
        if int(ds.labels.max()) >= spec["num_classes"] or int(ds.labels.min()) < 0:
            raise RuntimeError(f"{name} {split}: labels outside [0, {spec['num_classes']})")
        out.append(ds)
    return out[0], out[1], spec


def dataset_checksums(name: str, root: str = "data") -> dict[str, str]:
    tr, te, _ = load_dataset(name, root=root, download=False)
    return {"train_sha256": tr.sha256(), "test_sha256": te.sha256(), "n_train": len(tr), "n_test": len(te)}

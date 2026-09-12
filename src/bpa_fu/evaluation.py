"""Benign accuracy (BA) and attack success rate (ASR).

ASR is computed over test samples whose true label differs from the target class, so that
samples that would be classified as the target class anyway do not inflate the metric.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .datasets import TensorImageDataset
from .triggers import PatchTrigger


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test: TensorImageDataset,
    trigger: PatchTrigger,
    target_label: int,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    model.eval()
    model.to(device)
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    asr_hit = asr_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
        keep = y != target_label
        if int(keep.sum()) > 0:
            xt = trigger.apply(x[keep])
            pt = model(xt).argmax(1)
            asr_hit += int((pt == target_label).sum())
            asr_total += int(keep.sum())
    return {"BA": correct / total, "ASR": asr_hit / asr_total, "n_test": total, "n_asr": asr_total}

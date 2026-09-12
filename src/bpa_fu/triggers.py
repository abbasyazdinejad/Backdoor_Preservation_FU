"""Patch trigger used throughout the paper.

A square patch of side `size` pixels with constant value `value` (in [0,1] pixel space)
is written in the bottom-right corner, `offset` pixels away from the border, on all channels.
With size=4, offset=1 on 32x32 inputs the patch covers 16/1024 = 1.5625% of the input area.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class PatchTrigger:
    size: int = 4
    value: float = 1.0
    offset: int = 1

    def region(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Return (row0, row1, col0, col1) of the patch (half-open)."""
        return h - self.offset - self.size, h - self.offset, w - self.offset - self.size, w - self.offset

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to a float image (C,H,W) or batch (B,C,H,W) in [0,1]. Returns a copy."""
        x = x.clone()
        h, w = x.shape[-2], x.shape[-1]
        r0, r1, c0, c1 = self.region(h, w)
        if r0 < 0 or c0 < 0:
            raise ValueError("trigger does not fit in the image")
        x[..., r0:r1, c0:c1] = self.value
        return x

    def apply_uint8(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to uint8 images (value scaled to 0..255)."""
        x = x.clone()
        h, w = x.shape[-2], x.shape[-1]
        r0, r1, c0, c1 = self.region(h, w)
        x[..., r0:r1, c0:c1] = int(round(self.value * 255))
        return x

    def area_fraction(self, h: int = 32, w: int = 32) -> float:
        return (self.size * self.size) / float(h * w)

    def to_dict(self) -> dict:
        return asdict(self)

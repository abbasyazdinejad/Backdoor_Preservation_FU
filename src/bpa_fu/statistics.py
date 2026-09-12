"""Aggregation over seeds: mean, sample std, and 95% confidence interval (t distribution)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def summarize(values, confidence: float = 0.95) -> dict[str, float]:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    n = len(v)
    out = {"n": n, "mean": float(v.mean()) if n else float("nan")}
    if n >= 2:
        sd = float(v.std(ddof=1))
        half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / math.sqrt(n))
    else:
        sd, half = float("nan"), float("nan")
    out.update({"std": sd, "ci95_half": half, "ci95_low": out["mean"] - half, "ci95_high": out["mean"] + half})
    return out


def fmt_pct(mean: float, std: float | None = None, digits: int = 2) -> str:
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{100 * mean:.{digits}f}"
    return f"{100 * mean:.{digits}f} $\\pm$ {100 * std:.{digits}f}"

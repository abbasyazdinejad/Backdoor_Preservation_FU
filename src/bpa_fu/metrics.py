"""Derived security metrics.

All inputs are fractions in [0, 1] (not percentages).
"""
from __future__ import annotations

import math
from typing import Sequence

BPR_MIN_DENOMINATOR = 0.05  # BPR is undefined when the compromised model is within 5 pp of the counterfactual


def asr_reduction(asr_before: float, asr_after: float) -> float:
    """Absolute reduction of ASR (positive = safer)."""
    return asr_before - asr_after


def backdoor_preservation_ratio(asr_after: float, asr_before: float, asr_cf: float) -> float:
    """BPR = (ASR' - ASR_cf) / (ASR* - ASR_cf); NaN when the denominator is too small."""
    denom = asr_before - asr_cf
    if not math.isfinite(denom) or abs(denom) < BPR_MIN_DENOMINATOR:
        return float("nan")
    return (asr_after - asr_cf) / denom


def counterfactual_gap(asr_after: float, asr_cf: float) -> float:
    """ASR' - ASR_cf (positive = residual backdoor above the counterfactual level)."""
    return asr_after - asr_cf


def non_regression_violation(asr_by_alpha: Sequence[tuple[float, float]], tau: float) -> float:
    """DEPRECATED legacy diagnostic (V_tau) from the earlier manuscript version; NOT used in the final paper.

    V_tau = 1/(J-1) * sum_j max(0, ASR(alpha_{j+1}) - ASR(alpha_j) - tau)
    where alpha_1 < ... < alpha_J are nested deletion fractions. The final manuscript replaces this
    threshold-based score by within-seed paired differences with paired bootstrap intervals
    (latex_tables.paired_fraction_differences / paired_bootstrap_ci, Eq. (3) of the paper); see
    audit/statistical_method_audit.md. The function is kept only for its regression test and is not
    called by any table, figure or notebook code.
    """
    pts = sorted(asr_by_alpha, key=lambda t: t[0])
    if len(pts) < 2:
        raise ValueError("at least two deletion fractions are required")
    alphas = [a for a, _ in pts]
    if len(set(alphas)) != len(alphas):
        raise ValueError("deletion fractions must be distinct")
    total = 0.0
    for (a0, s0), (a1, s1) in zip(pts[:-1], pts[1:]):
        total += max(0.0, s1 - s0 - tau)
    return total / (len(pts) - 1)


def security_improvement_ok(asr_before: float, asr_after: float, margin: float) -> bool:
    """Primary endpoint 1: ASR reduced by at least `margin`."""
    return (asr_before - asr_after) >= margin


def counterfactual_proximity_ok(asr_after: float, asr_cf: float, tol: float) -> bool:
    """Primary endpoint 2: ASR' within `tol` of the counterfactual ASR."""
    return asr_after <= asr_cf + tol

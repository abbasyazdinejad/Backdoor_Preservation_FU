"""Descriptive alignment measurements of unlearning updates in parameter space.

Coordinate system: the compromised model theta*.
  u        = theta' - theta*                       update of an unlearning procedure
  Delta_M  = sum_{t in T_stored} sum_{i in M_del} w_i Delta_i^(t)
             accumulated stored malicious-update proxy: FedAvg-weighted sum of the stored updates of the deleted
             clients over the stored rounds only (a proxy for their contribution, not the contribution itself)
  Delta_R  = same sum over the retained clients
  d        = theta_cf0 - theta*                    direction to the same-initialisation counterfactual
Scalars (global and per parameter group):
  cos(u, -Delta_M), cos(u, Delta_R), cos(u, d)
  kappa   = <u, -Delta_M> / ||Delta_M||^2          projection coefficient: the component of u along -Delta_M in units of
                                                   ||Delta_M||; kappa = 1 means the projected component equals the proxy,
                                                   an orthogonal residual may remain
  resid   = ||u - kappa (-Delta_M)|| / ||u||       orthogonal residual fraction of the update (= sqrt(1 - cos^2))
  r_g     = ||theta'_g - theta_cf0_g|| / ||theta*_g - theta_cf0_g||   relative residual Euclidean distance to theta_cf0
  energy_g = ||u_g||^2 / ||u||^2                   normalised group-wise update energy (sums to 1 over groups)
Groups are the stable module groups of the architecture (models.parameter_groups): conv1/conv2/fc1/fc2 for SimpleCNN,
stem/layer1..layer4/fc for ResNet-18. Only trainable parameters enter the vectors (BatchNorm running statistics are
buffers and are excluded); for SimpleCNN, which has no buffers, this equals the earlier whole-state-dict computation.
CSV column names keep the historical prefix 'rho' for kappa.
"""
from __future__ import annotations

import torch

from .models import PARAMETER_GROUPS

# default groups (SimpleCNN); kept as a module constant for backward compatibility of callers
LAYERS = [g for g, _ in PARAMETER_GROUPS["simplecnn"]]
DEFAULT_GROUPS = PARAMETER_GROUPS["simplecnn"]


def _flat(state: dict, prefixes: list[str] | None = None, keys: list[str] | None = None) -> torch.Tensor:
    ks = sorted(k for k in (keys if keys is not None else state) if k in state and state[k].dtype.is_floating_point
                and (prefixes is None or any(k.startswith(p) for p in prefixes)))
    if not ks:
        return torch.zeros(0)
    return torch.cat([state[k].detach().float().flatten() for k in ks])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def malicious_contribution(history: dict, deleted: list[int]) -> tuple[dict, dict]:
    """Return (Delta_M, Delta_R): FedAvg-weighted sums of the stored updates of the deleted / retained clients."""
    dm, dr = None, None
    for upd, sizes in zip(history["client_updates"], history["client_sizes"]):
        total = float(sum(sizes.values()))
        for c, u in upd.items():
            w = sizes[c] / total
            target = "m" if c in deleted else "r"
            acc = dm if target == "m" else dr
            if acc is None:
                acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in u.items() if v.dtype.is_floating_point}
            for k in acc:
                acc[k] += w * u[k].float()
            if target == "m":
                dm = acc
            else:
                dr = acc
    return dm, dr


def check_groups_cover(keys: list[str], groups: list[tuple[str, list[str]]]) -> None:
    """Every key must match exactly one group (raises otherwise)."""
    for k in keys:
        n = sum(any(k.startswith(p) for p in ps) for _, ps in groups)
        if n != 1:
            raise ValueError(f"parameter {k} matches {n} groups; groups must partition the parameters")


def alignment_metrics(theta_star: dict, theta_prime: dict, theta_cf0: dict | None, delta_m: dict, delta_r: dict | None = None,
                      groups: list[tuple[str, list[str]]] | None = None, param_keys: list[str] | None = None) -> dict:
    groups = DEFAULT_GROUPS if groups is None else groups
    keys = [k for k in (param_keys if param_keys is not None else theta_star) if theta_star[k].dtype.is_floating_point]
    check_groups_cover(keys, groups)
    out = {}
    u = {k: theta_prime[k].float() - theta_star[k].float() for k in keys}
    u_all = _flat(u, None, keys); u_all_sq = float(u_all @ u_all)
    for name, pre in [("global", None)] + [(g, ps) for g, ps in groups]:
        uf, mf = _flat(u, pre, keys), _flat(delta_m, pre, keys)
        out[f"{name}_u_norm"] = float(uf.norm()); out[f"{name}_dm_norm"] = float(mf.norm())
        out[f"{name}_cos_u_negdm"] = _cos(uf, -mf)
        kappa = float((uf @ (-mf)) / (mf @ mf)) if float(mf.norm()) > 0 else float("nan")
        out[f"{name}_rho"] = kappa
        out[f"{name}_resid_frac"] = float((uf - kappa * (-mf)).norm() / uf.norm()) if float(uf.norm()) > 0 and kappa == kappa else float("nan")
        if pre is not None:
            out[f"{name}_energy"] = float(uf @ uf) / u_all_sq if u_all_sq > 0 else float("nan")
        if delta_r is not None:
            rf = _flat(delta_r, pre, keys); out[f"{name}_cos_u_dr"] = _cos(uf, rf)
        if theta_cf0 is not None:
            d = {k: theta_cf0[k].float() - theta_star[k].float() for k in keys}
            df = _flat(d, pre, keys)
            res = {k: theta_prime[k].float() - theta_cf0[k].float() for k in keys}
            out[f"{name}_cos_u_d"] = _cos(uf, df)
            out[f"{name}_d_norm"] = float(df.norm())
            out[f"{name}_rel_dist_cf"] = float(_flat(res, pre, keys).norm() / df.norm()) if float(df.norm()) > 0 else float("nan")
    return out

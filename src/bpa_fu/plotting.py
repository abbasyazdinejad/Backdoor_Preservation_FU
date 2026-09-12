"""Programmatic figures from the authoritative result file, each with a provenance sidecar."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .latex_tables import DATASET_NAMES, METHOD_NAMES, METHOD_ORDER, per_seed_metrics
from .statistics import summarize

SHORT = {"before": "Compromised", "short_retrain": "Fine-tuning (retained)", "federaser": "FedEraser", "grad_negation": "Grad. negation", "fine_pruning": "Fine-Pruning", "full_retrain": "Retraining (indep. init.)", "cf_sameinit": "Counterfactual (same init.)"}
COLORS = {"before": "#4d4d4d", "short_retrain": "#1b9e77", "federaser": "#7570b3", "grad_negation": "#d95f02", "fine_pruning": "#a6761d", "full_retrain": "#e7298a", "cf_sameinit": "#66a61e"}
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "serif"})


def _sidecar(fig_path: Path, results_file: str, run_ids: list[str], script: str, cfg: dict) -> None:
    h = hashlib.sha256(open(fig_path, "rb").read()).hexdigest()
    with open(fig_path.with_suffix(".provenance.json"), "w") as f:
        json.dump({"figure": str(fig_path), "figure_sha256": h, "input_result_file": results_file,
                   "input_result_sha256": hashlib.sha256(open(results_file, "rb").read()).hexdigest(),
                   "input_run_ids": sorted(run_ids), "plotting_script": script, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": cfg}, f, indent=2)


def fig_methods(df: pd.DataFrame, cfgs: list, out: Path, results_file: str, alpha: float = 1.0) -> Path:
    """Grouped bars: ASR and BA of every method at alpha=1 per dataset with 95% CI error bars."""
    exps = [c for c in cfgs]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    width = 0.12
    used = []
    for ax, metric in zip(axes, ("ASR", "BA")):
        for i, m in enumerate(METHOD_ORDER):
            means, halves = [], []
            for cfg in exps:
                sub = df[(df.exp_id == cfg.exp_id) & (df.method == m) & (np.isclose(df.alpha, alpha) | (m == "before"))]
                s = summarize(sub[metric]); means.append(100 * s["mean"]); halves.append(0 if np.isnan(s["ci95_half"]) else 100 * s["ci95_half"])
                used += sub.run_id.tolist()
            x = np.arange(len(exps)) + (i - 3) * width
            ax.bar(x, means, width, yerr=halves, capsize=2, color=COLORS[m], label=SHORT[m], edgecolor="black", linewidth=0.4)
        ax.set_xticks(np.arange(len(exps)))
        ax.set_xticklabels([DATASET_NAMES[c.dataset] + ("" if c.partition["scheme"] == "iid" else "\n(Dir)") for c in exps])
        ax.set_ylabel(f"{metric} (%)"); ax.set_ylim(0, 105); ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].set_title("Attack success rate after removing all identified clients", fontsize=9)
    axes[1].set_title("Benign accuracy", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out, dpi=300); plt.close(fig)
    _sidecar(out, results_file, used, "bpa_fu.plotting.fig_methods", {"alpha": alpha, "exp_ids": [c.exp_id for c in exps]})
    return out


def fig_fraction(df: pd.DataFrame, cfgs: list, out: Path, results_file: str, methods=("short_retrain", "federaser", "grad_negation", "fine_pruning", "cf_sameinit")) -> Path:
    """ASR versus nested deletion fraction, one panel per experiment, mean +- 95% CI over seeds."""
    n = len(cfgs)
    fig, axes = plt.subplots(1, n, figsize=(max(7.0, 1.9 * n + 0.8), 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    used = []
    for ax, cfg in zip(axes, cfgs):
        sub = df[df.exp_id == cfg.exp_id]
        b = summarize(sub[sub.method == "before"].ASR)
        ax.axhline(100 * b["mean"], color=COLORS["before"], linestyle="--", linewidth=1, label=SHORT["before"])
        for m in methods:
            r = sub[sub.method == m]
            if r.empty:
                continue
            alphas = sorted(set(round(a, 4) for a in r.alpha))
            s = [summarize(r[np.isclose(r.alpha, a)].ASR) for a in alphas]
            ax.errorbar(alphas, [100 * x["mean"] for x in s], yerr=[0 if np.isnan(x["ci95_half"]) else 100 * x["ci95_half"] for x in s],
                        marker="o", markersize=3, capsize=2, color=COLORS[m], label=SHORT[m], linewidth=1)
            used += r.run_id.tolist()
        ax.set_title(DATASET_NAMES[cfg.dataset] + ("" if cfg.partition["scheme"] == "iid" else " (Dir)"), fontsize=9)
        ax.set_xlabel(r"removed fraction $\alpha$"); ax.set_ylim(-3, 105); ax.grid(linestyle=":", alpha=0.5)
        ax.set_xticks(sorted(set(round(a, 4) for a in sub[sub.method != "before"].alpha)))
    axes[0].set_ylabel("ASR (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out, dpi=300); plt.close(fig)
    _sidecar(out, results_file, used, "bpa_fu.plotting.fig_fraction", {"exp_ids": [c.exp_id for c in cfgs], "methods": list(methods)})
    return out


def fig_tradeoff(df: pd.DataFrame, cfgs: list, out: Path, results_file: str, alpha: float = 1.0) -> Path:
    """Per-seed points: change in BA vs. ASR reduction relative to the compromised model (alpha=1)."""
    d = per_seed_metrics(df)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    markers = {"cifar10": "o", "mnist": "s", "gtsrb": "^"}
    used = []
    for cfg in cfgs:
        sub = d[(d.exp_id == cfg.exp_id) & np.isclose(d.alpha, alpha)]
        for m in METHOD_ORDER[1:]:
            r = sub[sub.method == m]
            if r.empty:
                continue
            mk = markers[cfg.dataset] if cfg.partition["scheme"] == "iid" else "D"
            ax.scatter(100 * r.dBA, 100 * r.asr_reduction, color=COLORS[m], marker=mk, s=18, edgecolor="black", linewidth=0.3, alpha=0.9)
            used += r.run_id.tolist()
    ax.axhline(0, color="gray", linewidth=0.6); ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel(r"$\Delta$BA (pp, relative to compromised)"); ax.set_ylabel("ASR reduction (pp)")
    ax.grid(linestyle=":", alpha=0.5)
    from matplotlib.lines import Line2D
    h1 = [Line2D([0], [0], marker="o", linestyle="", color=COLORS[m], markeredgecolor="black", markersize=5, label=SHORT[m]) for m in METHOD_ORDER[1:]]
    h2 = [Line2D([0], [0], marker=markers[k], linestyle="", color="gray", markersize=5, label=DATASET_NAMES[k]) for k in markers]
    h2.append(Line2D([0], [0], marker="D", linestyle="", color="gray", markersize=5, label="CIFAR-10 (Dir)"))
    ax.legend(handles=h1 + h2, fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.02))
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    _sidecar(out, results_file, used, "bpa_fu.plotting.fig_tradeoff", {"alpha": alpha, "exp_ids": [c.exp_id for c in cfgs]})
    return out


def fig_partition(counts: np.ndarray, out: Path, title: str) -> Path:
    """Heat map of per-client class counts (notebook diagnostic)."""
    fig, ax = plt.subplots(figsize=(4, 2.6))
    im = ax.imshow(counts, aspect="auto", cmap="Blues")
    ax.set_xlabel("class"); ax.set_ylabel("client"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def fig_alignment(dfa: pd.DataFrame, cfgs: list, out: Path, results_file: str, layer_exp: str = "cifar10_iid") -> Path:
    """(a) ASR reduction against the projection coefficient kappa for every run; (b) normalised layer-wise update energy
    ||u_l||^2/||u||^2 at alpha = 1 (mean and 95% CI over seeds) for one experiment."""
    from .latex_tables import ALIGN_METHODS, LAYER_NAMES
    main = dfa[dfa.exp_id.isin([c.exp_id for c in cfgs]) & dfa.method.isin(ALIGN_METHODS)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for m in ALIGN_METHODS:
        for full, mk in ((True, "o"), (False, "^")):
            r = main[(main.method == m) & (np.isclose(main.alpha, 1.0) == full)]
            ax.scatter(r.global_rho, 100 * r.asr_reduction, s=14 if full else 9, marker=mk, color=COLORS[m], alpha=0.85 if full else 0.45,
                       edgecolor="black", linewidth=0.3, label=SHORT[m] if full else None)
    ax.axvline(0, color="gray", linewidth=0.6); ax.axvline(1, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"projection coefficient $\kappa=\langle u,-\Delta_M\rangle/\|\Delta_M\|^2$"); ax.set_ylabel("ASR reduction (pp)")
    ax.grid(linestyle=":", alpha=0.5); ax.legend(fontsize=6.5, frameon=False, loc="center right")
    ax.text(0.02, 0.97, r"$\circ$: $\alpha=1$; $\triangle$: $\alpha<1$", transform=ax.transAxes, fontsize=6.5, va="top")
    ax = axes[1]
    sub = dfa[(dfa.exp_id == layer_exp) & np.isclose(dfa.alpha, 1.0) & dfa.method.isin(ALIGN_METHODS)]
    width = 0.19
    for i, m in enumerate(ALIGN_METHODS):
        r = sub[sub.method == m]
        means, halves = [], []
        for l in LAYER_NAMES:
            s_ = summarize(r[f"{l}_energy"]); means.append(s_["mean"]); halves.append(0 if np.isnan(s_["ci95_half"]) else s_["ci95_half"])
        ax.bar(np.arange(len(LAYER_NAMES)) + (i - 1.5) * width, means, width, yerr=halves, capsize=2, color=COLORS[m], edgecolor="black", linewidth=0.3, label=SHORT[m])
    ax.set_xticks(np.arange(len(LAYER_NAMES))); ax.set_xticklabels(LAYER_NAMES); ax.set_ylabel(r"update energy $\|u_\ell\|^2/\|u\|^2$"); ax.set_ylim(0, 1)
    ax.set_title(f"{DATASET_NAMES[layer_exp.split('_')[0]]}, $\\alpha=1$", fontsize=9); ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    _sidecar(out, results_file, main.run_id.tolist(), "bpa_fu.plotting.fig_alignment", {"exp_ids": [c.exp_id for c in cfgs], "layer_exp": layer_exp})
    return out

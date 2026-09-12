"""LaTeX table fragments generated from an aggregated result file
(for the final paper: results_final/processed/all_results.csv).

Every table function returns (latex_string, values_dict) where values_dict maps
"<table>:<row>:<col>" -> numeric value (fraction in [0,1] or pp) so that the paper text can be
cross-checked against the authoritative file (see scripts/check_paper_numbers.py).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import backdoor_preservation_ratio
from .statistics import summarize

DATASET_NAMES = {"cifar10": "CIFAR-10", "mnist": "MNIST", "gtsrb": "GTSRB"}
METHOD_NAMES = {"before": "Compromised (before unlearning)", "short_retrain": "Retained-data fine-tuning",
                "grad_negation": "Gradient negation (identified data)", "federaser": "FedEraser (tensor-wise impl.)",
                "full_retrain": "Retraining, independent init.", "cf_sameinit": "Retraining, same init. (counterfactual)", "fine_pruning": "Fine-Pruning (mitigation baseline)"}
METHOD_ORDER = ["before", "short_retrain", "federaser", "grad_negation", "fine_pruning", "full_retrain", "cf_sameinit"]
SHORT_METHOD = {"before": "Compromised", "short_retrain": "Fine-tuning (retained)", "federaser": "FedEraser", "grad_negation": "Gradient negation", "fine_pruning": "Fine-Pruning", "full_retrain": "Retraining (indep. init.)", "cf_sameinit": "Counterfactual (same init.)"}


def _z(v: float, digits: int = 1) -> float:
    """Round and remove negative zero."""
    r = round(v, digits)
    return 0.0 if r == 0 else r


def _pm(s: dict, digits: int = 1) -> str:
    if s["n"] == 0 or math.isnan(s["mean"]):
        return "--"
    if s["n"] < 2:
        return f"{_z(100 * s['mean'], digits):.{digits}f}"
    return f"{_z(100 * s['mean'], digits):.{digits}f} $\\pm$ {_z(100 * s['std'], digits):.{digits}f}"


def _ci(s: dict, digits: int = 1) -> str:
    if s["n"] < 2:
        return "--"
    return f"[{_z(100 * s['ci95_low'], digits):.{digits}f}, {_z(100 * s['ci95_high'], digits):.{digits}f}]"


def per_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-run ASR reduction, counterfactual gap and BPR (each relative to the same seed)."""
    before = df[df.method == "before"].set_index(["exp_id", "seed"])[["BA", "ASR"]].rename(columns={"BA": "BA_before", "ASR": "ASR_before"})
    d = df[df.method != "before"].join(before, on=["exp_id", "seed"])
    d["asr_reduction"] = d.ASR_before - d.ASR
    d["cf_gap"] = d.ASR - d.counterfactual_ASR
    d["dBA"] = d.BA - d.BA_before
    d["BPR"] = [backdoor_preservation_ratio(a, b, c) for a, b, c in zip(d.ASR, d.ASR_before, d.counterfactual_ASR)]
    return d


def table_setup(df: pd.DataFrame, cfgs: list) -> tuple[str, dict]:
    """Table: compromised models before unlearning (mean +- std over seeds) and counterfactual clean reference."""
    vals = {}
    lines = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Experiment} & \textbf{BA} & \textbf{ASR} & \textbf{BA$_{\mathrm{cf}}$} & \textbf{ASR$_{\mathrm{cf}}$} \\", r"\midrule"]
    for cfg in cfgs:
        sub = df[(df.exp_id == cfg.exp_id)]
        b = sub[sub.method == "before"]
        cf = sub[(sub.method == "cf_sameinit") & np.isclose(sub.alpha, 1.0)]
        sba, sasr, cba, casr = summarize(b.BA), summarize(b.ASR), summarize(cf.BA), summarize(cf.ASR)
        name = DATASET_NAMES[cfg.dataset]
        part = "IID" if cfg.partition["scheme"] == "iid" else "Dir."
        lines.append(f"{name} {part} & {_pm(sba)} & {_pm(sasr)} & {_pm(cba)} & {_pm(casr)} \\\\")
        for k, s in (("BA", sba), ("ASR", sasr), ("cfBA", cba), ("cfASR", casr)):
            vals[f"setup:{cfg.exp_id}:{k}_mean"] = s["mean"]; vals[f"setup:{cfg.exp_id}:{k}_std"] = s["std"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def table_methods(df: pd.DataFrame, cfgs: list, alpha: float = 1.0) -> tuple[str, dict]:
    """Main comparison at full removal of the identified clients (alpha = 1)."""
    d = per_seed_metrics(df)
    vals = {}
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Method} & \textbf{BA (\%)} & \textbf{ASR (\%)} & \textbf{ASR 95\% CI} & \textbf{$\Delta$ASR (pp)} & \textbf{Gap to cf (pp)} & \textbf{BPR} \\", r"\midrule"]
    for cfg in cfgs:
        sub = d[(d.exp_id == cfg.exp_id) & np.isclose(d.alpha, alpha)]
        b = df[(df.exp_id == cfg.exp_id) & (df.method == "before")]
        name = DATASET_NAMES[cfg.dataset] + ("" if cfg.partition["scheme"] == "iid" else " (Dir)")
        rows = []
        sba, sasr = summarize(b.BA), summarize(b.ASR)
        rows.append(f" & {METHOD_NAMES['before']} & {_pm(sba)} & {_pm(sasr)} & {_ci(sasr)} & -- & -- & -- \\\\")
        vals[f"methods:{cfg.exp_id}:before:BA"] = sba["mean"]; vals[f"methods:{cfg.exp_id}:before:ASR"] = sasr["mean"]
        for m in METHOD_ORDER[1:]:
            r = sub[sub.method == m]
            if r.empty:
                continue
            s_ba, s_asr, s_red, s_gap, s_bpr = summarize(r.BA), summarize(r.ASR), summarize(r.asr_reduction), summarize(r.cf_gap), summarize(r.BPR)
            bpr = "--" if m == "cf_sameinit" or s_bpr["n"] == 0 else f"{_z(s_bpr['mean'], 2):.2f}"
            gap = "--" if m == "cf_sameinit" else f"{_z(100 * s_gap['mean'], 1):+.1f}"
            rows.append(f" & {METHOD_NAMES[m]} & {_pm(s_ba)} & {_pm(s_asr)} & {_ci(s_asr)} & {100 * s_red['mean']:+.1f} & {gap} & {bpr} \\\\")
            for k, s in (("BA", s_ba), ("ASR", s_asr), ("dASR", s_red), ("gap", s_gap), ("BPR", s_bpr)):
                vals[f"methods:{cfg.exp_id}:{m}:{k}"] = s["mean"]
                vals[f"methods:{cfg.exp_id}:{m}:{k}_std"] = s["std"]
                vals[f"methods:{cfg.exp_id}:{m}:{k}_n"] = s["n"]
        lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{{name}}}")
        lines += rows
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines), vals


def table_fraction(df: pd.DataFrame, cfgs: list, methods=("short_retrain", "federaser", "grad_negation", "fine_pruning", "cf_sameinit")) -> tuple[str, dict]:
    """ASR (mean +- std) versus nested deletion fraction, with the largest within-seed paired increase between adjacent
    fractions (mean over seeds with a paired bootstrap 95% CI) per method."""
    vals = {}
    alphas = sorted(set(round(a, 4) for a in df[df.method != "before"].alpha))
    head = " & ".join([f"$\\alpha={a:g}$" for a in alphas])
    lines = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{ll" + "c" * len(alphas) + "c}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Method} & " + head + r" & largest paired $\Delta$ASR (pp) \\", r"\midrule"]
    for cfg in cfgs:
        sub = df[df.exp_id == cfg.exp_id]
        name = DATASET_NAMES[cfg.dataset] + ("" if cfg.partition["scheme"] == "iid" else " (Dir)")
        rows = []
        for m in methods:
            r = sub[sub.method == m]
            if r.empty:
                continue
            cells = []
            for a in alphas:
                ra = r[np.isclose(r.alpha, a)]
                st = summarize(ra.ASR); cells.append(_pm(st))
                vals[f"fraction:{cfg.exp_id}:{m}:{a:g}:ASR"] = st["mean"]; vals[f"fraction:{cfg.exp_id}:{m}:{a:g}:ASR_std"] = st["std"]
                vals[f"fraction:{cfg.exp_id}:{m}:{a:g}:BA"] = summarize(ra.BA)["mean"]
            pairs = paired_fraction_differences(df, cfg.exp_id, m)
            for pr in pairs:
                for k in ("mean", "median", "lo", "hi", "max", "n"):
                    vals[f"paired:{cfg.exp_id}:{m}:{pr['alpha0']:g}-{pr['alpha1']:g}:{k}"] = pr[k]
            worst = max(pairs, key=lambda pr: pr["mean"]) if pairs else None
            if worst is not None:
                vals[f"paired:{cfg.exp_id}:{m}:largest_mean"] = worst["mean"]; vals[f"paired:{cfg.exp_id}:{m}:largest_lo"] = worst["lo"]; vals[f"paired:{cfg.exp_id}:{m}:largest_hi"] = worst["hi"]
                vals[f"paired:{cfg.exp_id}:{m}:largest_pair"] = f"{worst['alpha0']:g}-{worst['alpha1']:g}"
                cell = f"{_z(100 * worst['mean'], 1):+.1f} [{_z(100 * worst['lo'], 1):+.1f}, {_z(100 * worst['hi'], 1):+.1f}]"
            else:
                cell = "--"
            rows.append(f" & {SHORT_METHOD[m]} & " + " & ".join(cells) + f" & {cell} \\\\")
        lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{{name}}}")
        lines += rows
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines), vals


def table_paired_full(df: pd.DataFrame, cfgs: list, methods=("short_retrain", "federaser", "grad_negation", "fine_pruning", "cf_sameinit")) -> tuple[str, dict]:
    """Every adjacent-fraction paired difference (mean [95% CI], pp), for the supplement / processed results."""
    vals = {}
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"\textbf{Dataset} & \textbf{Method} & $0.25\to0.5$ & $0.5\to0.75$ & $0.75\to1$ \\", r"\midrule"]
    for cfg in cfgs:
        for m in methods:
            pairs = paired_fraction_differences(df, cfg.exp_id, m)
            if not pairs:
                continue
            cells = [f"{_z(100 * pr['mean'], 1):+.1f} [{_z(100 * pr['lo'], 1):+.1f}, {_z(100 * pr['hi'], 1):+.1f}]" for pr in pairs]
            lines.append(f"{DATASET_NAMES[cfg.dataset]}{'' if cfg.partition['scheme'] == 'iid' else ' (Dir)'} & {SHORT_METHOD[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def table_cost(df: pd.DataFrame, cfgs: list, alpha: float = 1.0) -> tuple[str, dict]:
    """Wall-clock runtime of every method relative to full retraining (same seed), mean over seeds."""
    vals = {}
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{l" + "c" * len(cfgs) + "}", r"\toprule",
             r"\textbf{Method} & " + " & ".join(DATASET_NAMES[c.dataset] + ("" if c.partition["scheme"] == "iid" else " (Dir)") for c in cfgs) + r" \\", r"\midrule"]
    for m in [mm for mm in METHOD_ORDER[1:] if mm != "cf_sameinit"]:
        cells = []
        for cfg in cfgs:
            sub = df[(df.exp_id == cfg.exp_id) & np.isclose(df.alpha, alpha)]
            fr = sub[sub.method == "full_retrain"].set_index("seed").runtime
            r = sub[sub.method == m].set_index("seed").runtime
            if r.empty:
                cells.append("--"); continue
            rel = (r / fr.reindex(r.index)).mean()
            cells.append(f"{r.mean():.0f}\\,s ({100 * rel:.0f}\\%)")
            vals[f"cost:{cfg.exp_id}:{m}:sec"] = float(r.mean()); vals[f"cost:{cfg.exp_id}:{m}:rel"] = float(rel)
        lines.append(f"{SHORT_METHOD[m].replace('Gradient', 'Grad.').replace(' (cf.)', '')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def table_endpoints(df: pd.DataFrame, cfgs: list, alpha: float = 1.0, margin: float = 0.5, tol: float = 0.05) -> tuple[str, dict]:
    """Primary endpoints per method: number of seeds satisfying E1 (ASR reduction >= margin) and E2 (ASR' <= ASR_cf + tol)."""
    d = per_seed_metrics(df)
    vals = {}
    methods = ["short_retrain", "federaser", "grad_negation", "fine_pruning"]
    abbr = {"short_retrain": "FT", "federaser": "FE", "grad_negation": "GN", "fine_pruning": "FP"}
    lines = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{l" + "cc" * len(methods) + "}", r"\toprule",
             r"\textbf{Experiment} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{abbr[m]}}}" for m in methods) + r" \\",
             " & " + " & ".join(["E1 & E2"] * len(methods)) + r" \\", r"\midrule"]
    for cfg in cfgs:
        cells = []
        for m in methods:
            r = d[(d.exp_id == cfg.exp_id) & np.isclose(d.alpha, alpha) & (d.method == m)]
            if r.empty:
                cells += ["--", "--"]; continue
            e1 = (r.asr_reduction >= margin).mean(); e2 = (r.ASR <= r.counterfactual_ASR + tol).mean()
            cells += [f"{int(round(e1 * len(r)))}/{len(r)}", f"{int(round(e2 * len(r)))}/{len(r)}"]
            vals[f"endpoints:{cfg.exp_id}:{m}:E1"] = float(e1); vals[f"endpoints:{cfg.exp_id}:{m}:E2"] = float(e2)
        name = DATASET_NAMES[cfg.dataset] + ("" if cfg.partition["scheme"] == "iid" else " (Dir)")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def table_fesens(df: pd.DataFrame, exp_ids=("cifar10_iid", "cifar10_iid_federaser_r1", "cifar10_iid_federaser_dt1"), labels=("defaults ($\\Delta t{=}2$, $r{=}0.5$)", "$r{=}1$ ($\\Delta t{=}2$)", "$\\Delta t{=}1$ ($r{=}0.5$)")) -> tuple[str, dict]:
    """FedEraser sensitivity to the calibration ratio / retaining interval (CIFAR-10 IID, alpha=1)."""
    vals = {}
    for e in exp_ids:
        if not ((df.exp_id == e) & (df.method == "federaser")).any():
            raise ValueError(f"FedEraser sensitivity experiment {e} has no completed runs")
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccc}", r"\toprule", r"\textbf{Setting} & \textbf{BA} & \textbf{ASR} & \textbf{BA$_{\mathrm{cf}}$} & \textbf{ASR$_{\mathrm{cf}}$} \\", r"\midrule"]
    for e, lab in zip(exp_ids, labels):
        r = df[(df.exp_id == e) & (df.method == "federaser") & np.isclose(df.alpha, 1.0)]
        cf = df[(df.exp_id == e) & (df.method == "cf_sameinit") & np.isclose(df.alpha, 1.0)]
        sba, sasr, cba, casr = summarize(r.BA), summarize(r.ASR), summarize(cf.BA), summarize(cf.ASR)
        lines.append(f"{lab} & {_pm(sba)} & {_pm(sasr)} & {_pm(cba)} & {_pm(casr)} \\\\")
        for k, s_ in (("BA", sba), ("ASR", sasr), ("cfBA", cba), ("cfASR", casr)):
            vals[f"fesens:{e}:{k}"] = s_["mean"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


ALIGN_METHODS = ["short_retrain", "fine_pruning", "grad_negation", "federaser"]
LAYER_NAMES = ["conv1", "conv2", "fc1", "fc2"]
GROUP_ORDER = ["conv1", "conv2", "fc1", "fc2", "stem", "layer1", "layer2", "layer3", "layer4", "fc"]


def energy_groups(rows: pd.DataFrame) -> list[str]:
    """Parameter groups present in an alignment dataframe (architecture-independent), in canonical order."""
    present = [c[:-len("_energy")] for c in rows.columns if c.endswith("_energy") and c != "global_energy"]
    present = [g for g in present if rows[f"{g}_energy"].notna().any()]
    return [g for g in GROUP_ORDER if g in present] + sorted(g for g in present if g not in GROUP_ORDER)


def table_resnet(df: pd.DataFrame, dfa: pd.DataFrame | None, cfg, alpha: float = 1.0, margin: float = 0.5, tol: float = 0.05) -> tuple[str, dict]:
    """Compact architecture-check table for one experiment (ResNet-18): BA, ASR, dASR, gap, BPR, E1/E2 counts,
    kappa (if the alignment file is given) and runtime. Values keys use the experiment id like the other tables."""
    d = per_seed_metrics(df); vals = {}
    sub = d[(d.exp_id == cfg.exp_id) & np.isclose(d.alpha, alpha)]
    b = df[(df.exp_id == cfg.exp_id) & (df.method == "before")]
    SHORT = {"before": "Compromised", "short_retrain": "Fine-tuning", "federaser": "FedEraser", "grad_negation": "Grad. negation", "fine_pruning": "Fine-Pruning", "full_retrain": "Retrain (indep.)", "cf_sameinit": "Counterfactual"}
    lines = [r"\setlength{\tabcolsep}{1.4pt}", r"\begin{tabular}{lccccccc}", r"\toprule",
             r"\textbf{Method} & \textbf{BA (\%)} & \textbf{ASR (\%)} & \textbf{$\Delta$ASR} & \textbf{Gap} & \textbf{BPR} & \textbf{E1/E2} & $\kappa$ \\", r"\midrule"]
    sba, sasr = summarize(b.BA), summarize(b.ASR)
    vals[f"methods:{cfg.exp_id}:before:BA"] = sba["mean"]; vals[f"methods:{cfg.exp_id}:before:ASR"] = sasr["mean"]
    vals[f"methods:{cfg.exp_id}:before:BA_std"] = sba["std"]; vals[f"methods:{cfg.exp_id}:before:ASR_std"] = sasr["std"]
    lines.append(f"{SHORT['before']} & {_pm(sba)} & {_pm(sasr)} & -- & -- & -- & -- & -- \\\\")
    for m in METHOD_ORDER[1:]:
        r = sub[sub.method == m]
        if r.empty:
            continue
        s_ba, s_asr, s_red, s_gap, s_bpr = summarize(r.BA), summarize(r.ASR), summarize(r.asr_reduction), summarize(r.cf_gap), summarize(r.BPR)
        raw = df[(df.exp_id == cfg.exp_id) & (df.method == m) & np.isclose(df.alpha, alpha)]
        s_t = summarize(raw.runtime) if "runtime" in raw else {"mean": float("nan"), "n": 0}
        bpr = "--" if m == "cf_sameinit" or s_bpr["n"] == 0 else f"{_z(s_bpr['mean'], 2):.2f}"
        gap = "--" if m == "cf_sameinit" else f"{_z(100 * s_gap['mean'], 1):+.1f}"
        if m in ("short_retrain", "federaser", "grad_negation", "fine_pruning"):
            e1 = (r.asr_reduction >= margin).mean(); e2 = (r.ASR <= r.counterfactual_ASR + tol).mean()
            ep = f"{int(round(e1 * len(r)))}/{len(r)}, {int(round(e2 * len(r)))}/{len(r)}"
            vals[f"endpoints:{cfg.exp_id}:{m}:E1"] = float(e1); vals[f"endpoints:{cfg.exp_id}:{m}:E2"] = float(e2)
        else:
            ep = "--"
        kap = "--"
        if dfa is not None and m in ("short_retrain", "federaser", "grad_negation", "fine_pruning"):
            ra = dfa[(dfa.exp_id == cfg.exp_id) & (dfa.method == m) & np.isclose(dfa.alpha, alpha)]
            if not ra.empty:
                sk = summarize(ra.global_rho); kap = f"{_z(sk['mean'], 2):.2f} $\\pm$ {_z(sk['std'], 2):.2f}"
                for c in ("global_cos_u_negdm", "global_rho", "global_resid_frac", "global_cos_u_dr", "global_cos_u_d", "global_rel_dist_cf"):
                    st = summarize(ra[c]); vals[f"align:{cfg.exp_id}:{m}:{c}"] = st["mean"]; vals[f"align:{cfg.exp_id}:{m}:{c}_std"] = st["std"]
                for g in energy_groups(ra):
                    st = summarize(ra[f"{g}_energy"]); vals[f"energy:{cfg.exp_id}:{m}:{g}"] = st["mean"]
        t = "--" if s_t["n"] == 0 else f"{s_t['mean']:.0f}"
        lines.append(f"{SHORT[m]} & {_pm(s_ba)} & {_pm(s_asr)} & {100 * s_red['mean']:+.1f} & {gap} & {bpr} & {ep} & {kap} \\\\")
        for k, s_ in (("BA", s_ba), ("ASR", s_asr), ("dASR", s_red), ("gap", s_gap), ("BPR", s_bpr)):
            vals[f"methods:{cfg.exp_id}:{m}:{k}"] = s_["mean"]; vals[f"methods:{cfg.exp_id}:{m}:{k}_std"] = s_["std"]; vals[f"methods:{cfg.exp_id}:{m}:{k}_n"] = s_["n"]
        if s_t["n"]:
            vals[f"cost:{cfg.exp_id}:{m}:sec"] = s_t["mean"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Spearman correlation with a bootstrap (over runs) 95% interval."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    r = float(stats.spearmanr(x, y).statistic)
    rng = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        i = rng.randint(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            bs.append(stats.spearmanr(x[i], y[i]).statistic)
    if not bs:
        return r, float("nan"), float("nan")
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def table_alignment(dfa: pd.DataFrame, cfgs: list, sens_exps=("cifar10_iid_federaser_r1", "cifar10_iid_federaser_dt1")) -> tuple[str, dict]:
    """Alignment measurements at alpha = 1 (mean +- std over seeds) per experiment and method, plus FedEraser variants.
    Also computes row-wise and cluster-aware (experiment-seed) Spearman correlations."""
    vals = {}
    cols = ["global_cos_u_negdm", "global_rho", "global_resid_frac", "global_cos_u_dr", "global_cos_u_d", "global_rel_dist_cf", "asr_reduction"]
    head = (r"\textbf{Experiment} & \textbf{Method} & $\cos(u,-\Delta_M)$ & $\kappa$ & $\|u_\perp\|/\|u\|$ & $\cos(u,\Delta_R)$ & $\cos(u,d)$ & $r_{\mathrm{all}}$ & $\Delta$ASR (pp) \\")
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llccccccc}", r"\toprule", head, r"\midrule"]

    def fmt(s_, digits=2, scale=1.0):
        return "--" if s_["n"] == 0 or math.isnan(s_["mean"]) else f"{_z(scale * s_['mean'], digits):.{digits}f} $\\pm$ {_z(scale * s_['std'], digits):.{digits}f}"

    def block(exp_id, label, methods, name_fn):
        rows = []
        for m in methods:
            r = dfa[(dfa.exp_id == exp_id) & (dfa.method == m) & np.isclose(dfa.alpha, 1.0)]
            if r.empty:
                continue
            cells = []
            for c in cols:
                st = summarize(r[c]); vals[f"align:{exp_id}:{m}:{c}"] = st["mean"]; vals[f"align:{exp_id}:{m}:{c}_std"] = st["std"]; vals[f"align:{exp_id}:{m}:{c}_ci"] = st["ci95_half"]
                cells.append(fmt(st, 1, 100.0) if c == "asr_reduction" else fmt(st))
            for l in energy_groups(r):
                st = summarize(r[f"{l}_energy"]); vals[f"align:{exp_id}:{m}:{l}_energy"] = st["mean"]; vals[f"align:{exp_id}:{m}:{l}_energy_std"] = st["std"]
                st = summarize(r[f"{l}_rel_dist_cf"]); vals[f"align:{exp_id}:{m}:{l}_rel_dist_cf"] = st["mean"]
            rows.append(f" & {name_fn(m)} & " + " & ".join(cells) + r" \\")
        if rows:
            lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{{label}}}"); lines.extend(rows); lines.append(r"\midrule")

    for cfg in cfgs:
        label = DATASET_NAMES[cfg.dataset] + ("" if cfg.partition["scheme"] == "iid" else " (Dir)")
        block(cfg.exp_id, label, ALIGN_METHODS, lambda m: SHORT_METHOD[m])
    sens_labels = {"cifar10_iid": "FedEraser, $\\Delta t{=}2$, $r{=}0.5$", "cifar10_iid_federaser_r1": "FedEraser, $\\Delta t{=}2$, $r{=}1$", "cifar10_iid_federaser_dt1": "FedEraser, $\\Delta t{=}1$, $r{=}0.5$"}
    rows = []
    for e in ("cifar10_iid",) + tuple(sens_exps):
        r = dfa[(dfa.exp_id == e) & (dfa.method == "federaser") & np.isclose(dfa.alpha, 1.0)]
        if r.empty:
            continue
        cells = []
        for c in cols:
            st = summarize(r[c]); vals[f"align:{e}:federaser:{c}"] = st["mean"]; vals[f"align:{e}:federaser:{c}_std"] = st["std"]
            cells.append(fmt(st, 1, 100.0) if c == "asr_reduction" else fmt(st))
        st = summarize(r["global_u_norm"]); vals[f"align:{e}:federaser:u_norm"] = st["mean"]
        rows.append(f" & {sens_labels[e]} & " + " & ".join(cells) + r" \\")
    if rows:
        lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{CIFAR-10 variants}}"); lines.extend(rows); lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    # correlations: row-wise bootstrap (previous) and cluster bootstrap over experiment-seed (used in the paper)
    main = dfa[dfa.exp_id.isin([c.exp_id for c in cfgs]) & dfa.method.isin(ALIGN_METHODS)].copy()
    main["cluster"] = main.exp_id + "/" + main.seed.astype(str)
    r_, lo, hi = spearman_ci(main.global_rho, main.asr_reduction); vals.update({"align:spearman:all:rho": r_, "align:spearman:all:lo": lo, "align:spearman:all:hi": hi, "align:spearman:all:n": len(main)})
    r_, lo, hi, ncl = spearman_ci_cluster(main.global_rho, main.asr_reduction, main.cluster); vals.update({"align:spearman:allcl:rho": r_, "align:spearman:allcl:lo": lo, "align:spearman:allcl:hi": hi, "align:spearman:allcl:n": ncl})
    for m in ALIGN_METHODS:
        sub = main[main.method == m]
        r_, lo, hi = spearman_ci(sub.global_rho, sub.asr_reduction); vals.update({f"align:spearman:{m}:rho": r_, f"align:spearman:{m}:lo": lo, f"align:spearman:{m}:hi": hi, f"align:spearman:{m}:n": len(sub)})
        r_, lo, hi, ncl = spearman_ci_cluster(sub.global_rho, sub.asr_reduction, sub.cluster); vals.update({f"align:spearman:{m}cl:rho": r_, f"align:spearman:{m}cl:lo": lo, f"align:spearman:{m}cl:hi": hi, f"align:spearman:{m}cl:n": ncl})
    fe = dfa[(dfa.method == "federaser") & np.isclose(dfa.alpha, 1.0) & dfa.cf0_available.astype(bool)].copy()
    if len(fe) > 3:
        fe["cluster"] = fe.exp_id.str.replace("_federaser_r1", "").str.replace("_federaser_dt1", "") + "/" + fe.seed.astype(str)
        r_, lo, hi = spearman_ci(fe.global_cos_u_d, fe.BA); vals.update({"align:spearman:fe_cos_d_BA": r_, "align:spearman:fe_cos_d_BA_lo": lo, "align:spearman:fe_cos_d_BA_hi": hi, "align:spearman:fe_cos_d_BA_n": len(fe)})
        r_, lo, hi, ncl = spearman_ci_cluster(fe.global_cos_u_d, fe.BA, fe.cluster); vals.update({"align:spearman:fe_cos_d_BAcl": r_, "align:spearman:fe_cos_d_BAcl_lo": lo, "align:spearman:fe_cos_d_BAcl_hi": hi, "align:spearman:fe_cos_d_BAcl_n": ncl})
    return "\n".join(lines), vals


def table_energy(dfa: pd.DataFrame, cfgs: list, exp_id: str = "cifar10_iid") -> tuple[str, dict]:
    """Normalised layer-wise update energy ||u_l||^2/||u||^2 at alpha = 1 (mean +- std over seeds) for one experiment."""
    vals = {}
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "c" * len(LAYER_NAMES) + "}", r"\toprule", r"\textbf{Method} & " + " & ".join(LAYER_NAMES) + r" \\", r"\midrule"]
    for m in ALIGN_METHODS:
        r = dfa[(dfa.exp_id == exp_id) & (dfa.method == m) & np.isclose(dfa.alpha, 1.0)]
        if r.empty:
            continue
        cells = []
        for l in LAYER_NAMES:
            st = summarize(r[f"{l}_energy"]); cells.append(f"{_z(st['mean'], 2):.2f} $\\pm$ {_z(st['std'], 2):.2f}"); vals[f"energy:{exp_id}:{m}:{l}"] = st["mean"]
        lines.append(f"{SHORT_METHOD[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def table_endpoint_sensitivity(df: pd.DataFrame, cfgs: list, alpha: float = 1.0, margins=(0.3, 0.5, 0.7), tols=(0.02, 0.05, 0.10)) -> tuple[str, dict]:
    """Seeds satisfying E1 / E2 for alternative thresholds, pooled over the IID experiments (per method)."""
    d = per_seed_metrics(df); vals = {}
    iid = [c.exp_id for c in cfgs if c.partition["scheme"] == "iid"]
    sub = d[d.exp_id.isin(iid) & np.isclose(d.alpha, alpha)]
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{l" + "c" * (len(margins) + len(tols)) + "}", r"\toprule",
             r"\textbf{Method} & " + " & ".join(f"$\\mu{{=}}{int(100*m)}$" for m in margins) + " & " + " & ".join(f"$\\epsilon{{=}}{int(100*t)}$" for t in tols) + r" \\",
             " & " + r"\multicolumn{" + str(len(margins)) + r"}{c}{E1} & \multicolumn{" + str(len(tols)) + r"}{c}{E2} \\", r"\midrule"]
    for m in ["short_retrain", "federaser", "grad_negation", "fine_pruning"]:
        r = sub[sub.method == m]
        if r.empty:
            continue
        cells = []
        for mu in margins:
            k = int((r.asr_reduction >= mu).sum()); cells.append(f"{k}/{len(r)}"); vals[f"epsens:{m}:E1:{mu:g}"] = k / len(r)
        for t in tols:
            k = int((r.ASR <= r.counterfactual_ASR + t).sum()); cells.append(f"{k}/{len(r)}"); vals[f"epsens:{m}:E2:{t:g}"] = k / len(r)
        lines.append(f"{SHORT_METHOD[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def table_subsets(dfs: pd.DataFrame) -> tuple[str, dict]:
    """Attacker-subset sensitivity: for every subset size, ASR of the nested (predefined) subset vs the range over all
    subsets of that size (mean over seeds of the per-seed min and max), per experiment and method."""
    vals = {}
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llccc}", r"\toprule", r"\textbf{Experiment} & \textbf{Method} & $|\mathcal{M}_{\mathrm{del}}|$ & nested & all subsets: mean [min, max] \\", r"\midrule"]
    for e in sorted(dfs.exp_id.unique()):
        for m in ("short_retrain", "federaser"):
            sub = dfs[(dfs.exp_id == e) & (dfs.method == m)]
            if sub.empty:
                continue
            for k in sorted(sub.n_deleted.unique()):
                sk = sub[sub.n_deleted == k]
                per_seed = sk.groupby("seed").ASR.agg(["mean", "min", "max"])
                nested = sk[sk.nested].groupby("seed").ASR.mean()
                name = {"cifar10_iid": "CIFAR-10", "cifar10_dirichlet": "CIFAR-10 (Dir)"}.get(e, e)
                lines.append(f"{name} & {SHORT_METHOD[m].replace(' (retained)', '')} & {k} & {100*nested.mean():.1f} & {100*per_seed['mean'].mean():.1f} [{100*per_seed['min'].mean():.1f}, {100*per_seed['max'].mean():.1f}] \\\\")
                vals[f"subset:{e}:{m}:{k}:nested"] = nested.mean(); vals[f"subset:{e}:{m}:{k}:mean"] = per_seed["mean"].mean(); vals[f"subset:{e}:{m}:{k}:min"] = per_seed["min"].mean(); vals[f"subset:{e}:{m}:{k}:max"] = per_seed["max"].mean()
                vals[f"subset:{e}:{m}:{k}:range"] = (per_seed["max"] - per_seed["min"]).mean()
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


def table_negation_phases(dfn: pd.DataFrame, cfgs: list) -> tuple[str, dict]:
    """Gradient negation: BA/ASR after the ascent phase and after the recovery phase (alpha = 1, mean +- std over seeds)."""
    vals = {}
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccc}", r"\toprule", r"\textbf{Experiment} & BA$_{\mathrm{asc}}$ & ASR$_{\mathrm{asc}}$ & BA$_{\mathrm{rec}}$ & ASR$_{\mathrm{rec}}$ \\", r"\midrule"]
    for cfg in cfgs:
        r = dfn[(dfn.exp_id == cfg.exp_id) & np.isclose(dfn.alpha, 1.0)]
        if r.empty:
            continue
        cells = []
        for c in ("BA_before", "ASR_before", "BA_after_ascent", "ASR_after_ascent", "BA_after_recovery", "ASR_after_recovery"):
            st = summarize(r[c]); vals[f"negphase:{cfg.exp_id}:{c}"] = st["mean"]
            if c not in ("BA_before", "ASR_before"):
                cells.append(_pm(st))
        vals[f"negphase:{cfg.exp_id}:reproduced"] = float(r.reproduced_checkpoint_identical.mean())
        lines.append(f"{DATASET_NAMES[cfg.dataset]}{'' if cfg.partition['scheme'] == 'iid' else ' (Dir)'} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), vals


# ---------------------------------------------------------------------------------------------------------------
# Paired (within-seed) analysis of adjacent deletion fractions and cluster-aware correlation
# ---------------------------------------------------------------------------------------------------------------

def paired_bootstrap_ci(diffs, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float, float]:
    """Mean, median and percentile-bootstrap 95% CI of the mean, resampling seeds (paired differences) with replacement."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.median(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_fraction_differences(df: pd.DataFrame, exp_id: str, method: str) -> list[dict]:
    """For adjacent deletion fractions, the within-seed ASR difference ASR(alpha_{j+1}) - ASR(alpha_j) (fraction units)."""
    sub = df[(df.exp_id == exp_id) & (df.method == method)]
    alphas = sorted(set(round(a, 4) for a in sub.alpha))
    out = []
    for a0, a1 in zip(alphas[:-1], alphas[1:]):
        diffs = []
        for s in sorted(sub.seed.unique()):
            r0 = sub[(sub.seed == s) & np.isclose(sub.alpha, a0)]; r1 = sub[(sub.seed == s) & np.isclose(sub.alpha, a1)]
            if len(r0) == 1 and len(r1) == 1:
                diffs.append(float(r1.ASR.iloc[0]) - float(r0.ASR.iloc[0]))
        mean, med, lo, hi = paired_bootstrap_ci(diffs)
        out.append({"alpha0": a0, "alpha1": a1, "n": len(diffs), "mean": mean, "median": med, "lo": lo, "hi": hi, "max": max(diffs) if diffs else float("nan")})
    return out


def spearman_ci_cluster(x, y, clusters, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, int]:
    """Spearman correlation with a cluster bootstrap: whole clusters (experiment-seed) are resampled with replacement."""
    from scipy import stats
    x, y, c = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y); x, y, c = x[ok], y[ok], c[ok]
    ids = np.unique(c)
    if len(x) < 4 or len(ids) < 2:
        return float("nan"), float("nan"), float("nan"), int(len(ids))
    r = float(stats.spearmanr(x, y).statistic)
    rng = np.random.RandomState(seed); bs = []
    groups = {i: np.where(c == i)[0] for i in ids}
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([groups[i] for i in pick])
        if len(set(x[idx])) > 1 and len(set(y[idx])) > 1:
            bs.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if not bs:
        return r, float("nan"), float("nan"), int(len(ids))
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(ids))


def table_fedup(df: pd.DataFrame, dfa: pd.DataFrame | None, cfgs: list, alpha: float = 1.0, margin: float = 0.5, tol: float = 0.05) -> tuple[str, dict]:
    """FedUP baseline (v8) at alpha = 1, transposed (metrics as rows, one column per architecture): mean +- std,
    ASR 95% CI, dASR, gap, BPR, E1/E2, kappa, runtime and server storage. Values keys: fedup:<exp_id>:<name>; the
    per-seed Markdown table is returned under the key 'fedup:per_seed_md'."""
    d = per_seed_metrics(df); vals = {}
    arch = {"cifar10_iid": "SimpleCNN", "cifar10_iid_resnet18": "ResNet-18"}
    cols, cells = [], {k: [] for k in ("BA", "ASR", "CI", "dASR", "gap", "BPR", "E", "kappa", "time", "storage")}
    per_seed = [r"| Model | seed | BA (%) | ASR (%) | counterfactual ASR (%) | dASR (pp) | gap (pp) | BPR | E1 | E2 | runtime (s) | pruned weights |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in cfgs:
        r = d[(d.exp_id == cfg.exp_id) & np.isclose(d.alpha, alpha) & (d.method == "fedup")]
        if r.empty:
            continue
        cols.append(arch.get(cfg.exp_id, cfg.exp_id))
        raw = df[(df.exp_id == cfg.exp_id) & (df.method == "fedup") & np.isclose(df.alpha, alpha)]
        s_ba, s_asr, s_red, s_gap, s_bpr, s_t = summarize(r.BA), summarize(r.ASR), summarize(r.asr_reduction), summarize(r.cf_gap), summarize(r.BPR), summarize(raw.runtime)
        e1 = (r.asr_reduction >= margin).mean(); e2 = (r.ASR <= r.counterfactual_ASR + tol).mean()
        kap = "--"
        if dfa is not None:
            ra = dfa[(dfa.exp_id == cfg.exp_id) & (dfa.method == "fedup") & np.isclose(dfa.alpha, alpha)]
            if not ra.empty:
                sk = summarize(ra.global_rho); kap = f"{_z(sk['mean'], 2):.2f} $\\pm$ {_z(sk['std'], 2):.2f}"
                for c in ("global_cos_u_negdm", "global_rho", "global_resid_frac", "global_cos_u_dr", "global_cos_u_d", "global_rel_dist_cf"):
                    st = summarize(ra[c]); vals[f"fedup:{cfg.exp_id}:{c}"] = st["mean"]; vals[f"fedup:{cfg.exp_id}:{c}_std"] = st["std"]
        storage = raw.server_storage_bytes.mean() / 2**20 if "server_storage_bytes" in raw and raw.server_storage_bytes.notna().any() else float("nan")
        cells["BA"].append(_pm(s_ba)); cells["ASR"].append(_pm(s_asr)); cells["CI"].append(_ci(s_asr)); cells["dASR"].append(f"{100 * s_red['mean']:+.1f}")
        cells["gap"].append(f"{_z(100 * s_gap['mean'], 1):+.1f}"); cells["BPR"].append(f"{_z(s_bpr['mean'], 2):.2f}"); cells["E"].append(f"{int(round(e1 * len(r)))}/{len(r)}, {int(round(e2 * len(r)))}/{len(r)}")
        cells["kappa"].append(kap); cells["time"].append(f"{s_t['mean']:.0f}"); cells["storage"].append(f"{storage:.0f}")
        for k, st in (("BA", s_ba), ("ASR", s_asr), ("dASR", s_red), ("gap", s_gap), ("BPR", s_bpr), ("sec", s_t)):
            vals[f"fedup:{cfg.exp_id}:{k}"] = st["mean"]; vals[f"fedup:{cfg.exp_id}:{k}_std"] = st["std"]; vals[f"fedup:{cfg.exp_id}:{k}_cilo"] = st["ci95_low"]; vals[f"fedup:{cfg.exp_id}:{k}_cihi"] = st["ci95_high"]
        vals[f"fedup:{cfg.exp_id}:E1"] = float(e1); vals[f"fedup:{cfg.exp_id}:E2"] = float(e2); vals[f"fedup:{cfg.exp_id}:storage_mb"] = float(storage); vals[f"fedup:{cfg.exp_id}:n"] = int(len(r))
        vals[f"fedup:{cfg.exp_id}:maxgap"] = float(r.cf_gap.max()); vals[f"fedup:{cfg.exp_id}:maxbpr"] = float(r.BPR.max())
        for _, row in r.sort_values("seed").iterrows():
            rr = raw[raw.seed == row.seed].iloc[0]
            per_seed.append(f"| {arch.get(cfg.exp_id, cfg.exp_id)} | {int(row.seed)} | {100*row.BA:.2f} | {100*row.ASR:.2f} | {100*row.counterfactual_ASR:.2f} | {100*row.asr_reduction:+.2f} | {100*row.cf_gap:+.2f} | {row.BPR:.3f} | {int(row.asr_reduction >= margin)} | {int(row.ASR <= row.counterfactual_ASR + tol)} | {rr.runtime:.0f} | {int(rr.fedup_n_pruned) if 'fedup_n_pruned' in rr and rr.fedup_n_pruned == rr.fedup_n_pruned else '--'} |")
    labels = [("BA", "BA (\\%)"), ("ASR", "ASR (\\%)"), ("CI", "ASR 95\\% CI (\\%)"), ("dASR", "$\\Delta$ASR (pp)"), ("gap", "Gap to counterfactual (pp)"), ("BPR", "BPR"), ("E", "E1 / E2 (seeds of 5)"), ("kappa", "$\\kappa$"), ("time", "Wall-clock time (s)"), ("storage", "Server storage (MB)")]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", r"\textbf{Quantity} & " + " & ".join(f"\\textbf{{{c}}}" for c in cols) + r" \\", r"\midrule"]
    for key, lab in labels:
        lines.append(f"{lab} & " + " & ".join(cells[key]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    vals["fedup:per_seed_md"] = "\n".join(per_seed)
    return "\n".join(lines), vals

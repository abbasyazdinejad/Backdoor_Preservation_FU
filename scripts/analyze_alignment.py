#!/usr/bin/env python3
"""Compute the counterfactual-alignment measurements for every unlearning run from the verified checkpoints.

  python scripts/analyze_alignment.py --configs configs/cifar10_iid.json ... --out results/processed/legacy_v1/alignment.csv
Per run: global and per-layer cos(u, -Delta_M), rho, cos(u, Delta_R), and (alpha = 1 only, or where a same-init
counterfactual exists) cos(u, d) and the relative residual distance to the same-initialisation counterfactual.
Checkpoint checksums are verified against the run records before use.
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, torch
from bpa_fu.experiment import ExperimentConfig, load_history
from bpa_fu.alignment import malicious_contribution, alignment_metrics
from bpa_fu.models import build_model, parameter_groups, parameter_keys
from bpa_fu.datasets import DATASET_SPECS
from bpa_fu.reproducibility import state_dict_sha256, file_sha256

p = argparse.ArgumentParser(); p.add_argument("--configs", nargs="+", required=True); p.add_argument("--out", default="results/processed/legacy_v1/alignment.csv")
p.add_argument("--sameinit-from", default="cifar10_iid", help="exp_id whose cf_sameinit is reused for the CIFAR-10 FedEraser sensitivity experiments (identical initialisation)")
p.add_argument("--methods", nargs="*", default=None, help="methods to analyse (default: every method of the config except cf_sameinit)")
p.add_argument("--cf-from", nargs="*", default=None, help="results tree(s) holding the same-initialisation counterfactual records (v8: reuse of existing cf_sameinit runs)")
a = p.parse_args()

def load_ckpt(rec):
    sd = torch.load(rec["checkpoint_path"], map_location="cpu")
    assert state_dict_sha256(sd) == rec["checkpoint_sha256"], f"checksum mismatch {rec['run_id']}"
    return sd

rows = []
for cpath in a.configs:
    cfg = ExperimentConfig.load(cpath)
    spec = DATASET_SPECS[cfg.dataset]; _m = build_model(cfg.architecture, spec["in_channels"], spec["num_classes"])
    groups, pkeys = parameter_groups(_m), parameter_keys(_m)   # architecture-independent stable module groups; parameters only
    for seed in cfg.seeds:
        res_dir, _, _ = cfg.seed_dirs(seed)
        recs = {}
        for f in glob.glob(str(res_dir / "*.json")):
            if f.endswith("partition_and_poison.json"): continue
            r = json.load(open(f))
            if "method" not in r: continue
            recs[(r["method"], float(r["alpha"]))] = r
        if ("before", 0.0) not in recs and cfg.compromised_results_dir:
            recs[("before", 0.0)] = json.load(open(f"{cfg.compromised_results_dir}/{cfg.exp_id}/seed{seed}/compromised.json"))
        for d in (a.cf_from or []):
            for f in glob.glob(f"{d}/{cfg.exp_id}/seed{seed}/cf_sameinit_alpha*.json"):
                r = json.load(open(f))
                if r.get("status") == "completed" and ("cf_sameinit", float(r["alpha"])) not in recs: recs[("cf_sameinit", float(r["alpha"]))] = r
        comp = recs[("before", 0.0)]; theta_star = load_ckpt(comp)
        assert file_sha256(comp["history_path"]) == comp["history_sha256"]
        hist = load_history(comp["history_path"])
        hist_d = {"client_updates": hist.client_updates, "client_sizes": hist.client_sizes}
        for alpha in cfg.alphas:
            deleted = cfg.deletion_set(alpha)
            dm, dr = malicious_contribution(hist_d, deleted)
            cf0_rec = recs.get(("cf_sameinit", float(alpha)))
            if cf0_rec is None and cfg.exp_id != a.sameinit_from and cfg.dataset == "cifar10" and cfg.partition["scheme"] == "iid":
                alt_path = f"{cfg.results_dir}/{a.sameinit_from}/seed{seed}/cf_sameinit_alpha{alpha:.2f}.json"
                if os.path.exists(alt_path):
                    alt = json.load(open(alt_path))
                    if alt["compromised_checkpoint_sha256"] == comp["checkpoint_sha256"]:
                        cf0_rec = alt
            theta_cf0 = load_ckpt(cf0_rec) if cf0_rec else None
            for m in (a.methods or [mm for mm in cfg.methods if mm != "cf_sameinit"]):
                r = recs.get((m, float(alpha)))
                if r is None or r.get("status") != "completed": raise SystemExit(f"missing run {cfg.exp_id} seed {seed} {m} {alpha}")
                theta_p = load_ckpt(r)
                met = alignment_metrics(theta_star, theta_p, theta_cf0, dm, dr, groups=groups, param_keys=pkeys)
                rows.append(dict(exp_id=cfg.exp_id, dataset=cfg.dataset, architecture=cfg.architecture, partition=cfg.partition["scheme"], seed=seed, method=m, alpha=alpha, run_id=r["run_id"],
                                 BA=r["BA"], ASR=r["ASR"], ASR_before=comp["ASR"], asr_reduction=comp["ASR"] - r["ASR"],
                                 cf0_available=theta_cf0 is not None, cf0_BA=(cf0_rec or {}).get("BA"), cf0_ASR=(cf0_rec or {}).get("ASR"),
                                 history_rounds=len(hist.rounds), history_every=cfg.history_every, **met))
            print(cfg.exp_id, seed, alpha, "done", flush=True)
df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out), exist_ok=True); df.to_csv(a.out, index=False)
print("wrote", a.out, len(df), "rows")

#!/usr/bin/env python3
"""Collect the attacker-subset study records into results/processed/corrected_v2/attacker_subsets.csv."""
import glob, json, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
rows = []
for f in sorted(glob.glob("results/raw/attacker_subsets_v2/*/seed*/*_del*.json")):
    r = json.load(open(f))
    if r.get("status") != "completed": raise SystemExit(f"incomplete {f}")
    d = sorted(r["deleted_clients"]); nested = d == list(range(len(d)))
    rows.append(dict(exp_id=r["exp_id"], seed=r["seed"], method=r["method"], subset="-".join(map(str, d)), n_deleted=len(d), nested=nested, alpha=r["alpha"], BA=r["BA"], ASR=r["ASR"], run_id=r["run_id"], checkpoint_sha256=r["checkpoint_sha256"]))
df = pd.DataFrame(rows); os.makedirs("results/processed/corrected_v2", exist_ok=True); df.to_csv("results/processed/corrected_v2/attacker_subsets.csv", index=False)
print("wrote results/processed/corrected_v2/attacker_subsets.csv", len(df), "rows")

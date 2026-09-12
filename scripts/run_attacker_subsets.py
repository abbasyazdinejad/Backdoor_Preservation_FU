#!/usr/bin/env python3
"""Attacker-subset sensitivity study: every subset of the malicious clients (sizes 1..M) for the given methods,
using the unchanged compromised checkpoints. Records: <results_dir>/<exp_id>/seed<k>/<method>_del<ids>.json
  python scripts/run_attacker_subsets.py configs/v2/subsets_cifar10_iid.json --methods short_retrain federaser
"""
import argparse, itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bpa_fu.experiment import ExperimentConfig, run_unlearning
p = argparse.ArgumentParser(); p.add_argument("config"); p.add_argument("--methods", nargs="+", required=True); p.add_argument("--seeds", type=int, nargs="*"); a = p.parse_args()
cfg = ExperimentConfig.load(a.config); M = sorted(cfg.malicious_clients)
subsets = [list(s) for k in range(1, len(M) + 1) for s in itertools.combinations(M, k)]
print(f"{len(subsets)} subsets x {len(a.methods)} methods x {len(a.seeds or cfg.seeds)} seeds")
for seed in (a.seeds or cfg.seeds):
    for sub in subsets:
        for m in a.methods:
            r = run_unlearning(cfg, seed, m, alpha=len(sub) / len(M), deleted=sub)
            print(r["run_id"], f"BA={r['BA']:.4f} ASR={r['ASR']:.4f}", flush=True)
print("SUBSETS_DONE")

#!/usr/bin/env python3
"""Run (part of) an experiment defined by a config file.

Examples
  python scripts/run_experiment.py configs/cifar10_iid.json                    # all seeds/methods/alphas
  python scripts/run_experiment.py configs/cifar10_iid.json --seeds 0 --methods federaser --alphas 1.0
  python scripts/run_experiment.py configs/cifar10_iid.json --train-only --seeds 0
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from bpa_fu.experiment import ExperimentConfig, run_all, train_compromised

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--seeds", type=int, nargs="*")
p.add_argument("--methods", nargs="*")
p.add_argument("--alphas", type=float, nargs="*")
p.add_argument("--train-only", action="store_true")
p.add_argument("--force", action="store_true")
a = p.parse_args()
cfg = ExperimentConfig.load(a.config)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if a.train_only:
    for s in (a.seeds or cfg.seeds):
        train_compromised(cfg, s, force=a.force)
else:
    run_all(cfg, seeds=a.seeds, methods=a.methods, alphas=a.alphas, force=a.force)

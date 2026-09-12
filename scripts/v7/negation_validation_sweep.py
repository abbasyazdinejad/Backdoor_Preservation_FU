#!/usr/bin/env python3
"""Validation-split sweep of the gradient-negation loss-threshold multiplier (Rule G of
audit/v7_hyperparameter_selection_protocol.md). Uses the pilot compromised model of the given config (trained with
eval_split = "validation"); every BA/ASR below is computed on the held-out validation split, never on the test set.
  python scripts/v7/negation_validation_sweep.py --config configs/v7/pilot_negval_cifar10_simplecnn.json --out results/raw/pilots_hyperparameter_selection/negation_validation_sweep_simplecnn.csv
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd, torch
from bpa_fu.experiment import ExperimentConfig, _prepare, _model_factory, train_compromised
from bpa_fu.unlearning.gradient_negation import gradient_negation
from bpa_fu.unlearning.short_retraining import short_retraining
from bpa_fu.evaluation import evaluate
from bpa_fu.reproducibility import get_device, set_seed
p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--out", required=True)
p.add_argument("--mults", type=float, nargs="+", default=[1.0, 2.0, 4.0]); p.add_argument("--lr", type=float, default=0.002); p.add_argument("--radius", type=float, default=1.0)
a = p.parse_args()
cfg = ExperimentConfig.load(a.config); assert cfg.eval_split == "validation", "Rule G must be applied on the validation split"
seed = cfg.seeds[0]; device = get_device(cfg.device)
comp = train_compromised(cfg, seed)   # trains the pilot model if absent; verifies artifacts otherwise
set_seed(seed); train, val, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
deleted = cfg.deletion_set(1.0); retained = [c for c in range(cfg.num_clients) if c not in deleted]
rows = []
def fresh():
    m = _model_factory(cfg, spec)().to(device); m.load_state_dict(torch.load(comp["checkpoint_path"])); return m
set_seed(seed); m_ft = short_retraining(fresh(), client_ds, retained, deleted, cfg.local_cfg(), device, seed, rounds=cfg.methods["short_retrain"]["rounds"])
e_ft = evaluate(m_ft, val, trig, cfg.target_label, device)
rows.append(dict(setting="fine_tuning_reference", threshold_mult=float("nan"), ascent_lr=float("nan"), radius=float("nan"), val_BA=e_ft["BA"], val_ASR=e_ft["ASR"], eval_split="validation", n_eval=len(val)))
print(rows[-1], flush=True)
for mult in a.mults:
    set_seed(seed); t0 = time.time()
    m = gradient_negation(fresh(), client_ds, retained, deleted, cfg.local_cfg(), device, seed, ascent_epochs=5, ascent_lr=a.lr, radius=a.radius,
                          loss_threshold=mult * math.log(spec["num_classes"]), recovery_rounds=5, check_every=1)
    e = evaluate(m, val, trig, cfg.target_label, device)
    rows.append(dict(setting="grad_negation", threshold_mult=mult, ascent_lr=a.lr, radius=a.radius, val_BA=e["BA"], val_ASR=e["ASR"], eval_split="validation", n_eval=len(val), sec=time.time() - t0))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out), exist_ok=True); df.to_csv(a.out, index=False)
ok = df[(df.setting == "grad_negation") & (df.val_BA >= e_ft["BA"] - 0.01)]
sel = ok.threshold_mult.max() if len(ok) else float("nan")
print(f"RULE_G: fine-tuning val BA={e_ft['BA']:.4f}; selected multiplier={sel}")
json.dump({"config": a.config, "rule": "largest multiplier with val_BA >= fine-tuning val_BA - 0.01", "fine_tuning_val_BA": e_ft["BA"], "selected_multiplier": None if sel != sel else float(sel),
           "candidates": a.mults, "ascent_lr": a.lr, "radius": a.radius, "eval_split": "validation"}, open(a.out.replace(".csv", "_selection.json"), "w"), indent=2)

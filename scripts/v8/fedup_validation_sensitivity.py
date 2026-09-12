#!/usr/bin/env python3
"""FedUP pruning-percentage sensitivity on the validation split of a pilot model (robustness check only; P = 10% is
taken from the source and is not selected here). Every BA/ASR below is computed on the held-out validation split.
  python scripts/v8/fedup_validation_sensitivity.py --config configs/v8/pilot_fedup_sens_simplecnn.json --out results/raw/pilots_fedup_sensitivity/fedup_sensitivity_simplecnn.csv
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd, torch
from bpa_fu.experiment import ExperimentConfig, _prepare, _model_factory, train_compromised, reconstruct_last_round, Logger
from bpa_fu.unlearning.fedup import fedup
from bpa_fu.unlearning.short_retraining import short_retraining
from bpa_fu.evaluation import evaluate
from bpa_fu.reproducibility import get_device, set_seed
p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--out", required=True); p.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20]); a = p.parse_args()
cfg = ExperimentConfig.load(a.config); assert cfg.eval_split == "validation"
seed = cfg.seeds[0]; device = get_device(cfg.device)
comp = train_compromised(cfg, seed)
set_seed(seed); train, val, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
deleted = cfg.deletion_set(1.0); retained = [c for c in range(cfg.num_clients) if c not in deleted]
os.makedirs(os.path.dirname(a.out), exist_ok=True); log = Logger(os.path.join(cfg.logs_dir, cfg.exp_id, "fedup_sensitivity.log") if False else __import__("pathlib").Path(a.out.replace(".csv", ".log")))
last, gprev, info = reconstruct_last_round(cfg, seed, comp, client_ds, spec, device, log=log)
rows = []
def fresh():
    m = _model_factory(cfg, spec)().to(device); m.load_state_dict(torch.load(comp["checkpoint_path"])); return m
set_seed(seed); m_ft = short_retraining(fresh(), client_ds, retained, deleted, cfg.local_cfg(), device, seed, rounds=5)
e = evaluate(m_ft, val, trig, cfg.target_label, device)
rows.append(dict(setting="fine_tuning_reference", prune_fraction=float("nan"), val_BA=e["BA"], val_ASR=e["ASR"], eval_split="validation", n_eval=len(val)))
print(rows[-1], flush=True)
for f in a.fracs:
    set_seed(seed); t0 = time.time()
    m = fedup(fresh(), client_ds, retained, deleted, cfg.local_cfg(), device, seed, last_round_models=last, global_prev=gprev, prune_fraction=f, recovery_rounds=5, log=log)
    e = evaluate(m, val, trig, cfg.target_label, device)
    rows.append(dict(setting="fedup", prune_fraction=f, val_BA=e["BA"], val_ASR=e["ASR"], eval_split="validation", n_eval=len(val), n_pruned=m.fedup_n_pruned, sec=time.time() - t0))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False); json.dump(info, open(a.out.replace(".csv", "_reconstruction.json"), "w"), indent=2); print("wrote", a.out)

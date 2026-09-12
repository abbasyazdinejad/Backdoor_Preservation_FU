#!/usr/bin/env python3
"""Gradient negation: BA/ASR immediately after the ascent phase and after the recovery phase, reproduced from the
unchanged compromised checkpoints with the identical seeds. The reproduced final model must match the stored run's
checkpoint checksum (determinism check); otherwise the record is flagged.
  python scripts/ablate_negation_phases.py --configs configs/cifar10_iid.json ... --out results/processed/corrected_v2/negation_phases.csv
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, torch
from bpa_fu.experiment import ExperimentConfig, _prepare, _model_factory
from bpa_fu.unlearning.gradient_negation import gradient_negation
from bpa_fu.evaluation import evaluate
from bpa_fu.reproducibility import get_device, set_seed, state_dict_sha256
p = argparse.ArgumentParser(); p.add_argument("--configs", nargs="+", required=True); p.add_argument("--out", required=True); a = p.parse_args()
rows = []
for cpath in a.configs:
    cfg = ExperimentConfig.load(cpath); device = get_device(cfg.device); params = dict(cfg.methods["grad_negation"])
    for seed in cfg.seeds:
        comp = json.load(open(f"results/raw/legacy_v1/{cfg.exp_id}/seed{seed}/compromised.json"))
        set_seed(seed); train, test, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
        for alpha in cfg.alphas:
            rec = json.load(open(f"results/raw/legacy_v1/{cfg.exp_id}/seed{seed}/grad_negation_alpha{alpha:.2f}.json"))
            deleted = cfg.deletion_set(alpha); retained = [c for c in range(cfg.num_clients) if c not in deleted]
            model = _model_factory(cfg, spec)().to(device); model.load_state_dict(torch.load(comp["checkpoint_path"]))
            set_seed(seed)
            # ascent only
            t0 = time.time()
            m_asc = gradient_negation(model, client_ds, retained, deleted, cfg.local_cfg(), device, seed, num_classes=spec["num_classes"], **dict(params, recovery_rounds=0))
            e_asc = evaluate(m_asc, test, trig, cfg.target_label, device)
            # full method (ascent + recovery), identical seeds -> must reproduce the stored checkpoint
            model2 = _model_factory(cfg, spec)().to(device); model2.load_state_dict(torch.load(comp["checkpoint_path"]))
            set_seed(seed)
            m_full = gradient_negation(model2, client_ds, retained, deleted, cfg.local_cfg(), device, seed, num_classes=spec["num_classes"], **params)
            e_full = evaluate(m_full, test, trig, cfg.target_label, device)
            sha = state_dict_sha256({k: v.cpu() for k, v in m_full.state_dict().items()})
            rows.append(dict(exp_id=cfg.exp_id, seed=seed, alpha=alpha, BA_before=comp["BA"], ASR_before=comp["ASR"], BA_after_ascent=e_asc["BA"], ASR_after_ascent=e_asc["ASR"],
                             BA_after_recovery=e_full["BA"], ASR_after_recovery=e_full["ASR"], stored_BA=rec["BA"], stored_ASR=rec["ASR"],
                             reproduced_checkpoint_identical=(sha == rec["checkpoint_sha256"]), runtime_sec=time.time() - t0))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
pd.DataFrame(rows).to_csv(a.out, index=False); print("NEGATION_PHASES_DONE", len(rows))

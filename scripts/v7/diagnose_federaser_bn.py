#!/usr/bin/env python3
"""Diagnostic for the ResNet-18 FedEraser records: does the utility collapse come from the BatchNorm running statistics
(buffers that the tensor-wise calibration rescales like parameters) rather than from the weights?
For every FedEraser record of the given config: (a) re-evaluate the stored checkpoint (must reproduce the record);
(b) re-estimate every BatchNorm running mean/variance by forward passes over the retained clients' data with the
parameters frozen (cumulative moving average, no gradient step), then evaluate again. This is a diagnostic evaluation
of an existing checkpoint, not an unlearning procedure; it accesses retained-client data only. Rows go to --out.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd, torch
from torch.utils.data import ConcatDataset, DataLoader
from bpa_fu.experiment import ExperimentConfig, _prepare, _model_factory
from bpa_fu.evaluation import evaluate
from bpa_fu.reproducibility import get_device, set_seed, state_dict_sha256
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/v7/cifar10_iid_resnet18.json"); p.add_argument("--out", default="results/processed/resnet18/federaser_bn_diagnostic.csv"); p.add_argument("--method", default="federaser"); a = p.parse_args()
cfg = ExperimentConfig.load(a.config); device = get_device(cfg.device); rows = []
for seed in cfg.seeds:
    rec = json.load(open(f"{cfg.results_dir}/{cfg.exp_id}/seed{seed}/{a.method}_alpha1.00.json"))
    set_seed(seed); train, test, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
    deleted = cfg.deletion_set(1.0); retained = [c for c in range(cfg.num_clients) if c not in deleted]
    model = _model_factory(cfg, spec)().to(device); sd = torch.load(rec["checkpoint_path"], map_location="cpu")
    assert state_dict_sha256(sd) == rec["checkpoint_sha256"]; model.load_state_dict(sd)
    e0 = evaluate(model, test, trig, cfg.target_label, device)
    assert abs(e0["BA"] - rec["BA"]) < 1e-6 and abs(e0["ASR"] - rec["ASR"]) < 1e-6
    bn = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in bn: m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad():
        for x, _ in DataLoader(ConcatDataset([client_ds[c] for c in retained]), batch_size=256, shuffle=False):
            model(x.to(device))
    e1 = evaluate(model, test, trig, cfg.target_label, device)
    rows.append(dict(exp_id=cfg.exp_id, seed=seed, method=a.method, BA_stored=e0["BA"], ASR_stored=e0["ASR"], BA_bn_recalibrated=e1["BA"], ASR_bn_recalibrated=e1["ASR"], n_bn_layers=len(bn)))
    print(rows[-1], flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out)

#!/usr/bin/env python3
"""Train the same-initialisation counterfactual theta_cf0 for every (experiment, seed) at alpha = 1:
start from the stored round-0 global model of the compromised training (history.global_states[0]) and run the
identical FedAvg protocol on the retained clients. Records are written next to the other run records as
cf_sameinit_alpha1.00.json (method 'cf_sameinit'); they are analysis artifacts and are not part of all_results.csv.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from bpa_fu.experiment import ExperimentConfig, _prepare, _model_factory, load_history, Logger
from bpa_fu.federated_training import run_fedavg
from bpa_fu.evaluation import evaluate
from bpa_fu.reproducibility import get_device, set_seed, state_dict_sha256, write_json, file_sha256

p = argparse.ArgumentParser(); p.add_argument("configs", nargs="+"); a = p.parse_args()
for cpath in a.configs:
    cfg = ExperimentConfig.load(cpath); device = get_device(cfg.device)
    for seed in cfg.seeds:
        res_dir, ckpt_dir, log_dir = cfg.seed_dirs(seed)
        rec_path = res_dir / "cf_sameinit_alpha1.00.json"
        if rec_path.exists() and json.load(open(rec_path)).get("status") == "completed":
            print("skip", rec_path); continue
        comp = json.load(open(res_dir / "compromised.json"))
        hist = load_history(comp["history_path"]); assert file_sha256(comp["history_path"]) == comp["history_sha256"]
        assert hist.rounds[0] == 0
        log = Logger(log_dir / "cf_sameinit_alpha1.00.log"); t0 = time.time()
        set_seed(seed)
        train, test, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
        deleted = cfg.deletion_set(1.0); retained = [c for c in range(cfg.num_clients) if c not in deleted]
        model = _model_factory(cfg, spec)().to(device)
        model.load_state_dict({k: v.to(device) for k, v in hist.global_states[0].items()})
        init_sha = state_dict_sha256({k: v.cpu() for k, v in model.state_dict().items()})
        model, _ = run_fedavg(model, client_ds, retained, cfg.rounds, cfg.local_cfg(), device, seed=seed + 17, log=log)
        ev = evaluate(model, test, trig, cfg.target_label, device)
        sd = {k: v.cpu() for k, v in model.state_dict().items()}; ckpt = ckpt_dir / "cf_sameinit_alpha1.00.pt"; torch.save(sd, ckpt)
        rec = {"run_id": f"{cfg.exp_id}/seed{seed}/cf_sameinit_alpha1.00", "exp_id": cfg.exp_id, "seed": seed, "method": "cf_sameinit", "alpha": 1.0,
               "deleted_clients": deleted, "retained_clients": retained, "config_path": cfg.path, "init_from": "history.global_states[0]", "init_sha256": init_sha,
               "compromised_checkpoint_sha256": comp["checkpoint_sha256"], "BA": ev["BA"], "ASR": ev["ASR"], "checkpoint_path": str(ckpt),
               "checkpoint_sha256": state_dict_sha256(sd), "runtime_sec": time.time() - t0, "status": "completed", "purpose": "same-initialisation counterfactual for the alignment analysis (not part of all_results.csv)"}
        write_json(rec, rec_path); log(f"DONE cf_sameinit: BA={ev['BA']:.4f} ASR={ev['ASR']:.4f}"); log.close()
print("ALL_SAMEINIT_DONE")

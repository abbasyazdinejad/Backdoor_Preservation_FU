#!/usr/bin/env python3
"""Collect raw run records into a long-format result file and manifest.

  python scripts/aggregate_results.py --configs configs/cifar10_iid.json configs/mnist_iid.json ... \
         --out results/processed/legacy_v1/all_results.csv --manifest results/processed/legacy_v1/reproduction_manifest.json
The defaults write the intermediate results/processed/legacy_v1/ files; the authoritative file for the final paper is
results/processed/final_matrix/all_results.csv, produced by scripts/run_validation.sh from configs/final/*.json.
Fails (non-zero exit) when validation fails.
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd
from bpa_fu.experiment import ExperimentConfig
from bpa_fu.validation import validate_results
from bpa_fu.reproducibility import environment_info, git_commit_hash, write_json, file_sha256, state_dict_sha256


def collect(cfg: ExperimentConfig, require_complete: bool, verify_checksums: bool, args=None):
    rows, manifest = [], []
    for seed in cfg.seeds:
        res_dir, _, _ = cfg.seed_dirs(seed)
        recs = {}
        for p in sorted(glob.glob(str(res_dir / "*.json"))):
            if p.endswith("partition_and_poison.json"):
                continue
            with open(p) as f:
                r = json.load(f)
            if "method" not in r or "alpha" not in r:          # auxiliary files (e.g. last_round_reconstruction.json)
                continue
            recs[(r["method"], float(r["alpha"]))] = r
        comp = recs.get(("before", 0.0))
        if comp is None and cfg.compromised_results_dir:                       # v8: compromised record kept in the source tree
            src = os.path.join(cfg.compromised_results_dir, cfg.exp_id, f"seed{seed}", "compromised.json")
            if os.path.exists(src):
                with open(src) as f:
                    comp = json.load(f); comp["promoted_from"] = src
        if comp is None or comp.get("status") != "completed":
            if require_complete:
                raise SystemExit(f"{cfg.exp_id} seed {seed}: compromised run missing or not completed")
            continue
        base = dict(exp_id=cfg.exp_id, dataset=cfg.dataset, architecture=cfg.architecture,
                    partition=(cfg.partition["scheme"] if cfg.partition["scheme"] == "iid" else f"dirichlet{cfg.partition['beta']}"),
                    num_clients=cfg.num_clients, num_malicious=len(cfg.malicious_clients), poison_rate=cfg.poison_rate,
                    rounds=cfg.rounds, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size, optimizer=cfg.optimizer, lr=cfg.lr,
                    config_path=cfg.path, compromised_checkpoint_sha256=comp["checkpoint_sha256"])
        if verify_checksums:
            import torch
            sd = torch.load(comp["checkpoint_path"])
            if state_dict_sha256(sd) != comp["checkpoint_sha256"]:
                raise SystemExit(f"checksum mismatch for {comp['checkpoint_path']}")
        rows.append(dict(base, alpha=0.0, seed=seed, method="before", BA=comp["BA"], ASR=comp["ASR"], counterfactual_BA=float("nan"),
                         counterfactual_ASR=float("nan"), runtime=comp["runtime_sec"], status=comp["status"], checkpoint_path=comp["checkpoint_path"],
                         checkpoint_sha256=comp["checkpoint_sha256"], run_id=comp["run_id"], deleted_clients="", num_parameters=comp.get("num_parameters")))
        manifest.append(_manifest_entry(comp, cfg, base))
        for a_ in cfg.alphas:
            cf = recs.get((args.counterfactual_method, float(a_)))
            if cf is None and getattr(args, "counterfactual_from", None):       # v8: reuse the existing same-init counterfactual of the same seed
                cands = [os.path.join(d, cfg.exp_id, f"seed{seed}", f"{args.counterfactual_method}_alpha{float(a_):.2f}.json") for d in args.counterfactual_from]
                src = next((c for c in cands if os.path.exists(c)), None)
                if src is not None:
                    with open(src) as f:
                        cf = json.load(f)
                    if cf.get("compromised_checkpoint_sha256") != comp["checkpoint_sha256"]:
                        raise SystemExit(f"{src}: counterfactual belongs to a different compromised model")
                    cf["promoted_from"] = src; recs[(args.counterfactual_method, float(a_))] = cf
            methods_here = list(cfg.methods) + ([args.counterfactual_method] if args.counterfactual_method not in cfg.methods and (args.counterfactual_method, float(a_)) in recs else [])
            for m in methods_here:
                r = recs.get((m, float(a_)))
                if r is None or r.get("status") != "completed":
                    if require_complete:
                        raise SystemExit(f"{cfg.exp_id} seed {seed}: run {m} alpha={a_} missing or not completed")
                    continue
                if verify_checksums:
                    sd = torch.load(r["checkpoint_path"])
                    if state_dict_sha256(sd) != r["checkpoint_sha256"]:
                        raise SystemExit(f"checksum mismatch for {r['checkpoint_path']}")
                rows.append(dict(base, alpha=float(a_), seed=seed, method=m, BA=r["BA"], ASR=r["ASR"],
                                 counterfactual_BA=(cf["BA"] if cf and cf.get("status") == "completed" else float("nan")),
                                 counterfactual_ASR=(cf["ASR"] if cf and cf.get("status") == "completed" else float("nan")),
                                 runtime=r["runtime_sec"], status=r["status"], checkpoint_path=r["checkpoint_path"], checkpoint_sha256=r["checkpoint_sha256"],
                                 run_id=r["run_id"], deleted_clients=" ".join(map(str, r["deleted_clients"])), num_parameters=comp.get("num_parameters"),
                                 # method-specific columns appear only when a record carries them, so result files of other
                                 # experiments regenerate byte-identically
                                 **{k: r[k] for k in ("server_storage_bytes", "fedup_n_pruned") if k in r}))
                manifest.append(_manifest_entry(r, cfg, base, cf))
    return rows, manifest


def _manifest_entry(r, cfg, base, cf=None):
    return {"experiment_id": r["run_id"], "config_path": cfg.path, "dataset": cfg.dataset,
            "dataset_train_sha256": r.get("dataset_train_sha256"), "architecture": cfg.architecture,
            "fl_setting": {k: base[k] for k in ("partition", "num_clients", "num_malicious", "poison_rate", "rounds", "local_epochs", "batch_size", "optimizer", "lr")},
            "unlearning_method": r["method"], "removal_fraction": r["alpha"], "deleted_clients": r.get("deleted_clients"), "seed": r["seed"],
            "checkpoint_path": r.get("checkpoint_path"), "checkpoint_sha256": r.get("checkpoint_sha256"), "raw_log_path": r.get("raw_log_path"),
            "start_time": r.get("start_time"), "end_time": r.get("end_time"), "status": r.get("status"), "BA": r.get("BA"), "ASR": r.get("ASR"),
            "counterfactual_BA": (cf or {}).get("BA"), "counterfactual_ASR": (cf or {}).get("ASR"), "runtime_sec": r.get("runtime_sec"),
            "software": {k: (r.get("environment") or {}).get(k) for k in ("python", "torch", "torchvision", "numpy", "platform", "cpu_brand", "memory_gib", "os_version")},
            "device": r.get("device"), "method_params": r.get("method_params"), "history_rounds_used": r.get("history_rounds_used")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--out", default="results/processed/legacy_v1/all_results.csv")
    p.add_argument("--manifest", default="results/processed/legacy_v1/reproduction_manifest.json")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--verify-checksums", action="store_true")
    p.add_argument("--counterfactual-method", default="cf_sameinit", help="method whose run of the same (exp, seed, alpha) provides counterfactual_BA/ASR")
    p.add_argument("--counterfactual-from", nargs="*", default=None, help="results tree(s) from which an existing counterfactual record of the same seed is reused (checksum-linked to the compromised model)")
    a = p.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.chdir(root)
    rows, manifest = [], []
    cfgs = [ExperimentConfig.load(c) for c in a.configs]
    for cfg in cfgs:
        r, m = collect(cfg, require_complete=not a.allow_incomplete, verify_checksums=a.verify_checksums, args=a)
        rows += r; manifest += m
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no completed runs found")
    if not a.allow_incomplete:
        validate_results(df, {c.exp_id: c.seeds for c in cfgs}, {c.exp_id: list(c.methods) + ([a.counterfactual_method] if a.counterfactual_from and a.counterfactual_method not in c.methods else []) for c in cfgs}, {c.exp_id: c.alphas for c in cfgs}, counterfactual_method=a.counterfactual_method)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.sort_values(["exp_id", "seed", "alpha", "method"]).to_csv(a.out, index=False)
    composition = {m: int(n) for m, n in df.groupby("method").size().items()}
    write_json({"generated": environment_info()["timestamp"], "counterfactual_method": a.counterfactual_method,
                "run_composition": {"total_rows": int(len(df)), "by_method": composition,
                                    "note": "compromised-model records = method 'before'; cf_sameinit = same-initialisation counterfactual; full_retrain = independently initialised retraining; analyses outside this matrix (attacker subsets, negation phases, Fine-Pruning channels, alignment rows) are listed in audit/run_count_reconciliation.md"}, "git_commit": git_commit_hash(root), "environment": environment_info(),
                "n_runs": len(manifest), "configs": [c.path for c in cfgs], "runs": manifest}, a.manifest)
    print(f"wrote {a.out} ({len(df)} rows) and {a.manifest}; validation={'passed' if not a.allow_incomplete else 'skipped (incomplete allowed)'}")


if __name__ == "__main__":
    main()

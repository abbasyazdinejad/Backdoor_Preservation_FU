#!/usr/bin/env python3
"""Assemble the final results tree (results/raw/final_matrix) from the unchanged v1 records and the corrected v2 records.

For every main experiment and seed: compromised.json and partition_and_poison.json, short_retrain / grad_negation /
fine_pruning / full_retrain records are copied from results/raw/legacy_v1 (unchanged methods); federaser and cf_sameinit
records are copied from results/raw/corrected_v2 (corrected FedEraser, same-initialisation counterfactuals). Every copied record
gets a `promoted_from` field and results/manifests/final_matrix_PROVENANCE.json lists every source. Nothing in results/ or results_v2/ is
modified. Checkpoint paths inside the records keep pointing to checkpoints/ or checkpoints_v2/.
"""
import glob, json, os, shutil, sys, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
V1 = {"before": "results/raw/legacy_v1", "short_retrain": "results/raw/legacy_v1", "grad_negation": "results/raw/legacy_v1", "fine_pruning": "results/raw/legacy_v1", "full_retrain": "results/raw/legacy_v1"}
V2 = {"federaser": "results/raw/corrected_v2", "cf_sameinit": "results/raw/corrected_v2"}
EXPS = ["cifar10_iid", "mnist_iid", "gtsrb_iid", "cifar10_dirichlet", "cifar10_iid_federaser_r1", "cifar10_iid_federaser_dt1"]
prov = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "records": []}
n = 0
for e in EXPS:
    for sd in sorted(glob.glob(f"results/raw/legacy_v1/{e}/seed*")):
        seed = os.path.basename(sd); dst = f"results/raw/final_matrix/{e}/{seed}"; os.makedirs(dst, exist_ok=True)
        shutil.copy(f"{sd}/partition_and_poison.json", dst)
        for f in sorted(glob.glob(f"{sd}/*.json")) + sorted(glob.glob(f"results/raw/corrected_v2/{e}/{seed}/*.json")):
            if f.endswith("partition_and_poison.json"): continue
            r = json.load(open(f)); m = r["method"]
            src_tree = V1.get(m) or V2.get(m)
            if src_tree is None or not f.startswith(src_tree): continue   # skip superseded v1 federaser / v1 cf_sameinit(alpha=1)
            if r.get("status") != "completed": raise SystemExit(f"not completed: {f}")
            r["promoted_from"] = f
            json.dump(r, open(os.path.join(dst, os.path.basename(f)), "w"), indent=2, sort_keys=True)
            prov["records"].append({"final": os.path.join(dst, os.path.basename(f)), "source": f, "method": m, "checkpoint_sha256": r.get("checkpoint_sha256")}); n += 1
prov["superseded"] = {"v1_federaser_records": sorted(glob.glob("results/raw/legacy_v1/*/seed*/federaser_*.json")), "v1_cf_sameinit_alpha1_records": sorted(glob.glob("results/raw/legacy_v1/*/seed*/cf_sameinit_*.json")),
                      "note": "superseded by results_v2 (corrected FedEraser: first stored round uncalibrated; cf_sameinit re-run for every fraction). The v1 alpha=1 cf_sameinit checkpoints are byte-identical to the v2 ones (see audit/federaser_correction_comparison.md)."}
os.makedirs("results_final", exist_ok=True); json.dump(prov, open("results/manifests/final_matrix_PROVENANCE.json", "w"), indent=2)
print(f"promoted {n} records into results/raw/final_matrix")

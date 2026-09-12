#!/usr/bin/env python3
"""Package large artifacts for archival release (see audit/release_plan.md).

  python scripts/package_artifacts.py --out release/            # writes two tar archives + checksum lists
Checkpoint checksums are taken from the run records so that the archive can be verified against the Git repository.
"""
import argparse, glob, hashlib, json, os, subprocess, sys, tarfile
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
p = argparse.ArgumentParser(); p.add_argument("--out", default="release"); p.add_argument("--with-histories", action="store_true"); a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
recs = [json.load(open(f)) for f in glob.glob("results/raw/legacy_v1/*/seed*/*.json") if not f.endswith("partition_and_poison.json")]
ckpts = sorted({r["checkpoint_path"]: r["checkpoint_sha256"] for r in recs if r.get("status") == "completed"}.items())
with open(os.path.join(a.out, "checksums.sha256"), "w") as f:
    for path, sha in ckpts:
        f.write(f"{sha}  {path}\n")
with tarfile.open(os.path.join(a.out, "bpa_fu_checkpoints_v1.tar"), "w") as tar:
    for path, _ in ckpts:
        tar.add(path)
    tar.add(os.path.join(a.out, "checksums.sha256"), arcname="checksums.sha256")
print(f"wrote {a.out}/bpa_fu_checkpoints_v1.tar with {len(ckpts)} checkpoints")
if a.with_histories:
    hists = sorted({r["history_path"]: r["history_sha256"] for r in recs if r.get("history_path")}.items())
    with open(os.path.join(a.out, "histories.sha256"), "w") as f:
        for path, sha in hists:
            f.write(f"{sha}  {path}\n")
    with tarfile.open(os.path.join(a.out, "bpa_fu_federaser_histories_v1.tar"), "w") as tar:
        for path, _ in hists:
            tar.add(path)
        tar.add(os.path.join(a.out, "histories.sha256"), arcname="histories.sha256")
    print(f"wrote {a.out}/bpa_fu_federaser_histories_v1.tar with {len(hists)} histories")

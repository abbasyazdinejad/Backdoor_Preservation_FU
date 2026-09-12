#!/usr/bin/env python3
"""Download and verify an artifact archive released on an archival service (see audit/release_plan.md).

  python scripts/fetch_artifacts.py <URL-of-bpa_fu_checkpoints_v1.tar>
Every extracted checkpoint is verified against the sha256 recorded in the run records under results/raw/legacy_v1/.
"""
import glob, hashlib, json, os, sys, tarfile, urllib.request
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
url = sys.argv[1]; local = os.path.basename(url)
if not os.path.exists(local):
    print("downloading", url); urllib.request.urlretrieve(url, local)
with tarfile.open(local) as tar:
    tar.extractall(".")
expected = {r["checkpoint_path"]: r["checkpoint_sha256"] for f in glob.glob("results/raw/legacy_v1/*/seed*/*.json") if not f.endswith("partition_and_poison.json") for r in [json.load(open(f))] if r.get("checkpoint_path")}
bad = 0
for path, sha in expected.items():
    if not os.path.exists(path):
        print("missing", path); bad += 1; continue
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != sha:
        print("CHECKSUM MISMATCH", path); bad += 1
print(f"verified {len(expected) - bad} of {len(expected)} checkpoints"); sys.exit(1 if bad else 0)

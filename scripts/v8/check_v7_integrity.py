#!/usr/bin/env python3
"""Hard gate: every v7 evidence file (run records, processed result files, manifests, release checksum lists) must be
byte-identical to its pre-v8 SHA-256 recorded in results/manifests/v7_evidence_hashes.sha256. New fields belong in v8-derived files
only. Exit code 1 on any difference or missing file.
  python scripts/v8/check_v7_integrity.py
"""
import hashlib, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REC = "results/manifests/v7_evidence_hashes.sha256"
bad, missing, n = [], [], 0
for line in open(REC):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    want, path = line.split("  ", 1)
    if not os.path.exists(path):
        missing.append(path); continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    n += 1
    if h.hexdigest() != want:
        bad.append(path)
print(f"V7 INTEGRITY: {n} files checked, {len(bad)} modified, {len(missing)} missing")
if bad:
    print("MODIFIED:", *bad[:20], sep="\n  ")
if missing:
    print("MISSING:", *missing[:20], sep="\n  ")
print("V7 INTEGRITY CHECK:", "PASSED" if not bad and not missing else "FAILED")
sys.exit(0 if not bad and not missing else 1)

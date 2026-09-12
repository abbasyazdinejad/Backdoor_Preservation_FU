#!/usr/bin/env python3
"""Cross-platform comparison of a regenerated value store against the one shipped with the repository.

Floating-point arithmetic is not bit-identical across platforms, so numbers are compared with a strict absolute
tolerance while keys and structure must match exactly. The tolerance is far below any reported precision: every
number the study reports is printed to at most four significant decimals.

    python scripts/check_value_store.py <regenerated table_values.json> [--reference results/tables/table_values.json]
                                        [--atol 1e-12]

Exit status 0 when the stores agree, 1 otherwise.
"""
import argparse, json, math, os, sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
p = argparse.ArgumentParser()
p.add_argument("candidate")
p.add_argument("--reference", default="results/tables/table_values.json")
p.add_argument("--atol", type=float, default=1e-12)
a = p.parse_args()

new = json.load(open(a.candidate))
ref = json.load(open(a.reference))


def compare(x, y, path=""):
    """Recursive semantic comparison. Returns a list of (path, candidate, reference) differences."""
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return [(path or "<root>", f"key sets differ: {sorted(set(x) ^ set(y))[:5]}", "")]
        out = []
        for k in x:
            out += compare(x[k], y[k], f"{path}/{k}" if path else k)
        return out
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return [(path, f"length {len(x)}", f"length {len(y)}")]
        out = []
        for i, (u, v) in enumerate(zip(x, y)):
            out += compare(u, v, f"{path}[{i}]")
        return out
    if isinstance(x, bool) or isinstance(y, bool):
        return [] if x == y else [(path, x, y)]
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if x == y:
            return []
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            return []
        return [] if abs(x - y) <= a.atol else [(path, x, y)]
    return [] if x == y else [(path, x, y)]


diffs = compare(new, ref)
worst, worst_key, drifted = 0.0, None, 0
for k in new:
    u, v = new.get(k), ref.get(k)
    if isinstance(u, (int, float)) and isinstance(v, (int, float)) and not isinstance(u, bool):
        if isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v):
            continue
        d = abs(u - v)
        if d > 0:
            drifted += 1
        if d > worst:
            worst, worst_key = d, k

print(f"value store: {len(new)} entries in the candidate, {len(ref)} in the reference")
print(f"keys identical: {set(new) == set(ref)}")
print(f"bit-identical values: {len(new) - drifted} | differing in the last bits: {drifted}")
if worst_key:
    print(f"largest absolute difference: {worst:g} at {worst_key} (tolerance {a.atol:g})")
if diffs:
    print(f"VALUE STORE CHECK: FAILED, {len(diffs)} difference(s) beyond the tolerance")
    for d in diffs[:10]:
        print("   ", d)
    sys.exit(1)
print(f"VALUE STORE CHECK: PASSED (all values within {a.atol:g} absolute)")

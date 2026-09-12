"""Consistency assertions for an aggregated result file.

The authoritative file for the final paper is results_final/processed/all_results.csv;
results/processed/all_results.csv is the default (intermediate) output of scripts/aggregate_results.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["exp_id", "dataset", "architecture", "partition", "alpha", "seed", "method", "BA", "ASR",
                    "counterfactual_BA", "counterfactual_ASR", "runtime", "status", "config_path", "checkpoint_path",
                    "compromised_checkpoint_sha256"]


class ValidationError(AssertionError):
    pass


def validate_results(df: pd.DataFrame, expected_seeds: dict[str, list[int]], expected_methods: dict[str, list[str]], expected_alphas: dict[str, list[float]], counterfactual_method: str = "full_retrain") -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"missing columns: {missing_cols}")
    if (df.status != "completed").any():
        bad = df[df.status != "completed"][["exp_id", "seed", "method", "alpha", "status"]]
        raise ValidationError(f"runs with non-completed status:\n{bad}")
    for col in ("BA", "ASR"):
        v = df[col].astype(float)
        if v.isna().any() or (~np.isfinite(v)).any() or (v < 0).any() or (v > 1).any():
            raise ValidationError(f"{col} out of range [0,1] or NaN/inf")
    unl = df[df.method != "before"]
    for col in ("counterfactual_BA", "counterfactual_ASR"):
        v = unl[col].astype(float)
        if v.isna().any() or (~np.isfinite(v)).any() or (v < 0).any() or (v > 1).any():
            raise ValidationError(f"{col} missing or out of range for unlearning runs")
    key = ["exp_id", "seed", "method", "alpha"]
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        raise ValidationError(f"duplicate runs:\n{df[dup][key]}")
    for exp, seeds in expected_seeds.items():
        sub = df[df.exp_id == exp]
        if sub.empty:
            raise ValidationError(f"missing experiment {exp}")
        have = set(sub.seed.astype(int))
        if set(seeds) - have:
            raise ValidationError(f"{exp}: missing seeds {sorted(set(seeds) - have)}")
        for s in seeds:
            ss = sub[sub.seed == s]
            if not (ss.method == "before").any():
                raise ValidationError(f"{exp} seed {s}: missing compromised (before) run")
            for a in expected_alphas[exp]:
                for m in expected_methods[exp]:
                    if not ((ss.method == m) & (np.isclose(ss.alpha.astype(float), a))).any():
                        raise ValidationError(f"{exp} seed {s}: missing run method={m} alpha={a}")
        # every unlearning run must start from the same compromised checkpoint as the 'before' run of its seed
        for s in seeds:
            ss = sub[sub.seed == s]
            h = set(ss.compromised_checkpoint_sha256.astype(str))
            if len(h) != 1:
                raise ValidationError(f"{exp} seed {s}: unlearning runs use different compromised baselines (checkpoint hashes {h})")
        # the same experiment must have a single config path and architecture/partition
        for col in ("config_path", "architecture", "partition", "dataset"):
            if sub[col].nunique() != 1:
                raise ValidationError(f"{exp}: mismatched {col} across runs: {sub[col].unique()}")
    # counterfactual columns must equal the counterfactual run of the same (exp, seed, alpha)
    fr = df[df.method == counterfactual_method].set_index(["exp_id", "seed", "alpha"])
    for _, r in unl.iterrows():
        k = (r.exp_id, r.seed, r.alpha)
        if k not in fr.index:
            raise ValidationError(f"no counterfactual ({counterfactual_method}) for {k}")
        if abs(float(fr.loc[k, "ASR"]) - float(r.counterfactual_ASR)) > 1e-9 or abs(float(fr.loc[k, "BA"]) - float(r.counterfactual_BA)) > 1e-9:
            raise ValidationError(f"counterfactual columns do not match the {counterfactual_method} run for {k}")


def assert_table_matches(table_values: dict[str, float], df_values: dict[str, float], tol: float = 5e-5) -> None:
    for k, v in table_values.items():
        if k not in df_values:
            raise ValidationError(f"table value {k} has no counterpart in the result file")
        if abs(float(v) - float(df_values[k])) > tol:
            raise ValidationError(f"table value {k}={v} differs from result file {df_values[k]}")

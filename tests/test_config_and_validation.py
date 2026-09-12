import json, os, tempfile, pandas as pd, pytest
from bpa_fu.experiment import ExperimentConfig
from bpa_fu.validation import validate_results, ValidationError

def test_config_roundtrip():
    cfg = ExperimentConfig(exp_id="x", dataset="mnist", seeds=[0, 1], alphas=[0.5, 1.0])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.json"); cfg.save(p); cfg2 = ExperimentConfig.load(p)
    assert cfg2.seeds == [0, 1] and cfg2.alphas == [0.5, 1.0] and cfg2.dataset == "mnist" and cfg2.trigger == cfg.trigger

def _df():
    rows = []
    for s in (0, 1, 2):
        rows.append(dict(exp_id="e", dataset="mnist", architecture="simplecnn", partition="iid", alpha=0.0, seed=s, method="before", BA=0.99, ASR=0.99, counterfactual_BA=float("nan"), counterfactual_ASR=float("nan"), runtime=1.0, status="completed", config_path="c", checkpoint_path="p", compromised_checkpoint_sha256="h"))
        for a in (0.5, 1.0):
            for m in ("short_retrain", "full_retrain"):
                rows.append(dict(exp_id="e", dataset="mnist", architecture="simplecnn", partition="iid", alpha=a, seed=s, method=m, BA=0.98, ASR=0.5 if m == "short_retrain" else 0.05, counterfactual_BA=0.98, counterfactual_ASR=0.05, runtime=1.0, status="completed", config_path="c", checkpoint_path="p", compromised_checkpoint_sha256="h"))
    return pd.DataFrame(rows)

def test_validation_passes_and_detects_problems():
    df = _df(); validate_results(df, expected_seeds={"e": [0, 1, 2]}, expected_methods={"e": ["short_retrain", "full_retrain"]}, expected_alphas={"e": [0.5, 1.0]})
    with pytest.raises(ValidationError, match="duplicate"):
        validate_results(pd.concat([df, df.iloc[[5]]]), expected_seeds={"e": [0, 1, 2]}, expected_methods={"e": ["short_retrain", "full_retrain"]}, expected_alphas={"e": [0.5, 1.0]})
    with pytest.raises(ValidationError, match="missing"):
        validate_results(df[df.seed != 2], expected_seeds={"e": [0, 1, 2]}, expected_methods={"e": ["short_retrain", "full_retrain"]}, expected_alphas={"e": [0.5, 1.0]})
    bad = df.copy(); bad.loc[3, "ASR"] = 1.5
    with pytest.raises(ValidationError, match="range"):
        validate_results(bad, expected_seeds={"e": [0, 1, 2]}, expected_methods={"e": ["short_retrain", "full_retrain"]}, expected_alphas={"e": [0.5, 1.0]})
    bad = df.copy(); bad.loc[4, "compromised_checkpoint_sha256"] = "other"
    with pytest.raises(ValidationError, match="baseline"):
        validate_results(bad, expected_seeds={"e": [0, 1, 2]}, expected_methods={"e": ["short_retrain", "full_retrain"]}, expected_alphas={"e": [0.5, 1.0]})
    bad = df.copy(); bad.loc[4, "status"] = "failed"
    with pytest.raises(ValidationError, match="status"):
        validate_results(bad, expected_seeds={"e": [0, 1, 2]}, expected_methods={"e": ["short_retrain", "full_retrain"]}, expected_alphas={"e": [0.5, 1.0]})

"""Completed records are reused only when their checkpoint (and history, where required) exist and match."""
import json, os, tempfile, torch, pytest
from bpa_fu.experiment import ExperimentConfig, verify_record_artifacts, ArtifactMissingError, train_compromised, run_unlearning
from bpa_fu.reproducibility import state_dict_sha256, file_sha256


def _setup(tmp, with_history=True, corrupt=None):
    res = os.path.join(tmp, "res"); ck = os.path.join(tmp, "ck")
    os.makedirs(os.path.join(res, "e", "seed0")); os.makedirs(os.path.join(ck, "e", "seed0"))
    sd = {"w": torch.arange(4.0)}; ck_path = os.path.join(ck, "e", "seed0", "compromised.pt"); torch.save(sd, ck_path)
    hist_path = os.path.join(ck, "e", "seed0", "history.pt")
    if with_history:
        torch.save({"rounds": [0]}, hist_path)
    rec = {"run_id": "e/seed0/compromised", "exp_id": "e", "seed": 0, "method": "before", "alpha": 0.0, "status": "completed",
           "checkpoint_path": ck_path, "checkpoint_sha256": state_dict_sha256(sd), "history_path": hist_path,
           "history_sha256": file_sha256(hist_path) if with_history else "0" * 64, "BA": 0.5, "ASR": 0.5}
    if corrupt == "checkpoint":
        torch.save({"w": torch.zeros(4)}, ck_path)
    if corrupt == "history":
        torch.save({"rounds": [1]}, hist_path)
    json.dump(rec, open(os.path.join(res, "e", "seed0", "compromised.json"), "w"))
    cfg = ExperimentConfig(exp_id="e", dataset="mnist", results_dir=res, checkpoints_dir=ck, logs_dir=os.path.join(tmp, "logs"))
    return cfg, rec


def test_valid_record_is_reused():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rec = _setup(tmp)
        out = verify_record_artifacts(cfg, rec, need_history=True)
        assert out["checkpoint_path"] == rec["checkpoint_path"]
        assert train_compromised(cfg, 0)["checkpoint_sha256"] == rec["checkpoint_sha256"]   # no training happens


def test_missing_checkpoint_fails_explicitly():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rec = _setup(tmp); os.remove(rec["checkpoint_path"])
        with pytest.raises(ArtifactMissingError, match="missing"):
            verify_record_artifacts(cfg, rec)
        with pytest.raises(ArtifactMissingError):
            train_compromised(cfg, 0)


def test_missing_history_fails_only_when_required():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rec = _setup(tmp, with_history=False)
        verify_record_artifacts(cfg, rec, need_history=False)
        with pytest.raises(ArtifactMissingError, match="history"):
            verify_record_artifacts(cfg, rec, need_history=True)


def test_checksum_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rec = _setup(tmp, corrupt="checkpoint")
        with pytest.raises(ArtifactMissingError, match="does not match"):
            verify_record_artifacts(cfg, rec)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rec = _setup(tmp, corrupt="history")
        with pytest.raises(ArtifactMissingError, match="history"):
            verify_record_artifacts(cfg, rec, need_history=True)


def test_compromised_checkpoints_dir_resolution():
    """A record from another tree whose recorded path does not exist is resolved under compromised_checkpoints_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rec = _setup(tmp)
        other = os.path.join(tmp, "elsewhere"); os.makedirs(os.path.join(other, "e", "seed0"))
        os.rename(rec["checkpoint_path"], os.path.join(other, "e", "seed0", "compromised.pt"))
        os.rename(rec["history_path"], os.path.join(other, "e", "seed0", "history.pt"))
        with pytest.raises(ArtifactMissingError):
            verify_record_artifacts(cfg, rec, need_history=True)
        cfg2 = ExperimentConfig(exp_id="e", dataset="mnist", results_dir=os.path.join(tmp, "res2"), checkpoints_dir=os.path.join(tmp, "ck2"),
                                logs_dir=os.path.join(tmp, "logs"), compromised_results_dir=cfg.results_dir, compromised_checkpoints_dir=other)
        out = train_compromised(cfg2, 0)
        assert out["checkpoint_path"].startswith(other) and out["history_path"].startswith(other)


def test_completed_unlearning_record_without_checkpoint_is_not_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rec = _setup(tmp)
        bad = dict(rec, run_id="e/seed0/short_retrain_alpha1.00", method="short_retrain", alpha=1.0, checkpoint_path=os.path.join(tmp, "nope.pt"))
        json.dump(bad, open(os.path.join(cfg.results_dir, "e", "seed0", "short_retrain_alpha1.00.json"), "w"))
        with pytest.raises(ArtifactMissingError, match="never skipped silently"):
            run_unlearning(cfg, 0, "short_retrain", 1.0)

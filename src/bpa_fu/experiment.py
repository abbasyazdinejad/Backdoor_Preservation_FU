"""Experiment orchestration.

One *experiment* (config) fixes dataset, architecture, FL protocol, poisoning and the
method hyperparameters. For each seed the pipeline is:

  1. train_compromised(cfg, seed)   FedAvg with malicious clients, storing update history
  2. run_unlearning(cfg, seed, method, alpha)   for every method and nested deletion fraction

Every run writes a JSON record under results/raw/<exp_id>/seed<seed>/ and a checkpoint under
checkpoints/<exp_id>/seed<seed>/. Records carry config path, checksums, timings and status.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import load_dataset
from .evaluation import evaluate
from .federated_training import LocalTrainConfig, UpdateHistory, run_fedavg, local_train, state_delta, fedavg_aggregate
from .models import build_model, count_parameters, parameter_groups, clone_state
from .partitions import check_partition, make_partition, partition_statistics
from .poisoning import build_client_datasets
from .reproducibility import environment_info, file_sha256, get_device, set_seed, state_dict_sha256, write_json
from .triggers import PatchTrigger
from .unlearning import METHODS


@dataclass
class ExperimentConfig:
    exp_id: str
    dataset: str
    data_root: str = "data"
    architecture: str = "simplecnn"
    num_clients: int = 20
    malicious_clients: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    poison_rate: float = 0.5
    target_label: int = 0
    trigger: dict = field(default_factory=lambda: {"size": 4, "value": 1.0, "offset": 1})
    partition: dict = field(default_factory=lambda: {"scheme": "iid", "beta": None})
    rounds: int = 30
    local_epochs: int = 2
    batch_size: int = 64
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    clients_per_round: float = 1.0
    history_every: int = 2
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    alphas: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    methods: dict = field(default_factory=lambda: {
        "short_retrain": {"rounds": 5},
        "grad_negation": {"ascent_epochs": 2, "ascent_lr": 0.01, "radius": 5.0, "recovery_rounds": 2},
        "federaser": {"calibration_ratio": 0.5},
        "full_retrain": {},
    })
    device: str = "auto"
    results_dir: str = "results/raw"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    notes: str = ""
    # optional: read the compromised model/history records from another results/checkpoints tree (read-only)
    compromised_results_dir: str | None = None
    compromised_checkpoints_dir: str | None = None
    # hyperparameter-selection support (v7): hold out `validation_holdout` training images (deterministic, seeded)
    # before partitioning, and evaluate on that validation split instead of the test set when eval_split == "validation".
    # The test set is never loaded for evaluation in that mode. Records carry eval_split so that validation-based
    # pilot numbers cannot be confused with test-set results.
    validation_holdout: int = 0
    eval_split: str = "test"
    eval_every: int = 5   # evaluation cadence (rounds) of the compromised-training trajectory

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = ExperimentConfig(**d)
        cfg._path = str(path)  # type: ignore[attr-defined]
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.eval_split not in ("test", "validation"):
            raise ValueError(f"eval_split must be 'test' or 'validation', got {self.eval_split!r}")
        if self.eval_split == "validation" and self.validation_holdout <= 0:
            raise ValueError("eval_split='validation' requires validation_holdout > 0")
        if self.validation_holdout < 0:
            raise ValueError("validation_holdout must be >= 0")

    def save(self, path: str | Path) -> None:
        write_json(asdict(self), path)

    @property
    def path(self) -> str:
        return getattr(self, "_path", "")

    def local_cfg(self) -> LocalTrainConfig:
        return LocalTrainConfig(self.local_epochs, self.batch_size, self.optimizer, self.lr, self.momentum, self.weight_decay)

    def trigger_obj(self) -> PatchTrigger:
        return PatchTrigger(**self.trigger)

    def deletion_set(self, alpha: float) -> list[int]:
        """Nested deletion target: the first round(alpha*M) malicious clients in id order."""
        m = sorted(self.malicious_clients)
        k = int(round(alpha * len(m)))
        if not (0 < k <= len(m)):
            raise ValueError(f"alpha={alpha} selects {k} of {len(m)} malicious clients")
        return m[:k]

    def seed_dirs(self, seed: int) -> tuple[Path, Path, Path]:
        return (Path(self.results_dir) / self.exp_id / f"seed{seed}",
                Path(self.checkpoints_dir) / self.exp_id / f"seed{seed}",
                Path(self.logs_dir) / self.exp_id / f"seed{seed}")


class ArtifactMissingError(RuntimeError):
    """A completed run record exists but a required checkpoint or history file is absent or does not match its checksum."""


def _resolve_artifact(cfg: "ExperimentConfig", rec: dict[str, Any], key: str) -> Path | None:
    """Return the path of a recorded artifact, trying the configured compromised_checkpoints_dir when the record was
    produced in another result tree. Returns None when no candidate exists."""
    p = rec.get(key)
    if not p:
        return None
    cand = [Path(p)]
    if cfg.compromised_checkpoints_dir and rec.get("method") == "before":
        cand.append(Path(cfg.compromised_checkpoints_dir) / cfg.exp_id / f"seed{rec['seed']}" / Path(p).name)
    for c in cand:
        if c.exists():
            return c
    return None


def verify_record_artifacts(cfg: "ExperimentConfig", rec: dict[str, Any], need_history: bool = False) -> dict[str, Any]:
    """Check that the artifacts referenced by a completed record exist and match their recorded checksums.
    Returns a copy of the record with resolved paths. Raises ArtifactMissingError otherwise."""
    if rec.get("status") != "completed":
        raise ArtifactMissingError(f"record {rec.get('run_id')} is not completed")
    out = dict(rec)
    ck = _resolve_artifact(cfg, rec, "checkpoint_path")
    if ck is None:
        raise ArtifactMissingError(
            f"record {rec['run_id']} is completed but its checkpoint {rec.get('checkpoint_path')} is missing. "
            "Either fetch the checkpoint archive (scripts/fetch_artifacts.py, see RELEASE.md) into checkpoints/ and checkpoints_v2/, "
            "or train from scratch in empty result/checkpoint directories (configs/repro/*.json). Runs are never skipped silently.")
    sd = torch.load(ck, map_location="cpu")
    if state_dict_sha256(sd) != rec["checkpoint_sha256"]:
        raise ArtifactMissingError(f"checkpoint {ck} does not match the checksum recorded for {rec['run_id']}; refusing to reuse it")
    out["checkpoint_path"] = str(ck)
    if need_history:
        hp = _resolve_artifact(cfg, rec, "history_path")
        if hp is None:
            raise ArtifactMissingError(
                f"record {rec['run_id']} is completed but its stored update history {rec.get('history_path')} is missing "
                "(required by FedEraser and the same-initialisation counterfactual). Fetch the history archive (RELEASE.md) or train from scratch.")
        if file_sha256(hp) != rec["history_sha256"]:
            raise ArtifactMissingError(f"history {hp} does not match the checksum recorded for {rec['run_id']}")
        out["history_path"] = str(hp)
    return out


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.f.write(line + "\n")
        self.f.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.f.close()


def split_validation(train, n_val: int, seed: int):
    """Deterministically hold out n_val training images (never test images) as a validation split."""
    if n_val <= 0:
        return train, None
    if n_val >= len(train):
        raise ValueError("validation_holdout must be smaller than the training set")
    perm = np.random.RandomState(seed + 31337).permutation(len(train))
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    from .datasets import TensorImageDataset
    return (TensorImageDataset(train.images[tr_idx].clone(), train.labels[tr_idx].clone()),
            TensorImageDataset(train.images[val_idx].clone(), train.labels[val_idx].clone()))


def _prepare(cfg: ExperimentConfig, seed: int):
    """Load data, partition, poison. Deterministic given (cfg, seed). With eval_split == "validation" the returned
    evaluation set is the held-out validation split of the training data and the test set is not used."""
    train, test, spec = load_dataset(cfg.dataset, cfg.data_root, download=False)
    train, val = split_validation(train, cfg.validation_holdout, seed)
    if cfg.eval_split == "validation":
        test = val   # evaluation set = validation split; the test images are never evaluated in this mode
    labels = train.labels.numpy()
    parts = make_partition(labels, cfg.num_clients, cfg.partition["scheme"], seed, cfg.partition.get("beta"))
    check_partition(parts, len(train))
    trig = cfg.trigger_obj()
    client_ds, records = build_client_datasets(train, parts, cfg.malicious_clients, cfg.poison_rate, cfg.target_label, trig, seed)
    return train, test, spec, parts, client_ds, records, trig


def _model_factory(cfg: ExperimentConfig, spec: dict):
    return lambda: build_model(cfg.architecture, spec["in_channels"], spec["num_classes"])


def train_compromised(cfg: ExperimentConfig, seed: int, force: bool = False) -> dict[str, Any]:
    """Train the backdoored global model with FedAvg and store the update history."""
    if cfg.compromised_results_dir is not None:
        src = Path(cfg.compromised_results_dir) / cfg.exp_id / f"seed{seed}" / "compromised.json"
        if not src.exists():
            raise FileNotFoundError(f"compromised record not found in the source tree: {src}")
        with open(src) as f:
            rec = json.load(f)
        if rec.get("status") != "completed":
            raise RuntimeError(f"source compromised record is not completed: {src}")
        return verify_record_artifacts(cfg, rec, need_history=True)   # never reuse a record whose artifacts are absent
    res_dir, ckpt_dir, log_dir = cfg.seed_dirs(seed)
    rec_path = res_dir / "compromised.json"
    if rec_path.exists() and not force:
        with open(rec_path) as f:
            rec = json.load(f)
        if rec.get("status") == "completed":
            return verify_record_artifacts(cfg, rec, need_history=True)
    for d in (res_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    log = Logger(log_dir / "compromised.log")
    device = get_device(cfg.device)
    t_start = time.time()
    rec: dict[str, Any] = {"run_id": f"{cfg.exp_id}/seed{seed}/compromised", "exp_id": cfg.exp_id, "seed": seed,
                           "method": "before", "alpha": 0.0, "config_path": cfg.path, "status": "running",
                           "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": str(device),
                           "architecture": cfg.architecture, "eval_split": cfg.eval_split, "validation_holdout": cfg.validation_holdout}
    try:
        set_seed(seed)
        train, test, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
        set_seed(seed)
        model = _model_factory(cfg, spec)().to(device)
        log(f"dataset={cfg.dataset} arch={cfg.architecture} n_train={len(train)} n_eval={len(test)} eval_split={cfg.eval_split} clients={cfg.num_clients} malicious={cfg.malicious_clients} params={count_parameters(model)}")
        for r in records:
            log(f"client {r.client_id}: {len(r.poisoned_local_positions)} poisoned of {len(parts[r.client_id])}")
        traj = []

        def cb(t, m):
            if (t + 1) % cfg.eval_every == 0 or t == cfg.rounds - 1:
                e = evaluate(m, test, trig, cfg.target_label, device)
                traj.append({"round": t + 1, "BA": e["BA"], "ASR": e["ASR"]})
                log(f"  eval round {t + 1}: BA={e['BA']:.4f} ASR={e['ASR']:.4f}")

        model, hist = run_fedavg(model, client_ds, list(range(cfg.num_clients)), cfg.rounds, cfg.local_cfg(), device, seed,
                                 clients_per_round=cfg.clients_per_round, history_every=cfg.history_every, round_callback=cb, log=log)
        ev = evaluate(model, test, trig, cfg.target_label, device)
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        ckpt = ckpt_dir / "compromised.pt"
        torch.save(sd, ckpt)
        hist_path = ckpt_dir / "history.pt"
        torch.save({"rounds": hist.rounds, "global_states": hist.global_states, "client_updates": hist.client_updates,
                    "client_sizes": hist.client_sizes}, hist_path)
        meta = {"partition": [p.tolist() for p in parts],
                "partition_counts": partition_statistics(parts, train.labels.numpy(), spec["num_classes"]).tolist(),
                "poison_records": [{"client_id": r.client_id, "n_poisoned": int(len(r.poisoned_local_positions)),
                                    "poisoned_global_indices": r.poisoned_global_indices.tolist()} for r in records]}
        write_json(meta, res_dir / "partition_and_poison.json")
        rec.update({"BA": ev["BA"], "ASR": ev["ASR"], "n_test": ev["n_test"], "n_asr": ev["n_asr"], "trajectory": traj,
                    "checkpoint_path": str(ckpt), "checkpoint_sha256": state_dict_sha256(sd), "history_path": str(hist_path),
                    "history_rounds": hist.rounds, "history_sha256": file_sha256(hist_path), "num_parameters": count_parameters(model),
                    "parameter_groups": [g for g, _ in parameter_groups(model)], "checkpoint_bytes": ckpt.stat().st_size, "history_bytes": hist_path.stat().st_size,
                    "n_train_after_holdout": len(train), "n_eval": len(test),
                    "dataset_train_sha256": train.sha256(), "dataset_test_sha256": test.sha256(),
                    "poisoned_counts": {str(r.client_id): int(len(r.poisoned_local_positions)) for r in records},
                    "runtime_sec": time.time() - t_start, "status": "completed", "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "environment": environment_info(), "raw_log_path": str(log_dir / "compromised.log")})
        log(f"DONE compromised: BA={ev['BA']:.4f} ASR={ev['ASR']:.4f} runtime={rec['runtime_sec']:.0f}s")
    except Exception as e:  # record the failure, then re-raise: never silently continue
        rec.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc(), "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        write_json(rec, rec_path)
        log.close()
        raise
    write_json(rec, rec_path)
    log.close()
    return rec


def load_history(path: str | Path) -> UpdateHistory:
    d = torch.load(path)
    return UpdateHistory(rounds=d["rounds"], global_states=d["global_states"], client_updates=d["client_updates"], client_sizes=d["client_sizes"])


def reconstruct_last_round(cfg: ExperimentConfig, seed: int, comp: dict[str, Any], client_ds, spec: dict, device, log=None) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, Any]]:
    """Exactly reconstruct the local models of every client from the last training round (T-1) and the global model
    they started from, which history-based-last-round methods such as FedUP require but the stored history (every
    history_every rounds) does not contain when T is even.

    Steps: (1) aggregate the stored round T-2 on the same device as training -> theta^(T-1); (2) replay the data-order
    generator through rounds 0..T-2 by iterating DataLoaders built exactly as in local_train (no training); (3) re-execute the
    final round's local training for every client; (4) aggregate and require the result to match the stored compromised
    checkpoint checksum bit for bit. Raises RuntimeError otherwise (never falls back to an approximation)."""
    if cfg.clients_per_round != 1.0:
        raise RuntimeError("reconstruct_last_round assumes full participation")
    hist = load_history(comp["history_path"])
    if file_sha256(comp["history_path"]) != comp["history_sha256"]:
        raise RuntimeError("history checksum mismatch")
    T = cfg.rounds
    if hist.rounds[-1] != T - 2:
        raise RuntimeError(f"last stored round is {hist.rounds[-1]}, expected {T - 2}")
    participating = list(range(cfg.num_clients))
    sizes = {c: len(client_ds[c]) for c in participating}
    g_prev = {k: v.to(device) for k, v in hist.global_states[-1].items()}
    upd = {c: {k: v.to(device) for k, v in u.items()} for c, u in hist.client_updates[-1].items()}
    theta_prev = fedavg_aggregate(g_prev, upd, sizes)                      # theta^(T-1), same op and device as training
    gen = torch.Generator().manual_seed(seed + 15485863)
    lcfg = cfg.local_cfg()
    from torch.utils.data import DataLoader
    for _t in range(T - 1):                                                 # replay the data-order draws of rounds 0..T-2
        for c in participating:                                             # exactly as local_train constructs its loader
            loader = DataLoader(client_ds[c], batch_size=lcfg.batch_size, shuffle=True, drop_last=False, generator=gen, num_workers=0)
            for _e in range(lcfg.local_epochs):
                for _ in loader:
                    pass
    model = _model_factory(cfg, spec)().to(device)
    last_models: dict[int, dict[str, torch.Tensor]] = {}
    updates: dict[int, dict[str, torch.Tensor]] = {}
    for c in participating:
        model.load_state_dict(theta_prev)
        local_train(model, client_ds[c], lcfg, device, generator=gen)
        st = clone_state(model)
        last_models[c] = {k: v.cpu() for k, v in st.items()}
        updates[c] = state_delta(st, theta_prev)
    theta_T = fedavg_aggregate(theta_prev, updates, sizes)
    sha = state_dict_sha256({k: v.cpu() for k, v in theta_T.items()})
    ok = sha == comp["checkpoint_sha256"]
    info = {"stored_round_used": hist.rounds[-1], "reconstructed_round": T - 1, "reproduced_compromised_checkpoint": ok,
            "reconstructed_sha256": sha, "compromised_sha256": comp["checkpoint_sha256"], "n_clients": len(participating),
            "local_model_bytes_each": int(sum(v.numel() * v.element_size() for v in last_models[participating[0]].values()))}
    if log is not None:
        log(f"last-round reconstruction: round {T - 1} local models of {len(participating)} clients; aggregate reproduces compromised checkpoint: {ok}")
    if not ok:
        raise RuntimeError("last-round reconstruction does not reproduce the stored compromised checkpoint; refusing to run FedUP on approximate artifacts")
    return last_models, {k: v.cpu() for k, v in theta_prev.items()}, info


def run_unlearning(cfg: ExperimentConfig, seed: int, method: str, alpha: float, force: bool = False, deleted: list[int] | None = None) -> dict[str, Any]:
    """Run one unlearning method. `deleted` overrides the nested deletion set (attacker-subset study); its record is
    tagged with the client ids and alpha = |deleted| / |malicious|."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; known: {sorted(METHODS)}")
    res_dir, ckpt_dir, log_dir = cfg.seed_dirs(seed)
    if deleted is not None:
        deleted = sorted(deleted)
        if not set(deleted) <= set(cfg.malicious_clients) or not deleted:
            raise ValueError("explicit deletion set must be a non-empty subset of the malicious clients")
        alpha = len(deleted) / len(cfg.malicious_clients)
        tag = f"{method}_del{'-'.join(map(str, deleted))}"
    else:
        tag = f"{method}_alpha{alpha:.2f}"
    rec_path = res_dir / f"{tag}.json"
    if rec_path.exists() and not force:
        with open(rec_path) as f:
            rec = json.load(f)
        if rec.get("status") == "completed":
            return verify_record_artifacts(cfg, rec)   # skip only when the checkpoint exists and matches
    comp = train_compromised(cfg, seed)
    log = Logger(log_dir / f"{tag}.log")
    device = get_device(cfg.device)
    t_start = time.time()
    if deleted is None:
        deleted = cfg.deletion_set(alpha)
    retained = [c for c in range(cfg.num_clients) if c not in deleted]
    for d in (res_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"run_id": f"{cfg.exp_id}/seed{seed}/{tag}", "exp_id": cfg.exp_id, "seed": seed, "method": method,
                           "alpha": alpha, "deleted_clients": deleted, "retained_clients": retained, "config_path": cfg.path,
                           "compromised_checkpoint_sha256": comp["checkpoint_sha256"], "status": "running",
                           "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": str(device),
                           "architecture": cfg.architecture, "eval_split": cfg.eval_split, "validation_holdout": cfg.validation_holdout}
    try:
        set_seed(seed)
        train, test, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
        model = _model_factory(cfg, spec)().to(device)
        sd = torch.load(comp["checkpoint_path"])
        if state_dict_sha256(sd) != comp["checkpoint_sha256"]:
            raise RuntimeError("compromised checkpoint checksum mismatch")
        model.load_state_dict(sd)
        before = evaluate(model, test, trig, cfg.target_label, device)
        if abs(before["ASR"] - comp["ASR"]) > 1e-6 or abs(before["BA"] - comp["BA"]) > 1e-6:
            raise RuntimeError("re-evaluation of the compromised checkpoint does not match its record")
        params = dict(cfg.methods.get(method, {}))
        extra: dict[str, Any] = {}
        if method in ("federaser", "cf_sameinit"):
            hist = load_history(comp["history_path"])
            if file_sha256(comp["history_path"]) != comp["history_sha256"]:
                raise RuntimeError("history checksum mismatch")
            extra["history"] = hist
            rec["history_rounds_used"] = hist.rounds
        if method in ("full_retrain", "cf_sameinit"):
            extra["model_factory"] = _model_factory(cfg, spec)
            params["rounds"] = cfg.rounds
        if method == "grad_negation":
            extra["num_classes"] = spec["num_classes"]
        if method == "fedup":
            last_models, g_prev, info = reconstruct_last_round(cfg, seed, comp, client_ds, spec, device, log=log)
            extra["last_round_models"] = last_models; extra["global_prev"] = g_prev
            rec["last_round_reconstruction"] = info
            rec["server_storage_bytes"] = int(info["local_model_bytes_each"] * (info["n_clients"] + 1))
            write_json(info, res_dir / "last_round_reconstruction.json")
        log(f"method={method} alpha={alpha} deleted={deleted} params={params}")
        set_seed(seed)
        model = METHODS[method](model, client_ds, retained, deleted, cfg.local_cfg(), device, seed, log=log, **params, **extra)
        ev = evaluate(model, test, trig, cfg.target_label, device)
        sd_out = {k: v.cpu() for k, v in model.state_dict().items()}
        ckpt = ckpt_dir / f"{tag}.pt"
        torch.save(sd_out, ckpt)
        if hasattr(model, "n_pruned_channels"):
            rec["n_pruned_channels"] = int(model.n_pruned_channels)
            rec["pruned_layer"] = getattr(model, "pruned_layer", None)
        if hasattr(model, "fedup_n_pruned"):
            rec["fedup_n_pruned"] = int(model.fedup_n_pruned); rec["fedup_n_prunable"] = int(model.fedup_n_prunable)
        rec.update({"BA": ev["BA"], "ASR": ev["ASR"], "BA_before": before["BA"], "ASR_before": before["ASR"], "method_params": params,
                    "checkpoint_path": str(ckpt), "checkpoint_sha256": state_dict_sha256(sd_out), "checkpoint_bytes": ckpt.stat().st_size, "runtime_sec": time.time() - t_start,
                    "status": "completed", "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "raw_log_path": str(log_dir / f"{tag}.log"),
                    "environment": environment_info()})
        log(f"DONE {tag}: BA={ev['BA']:.4f} ASR={ev['ASR']:.4f} runtime={rec['runtime_sec']:.0f}s")
    except Exception as e:
        rec.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc(), "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        write_json(rec, rec_path)
        log.close()
        raise
    write_json(rec, rec_path)
    log.close()
    return rec


def run_all(cfg: ExperimentConfig, seeds=None, methods=None, alphas=None, force: bool = False) -> list[dict[str, Any]]:
    seeds = cfg.seeds if seeds is None else seeds
    methods = list(cfg.methods) if methods is None else methods
    alphas = cfg.alphas if alphas is None else alphas
    out = []
    for s in seeds:
        out.append(train_compromised(cfg, s, force=force))
        for a in alphas:
            for m in methods:
                out.append(run_unlearning(cfg, s, m, a, force=force))
    return out

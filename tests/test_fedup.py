"""FedUP baseline: mask construction, prunable layers, weighted average, exact last-round reconstruction."""
import json, os, tempfile
import numpy as np, pytest, torch
from bpa_fu.models import build_model, SimpleCNN
from bpa_fu.unlearning.fedup import fedup, fedup_mask, prunable_keys, weighted_average
from bpa_fu.federated_training import LocalTrainConfig, run_fedavg, fedavg_aggregate, local_train, state_delta
from bpa_fu.models import clone_state
from bpa_fu.datasets import TensorImageDataset
from bpa_fu.reproducibility import state_dict_sha256

CPU = torch.device("cpu")


def _clients(n_clients=4, n=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [TensorImageDataset(torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8), torch.randint(0, 10, (n,), generator=g)) for _ in range(n_clients)]


def test_prunable_keys_cover_dense_and_conv_only():
    assert prunable_keys(SimpleCNN(3, 10)) == ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"]
    ks = prunable_keys(build_model("resnet18", 3, 10))
    assert "conv1.weight" in ks and "fc.weight" in ks and "layer4.1.conv2.weight" in ks and "layer2.0.shortcut.0.weight" in ks
    assert not any("bn" in k or "bias" in k or "running" in k for k in ks) and len(ks) == 21


def test_mask_selects_top_fraction_per_layer_by_squared_difference_times_magnitude():
    torch.manual_seed(0)
    m = SimpleCNN(3, 10); keys = prunable_keys(m)
    ben = {k: torch.randn_like(v) for k, v in m.state_dict().items()}
    mal = {k: v + 0.01 * torch.rand_like(v) for k, v in ben.items()}      # small non-zero differences everywhere
    gprev = {k: torch.ones_like(v) for k, v in ben.items()}
    mal["fc2.weight"][0, :5] += 10.0        # five strongly conflicting weights
    masks = fedup_mask(mal, ben, gprev, keys, prune_fraction=0.01)
    for k in keys:
        assert masks[k].sum().item() == int(np.ceil(0.01 * masks[k].numel()))
    assert masks["fc2.weight"][0, :5].all()
    # magnitude weighting: a zero global weight is never selected before a non-zero one with equal difference
    g2 = {k: torch.ones_like(v) for k, v in ben.items()}; g2["fc2.weight"][0, 0] = 0.0
    masks2 = fedup_mask(mal, ben, g2, keys, prune_fraction=0.005)
    assert not masks2["fc2.weight"][0, 0]


def test_weighted_average_uses_sample_sizes():
    a = {"w": torch.tensor([1.0, 1.0]), "n": torch.tensor([1])}; b = {"w": torch.tensor([3.0, 3.0]), "n": torch.tensor([2])}
    out = weighted_average({0: a, 1: b}, {0: 3, 1: 1})
    assert torch.allclose(out["w"], torch.tensor([1.5, 1.5])) and out["n"].item() == 1


def test_fedup_prunes_benign_average_then_recovers_without_deleted_data():
    ds = _clients(); cfg = LocalTrainConfig(local_epochs=1, batch_size=16, lr=0.01)
    torch.manual_seed(0); g = {k: v.clone() for k, v in SimpleCNN(3, 10).state_dict().items()}
    last = {c: {k: v + 0.01 * (c + 1) for k, v in g.items()} for c in range(4)}
    class Boom(TensorImageDataset):
        def __getitem__(self, i): raise AssertionError("deleted client data accessed")
    ds_b = list(ds); ds_b[0] = Boom(ds[0].images, ds[0].labels)
    m = fedup(SimpleCNN(3, 10), ds_b, [1, 2, 3], [0], cfg, CPU, seed=0, last_round_models=last, global_prev=g, prune_fraction=0.1, recovery_rounds=0)
    sd = m.state_dict()
    for k in prunable_keys(m):
        assert (sd[k] == 0).sum().item() >= int(np.ceil(0.1 * sd[k].numel()))
    assert m.fedup_n_pruned > 0
    m2 = fedup(SimpleCNN(3, 10), ds_b, [1, 2, 3], [0], cfg, CPU, seed=0, last_round_models=last, global_prev=g, prune_fraction=0.1, recovery_rounds=1)
    assert isinstance(m2, SimpleCNN)
    with pytest.raises(ValueError):
        fedup(SimpleCNN(3, 10), ds, [1, 2, 3], [0], cfg, CPU, seed=0)


def test_last_round_reconstruction_reproduces_the_final_checkpoint():
    """Replaying the data-order generator and re-executing the last round from the stored round T-2 must give the
    stored final model bit for bit (the property reconstruct_last_round relies on)."""
    from bpa_fu.experiment import ExperimentConfig, reconstruct_last_round
    ds = _clients(n_clients=3, n=24); cfg = LocalTrainConfig(local_epochs=2, batch_size=8, lr=0.01)
    torch.manual_seed(5); m = SimpleCNN(3, 10)
    m, hist = run_fedavg(m, ds, [0, 1, 2], rounds=4, cfg=cfg, device=CPU, seed=7, history_every=2)
    final = clone_state(m); sha = state_dict_sha256(final)
    with tempfile.TemporaryDirectory() as tmp:
        hp = os.path.join(tmp, "history.pt")
        torch.save({"rounds": hist.rounds, "global_states": hist.global_states, "client_updates": hist.client_updates, "client_sizes": hist.client_sizes}, hp)
        from bpa_fu.reproducibility import file_sha256
        comp = {"history_path": hp, "history_sha256": file_sha256(hp), "checkpoint_sha256": sha}
        ecfg = ExperimentConfig(exp_id="t", dataset="cifar10", rounds=4, local_epochs=2, batch_size=8, lr=0.01, num_clients=3, malicious_clients=[0], seeds=[7], device="cpu")
        last, gprev, info = reconstruct_last_round(ecfg, 7, comp, ds, {"in_channels": 3, "num_classes": 10}, CPU)
        assert info["reproduced_compromised_checkpoint"] and set(last) == {0, 1, 2}
        # a tampered checksum must be refused
        comp["checkpoint_sha256"] = "0" * 64
        with pytest.raises(RuntimeError):
            reconstruct_last_round(ecfg, 7, comp, ds, {"in_channels": 3, "num_classes": 10}, CPU)

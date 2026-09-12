"""ResNet-18 (CIFAR variant) support: construction, parameter groups, FedAvg with BatchNorm buffers, FedEraser on a
ResNet state dict, checkpoint verification, architecture-independent alignment, Fine-Pruning adaptation, validation
holdout."""
import math, os, tempfile, json
import numpy as np, pytest, torch
from bpa_fu.models import build_model, count_parameters, parameter_groups, parameter_keys, ResNet18CIFAR, SimpleCNN
from bpa_fu.alignment import alignment_metrics, check_groups_cover, malicious_contribution
from bpa_fu.federated_training import fedavg_aggregate, run_fedavg, LocalTrainConfig, state_delta
from bpa_fu.unlearning.federaser import federaser
from bpa_fu.unlearning.fine_pruning import fine_pruning
from bpa_fu.datasets import TensorImageDataset
from bpa_fu.reproducibility import state_dict_sha256
from bpa_fu.experiment import ExperimentConfig, split_validation

CPU = torch.device("cpu")


def _clients(n_clients=4, n=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [TensorImageDataset(torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8), torch.randint(0, 10, (n,), generator=g)) for _ in range(n_clients)]


def test_resnet18_cifar_construction():
    m = build_model("resnet18", 3, 10)
    assert isinstance(m, ResNet18CIFAR)
    assert count_parameters(m) == 11_173_962            # CIFAR ResNet-18 with a 10-way classifier
    assert m.conv1.kernel_size == (3, 3) and m.conv1.stride == (1, 1) and m.conv1.padding == (1, 1)
    assert not any(isinstance(mod, torch.nn.MaxPool2d) for mod in m.modules())   # no ImageNet max-pooling
    x = torch.zeros(2, 3, 32, 32)
    assert m.features(x).shape == (2, 512, 4, 4) and m(x).shape == (2, 10)
    assert count_parameters(build_model("resnet18", 3, 43)) == 11_173_962 + 33 * 512 + 33
    with pytest.raises(ValueError):
        build_model("vgg", 3, 10)


def test_parameter_groups_partition_parameters():
    for arch in ("simplecnn", "resnet18"):
        m = build_model(arch, 3, 10)
        groups = parameter_groups(m)
        check_groups_cover(parameter_keys(m), groups)
        n = 0
        for _, ps in groups:
            n += sum(p.numel() for k, p in m.named_parameters() if any(k.startswith(x) for x in ps))
        assert n == count_parameters(m)
    assert [g for g, _ in parameter_groups("resnet18")] == ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
    assert [g for g, _ in parameter_groups("simplecnn")] == ["conv1", "conv2", "fc1", "fc2"]


def test_fedavg_aggregates_bn_buffers_and_keeps_int_counter():
    m = build_model("resnet18", 3, 10)
    g = {k: v.clone() for k, v in m.state_dict().items()}
    u0 = {k: torch.ones_like(v) if v.dtype.is_floating_point else torch.full_like(v, 7) for k, v in g.items()}
    u1 = {k: -torch.ones_like(v) if v.dtype.is_floating_point else torch.full_like(v, 9) for k, v in g.items()}
    out = fedavg_aggregate(g, {0: u0, 1: u1}, {0: 3, 1: 1})
    assert torch.allclose(out["bn1.running_mean"], g["bn1.running_mean"] + 0.5)      # float buffer aggregated
    assert torch.equal(out["bn1.num_batches_tracked"], g["bn1.num_batches_tracked"])  # int64 counter untouched
    assert set(out) == set(g)


def test_fedavg_and_federaser_on_resnet_state_dict():
    ds = _clients(); cfg = LocalTrainConfig(local_epochs=1, batch_size=16, lr=0.01)
    m = build_model("resnet18", 3, 10)
    m, hist = run_fedavg(m, ds, [0, 1, 2, 3], rounds=2, cfg=cfg, device=CPU, seed=0, history_every=1)
    assert hist.rounds == [0, 1]
    upd = hist.client_updates[0][0]
    assert set(upd) == set(m.state_dict()) and all(not v.dtype.is_floating_point or v.shape == m.state_dict()[k].shape for k, v in upd.items())
    m2 = federaser(build_model("resnet18", 3, 10), ds, [1, 2, 3], [0], cfg, CPU, seed=0, history=hist, calibration_ratio=0.5)
    sd = m2.state_dict()
    assert all(torch.isfinite(v).all() for v in sd.values() if v.dtype.is_floating_point)
    assert set(sd) == set(m.state_dict())


def test_checkpoint_checksum_roundtrip_resnet():
    m = build_model("resnet18", 3, 10)
    sd = {k: v.cpu() for k, v in m.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "ck.pt"); torch.save(sd, p)
        back = torch.load(p, map_location="cpu")
        assert state_dict_sha256(back) == state_dict_sha256(sd)
        back["fc.weight"][0, 0] += 1e-3
        assert state_dict_sha256(back) != state_dict_sha256(sd)


def test_alignment_is_architecture_independent():
    for arch in ("simplecnn", "resnet18"):
        torch.manual_seed(0)
        m_star, m_prime, m_cf = build_model(arch, 3, 10), build_model(arch, 3, 10), build_model(arch, 3, 10)
        ts, tp, tc = m_star.state_dict(), m_prime.state_dict(), m_cf.state_dict()
        keys = parameter_keys(m_star)
        upd = {c: {k: torch.randn_like(v) if v.dtype.is_floating_point else v for k, v in ts.items()} for c in range(4)}
        hist = {"client_updates": [upd], "client_sizes": [{0: 1, 1: 1, 2: 1, 3: 1}]}
        dm, dr = malicious_contribution(hist, [0, 1])
        met = alignment_metrics(ts, tp, tc, dm, dr, groups=parameter_groups(m_star), param_keys=keys)
        groups = [g for g, _ in parameter_groups(m_star)]
        assert abs(sum(met[f"{g}_energy"] for g in groups) - 1.0) < 1e-5
        assert all(np.isfinite(met[f"{g}_rho"]) for g in groups) and np.isfinite(met["global_cos_u_d"])
        # residual fraction equals sqrt(1 - cos^2) of u against -Delta_M
        c = met["global_cos_u_negdm"]; assert abs(met["global_resid_frac"] - math.sqrt(max(0.0, 1 - c * c))) < 1e-4
    # SimpleCNN: restricting to parameter keys equals the whole-state-dict computation (no buffers)
    m = SimpleCNN(3, 10); ts = m.state_dict(); tp = SimpleCNN(3, 10).state_dict()
    dm = {k: torch.randn_like(v) for k, v in ts.items()}
    a = alignment_metrics(ts, tp, None, dm); b = alignment_metrics(ts, tp, None, dm, param_keys=parameter_keys(m))
    assert a == b


def test_fine_pruning_resnet_path_masks_feature_channels(monkeypatch):
    import importlib; fp = importlib.import_module("bpa_fu.unlearning.fine_pruning")
    ds = _clients(n_clients=3, n=8); cfg = LocalTrainConfig(local_epochs=1, batch_size=8, lr=0.001)
    torch.manual_seed(0)
    m = build_model("resnet18", 3, 10); m.eval()
    x = torch.rand(4, 3, 32, 32); logits_before = m(x).clone()
    m = fine_pruning(m, ds, [1, 2], [0], cfg, CPU, seed=0, max_acc_drop=1.0, recovery_rounds=0)   # prunes every channel
    assert m.pruned_layer == "layer4" and m.n_pruned_channels == 512
    zero_cols = (m.fc.weight.abs().sum(0) == 0).nonzero().flatten().tolist()
    assert len(zero_cols) == 512
    m.eval(); out = m(x)
    assert torch.allclose(out, m.fc.bias.expand_as(out), atol=1e-6)          # only the bias remains
    # partial pruning: masking the pooled features equals zeroing the classifier columns (exact on the forward path)
    torch.manual_seed(1); m2 = build_model("resnet18", 3, 10); m2.eval()
    feats = torch.nn.functional.adaptive_avg_pool2d(m2.features(x), 1).flatten(1)
    mask = torch.ones(512); mask[[0, 5, 77]] = 0.0
    ref = (feats * mask) @ m2.fc.weight.T + m2.fc.bias
    with torch.no_grad():
        m2.fc.weight[:, [0, 5, 77]] = 0.0
    assert torch.allclose(m2(x), ref, atol=1e-5)
    # SimpleCNN path unchanged: uses conv2 and a cheap accuracy stub
    calls = {"n": 0}
    def fake_acc(model, loader, device):
        calls["n"] += 1
        return 1.0 if calls["n"] <= 10 else 0.0
    monkeypatch.setattr(fp, "_accuracy", fake_acc)
    m_cnn = fine_pruning(SimpleCNN(3, 10), ds, [1, 2], [0], cfg, CPU, seed=0, max_acc_drop=0.04, recovery_rounds=0)
    assert m_cnn.pruned_layer == "conv2" and m_cnn.n_pruned_channels == 10
    assert (m_cnn.conv2.weight.abs().sum((1, 2, 3)) == 0).sum().item() == 10


def test_validation_holdout_is_deterministic_and_disjoint():
    g = torch.Generator().manual_seed(3)
    train = TensorImageDataset(torch.randint(0, 256, (100, 3, 32, 32), generator=g, dtype=torch.uint8), torch.arange(100) % 10)
    tr1, va1 = split_validation(train, 20, seed=5); tr2, va2 = split_validation(train, 20, seed=5)
    assert len(tr1) == 80 and len(va1) == 20 and torch.equal(va1.images, va2.images) and torch.equal(tr1.labels, tr2.labels)
    ids = lambda d: {d.images[i].numpy().tobytes() for i in range(len(d))}
    assert not (ids(tr1) & ids(va1))
    tr3, va3 = split_validation(train, 20, seed=6); assert not torch.equal(va1.images, va3.images)
    with pytest.raises(ValueError):
        ExperimentConfig(exp_id="x", dataset="cifar10", eval_split="validation").validate()
    ExperimentConfig(exp_id="x", dataset="cifar10", eval_split="validation", validation_holdout=5000).validate()

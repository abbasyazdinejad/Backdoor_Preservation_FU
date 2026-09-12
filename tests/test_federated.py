import numpy as np, torch
from bpa_fu.federated_training import fedavg_aggregate, run_fedavg, LocalTrainConfig, UpdateHistory
from bpa_fu.unlearning.federaser import calibrate_update, federaser
from bpa_fu.models import SimpleCNN, count_parameters
from bpa_fu.datasets import TensorImageDataset

def test_fedavg_weighted_aggregate():
    g = {"w": torch.zeros(3)}
    upd = {0: {"w": torch.ones(3)}, 1: {"w": -torch.ones(3)}}
    out = fedavg_aggregate(g, upd, {0: 3, 1: 1})
    assert torch.allclose(out["w"], torch.full((3,), 0.5))

def test_parameter_counts():
    assert count_parameters(SimpleCNN(3, 10)) == 545098
    assert count_parameters(SimpleCNN(1, 10)) == 544522
    assert count_parameters(SimpleCNN(3, 43)) == 549355

def test_calibrate_update_layerwise_norm_and_direction():
    stored = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 2.0])}
    calib = {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 0.0])}
    out = calibrate_update(stored, calib)
    assert torch.allclose(out["a"], torch.tensor([0.0, 5.0])) and torch.allclose(out["b"], torch.tensor([2.0, 0.0]))

def _clients(n_clients=4, n=40):
    g = torch.Generator().manual_seed(1)
    return [TensorImageDataset(torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8), torch.randint(0, 10, (n,), generator=g)) for _ in range(n_clients)]

def test_history_recording_and_removed_client_excluded():
    ds = _clients(); cfg = LocalTrainConfig(local_epochs=1, batch_size=20, lr=0.01)
    m = SimpleCNN(3, 10)
    m, hist = run_fedavg(m, ds, [0, 1, 2, 3], rounds=4, cfg=cfg, device=torch.device("cpu"), seed=0, history_every=2)
    assert hist.rounds == [0, 2] and set(hist.client_updates[0]) == {0, 1, 2, 3}
    # FedEraser with client 0 deleted: only retained clients participate and are aligned with the history
    m2 = federaser(SimpleCNN(3, 10), ds, [1, 2, 3], [0], cfg, torch.device("cpu"), seed=0, history=hist, calibration_ratio=0.5)
    assert isinstance(m2, SimpleCNN)
    # short retraining on retained clients must not touch deleted client's data: monkeypatch dataset access
    class Boom(TensorImageDataset):
        def __getitem__(self, i): raise AssertionError("deleted client data accessed")
    ds_b = list(ds); ds_b[0] = Boom(ds[0].images, ds[0].labels)
    from bpa_fu.unlearning.short_retraining import short_retraining
    short_retraining(SimpleCNN(3, 10), ds_b, [1, 2, 3], [0], cfg, torch.device("cpu"), seed=0, rounds=1)
    federaser(SimpleCNN(3, 10), ds_b, [1, 2, 3], [0], cfg, torch.device("cpu"), seed=0, history=hist)
    import pytest
    with pytest.raises(ValueError):
        federaser(SimpleCNN(3, 10), ds, [1, 2, 3], [0], cfg, torch.device("cpu"), seed=0, history=None)

def test_counterfactual_uses_only_retained_clients():
    from bpa_fu.unlearning.full_retraining import full_retraining
    ds = _clients(); cfg = LocalTrainConfig(local_epochs=1, batch_size=20)
    class Boom(TensorImageDataset):
        def __getitem__(self, i): raise AssertionError("deleted client data accessed")
    ds[0] = Boom(ds[0].images, ds[0].labels)
    m = full_retraining(SimpleCNN(3, 10), ds, [1, 2, 3], [0], cfg, torch.device("cpu"), seed=0, rounds=1, model_factory=lambda: SimpleCNN(3, 10))
    assert isinstance(m, SimpleCNN)

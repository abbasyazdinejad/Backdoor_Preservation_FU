import numpy as np, torch, pytest
from bpa_fu.experiment import ExperimentConfig
from bpa_fu.evaluation import evaluate
from bpa_fu.triggers import PatchTrigger
from bpa_fu.datasets import TensorImageDataset

def test_nested_deletion_sets():
    cfg = ExperimentConfig(exp_id="t", dataset="cifar10", malicious_clients=[3, 0, 1, 2], alphas=[0.25, 0.5, 0.75, 1.0])
    sets = [cfg.deletion_set(a) for a in cfg.alphas]
    assert sets == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    for a, b in zip(sets[:-1], sets[1:]):
        assert set(a) < set(b)
    with pytest.raises(ValueError):
        cfg.deletion_set(0.0)

class Const(torch.nn.Module):
    """Predicts class 0 for triggered images (bottom-right patch bright) and true class otherwise."""
    def __init__(self, labels): super().__init__(); self.labels = labels; self.i = 0
    def forward(self, x):
        n = x.shape[0]; out = torch.zeros(n, 10)
        trig = x[:, :, 28, 28].mean(1) > 0.99
        lab = self.labels[self.i:self.i + n]; self.i += n
        out[torch.arange(n), lab] = 1.0
        out[trig] = 0; out[trig, 0] = 5.0
        return out

def test_ba_and_asr_exclude_target_class():
    imgs = torch.zeros(100, 3, 32, 32, dtype=torch.uint8); labels = torch.arange(100) % 10
    ds = TensorImageDataset(imgs, labels)
    m = Const(labels)
    # BA pass: model returns true label -> BA=1; ASR pass: triggered -> class 0 for all, but target-class samples excluded
    class Wrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.m = m; s.calls = 0
        def forward(s, x):
            s.calls += 1
            if s.calls % 2 == 1: m.i = 0  # BA batch
            else: m.i = 0
            return m(x)
    r = evaluate(Wrap(), ds, PatchTrigger(), target_label=0, device=torch.device("cpu"), batch_size=100)
    assert r["BA"] == 1.0 and r["ASR"] == 1.0 and r["n_asr"] == 90 and r["n_test"] == 100

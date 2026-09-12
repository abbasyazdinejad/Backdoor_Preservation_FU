"""Pilot-only hyperparameter sweep for gradient negation on the pilot30 CIFAR-10 checkpoint (seed 0, 30 rounds).
Writes logs/pilot/negation_sweep.csv. Used to pick the documented default (radius, loss threshold multiplier)."""
import sys, os, json, itertools, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, pandas as pd
from bpa_fu.experiment import ExperimentConfig, _prepare, _model_factory
from bpa_fu.unlearning.gradient_negation import gradient_negation
from bpa_fu.evaluation import evaluate
from bpa_fu.reproducibility import get_device, set_seed
cfg = ExperimentConfig.load("configs/pilot30_cifar10.json"); seed = 0; device = get_device()
comp = json.load(open("results/pilot/pilot30_cifar10/seed0/compromised.json"))
train, test, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
rows = []
for radius, mult in itertools.product([1.0, 3.0], [1.0, 2.0, 4.0]):
    set_seed(seed); m = _model_factory(cfg, spec)(); m.load_state_dict(torch.load(comp["checkpoint_path"]))
    t0 = time.time()
    m = gradient_negation(m, client_ds, list(range(4, 20)), [0, 1, 2, 3], cfg.local_cfg(), device, seed, ascent_epochs=5, ascent_lr=0.002, radius=radius,
                          loss_threshold=mult * math.log(10), recovery_rounds=5, check_every=1)
    e = evaluate(m, test, trig, 0, device)
    rows.append(dict(radius=radius, threshold_mult=mult, BA=e["BA"], ASR=e["ASR"], sec=time.time() - t0)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("logs/pilot/negation_sweep.csv", index=False)

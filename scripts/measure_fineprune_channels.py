#!/usr/bin/env python3
"""Direct measurement for Fine-Pruning: are the pruned channels the ones that respond to the trigger?
For every Fine-Pruning run (CIFAR-10, alpha = 1): recompute the pruning order of the compromised model from the retained
clients' activations (deterministic), take the first n_pruned channels of that order (n from the run record), and measure
on the test set the mean post-ReLU activation of each conv2 channel on clean inputs and on triggered inputs.
Reports, for pruned vs kept channels, the mean trigger-induced activation change and the share of the total
trigger-induced activation increase carried by the pruned channels.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from bpa_fu.experiment import ExperimentConfig, _prepare, _model_factory
from bpa_fu.unlearning.fine_pruning import _channel_activation_and_accuracy
from bpa_fu.reproducibility import get_device, set_seed
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/cifar10_iid.json"); p.add_argument("--out", default="results/processed/corrected_v2/fineprune_channels.csv"); a = p.parse_args()
cfg = ExperimentConfig.load(a.config); device = get_device(cfg.device); rows = []

@torch.no_grad()
def channel_means(model, loader, trig, target):
    model.eval(); clean = trig_ = None; n = 0
    for x, y in loader:
        keep = y != target; x = x[keep].to(device)
        if x.shape[0] == 0: continue
        for arr, inp in (("clean", x), ("trig", trig.apply(x))):
            h = model.pool(F.relu(model.conv1(inp))); act = F.relu(model.conv2(h)).mean(dim=(2, 3)).sum(0)
            if arr == "clean": clean = act if clean is None else clean + act
            else: trig_ = act if trig_ is None else trig_ + act
        n += x.shape[0]
    return (clean / n).cpu().numpy(), (trig_ / n).cpu().numpy()

for seed in cfg.seeds:
    for alpha in cfg.alphas:
        rec = json.load(open(f"results/raw/legacy_v1/{cfg.exp_id}/seed{seed}/fine_pruning_alpha{alpha:.2f}.json")); n_pruned = rec["n_pruned_channels"]
        comp = json.load(open(f"results/raw/legacy_v1/{cfg.exp_id}/seed{seed}/compromised.json"))
        set_seed(seed); train, test, spec, parts, client_ds, records, trig = _prepare(cfg, seed)
        deleted = cfg.deletion_set(alpha); retained = [c for c in range(cfg.num_clients) if c not in deleted]
        model = _model_factory(cfg, spec)().to(device); model.load_state_dict(torch.load(comp["checkpoint_path"]))
        acts, _ = _channel_activation_and_accuracy(model, DataLoader(ConcatDataset([client_ds[c] for c in retained]), batch_size=512), device)
        order = torch.argsort(acts).tolist(); pruned = set(order[:n_pruned]); kept = [c for c in range(64) if c not in pruned]
        clean, trigd = channel_means(model, DataLoader(test, batch_size=512), trig, cfg.target_label)
        delta = trigd - clean; pos = np.clip(delta, 0, None)
        rows.append(dict(exp_id=cfg.exp_id, seed=seed, alpha=alpha, n_pruned=n_pruned,
                         mean_delta_pruned=float(delta[list(pruned)].mean()), mean_delta_kept=float(delta[kept].mean()),
                         share_trigger_increase_pruned=float(pos[list(pruned)].sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
                         share_channels_pruned=n_pruned / 64, mean_clean_pruned=float(clean[list(pruned)].mean()), mean_clean_kept=float(clean[kept].mean()),
                         top10_trigger_channels_pruned=int(sum(c in pruned for c in np.argsort(-delta)[:10]))))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out)

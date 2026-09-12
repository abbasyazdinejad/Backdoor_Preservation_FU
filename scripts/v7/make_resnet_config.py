#!/usr/bin/env python3
"""Write configs/v7/cifar10_iid_resnet18.json from the frozen Rule R / Rule G selections (validation split only).
All outputs of that configuration go to results_resnet18/, checkpoints_resnet18/ and logs_resnet18/; nothing in the
authoritative SimpleCNN trees is read or written by it."""
import json, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
rr = json.load(open("results/raw/pilots_hyperparameter_selection/rule_R_selection.json"))
rg = json.load(open("results/raw/pilots_hyperparameter_selection/negation_validation_sweep_resnet18_selection.json"))
base = json.load(open("configs/final/cifar10_iid.json"))
c = dict(base)
c.update({"exp_id": "cifar10_iid_resnet18", "architecture": "resnet18", "lr": rr["selected_lr"], "rounds": rr["selected_T"], "seeds": [0, 1, 2, 3, 4], "alphas": [1.0],
          "eval_split": "test", "validation_holdout": 0, "eval_every": 5,
          "results_dir": "results/raw/resnet18", "checkpoints_dir": "checkpoints_resnet18", "logs_dir": "logs_resnet18",
          "notes": ("v7 architecture check: CIFAR-adapted ResNet-18, IID CIFAR-10, 20 clients (4 malicious), complete identification (alpha = 1), five seeds. "
                    f"Learning rate {rr['selected_lr']} and T = {rr['selected_T']} selected by Rule R and the gradient-negation multiplier {rg['selected_multiplier']} by Rule G on a "
                    "validation split of pilot seed 100 (audit/v7_hyperparameter_selection_protocol.md / _result.md) before any ResNet-18 test-set evaluation. "
                    "All other procedure parameters are the protocol values of the main study. Fine-Pruning uses the last-stage feature-channel adaptation.")})
c.pop("compromised_results_dir", None); c.pop("compromised_checkpoints_dir", None)
c["methods"] = {"short_retrain": {"rounds": 5}, "federaser": {"calibration_ratio": 0.5},
                "grad_negation": dict(base["methods"]["grad_negation"], loss_threshold_mult=float(rg["selected_multiplier"])),
                "full_retrain": {}, "fine_pruning": {"max_acc_drop": 0.04, "recovery_rounds": 5}, "cf_sameinit": {}}
json.dump(c, open("configs/v7/cifar10_iid_resnet18.json", "w"), indent=2)
print(json.dumps({k: c[k] for k in ("exp_id", "architecture", "lr", "rounds", "seeds", "alphas", "eval_split")}, indent=None)); print("grad_negation:", c["methods"]["grad_negation"])

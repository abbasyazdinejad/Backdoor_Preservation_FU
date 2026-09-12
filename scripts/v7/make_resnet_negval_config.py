#!/usr/bin/env python3
"""Write configs/v7/pilot_negval_resnet18.json: the validation-split configuration for Rule G on the ResNet-18 pilot model
selected by Rule R (same exp_id/results tree as that pilot, so the existing pilot model is verified and reused; nothing
is retrained and the test set is never evaluated)."""
import json, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
rr = json.load(open("results/raw/pilots_hyperparameter_selection/rule_R_selection.json"))
tag = "lr0p01" if rr["selected_lr"] == 0.01 else "lr0p05"
c = json.load(open(f"configs/v7/pilot_resnet18_{tag}.json"))
c["notes"] += (f" | Rule G sweep configuration: reuses this pilot model (30 rounds; Rule R selected T = {rr['selected_T']}, the sweep is applied to the "
               "30-round pilot checkpoint, which is the only ResNet-18 pilot checkpoint that exists; deviation from the protocol wording noted in audit/v7_hyperparameter_selection_result.md).")
c["methods"] = {"short_retrain": {"rounds": 5}, "grad_negation": dict(json.load(open("configs/final/cifar10_iid.json"))["methods"]["grad_negation"])}
json.dump(c, open("configs/v7/pilot_negval_resnet18.json", "w"), indent=2)
print("wrote configs/v7/pilot_negval_resnet18.json for", c["exp_id"])

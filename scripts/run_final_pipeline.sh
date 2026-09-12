#!/usr/bin/env bash
# Final pipeline after the main runs: corrected FedEraser + same-init counterfactuals (v2), attacker-subset study,
# negation-phase ablation, Fine-Pruning channel measurement, promotion, alignment analysis, validation gate.
set -euo pipefail
cd "$(dirname "$0")/.."
for c in cifar10_iid mnist_iid gtsrb_iid cifar10_dirichlet cifar10_iid_federaser_r1 cifar10_iid_federaser_dt1; do python3 scripts/run_experiment.py configs/v2/$c.json; done
python3 scripts/run_attacker_subsets.py configs/v2/subsets_cifar10_iid.json --methods short_retrain federaser
python3 scripts/run_attacker_subsets.py configs/v2/subsets_cifar10_dirichlet.json --methods short_retrain federaser
python3 scripts/aggregate_subsets.py
python3 scripts/ablate_negation_phases.py --configs configs/cifar10_iid.json configs/mnist_iid.json configs/gtsrb_iid.json configs/cifar10_dirichlet.json --out results/processed/corrected_v2/negation_phases.csv
python3 scripts/measure_fineprune_channels.py
python3 scripts/promote_results.py
python3 scripts/analyze_alignment.py --configs configs/final/cifar10_iid.json configs/final/mnist_iid.json configs/final/gtsrb_iid.json configs/final/cifar10_dirichlet.json configs/final/cifar10_iid_federaser_r1.json configs/final/cifar10_iid_federaser_dt1.json --out results/processed/final_matrix/alignment.csv
scripts/run_validation.sh

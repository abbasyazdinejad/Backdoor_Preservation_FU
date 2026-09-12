#!/usr/bin/env bash
# ResNet-18 architecture check (v7). Phase 1 trains the five compromised models sequentially (one process, because the
# stored update history of a ResNet-18 training is held in memory until it is written); phase 2 runs the unlearning
# procedures in two streams. Every record/checkpoint/log goes to results_resnet18/, checkpoints_resnet18/, logs_resnet18/.
set -euo pipefail
cd "$(dirname "$0")/../.."
CFG=configs/v7/cifar10_iid_resnet18.json
mkdir -p logs_resnet18
echo "phase 1: compromised models"
python3 scripts/run_experiment.py $CFG --train-only --seeds 0 1 2 3 4 > logs_resnet18/phase1_compromised.log 2>&1
echo "phase 2: unlearning procedures (two streams)"
python3 scripts/run_experiment.py $CFG --seeds 0 2 4 > logs_resnet18/phase2_streamA.log 2>&1 &
A=$!
python3 scripts/run_experiment.py $CFG --seeds 1 3 > logs_resnet18/phase2_streamB.log 2>&1 &
B=$!
wait $A; wait $B
echo "phase 3: aggregation and update-direction analysis"
python3 scripts/aggregate_results.py --configs $CFG --verify-checksums --out results/processed/resnet18/all_results_resnet18.csv --manifest results/processed/resnet18/reproduction_manifest_resnet18.json
python3 scripts/analyze_alignment.py --configs $CFG --out results/processed/resnet18/alignment_resnet18.csv --sameinit-from cifar10_iid_resnet18
echo "RESNET18_PIPELINE_DONE"

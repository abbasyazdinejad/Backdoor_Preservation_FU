#!/usr/bin/env bash
# Full reproduction of every experiment in the paper. Two streams run concurrently (the pipeline is
# partly CPU-bound, so two processes improve throughput on a single accelerator). Seeds run in order.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
( for s in 0 1 2 3 4; do python3 scripts/run_experiment.py configs/cifar10_iid.json --seeds $s; done ) > logs/stream_A.log 2>&1 &
( for s in 0 1 2 3 4; do python3 scripts/run_experiment.py configs/mnist_iid.json --seeds $s; done
  for s in 0 1 2 3 4; do python3 scripts/run_experiment.py configs/gtsrb_iid.json --seeds $s; done
  for s in 0 1 2 3 4; do python3 scripts/run_experiment.py configs/cifar10_dirichlet.json --seeds $s; done
  python3 scripts/run_experiment.py configs/cifar10_iid_federaser_r1.json
  python3 scripts/run_experiment.py configs/cifar10_iid_federaser_dt1.json ) > logs/stream_B.log 2>&1 &
wait
echo "ALL STREAMS FINISHED"

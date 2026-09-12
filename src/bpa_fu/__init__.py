"""bpa_fu: clean re-implementation of the Backdoor Preservation in Federated Unlearning pipeline.

Modules
-------
datasets            dataset loading (CIFAR-10, MNIST, GTSRB) with fixed preprocessing
partitions          deterministic IID / Dirichlet client partitioning
triggers            patch trigger definition
poisoning           malicious-client data poisoning (fixed poisoned index sets)
models              SimpleCNN (unified architecture) and parameter counting
federated_training  FedAvg simulation with optional update-history recording
unlearning          short_retraining, gradient_negation, federaser, full_retraining
evaluation          BA / ASR evaluation (target class excluded from ASR)
metrics             ASR reduction, BPR, tolerance-aware non-regression score
statistics          mean / std / 95% CI aggregation over seeds
experiment          experiment orchestration (one run = one (dataset, seed, method, alpha))
validation          schema / consistency assertions for the authoritative result file
plotting            programmatic figures with provenance sidecars
latex_tables        LaTeX table fragments generated from the authoritative result file
reproducibility     seeds, device selection, checksums, environment capture
"""
__version__ = "1.0.0"

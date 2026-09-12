# Backdoor preservation in federated unlearning

Implementation and experimental evidence for a study of what happens to a backdoor when a federated-learning operator
uses an unlearning procedure to remove the clients identified as its source.

The short answer the experiments give: continued training on the retained clients restores benign accuracy while
leaving most of the backdoor in place, so restored accuracy is not evidence of removal. Procedures that use the
malicious contribution explicitly do better, at a utility cost that depends on their budget.

**Every reported result can be regenerated and verified from the packaged run records in a few minutes, without
training and without downloading datasets.** Six parameter-space measurement files are verified at record level in the
default mode and require the separately archived checkpoints and datasets for recomputation from model weights. Open
the notebook and run it.

```bash
git clone <this repository> && cd backdoor-preservation-fu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
jupyter lab notebooks/reproduce_paper_results.ipynb      # or: jupyter notebook / VS Code
```

or head-less:

```bash
jupyter nbconvert --to notebook --execute notebooks/reproduce_paper_results.ipynb \
  --output executed.ipynb --ExecutePreprocessor.timeout=-1
```

The notebook finds the repository root itself, so it runs from the repository root, from `notebooks/`, from a parent
directory, from Jupyter Lab or from an editor.

## What the notebook does

It is organised around the results, not around file generation. Every result set, plot, statistic, ablation and
reported number is recomputed from the experimental records and displayed as a DataFrame or an inline figure, then
compared with the frozen reference that ships here.

```
results/raw/**            one JSON record per run
    -> aggregation        scripts/aggregate_results.py, scripts/aggregate_subsets.py
results/processed/**      the aggregated records everything is computed from
    -> analysis           src/bpa_fu/latex_tables.py, src/bpa_fu/plotting.py, src/bpa_fu/statistics.py
    -> the tables, plots and values shown in the notebook
```

Sections of the notebook: environment and root detection, modes, result inventory, raw records with duplicate and
missing-run checks, the main comparison, every result table, every plot with the numbers behind it, the complete
value verification, the attacker-subset analysis, the checkpoint-derived measurements, the statistics, eight
ablations, the baselines, a verification map and a summary.

## What is reproduced

| Result | Source records |
|---|---|
| Complete-identification comparison of every procedure | `results/processed/final_matrix/all_results.csv` |
| Endpoint satisfaction counts, the two operational requirements | same |
| Trajectories along nested deletion sets, with paired within-seed differences | same |
| Exhaustive attacker-subset study | `results/processed/corrected_v2/attacker_subsets.csv` |
| FedEraser calibration-budget sensitivity | `results/processed/final_matrix/all_results.csv` |
| Wall-clock cost and server storage | same |
| Update-direction measurements and group-wise update energy | `results/processed/final_matrix/alignment.csv` |
| Gradient-negation ascent and recovery phases | `results/processed/corrected_v2/negation_phases.csv` |
| Fine-Pruning channel measurement | `results/processed/corrected_v2/fineprune_channels.csv` |
| FedUP baseline | `results/processed/fedup/*.csv` |
| ResNet-18 architecture check and its BatchNorm diagnostic | `results/processed/resnet18/*.csv` |
| Hyperparameter-selection pilots, validation splits only | `results/raw/pilots_*` |
| All 1938 reported numerical values | the value store, recomputed from all of the above |

Four plots are regenerated and displayed inline: the attack-success and accuracy comparison across datasets, attack
success against the removed fraction, the per-seed security-utility scatter, and the update-direction analysis.


## Command line

```bash
python3 -m pytest tests -q                    # 41 unit and regression tests
python3 scripts/v8/check_v7_integrity.py      # 1544 evidence files against their recorded SHA-256
python3 scripts/aggregate_results.py --configs configs/final/*.json --out out/matrix.csv --manifest out/manifest.json
python3 scripts/aggregate_subsets.py          # rebuild the attacker-subset record
python3 scripts/check_value_store.py <regenerated table_values.json>   # cross-platform value comparison
python3 download_datasets.py data             # only needed for training from scratch
```

## Experimental setup

Twenty FedAvg clients with full participation, four of them malicious. Each malicious client replaces half of its
local samples whose label differs from the target class with triggered copies labelled with the target class. The
trigger is a four-by-four patch one pixel from the bottom-right corner. Datasets: CIFAR-10, MNIST and GTSRB, IID and
with a Dirichlet label partition on CIFAR-10. Five seeds everywhere. Two architectures: a small CNN and a
CIFAR-adapted ResNet-18.

Procedures compared: retained-data fine-tuning, a corrected tensor-wise FedEraser, bounded gradient negation on the
identified clients' data, Fine-Pruning as a post-hoc mitigation baseline, a FedUP reimplementation, and retraining
both from the same initialization (the counterfactual reference) and from an independent one.

`docs/METHODS.md` describes each procedure, the metrics, the two operational requirements and the statistical
protocol, and documents every adaptation made to the FedUP reimplementation.

## Continuous integration

`.github/workflows/ci.yml` runs the release verification on Ubuntu with Python 3.13 and `requirements-lock.txt`: the
41 tests, the evidence integrity gate, byte-identical rebuilds of the aggregated records, byte-identical rendered
artifacts, the cross-platform value-store comparison, and the full notebook in `VERIFY_EXISTING`. It downloads no
dataset and trains nothing, and it uploads the executed notebook as a build artifact.


## Environment

Python 3.10 or newer.

**Use `requirements-lock.txt` for release verification.** It pins the exact versions behind every recorded number, and
it is what continuous integration installs. `requirements.txt` gives loose ranges for ordinary development; those
ranges are not exhaustively tested, so a mismatch there is not evidence of a scientific difference.
`environment.yml` is the conda equivalent, and `pip install -e .` works if you prefer an installed package to
`PYTHONPATH`.

The implementation is written for CPU, CUDA and Apple MPS and selects the device automatically. **The reported
experiments and the release verification ran on Apple MPS**; the CPU path is exercised by the test suite and by
continuous integration on Ubuntu, and the CUDA path is not tested here. Training on a different device may produce
different last-bit values; the verification below is designed to tolerate that without hiding it.

### Cross-platform verification

Floating-point arithmetic is not bit-identical across platforms, so the notebook separates three kinds of check.

| Check | Requirement |
|---|---|
| aggregated records rebuilt from the raw runs | byte-identical |
| source-record hashes and provenance run IDs | exact |
| the value store `results/tables/table_values.json` | identical keys and structure; numbers within 1e-12 absolute |
| rendered table fragments and numerical macro files | byte-identical, against `results/tables/generated_artifacts.sha256` |
| plotted arrays, confidence intervals, series and panel inventory | exact, against the value store and the provenance sidecars |
| PNG bytes of the plots | reported, **not gated**: renderer, fonts and image metadata differ between operating systems |

A different PNG hash, or a value differing in the sixteenth decimal, is a platform artifact and does not fail the
scientific gate. A different key set, a different rendered number or a different source-record hash does.

## Licence and citation

MIT, see `LICENSE`. If you use this code or its results, please cite the accompanying study and this repository; see
`CITATION.cff`.

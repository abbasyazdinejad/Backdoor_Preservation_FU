# Methods, metrics and protocol

Reference for the implementation. Everything here is what the code does; the notebook shows what it produces.

## 1. Federated setup

Twenty clients, full participation every round, FedAvg with sample-size weights. Local training is SGD with momentum
0.9, batch size 64, two local epochs, no weight decay, learning rate 0.01 for the small CNN. The number of rounds was
fixed from pilot convergence curves: 40 for CIFAR-10 and GTSRB, 20 for MNIST. The server stores the global model and
every client update every second round, which is what history-based unlearning needs.

Partitions are IID (shuffled with the seed, equal shards) or Dirichlet with concentration 0.5 for the non-IID study.

Architectures: a small CNN with two 3x3 convolutions (32 and 64 channels), ReLU, 2x2 max pooling, a 128-unit dense
layer and a linear classifier, about 545k parameters; and a CIFAR-adapted ResNet-18 with a 3x3 stride-1 stem, no max
pooling, four stages of two residual blocks and global average pooling, about 11.2M parameters.

The clean accuracies are modest relative to tuned classification benchmarks. These are controlled security
evaluations: normalization and augmentation are left out to keep the protocol minimal and the trigger at a fixed
pixel value.

## 2. Threat model

Clients 0 to 3 are malicious, 20% of the federation. Each replaces a fraction 0.5 of its local samples whose label
differs from the target class by triggered copies labelled with the target class, with the poisoned indices drawn once
per seed and fixed for all rounds. The trigger is a 4x4 patch of value 1.0 on all channels, one pixel from the
bottom-right corner, covering 1.56% of the input. Malicious clients otherwise follow the protocol: they do not scale
their updates, do not deviate from local SGD and do not act after training.

Identification is assumed, not evaluated. The deletion request names a subset of the malicious clients; the nested
request removes the first k of them in id order, giving removed fractions 0.25, 0.5, 0.75 and 1. The attacker-subset
study evaluates every non-empty subset instead.

Twenty clients with four attackers is a controlled design, not a claim about deployment scale.

## 3. Procedures

All start from the same compromised checkpoint of the seed.

**Retained-data fine-tuning.** Five further FedAvg rounds with the retained clients only.

**FedEraser** (`src/bpa_fu/unlearning/federaser.py`). A tensor-wise implementation of the published algorithm. The
reconstruction starts from the stored initial global model; the first stored round aggregates the retained clients'
stored updates without calibration, because that model contains no contribution of the removed clients; at every later
stored round the retained clients calibrate for one epoch from the current reconstruction, each stored update is
replaced by the calibration update rescaled tensor by tensor to the stored norm, and the rescaled updates are
aggregated with sample-size weights. The source does not state whether the norm is per tensor or over the whole
update; the public reproduction uses per tensor and so do we. Retaining interval 2 and calibration ratio 0.5, as in
the source. The removed clients' stored updates are discarded and their data is never accessed.

**Bounded gradient negation** (`src/bpa_fu/unlearning/gradient_negation.py`). A stabilized gradient-ascent baseline
adapted from the published method, differing in two respects: the server runs the ascent on the union of the
identified clients' local data including their poisoned samples, rather than a cooperative departing client on its own
data; and the projection reference is the compromised model rather than the average of the other clients' models.
Plain SGD steps of size 0.002 projected onto an L2 ball of radius 1, stopping when the mean loss on the identified
data exceeds a threshold multiplier times the log of the class count, or after five epochs, followed by the same five
recovery rounds as fine-tuning.

**Fine-Pruning** (`src/bpa_fu/unlearning/fine_pruning.py`). A post-hoc mitigation baseline with the same data access
as fine-tuning, on CIFAR-10 only. Channels of the last convolutional stage are pruned in increasing order of their
mean activation on retained-client data until retained-data accuracy drops by more than four points, then the pruned
model is fine-tuned for five rounds. For the ResNet a last-stage feature-channel adaptation masks the output channels
of the last residual stage and folds the mask into the classifier input, exactly equivalent on the forward path. That
adaptation is our specification. Fine-Pruning is not an unlearning method; it answers whether an ordinary mitigation
removes more of the backdoor than the unlearning procedures with the same access.

**FedUP** (`src/bpa_fu/unlearning/fedup.py`). **A paper-derived reimplementation**, not an exact reproduction: the
source is a preprint with no official code. The server averages the malicious and the benign last-round local models,
ranks every weight of each dense and convolutional layer by the squared difference of the two averages times the
magnitude of the previous global weight, zeros the top fraction of each layer in the benign average, and runs recovery
rounds with the retained clients. Documented adaptations:

* the magnitude weighting, the sample-size-weighted averages and the restriction to weight tensors are our reading of
  the source;
* the pruning fraction is 10%, the value the source derives for IID data; a validation-split sensitivity over 5, 10
  and 20% is included and was **not** used to select it;
* recovery uses the same five rounds and local optimizer as fine-tuning rather than the source's Adam recovery;
* because the history stores updates every second round, the last-round local models are reconstructed by replaying
  the final round; the reconstruction reproduces the compromised checkpoint bit for bit for every seed and both
  architectures, and the pipeline asserts this.

**Retraining.** The identical protocol on the retained clients from scratch, both from the same round-zero parameters,
partition and data order as the compromised run (the counterfactual reference) and from an independent initialization
(to show the seed-to-seed range of retraining). Neither is an unlearning algorithm applied to the compromised model.

## 4. Metrics

**BA**, benign accuracy on the clean test set. **ASR**, attack success rate: the fraction of test samples whose true
label differs from the target class that are classified as the target class once the trigger is applied.

**Counterfactual reference.** The same-initialization retraining. Its ASR is the reference security level, not a
bound: an evaluated procedure can fall slightly below it.

**Two operational requirements**, both declared before the results were examined and both evaluated per seed against
the counterfactual of that seed:

* *security improvement*: the ASR reduction against the compromised model is at least a declared margin, 50 percentage
  points by default, half of a near-saturated attack;
* *counterfactual proximity*: the ASR is within a declared tolerance of the counterfactual, 5 percentage points by
  default, of the order of the seed-to-seed spread of the counterfactual ASR on the harder tasks.

Sensitivity to margins of 30, 50 and 70 points and tolerances of 2, 5 and 10 points is reported.

**Backdoor-preservation ratio.** The residual backdoor as a fraction of the counterfactual-relative effect. It is
defined only when the compromised model is meaningfully more vulnerable than the counterfactual, and is left undefined
when that difference is below 5 percentage points, which excludes the counterfactual and retraining rows and any
request for which the attack was ineffective. A ratio slightly below zero means an ASR slightly below the measured
reference, which is sampling variation, not negative preservation.

**Cost.** Wall-clock time per run in seconds and as a fraction of the independently initialized retraining of the same
seed, plus the server storage each procedure requires.

## 5. Parameter-space measurements

For every run the framework records the update `u` of the procedure, the sample-size-weighted sums of the stored
updates of the removed and the retained clients, the direction to the counterfactual, the projection coefficient of
`u` onto the negated malicious proxy, the orthogonal residual fraction, the relative residual distance to the
counterfactual, and the normalized group-wise energy of the update over the parameter groups of the network. These are
descriptive: they characterize where each procedure moves the parameters, and they are not used to establish causes.

Computing them requires the stored model weights, so they cannot be recomputed without the checkpoint archive.

## 6. Statistical protocol

Five seeds everywhere, controlling the partition, the poisoned indices, the initialization and the data order. Means,
sample standard deviations and untruncated Student-t 95% confidence intervals for the mean; an interval may extend
below zero for a bounded quantity and is not truncated. Paired differences over nested deletion sets use a percentile
bootstrap over seeds with 2000 resamples. Correlations use a cluster bootstrap over experiment-seed clusters with the
same number of resamples, because runs of one seed at different removed fractions are not independent; a row-wise
bootstrap is reported alongside for comparison. Wall-clock time is measured per run.

## 7. Hyperparameter selection

Protocol parameters (the fine-tuning budget, the FedEraser calibration ratio and retaining interval, the Fine-Pruning
rule, the retraining budget) are taken from the respective sources or matched across procedures, not tuned.

Two settings were selected on pilot runs, on a validation split of 5000 training images held out before partitioning,
never on the test set: the gradient-negation threshold multiplier, by a pre-registered rule taking the largest
multiplier whose validation accuracy after recovery is within one point of fine-tuning's; and the ResNet-18 learning
rate and round budget, by a plateau rule. For the ResNet-18 negation multiplier no candidate stayed within one point
of fine-tuning, so a fallback declared before any test evaluation applied. The FedUP pruning fraction was not
selected: it is the source value, with a validation-split sensitivity reported alongside.

The pilot records ship in `results/raw/pilots_hyperparameter_selection/` and
`results/raw/pilots_fedup_sensitivity/`, and the notebook recomputes the selected values from them.

## 8. Reproducibility controls

Every run writes a record with its configuration, seed, deletion set, accuracy, attack success rate, runtime,
environment and the SHA-256 checksum of its checkpoint; compromised-model records also carry the checksum of the
stored update history. Runs that depend on a stored artifact verify its checksum before use and stop rather than
retrain silently when an artifact is missing or altered. Tables, plots and numerical values are regenerated from the
records and a consistency gate fails when any value differs.

Scope statement: an earlier 390-run matrix was reproduced from empty directories with bit-identical accuracy and
attack success rate. The corrected final 480-run matrix, the ResNet-18 check and the FedUP baseline were executed once
and are checkpoint-verified; they were not rerun completely from scratch.

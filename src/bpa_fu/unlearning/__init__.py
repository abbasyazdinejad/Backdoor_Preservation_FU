"""Unlearning methods.

All methods take the compromised global model, the list of retained client ids, the
per-client datasets and the deletion target, and return the unlearned model.

short_retraining   federated fine-tuning of theta* on retained clients only (approximate)
gradient_negation  gradient ascent on the identified (deleted) data, bounded, then federated
                   fine-tuning on retained clients (approximate; needs the deleted data)
federaser          FedEraser (Liu et al., 2021): reconstruction from stored client updates
                   with calibration training on retained clients (approximate)
full_retraining    FedAvg from scratch on retained clients (counterfactual reference)
fine_pruning       Fine-Pruning (Liu et al. 2018) adapted to FL: post-hoc mitigation baseline, same data access
                   as short_retraining (retained clients only)
fedup              FedUP (Romandini et al. 2025): pruning of the benign average by the benign/malicious last-round
                   difference, then recovery rounds (needs the last-round local models, reconstructed exactly)
"""
from .short_retraining import short_retraining
from .gradient_negation import gradient_negation
from .federaser import federaser
from .full_retraining import full_retraining, cf_sameinit
from .fine_pruning import fine_pruning
from .fedup import fedup

METHODS = {
    "short_retrain": short_retraining,
    "grad_negation": gradient_negation,
    "federaser": federaser,
    "full_retrain": full_retraining,
    "fine_pruning": fine_pruning,
    "cf_sameinit": cf_sameinit,
    "fedup": fedup,
}

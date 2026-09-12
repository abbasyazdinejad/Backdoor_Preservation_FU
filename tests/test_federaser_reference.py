"""Deterministic regression test: federaser_reconstruct against a transparent toy implementation of Algorithm 1
(Liu et al. 2021, Eqs. 1-3, first retained round uncalibrated)."""
import torch, pytest
from bpa_fu.federated_training import UpdateHistory
from bpa_fu.unlearning.federaser import federaser_reconstruct, calibrate_update


def _toy_history(seed=0, n_clients=4, n_rounds=3):
    g = torch.Generator().manual_seed(seed)
    keys = {"w": (3, 2), "b": (3,)}
    globals_, updates, sizes = [], [], []
    for r in range(n_rounds):
        globals_.append({k: torch.randn(*s, generator=g) for k, s in keys.items()})
        updates.append({c: {k: torch.randn(*s, generator=g) for k, s in keys.items()} for c in range(n_clients)})
        sizes.append({c: 10 * (c + 1) for c in range(n_clients)})
    return UpdateHistory(rounds=list(range(0, 2 * n_rounds, 2)), global_states=globals_, client_updates=updates, client_sizes=sizes)


def _calib_fn(theta, c):
    """Deterministic stand-in for calibration training: a fixed function of the current model and client id."""
    return {k: -0.1 * v + 0.05 * (c + 1) for k, v in theta.items()}


def _reference(history, retained, calib_fn):
    """Literal transcription of Algorithm 1: per client, per tensor calibration; FedAvg weights over retained clients."""
    theta = {k: v.clone() for k, v in history.global_states[0].items()}
    for j in range(len(history.rounds)):
        stored, sizes = history.client_updates[j], history.client_sizes[j]
        tot = sum(sizes[c] for c in retained)
        agg = {k: torch.zeros_like(v) for k, v in theta.items()}
        for c in retained:
            w = sizes[c] / tot
            if j == 0:
                u_tilde = stored[c]                       # first reconstruction epoch: no calibration
            else:
                u_hat = calib_fn(theta, c)
                u_tilde = {k: stored[c][k].norm() * u_hat[k] / u_hat[k].norm() for k in theta}   # Eq. (1), tensor-wise
            for k in theta:
                agg[k] += w * u_tilde[k]                  # Eq. (2)
        theta = {k: theta[k] + agg[k] for k in theta}    # Eq. (3)
    return theta


def test_reconstruction_matches_reference():
    h = _toy_history(); retained = [1, 2, 3]
    out = federaser_reconstruct(h, retained, _calib_fn)
    ref = _reference(h, retained, _calib_fn)
    for k in ref:
        assert torch.allclose(out[k], ref[k], atol=1e-6), k


def test_first_round_uncalibrated_and_deleted_client_ignored():
    h = _toy_history(); retained = [1, 2, 3]
    out = federaser_reconstruct(h, retained, _calib_fn)
    # after the first (uncalibrated) round the model must equal theta0 + weighted retained stored updates only
    h1 = UpdateHistory(rounds=[0], global_states=h.global_states[:1], client_updates=h.client_updates[:1], client_sizes=h.client_sizes[:1])
    first = federaser_reconstruct(h1, retained, _calib_fn)
    tot = sum(h.client_sizes[0][c] for c in retained)
    for k in first:
        expected = h.global_states[0][k] + sum(h.client_sizes[0][c] / tot * h.client_updates[0][c][k] for c in retained)
        assert torch.allclose(first[k], expected, atol=1e-6)
    # the deleted client's updates must not influence the result
    h_mod = _toy_history(); h_mod.client_updates[1][0]["w"] += 100.0
    out2 = federaser_reconstruct(h_mod, retained, _calib_fn)
    for k in out:
        assert torch.allclose(out[k], out2[k], atol=1e-6)


def test_calibrate_update_magnitude_and_direction():
    stored = {"a": torch.tensor([3.0, 4.0])}; calib = {"a": torch.tensor([0.0, 2.0])}
    out = calibrate_update(stored, calib)
    assert torch.allclose(out["a"], torch.tensor([0.0, 5.0]))

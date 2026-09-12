import math, pytest
from bpa_fu.metrics import backdoor_preservation_ratio, non_regression_violation, asr_reduction, security_improvement_ok, counterfactual_proximity_ok
from bpa_fu.statistics import summarize

def test_bpr_and_zero_denominator():
    assert abs(backdoor_preservation_ratio(0.9, 0.95, 0.05) - (0.85 / 0.90)) < 1e-12
    assert math.isnan(backdoor_preservation_ratio(0.5, 0.06, 0.05))  # denominator 0.01 < 0.05 -> undefined
    assert math.isnan(backdoor_preservation_ratio(0.5, 0.05, 0.05))

def test_non_regression_violation():
    # regression test for the deprecated legacy V_tau diagnostic (not used in the final manuscript)
    pts = [(0.25, 0.90), (0.5, 0.92), (0.75, 0.91), (1.0, 0.95)]
    assert abs(non_regression_violation(pts, tau=0.0) - ((0.02 + 0.04) / 3)) < 1e-12
    assert abs(non_regression_violation(pts, tau=0.03) - (0.01 / 3)) < 1e-12
    assert non_regression_violation(pts, tau=0.1) == 0.0
    assert non_regression_violation(list(reversed(pts)), tau=0.0) == non_regression_violation(pts, tau=0.0)  # order-invariant
    with pytest.raises(ValueError):
        non_regression_violation([(0.5, 0.1)], 0.0)

def test_endpoints_and_stats():
    assert asr_reduction(0.95, 0.10) == pytest.approx(0.85)
    assert security_improvement_ok(0.95, 0.10, 0.5) and not security_improvement_ok(0.95, 0.90, 0.5)
    assert counterfactual_proximity_ok(0.12, 0.10, 0.05) and not counterfactual_proximity_ok(0.30, 0.10, 0.05)
    s = summarize([0.5, 0.6, 0.7])
    assert s["n"] == 3 and abs(s["mean"] - 0.6) < 1e-12 and abs(s["std"] - 0.1) < 1e-12 and s["ci95_half"] > 0

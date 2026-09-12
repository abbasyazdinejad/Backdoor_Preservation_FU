import numpy as np, pandas as pd
from bpa_fu.latex_tables import paired_bootstrap_ci, paired_fraction_differences, spearman_ci_cluster

def test_paired_bootstrap_basic():
    m, med, lo, hi = paired_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(m - 0.3) < 1e-12 and abs(med - 0.3) < 1e-12 and lo <= m <= hi and lo > 0

def test_paired_fraction_differences_within_seed():
    rows = []
    for s in range(3):
        for a, v in ((0.25, 0.9 + 0.01 * s), (0.5, 0.8 + 0.01 * s), (1.0, 0.1 + 0.01 * s)):
            rows.append(dict(exp_id="e", method="m", seed=s, alpha=a, ASR=v))
    out = paired_fraction_differences(pd.DataFrame(rows), "e", "m")
    assert len(out) == 2 and abs(out[0]["mean"] + 0.1) < 1e-9 and abs(out[1]["mean"] + 0.7) < 1e-9 and out[0]["n"] == 3

def test_cluster_bootstrap_uses_clusters():
    rng = np.random.RandomState(0); x = rng.randn(40); y = x + 0.1 * rng.randn(40); cl = np.repeat(np.arange(8), 5)
    r, lo, hi, n = spearman_ci_cluster(x, y, cl)
    assert n == 8 and r > 0.9 and lo <= r <= hi

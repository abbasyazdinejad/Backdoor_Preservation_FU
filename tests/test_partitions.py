import numpy as np
from bpa_fu.partitions import iid_partition, dirichlet_partition, check_partition, partition_statistics

def test_iid_partition_deterministic_and_disjoint():
    labels = np.arange(1000) % 10
    a = iid_partition(labels, 20, seed=3); b = iid_partition(labels, 20, seed=3); c = iid_partition(labels, 20, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))
    check_partition(a, 1000)
    assert {len(p) for p in a} == {50}

def test_dirichlet_partition_integrity_and_skew():
    rng = np.random.RandomState(0); labels = rng.randint(0, 10, 5000)
    parts = dirichlet_partition(labels, 10, beta=0.5, seed=1)
    check_partition(parts, 5000)
    m = partition_statistics(parts, labels, 10)
    assert m.sum() == 5000
    iid = partition_statistics(iid_partition(labels, 10, 1), labels, 10)
    assert m.std(axis=0).mean() > iid.std(axis=0).mean()  # label skew larger than IID
    assert all(np.array_equal(x, y) for x, y in zip(parts, dirichlet_partition(labels, 10, beta=0.5, seed=1)))

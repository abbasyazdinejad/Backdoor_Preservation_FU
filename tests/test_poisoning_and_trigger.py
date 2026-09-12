import numpy as np, torch
from bpa_fu.triggers import PatchTrigger
from bpa_fu.poisoning import build_client_datasets, poison_fraction
from bpa_fu.partitions import iid_partition

def test_trigger_location_and_size():
    t = PatchTrigger(size=4, value=1.0, offset=1)
    x = torch.zeros(3, 32, 32); y = t.apply(x)
    assert y[:, 27:31, 27:31].eq(1.0).all() and y.sum() == 3 * 16
    assert y[:, 31, :].eq(0).all() and y[:, :, 31].eq(0).all()  # offset row/col untouched
    assert abs(t.area_fraction() - 16 / 1024) < 1e-12
    xb = torch.zeros(5, 1, 32, 32); assert t.apply(xb)[:, :, 27:31, 27:31].eq(1.0).all()
    assert t.apply_uint8(torch.zeros(1, 32, 32, dtype=torch.uint8))[0, 27:31, 27:31].eq(255).all()

def test_poison_fraction_and_target_exclusion(tiny_dataset):
    parts = iid_partition(tiny_dataset.labels.numpy(), 6, seed=0)
    ds, recs = build_client_datasets(tiny_dataset, parts, [0, 1], 0.4, 0, PatchTrigger(), seed=0)
    assert [r.client_id for r in recs] == [0, 1]
    fr = poison_fraction(recs, parts)
    for cid in (0, 1):
        assert abs(fr[cid] - 0.4) < 0.02
        pos = recs[cid].poisoned_local_positions
        assert (ds[cid].labels[pos] == 0).all()
        assert (tiny_dataset.labels[parts[cid]][pos] != 0).all()  # only non-target samples were poisoned
        assert ds[cid].images[pos][:, :, 27:31, 27:31].eq(255).all()
    assert torch.equal(ds[2].images, tiny_dataset.images[parts[2]])  # benign clients untouched
    ds2, recs2 = build_client_datasets(tiny_dataset, parts, [0, 1], 0.4, 0, PatchTrigger(), seed=0)
    assert np.array_equal(recs2[0].poisoned_global_indices, recs[0].poisoned_global_indices)

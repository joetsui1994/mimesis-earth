# python/tests/test_agglomerate.py
import numpy as np
from mimesis_earth.agglomerate import region_grow


def path_graph(n, w=1.0):
    """0-1-2-...-(n-1) line graph; unit sizes."""
    nbr = {i: [] for i in range(n)}
    for i in range(n - 1):
        nbr[i].append((i + 1, w))
        nbr[i + 1].append((i, w))
    return nbr, np.ones(n)


def test_region_grow_splits_path_in_half():
    nbr, sizes = path_graph(6)
    assign = region_grow(nbr, sizes, np.array([3.0, 3.0]), [0, 5],
                         np.random.default_rng(0))
    assert set(assign.tolist()) == {0, 1}
    assert (assign == 0).sum() == 3 and (assign == 1).sum() == 3
    # contiguous: group 0 is a prefix, group 1 a suffix
    assert assign.tolist() == [0, 0, 0, 1, 1, 1]

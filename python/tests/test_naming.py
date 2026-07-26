import numpy as np

from mimesis_earth.naming import make_namer


def test_names_unique_and_wordlike():
    namer = make_namer(np.random.default_rng(40))
    names = [namer() for _ in range(300)]
    assert len(set(names)) == 300
    for n in names:
        assert n[0].isupper()
        assert 3 <= len(n) <= 20


def test_names_deterministic():
    a = [make_namer(np.random.default_rng(41))() for _ in range(10)]
    b = [make_namer(np.random.default_rng(41))() for _ in range(10)]
    assert a == b


def test_different_seeds_differ():
    a = [make_namer(np.random.default_rng(42))() for _ in range(10)]
    b = [make_namer(np.random.default_rng(43))() for _ in range(10)]
    assert a != b

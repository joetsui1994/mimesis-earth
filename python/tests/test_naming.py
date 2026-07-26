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


def test_names_unique_at_world_scale():
    # a levels=[8,6,9] world draws ~488 names; verify headroom beyond that
    namer = make_namer(np.random.default_rng(44))
    names = [namer() for _ in range(1000)]
    assert len(set(names)) == 1000
    assert not any("-" in n for n in names)  # no fallback triggered


def test_fallback_disambiguation(monkeypatch):
    from mimesis_earth import naming

    monkeypatch.setattr(naming, "ONSETS", ["b"] * 26)
    monkeypatch.setattr(naming, "VOWELS", ["a"] * 10)
    monkeypatch.setattr(naming, "CODAS", [""] * 11)
    monkeypatch.setattr(naming, "SUFFIXES", ["a"] * 10)
    namer = naming.make_namer(np.random.default_rng(45))
    names = [namer() for _ in range(20)]
    assert len(set(names)) == 20  # fallback keeps names unique, no crash

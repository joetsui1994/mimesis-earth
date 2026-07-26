import numpy as np

from mimesis_earth.noise import sample_vmf, sphere_noise, unit_vectors


def test_unit_vectors():
    v = unit_vectors(100, np.random.default_rng(1))
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)


def test_sphere_noise_normalized_and_deterministic():
    pts = unit_vectors(2000, np.random.default_rng(2))
    a = sphere_noise(pts, np.random.default_rng(3))
    b = sphere_noise(pts, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean()) < 1e-9
    assert abs(a.std() - 1.0) < 1e-9


def test_sphere_noise_is_smooth():
    # nearby points must have similar noise values
    rng = np.random.default_rng(4)
    base = unit_vectors(500, rng)
    eps = base + rng.normal(scale=1e-4, size=base.shape)
    eps /= np.linalg.norm(eps, axis=1, keepdims=True)
    na = sphere_noise(np.vstack([base, eps]), np.random.default_rng(5))
    diff = np.abs(na[:500] - na[500:])
    assert diff.max() < 0.05


def test_vmf_concentration():
    mu = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(6)
    tight = sample_vmf(mu, kappa=200.0, n=500, rng=rng)
    loose = sample_vmf(mu, kappa=1e-9, n=500, rng=np.random.default_rng(7))
    np.testing.assert_allclose(np.linalg.norm(tight, axis=1), 1.0, atol=1e-9)
    # tight samples hug mu; loose samples cover the sphere
    assert (tight @ mu).min() > 0.8
    assert (loose @ mu).min() < -0.5


def test_vmf_accepts_non_unit_mu():
    out = sample_vmf(np.array([0.0, 0.0, 5.0]), kappa=50.0, n=100,
                     rng=np.random.default_rng(8))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)
    assert (out @ np.array([0.0, 0.0, 1.0])).min() > 0.5


def test_vmf_deterministic():
    mu = np.array([0.0, 1.0, 0.0])
    a = sample_vmf(mu, 30.0, 50, np.random.default_rng(9))
    b = sample_vmf(mu, 30.0, 50, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)

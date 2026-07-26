"""Seeded random fields on the sphere: spectral noise and vMF direction sampling."""

import numpy as np


def unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sphere_noise(
    points: np.ndarray,
    rng: np.random.Generator,
    octaves: int = 4,
    waves_per_octave: int = 6,
    base_freq: float = 3.0,
) -> np.ndarray:
    """Smooth zero-mean unit-variance noise sampled at `points` on the unit sphere.

    Sum of random plane waves; smooth by construction, fully determined by rng.
    """
    out = np.zeros(len(points))
    for octave in range(octaves):
        freq = base_freq * 2.0**octave
        amp = 0.55**octave
        dirs = unit_vectors(waves_per_octave, rng)
        phases = rng.uniform(0, 2 * np.pi, waves_per_octave)
        for u, phase in zip(dirs, phases):
            out += amp * np.cos(freq * np.pi * (points @ u) + phase)
    return (out - out.mean()) / out.std()


def sample_vmf(
    mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample n unit vectors from a von Mises-Fisher distribution around mu.

    kappa -> 0 is uniform on the sphere; large kappa concentrates near mu.
    """
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    if kappa < 1e-6:
        return unit_vectors(n, rng)
    u = rng.uniform(size=n)
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    angle = rng.uniform(0, 2 * np.pi, n)
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(mu, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    r = np.sqrt(np.clip(1.0 - w**2, 0.0, None))
    return (
        w[:, None] * mu
        + (r * np.cos(angle))[:, None] * e1
        + (r * np.sin(angle))[:, None] * e2
    )

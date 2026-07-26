"""Synthetic demographics: a spatially correlated population density field."""

import numpy as np

from mimesis_earth.mesh import Mesh


def population_density(
    mesh: Mesh, land_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Relative population density per land atom (aligned with land_idx).

    A few gaussian 'city' kernels on top of a rural base, times lognormal
    noise -> log-normal-ish unit sizes with spatial autocorrelation."""
    n_cities = min(12, len(land_idx))
    centers = land_idx[rng.integers(0, len(land_idx), size=n_cities)]
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=n_cities)
    sigmas = rng.uniform(0.04, 0.18, size=n_cities)  # radians
    density = np.full(len(land_idx), 0.05)
    pts = mesh.points[land_idx]
    for c, w, s in zip(centers, weights, sigmas):
        ang = np.arccos(np.clip(pts @ mesh.points[int(c)], -1.0, 1.0))
        density += w * np.exp(-0.5 * (ang / s) ** 2)
    density *= rng.lognormal(mean=0.0, sigma=0.4, size=len(density))
    return density


def round_preserving_sum(x: np.ndarray, total: int) -> np.ndarray:
    """Scale positive weights x to integers summing exactly to `total`
    (largest-remainder method)."""
    scaled = x * (total / x.sum())
    base = np.floor(scaled).astype(np.int64)
    remainder = int(total - base.sum())
    order = np.argsort(-(scaled - base))
    base[order[:remainder]] += 1
    return base

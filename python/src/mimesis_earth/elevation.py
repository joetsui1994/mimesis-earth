"""Per-atom elevation on the sphere: continent bases + ridged mountains.

Phase-1 terrain: land is whatever rises above sea level (landmask.py picks
the sea-level quantile); border costs follow the same field's ridges."""

import numpy as np

from mimesis_earth.mesh import Mesh
from mimesis_earth.noise import sphere_noise
from mimesis_earth.spec import WorldSpec


def ridged_noise(
    points: np.ndarray, rng: np.random.Generator,
    octaves: int = 5, base_freq: float = 2.5,
) -> np.ndarray:
    """Mountain-chain noise: sharp crests, broad valleys. Zero-mean, unit-std."""
    n = sphere_noise(points, rng, octaves=octaves, base_freq=base_freq)
    r = (1.0 - np.abs(n) / np.abs(n).max()) ** 1.7
    return (r - r.mean()) / r.std()


def build_elevation(
    mesh: Mesh, seeds: np.ndarray, spec: WorldSpec, rng: np.random.Generator
) -> np.ndarray:
    """Unitless elevation per atom. Continent bumps around the landmass seeds
    preserve the islands/spread semantics; coast_ruggedness scales relief."""
    angle = np.arccos(np.clip(mesh.points @ seeds.T, -1.0, 1.0)).min(axis=1)
    base = -angle
    base = (base - base.mean()) / base.std()
    mountains = ridged_noise(mesh.points, rng)
    detail = sphere_noise(mesh.points, rng, octaves=4, base_freq=6.0)
    return base + spec.coast_ruggedness * (0.9 * mountains + 0.5 * detail)

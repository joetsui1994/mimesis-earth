"""Classify atoms into sea and landmass groups; bridge islands within a group."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mimesis_earth.mesh import Mesh
from mimesis_earth.noise import sample_vmf, sphere_noise, unit_vectors
from mimesis_earth.spec import WorldSpec


@dataclass
class LandMask:
    land: np.ndarray  # (n,) bool
    group: np.ndarray  # (n,) int landmass index; -1 for sea
    bridges: np.ndarray  # (b, 2) atom index pairs linking islands within a group


def build_landmask(mesh: Mesh, spec: WorldSpec, rng: np.random.Generator) -> LandMask:
    # spread=0 -> tightly clustered landmass seeds; spread=1 -> uniform
    kappa = 100.0 * (1.0 - spec.spread) ** 4
    for _ in range(10):
        center = unit_vectors(1, rng)[0]
        seeds = sample_vmf(center, kappa, spec.n_landmasses, rng)
        angle = np.arccos(np.clip(mesh.points @ seeds.T, -1.0, 1.0))  # (n, K)
        nearest = angle.argmin(axis=1)
        score = -angle.min(axis=1)
        score = (score - score.mean()) / score.std()
        score = score + 1.2 * spec.coast_ruggedness * sphere_noise(mesh.points, rng)
        threshold = np.quantile(score, 1.0 - spec.land_fraction)
        land = score > threshold
        group = np.where(land, nearest, -1)
        present = np.unique(group[land])
        if len(present) == spec.n_landmasses:
            bridges = _bridge_islands(mesh, group, spec.n_landmasses)
            return LandMask(land=land, group=group, bridges=bridges)
    raise RuntimeError(
        "could not place all landmasses; try a different seed or raise land_fraction"
    )


def _bridge_islands(mesh: Mesh, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Within each landmass group, link secondary islands to the largest island so
    flood-fill partitioning can cross the (small) sea gaps inside an archipelago."""
    bridges = []
    for g in range(n_groups):
        idx = np.flatnonzero(group == g)
        if len(idx) == 0:
            continue
        sub = mesh.adjacency[idx][:, idx]
        n_comp, labels = connected_components(sub, directed=False)
        if n_comp <= 1:
            continue
        sizes = np.bincount(labels)
        main_label = int(sizes.argmax())
        main = np.flatnonzero(labels == main_label)
        tree = cKDTree(mesh.points[idx[main]])
        for c in range(n_comp):
            if c == main_label:
                continue
            comp = np.flatnonzero(labels == c)
            dist, j = tree.query(mesh.points[idx[comp]])
            k = int(dist.argmin())
            bridges.append((int(idx[comp[k]]), int(idx[main[j[k]]])))
    return np.array(bridges, dtype=int).reshape(-1, 2)

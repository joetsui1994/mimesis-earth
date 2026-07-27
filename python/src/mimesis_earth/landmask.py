"""Classify atoms into sea and landmass groups; bridge islands within a group."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mimesis_earth.elevation import build_elevation
from mimesis_earth.mesh import Mesh
from mimesis_earth.noise import sample_vmf, unit_vectors
from mimesis_earth.spec import WorldSpec


@dataclass
class LandMask:
    land: np.ndarray  # (n,) bool
    group: np.ndarray  # (n,) int landmass index; -1 for sea
    bridges: np.ndarray  # (b, 2) atom index pairs linking islands within a group
    elevation: np.ndarray  # (n,) float per-atom elevation used to build this mask


def build_landmask(mesh: Mesh, spec: WorldSpec, rng: np.random.Generator) -> LandMask:
    n = len(mesh.points)
    n_land = max(1, int(round(spec.land_fraction * n)))
    k = spec.n_landmasses
    # spread=0 -> tightly clustered landmass seeds; spread=1 -> uniform.
    # Divided by n_landmasses so the seed cluster's angular footprint grows
    # with the number of seeds; each retry relaxes concentration further.
    base_kappa = 100.0 * (1.0 - spec.spread) ** 4 / k
    # Every landmass is guaranteed a small kernel of land around its seed;
    # the remaining land budget goes to the best-scoring atoms, so noise
    # still shapes coastlines but can never erase a whole landmass.
    per_seed = max(4, min(int(0.25 * n_land / k), n_land // k))
    for attempt in range(10):
        kappa = base_kappa * 0.7**attempt
        center = unit_vectors(1, rng)[0]
        seeds = sample_vmf(center, kappa, k, rng)
        elevation = build_elevation(mesh, seeds, spec, rng)
        angle = np.arccos(np.clip(mesh.points @ seeds.T, -1.0, 1.0))  # (n, K)
        nearest = angle.argmin(axis=1)
        score = elevation
        land = np.zeros(n, dtype=bool)
        for s in range(k):
            land[np.argpartition(angle[:, s], per_seed)[:per_seed]] = True
        remaining = n_land - int(land.sum())
        if remaining > 0:
            candidates = np.flatnonzero(~land)
            order = np.argsort(-score[candidates], kind="stable")
            land[candidates[order[:remaining]]] = True
        group = np.where(land, nearest, -1)
        present = np.unique(group[land])
        if len(present) == k:
            bridges = _bridge_islands(mesh, group, k)
            return LandMask(land=land, group=group, bridges=bridges, elevation=elevation)
    raise RuntimeError(
        "could not place all landmasses; raise spread, lower n_landmasses, "
        "or raise resolution"
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

"""The generation pipeline: mesh -> land mask -> partitions -> geometry -> attributes."""

import numpy as np
import shapely
from shapely.ops import unary_union

from mimesis_earth.attributes import population_density, round_preserving_sum
from mimesis_earth.geometry import R_EARTH_KM, atoms_polygon, xyz_to_lonlat
from mimesis_earth.landmask import build_landmask
from mimesis_earth.mesh import build_mesh
from mimesis_earth.naming import make_namer
from mimesis_earth.partition import (
    allocate_counts,
    child_counts,
    partition_atoms,
    redistribute_counts,
)
from mimesis_earth.spec import WorldSpec
from mimesis_earth.world import Unit, World


def generate(spec: WorldSpec) -> World:
    rng = np.random.default_rng(spec.seed)
    mesh = build_mesh(spec.resolution, rng)
    mask = build_landmask(mesh, spec, rng)
    roughness = spec.border_roughness_per_level()
    n_levels = len(spec.levels)

    # --- partition atoms level by level ---------------------------------
    # each entry: {"atoms": ndarray, "parent": index into previous level or None,
    #              "landmass": int (level 0 only)}
    level_nodes: list[list[dict]] = []
    group_sizes = np.array(
        [(mask.group == g).sum() for g in range(spec.n_landmasses)], dtype=float
    )
    counts0 = allocate_counts(spec.levels[0], group_sizes)
    top: list[dict] = []
    for g in range(spec.n_landmasses):
        idx = np.flatnonzero(mask.group == g)
        parts = partition_atoms(
            mesh, idx, int(counts0[g]), mask.bridges, roughness[0], rng
        )
        for atoms in parts:
            top.append({"atoms": atoms, "parent": None, "landmass": g})
    level_nodes.append(top)

    for level in range(1, n_levels):
        prev = level_nodes[level - 1]
        counts = child_counts(spec.levels[level], len(prev), spec.count_variance, rng)
        capacities = np.array([len(p["atoms"]) for p in prev])
        # preserve exact per-level totals even if a parent is atom-starved
        counts = redistribute_counts(counts, capacities)
        current: list[dict] = []
        for parent_index, parent in enumerate(prev):
            k = int(counts[parent_index])
            parts = partition_atoms(
                mesh, parent["atoms"], k, mask.bridges, roughness[level], rng
            )
            for atoms in parts:
                current.append({"atoms": atoms, "parent": parent_index})
        level_nodes.append(current)

    # --- population on leaves --------------------------------------------
    land_idx = np.flatnonzero(mask.land)
    atom_density = np.zeros(len(mesh.points))
    atom_density[land_idx] = population_density(mesh, land_idx, rng)
    leaf_weights = np.array(
        [
            float((atom_density[n["atoms"]] * mesh.areas[n["atoms"]]).sum())
            for n in level_nodes[-1]
        ]
    )
    leaf_pops = round_preserving_sum(leaf_weights, spec.total_population)

    # --- attributes + geometry, bottom-up --------------------------------
    namer = make_namer(rng)
    cell_cache: dict = {}
    unit_grids: list[list[Unit]] = [[] for _ in range(n_levels)]

    # names must be drawn in a deterministic order: level by level, node order
    names = [[namer() for _ in level_nodes[lvl]] for lvl in range(n_levels)]

    # leaf geometries from atoms; parent geometry = union of children
    geoms: list[list] = [[None] * len(level_nodes[lvl]) for lvl in range(n_levels)]
    for i, node in enumerate(level_nodes[-1]):
        geoms[-1][i] = atoms_polygon(mesh, node["atoms"], cell_cache)
    for lvl in range(n_levels - 2, -1, -1):
        children_of: list[list] = [[] for _ in level_nodes[lvl]]
        for i, node in enumerate(level_nodes[lvl + 1]):
            children_of[node["parent"]].append(geoms[lvl + 1][i])
        for i, childs in enumerate(children_of):
            geoms[lvl][i] = shapely.set_precision(unary_union(childs), 1e-9)

    # populations bottom-up
    pops: list[np.ndarray] = [None] * n_levels
    pops[-1] = leaf_pops
    for lvl in range(n_levels - 2, -1, -1):
        agg = np.zeros(len(level_nodes[lvl]), dtype=np.int64)
        for i, node in enumerate(level_nodes[lvl + 1]):
            agg[node["parent"]] += pops[lvl + 1][i]
        pops[lvl] = agg

    # build Unit objects top-down so ids exist before children need them
    id_grids: list[list[str]] = [[None] * len(level_nodes[lvl]) for lvl in range(n_levels)]
    child_counter: list[dict] = [dict() for _ in range(n_levels)]

    widths = []
    for lvl in range(n_levels):
        if lvl == 0:
            max_idx = len(level_nodes[0])
        else:
            parents_arr = np.array([n["parent"] for n in level_nodes[lvl]])
            max_idx = int(np.bincount(parents_arr).max())
        widths.append(max(2, len(str(max_idx))))

    for lvl in range(n_levels):
        letter = spec.level_names[lvl][0].upper()
        for i, node in enumerate(level_nodes[lvl]):
            if lvl == 0:
                index = i + 1
                uid = f"{letter}{index:0{widths[lvl]}d}"
                parent_id = None
            else:
                parent_pos = node["parent"]
                parent_id = id_grids[lvl - 1][parent_pos]
                index = child_counter[lvl].get(parent_pos, 0) + 1
                child_counter[lvl][parent_pos] = index
                uid = f"{parent_id}.{letter}{index:0{widths[lvl]}d}"
            id_grids[lvl][i] = uid
            atoms = node["atoms"]
            weights = mesh.areas[atoms]
            center = (mesh.points[atoms] * weights[:, None]).sum(axis=0)
            center /= np.linalg.norm(center)
            lon, lat = xyz_to_lonlat(center[None, :])
            unit_grids[lvl].append(
                Unit(
                    id=uid,
                    level=lvl,
                    level_name=spec.level_names[lvl],
                    parent_id=parent_id,
                    name=names[lvl][i],
                    population=int(pops[lvl][i]),
                    area_km2=float(mesh.areas[atoms].sum() * R_EARTH_KM**2),
                    centroid_lon=float(lon[0]),
                    centroid_lat=float(lat[0]),
                    geometry=geoms[lvl][i],
                    landmass=node.get("landmass"),
                )
            )

    units = [u for grid in unit_grids for u in grid]
    return World(spec=spec, units=units)

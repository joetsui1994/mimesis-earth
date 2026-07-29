"""The generation pipeline: mesh -> land mask -> partitions -> geometry -> attributes."""

import numpy as np
import shapely

from mimesis_earth.agglomerate import partition_world
from mimesis_earth.attributes import population_density, round_preserving_sum
from mimesis_earth.geometry import (
    PRECISION_GRID,
    R_EARTH_KM,
    atoms_polygon,
    snap_union,
    xyz_to_lonlat,
)
from mimesis_earth.landmask import build_landmask
from mimesis_earth.mesh import build_mesh
from mimesis_earth.naming import make_namer
from mimesis_earth.noise import sphere_noise
from mimesis_earth.spec import WorldSpec
from mimesis_earth.world import Unit, World, _rfc7946

# --- atom_cost: shapes the LEAF (district) border texture --------------
# Exponent coefficient for the coherent border-roughness term in atom_cost. A
# larger value sharpens the noise ridges the leaf partition settles borders on
# (more texture) without unbalancing it, since the clip below caps the contrast.
BORDER_NOISE_COST = 1.5

# Bound on the |atom_cost| exponent. atom_cost shapes leaf borders (they follow
# its crests) but ALSO governs reachability in the weighted-Voronoi leaf
# partition, so an unbounded field lets one low-cost basin dominate and starve
# the rest into slivers. Clipping keeps enough contrast to wiggle leaf borders
# (ratio up to e^2 ~ 7x) while keeping the leaf partition balanced.
COST_EXPONENT_CLIP = 1.0

# Frequency/persistence of the coherent noise in atom_cost. High persistence
# gives the mid-frequency octaves weight so leaf-border ridges are dense (leaf
# borders bend at fine scale rather than running straight). These shape the
# district texture; the macro wander of country/province borders comes instead
# from the region-grower's ridge bias (GROW_FIELD_FREQ below).
BORDER_NOISE_FREQ = 9.0
BORDER_NOISE_OCTAVES = 5
BORDER_NOISE_PERSISTENCE = 0.82

# --- grow_field: biases region-grow so COUNTRY/PROVINCE borders wander ---
# Base frequency of the low-frequency macro ridge field the region-grower
# follows when agglomerating districts->provinces->countries.
GROW_FIELD_FREQ = 4.0


def generate(spec: WorldSpec, _capture: dict | None = None) -> World:
    rng = np.random.default_rng(spec.seed)
    mesh = build_mesh(spec.resolution, rng)
    mask = build_landmask(mesh, spec, rng)
    # --- cost fields ------------------------------------------------------
    land_elevation = mask.elevation[mask.land]
    elev_z = (mask.elevation - land_elevation.mean()) / land_elevation.std()
    # leaf-border texture (atom scale): elevation crests + coherent noise, clipped
    leaf_noise = sphere_noise(
        mesh.points, np.random.default_rng([spec.seed, 0xB0DE]),
        octaves=BORDER_NOISE_OCTAVES, base_freq=BORDER_NOISE_FREQ,
        persistence=BORDER_NOISE_PERSISTENCE,
    )
    atom_cost = np.exp(np.clip(
        1.5 * spec.border_meander * elev_z
        + BORDER_NOISE_COST * spec.border_roughness * leaf_noise,
        -COST_EXPONENT_CLIP, COST_EXPONENT_CLIP,
    ))
    # macro ridge field for region-grow bias: low frequency so borders wander at
    # country scale; elevation term makes meander propagate to higher levels.
    grow_noise = sphere_noise(
        mesh.points, np.random.default_rng([spec.seed, 0x6600]),
        octaves=3, base_freq=GROW_FIELD_FREQ, persistence=0.7,
    )
    grow_field = spec.border_meander * elev_z + spec.border_roughness * grow_noise

    # --- bottom-up partition ---------------------------------------------
    level_nodes = partition_world(mesh, mask, spec, atom_cost, grow_field, rng)
    n_levels = len(spec.levels)

    if _capture is not None:
        _capture["mesh"] = mesh
        _capture["level_nodes"] = level_nodes
        _capture["elevation"] = mask.elevation
        _capture["bridges"] = mask.bridges

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
            geoms[lvl][i] = shapely.set_precision(snap_union(childs), PRECISION_GRID)

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

    # per-unit elevation in meters: nominal sea-level contour scaled so the
    # highest atom lands near 4500m; kernel-forced low atoms clamp to -100m.
    sea_level = np.quantile(mask.elevation, 1.0 - spec.land_fraction)
    scale = 4500.0 / max(mask.elevation.max() - sea_level, 1e-9)

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
            weighted_mean_elevation = float(
                (mask.elevation[atoms] * weights).sum() / weights.sum()
            )
            elevation_m = max(
                -100, int(round(scale * (weighted_mean_elevation - sea_level)))
            )
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
                    geometry=_rfc7946(geoms[lvl][i]),
                    landmass=node.get("landmass"),
                    elevation_m=elevation_m,
                )
            )

    units = [u for grid in unit_grids for u in grid]
    return World(spec=spec, units=units)

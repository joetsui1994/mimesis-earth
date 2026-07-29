"""The generation pipeline: mesh -> land mask -> partitions -> geometry -> attributes."""

import numpy as np
import shapely

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
from mimesis_earth.partition import (
    _island_analysis,
    allocate_counts,
    coupled_counts,
    honor_minimums,
    partition_atoms,
    plan_islands,
    redistribute_counts,
)
from mimesis_earth.spec import WorldSpec
from mimesis_earth.world import Unit, World, _rfc7946

# Exponent coefficient for the coherent border-roughness cost field. With the
# clip below, a larger value sharpens the noise ridges (borders follow them
# more decisively -> more wiggle) without unbalancing the partition, since the
# clip caps contrast regardless. 1.5 gives a clearly visible country-level
# effect (~+17% border tortuosity at max roughness) while keeping province
# balance strong; the previously-used per-edge noise was near-inert at that
# scale (long-path averaging + Lloyd re-smoothing washed it out).
BORDER_NOISE_COST = 1.5

# Bound on the |atom_cost| exponent. The cost field shapes borders (they
# follow its crests) but ALSO governs reachability in the weighted-Voronoi
# partition, so an unbounded field lets one low-cost basin dominate and
# starves the rest into slivers (and collapses a mainland into one giant
# province). Clipping keeps enough contrast to wiggle borders (ratio up to
# e^2 ~ 7x) while keeping partitions balanced.
COST_EXPONENT_CLIP = 1.0

# Coherent border field shape. base_freq puts the dominant wavelength near
# country-border scale (~20deg). High persistence gives the mid-frequency
# octaves more weight so ridges are DENSE -- without it, a border between two
# countries separated by a flat patch of the field gets a near-straight
# bisector, leaving "quite a few straight pairs"; dense ridges make nearly
# every border cross one and bend (straight-pair fraction ~18% -> ~5%).
BORDER_NOISE_FREQ = 9.0
BORDER_NOISE_OCTAVES = 5
BORDER_NOISE_PERSISTENCE = 0.82


def generate(spec: WorldSpec, _capture: dict | None = None) -> World:
    rng = np.random.default_rng(spec.seed)
    mesh = build_mesh(spec.resolution, rng)
    mask = build_landmask(mesh, spec, rng)
    # cost field: borders settle on cost crests (watersheds). Two additive
    # contributions in the exponent, both multiplicative on edge weight:
    #  - meander follows elevation crests, using the same field that defines
    #    land/sea via landmask.py's sea-level quantile. z-score against
    #    land-only elevation stats: atom_cost is only ever read on
    #    land-restricted subgraphs (partition_atoms is always called with a
    #    land/landmass atom subset), so normalizing against the full mesh --
    #    which mixes in the sea's much lower, differently-distributed
    #    elevation -- dilutes the std and washes out the contrast among land
    #    atoms where it actually matters.
    #  - roughness follows a coherent (spatially-correlated) random field.
    #    This is deliberately NOT the per-edge i.i.d. noise in _subgraph:
    #    that noise averages out over the long dijkstra paths that define
    #    coarse units and is then re-smoothed by Lloyd, so it barely wiggles
    #    country-scale borders however high it is set. A low-frequency
    #    coherent field survives both, giving roughness real, scale-
    #    independent effect. roughness is per-level, so the field is scaled
    #    per level below.
    land_elevation = mask.elevation[mask.land]
    elev_z = (mask.elevation - land_elevation.mean()) / land_elevation.std()
    # draw the coherent border field from an INDEPENDENT stream so it never
    # perturbs the main rng sequence (seeds, variance, population, names).
    # Consequence: border_roughness=0 reproduces the pre-field output exactly.
    # base_freq is raised above the sphere_noise default so the dominant
    # wavelength (~1/base_freq of the sphere) lands near country-border scale
    # -- at the default it's a ~60deg lobe, larger than a country, so country
    # borders sit inside one smooth lobe and stay near-straight.
    border_field = sphere_noise(
        mesh.points, np.random.default_rng([spec.seed, 0xB0DE]),
        octaves=BORDER_NOISE_OCTAVES, base_freq=BORDER_NOISE_FREQ,
        persistence=BORDER_NOISE_PERSISTENCE,
    )
    roughness = spec.border_roughness_per_level()

    def atom_cost_for(level: int) -> np.ndarray:
        expo = (
            1.5 * spec.border_meander * elev_z
            + BORDER_NOISE_COST * roughness[level] * border_field
        )
        return np.exp(np.clip(expo, -COST_EXPONENT_CLIP, COST_EXPONENT_CLIP))

    n_levels = len(spec.levels)

    # --- partition atoms level by level ---------------------------------
    # each entry: {"atoms": ndarray, "parent": index into previous level or None,
    #              "landmass": int (level 0 only)}
    level_nodes: list[list[dict]] = []
    group_sizes = np.array(
        [(mask.group == g).sum() for g in range(spec.n_landmasses)], dtype=float
    )
    counts0 = redistribute_counts(
        allocate_counts(spec.levels[0], group_sizes**spec.count_coupling),
        group_sizes.astype(int),
    )
    top: list[dict] = []
    for g in range(spec.n_landmasses):
        idx = np.flatnonzero(mask.group == g)
        parts = partition_atoms(
            mesh, idx, int(counts0[g]), mask.bridges, roughness[0], rng,
            size_variance=spec.size_variance, atom_cost=atom_cost_for(0),
        )
        for atoms in parts:
            top.append({"atoms": atoms, "parent": None, "landmass": g})
    level_nodes.append(top)

    counts_by_level: list = [None] * n_levels
    for level in range(1, n_levels):
        prev = level_nodes[level - 1]
        parent_sizes = np.array([len(p["atoms"]) for p in prev], dtype=float)
        level_total = spec.levels[level] * len(prev)
        counts = coupled_counts(
            level_total, parent_sizes, spec.count_coupling,
            spec.count_variance, rng,
        )
        capacities = parent_sizes.astype(int)
        counts = redistribute_counts(counts, capacities)
        # island-rich parents need enough children for one per island where
        # the level total allows; shortfalls degrade to island clustering.
        # Run the island analysis once per parent and reuse it below for the
        # actual split, instead of recomputing it inside plan_islands.
        analyses = [_island_analysis(mesh, p["atoms"], rng) for p in prev]
        minimums = np.array(
            [min(len(a[3]), int(capacities[i])) for i, a in enumerate(analyses)]
        )
        counts = honor_minimums(counts, minimums)
        assert counts.sum() == level_total
        assert (counts >= 1).all() and (counts <= capacities).all()
        counts_by_level[level] = counts
        current: list[dict] = []
        for parent_index, parent in enumerate(prev):
            k = int(counts[parent_index])
            for group_atoms, group_k in plan_islands(
                mesh, parent["atoms"], k, spec.count_coupling, rng,
                analysis=analyses[parent_index],
            ):
                parts = partition_atoms(
                    mesh, group_atoms, group_k, None, roughness[level], rng,
                    size_variance=spec.size_variance, atom_cost=atom_cost_for(level),
                )
                for atoms in parts:
                    current.append({"atoms": atoms, "parent": parent_index})
        level_nodes.append(current)

    if _capture is not None:
        _capture["mesh"] = mesh
        _capture["level_nodes"] = level_nodes
        _capture["counts_by_level"] = counts_by_level
        _capture["elevation"] = mask.elevation

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

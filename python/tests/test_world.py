import json

import numpy as np
import pytest
from shapely.ops import unary_union

from mimesis_earth import World, WorldSpec, generate

# small, fast spec used across tests
SPEC = WorldSpec(
    levels=[4, 3, 3],
    n_landmasses=2,
    resolution=6000,
    count_variance=0.0,
    seed=7,
)


@pytest.fixture(scope="module")
def world() -> World:
    return generate(SPEC)


def test_unit_counts_exact_when_variance_zero(world):
    assert len(world.units_at(0)) == 4
    assert len(world.units_at(1)) == 4 * 3
    assert len(world.units_at(2)) == 4 * 3 * 3


def test_ids_and_parents(world):
    for level in range(1, 3):
        parent_ids = {u.id for u in world.units_at(level - 1)}
        for u in world.units_at(level):
            assert u.parent_id in parent_ids
            assert u.id.startswith(u.parent_id + ".")
    for u in world.units_at(0):
        assert u.parent_id is None
    all_ids = [u.id for u in world.units]
    assert len(set(all_ids)) == len(all_ids)


def test_landmass_count(world):
    landmasses = {u.landmass for u in world.units_at(0)}
    assert landmasses == {0, 1}


def test_geometries_valid_and_in_range(world):
    for u in world.units:
        assert u.geometry.is_valid, u.id
        assert not u.geometry.is_empty, u.id
        minx, miny, maxx, maxy = u.geometry.bounds
        assert -180.0001 <= minx <= maxx <= 180.0001
        assert -90.0001 <= miny <= maxy <= 90.0001


def test_children_tile_parent_exactly(world):
    for level in range(1, 3):
        for parent in world.units_at(level - 1):
            children = [u for u in world.units_at(level) if u.parent_id == parent.id]
            merged = unary_union([c.geometry for c in children])
            assert parent.geometry.symmetric_difference(merged).area < 1e-9


def test_siblings_do_not_overlap(world):
    districts = world.units_at(2)
    by_parent: dict = {}
    for u in districts:
        by_parent.setdefault(u.parent_id, []).append(u)
    for sibs in by_parent.values():
        for i in range(len(sibs)):
            for j in range(i + 1, len(sibs)):
                inter = sibs[i].geometry.intersection(sibs[j].geometry)
                assert inter.area < 1e-9


def test_population_sums(world):
    assert sum(u.population for u in world.units_at(2)) == SPEC.total_population
    for level in range(1, 3):
        for parent in world.units_at(level - 1):
            child_sum = sum(
                u.population for u in world.units_at(level) if u.parent_id == parent.id
            )
            assert child_sum == parent.population


def test_areas_positive_and_consistent(world):
    for u in world.units:
        assert u.area_km2 > 0
    for parent in world.units_at(0):
        child_area = sum(
            u.area_km2 for u in world.units_at(1) if u.parent_id == parent.id
        )
        np.testing.assert_allclose(child_area, parent.area_km2, rtol=1e-6)


def test_deterministic_and_seed_sensitive():
    w1 = generate(SPEC)
    w2 = generate(SPEC)
    j1 = json.dumps(w1.geojson_dict(2), sort_keys=True)
    j2 = json.dumps(w2.geojson_dict(2), sort_keys=True)
    assert j1 == j2
    w3 = generate(SPEC.model_copy(update={"seed": 8}))
    assert json.dumps(w3.geojson_dict(2), sort_keys=True) != j1


def test_geojson_dict_structure(world):
    fc = world.geojson_dict(0)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 4
    f = fc["features"][0]
    props = f["properties"]
    for key in (
        "id", "name", "level", "level_name", "parent_id",
        "population", "area_km2", "centroid_lon", "centroid_lat",
    ):
        assert key in props
    assert f["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_exports(tmp_path, world):
    out = tmp_path / "w"
    world.to_geojson(out)
    files = sorted(p.name for p in out.iterdir())
    assert files == [
        "level0_country.geojson",
        "level1_province.geojson",
        "level2_district.geojson",
        "spec.json",
    ]
    loaded = json.loads((out / "level2_district.geojson").read_text())
    assert len(loaded["features"]) == 36
    world.to_csv(tmp_path / "units.csv")
    lines = (tmp_path / "units.csv").read_text().strip().split("\n")
    assert len(lines) == 1 + len(world.units)
    assert lines[0].startswith("id,level,level_name,parent_id,name,population")


def test_gdf_roundtrip(tmp_path, world):
    geopandas = pytest.importorskip("geopandas")
    gdf = world.gdf(level=2)
    assert len(gdf) == 36
    assert gdf.crs.to_epsg() == 4326
    path = tmp_path / "districts.gpkg"
    gdf.to_file(path)
    back = geopandas.read_file(path)
    assert len(back) == 36


def test_gdf_export_winding(tmp_path, world):
    geopandas = pytest.importorskip("geopandas")
    path = tmp_path / "gdf_export.geojson"
    world.gdf(level=0).to_file(path, driver="GeoJSON")
    back = geopandas.read_file(path)
    for g in back.geometry:
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for p in polys:
            assert p.exterior.is_ccw


def test_exported_geojson_valid_and_consistent(tmp_path, world):
    from shapely.geometry import shape

    out = tmp_path / "wexp"
    world.to_geojson(out)
    geoms_by_level = {}
    for level, name in enumerate(world.spec.level_names):
        fc = json.loads((out / f"level{level}_{name}.geojson").read_text())
        geoms = {}
        for f in fc["features"]:
            g = shape(f["geometry"])
            assert g.is_valid, f["properties"]["id"]
            assert not g.is_empty
            geoms[f["properties"]["id"]] = g
        geoms_by_level[level] = geoms
    # invariants must hold on the exported bytes, not just in memory
    for level in range(1, len(world.spec.level_names)):
        children_by_parent: dict = {}
        for u in world.units_at(level):
            children_by_parent.setdefault(u.parent_id, []).append(
                geoms_by_level[level][u.id]
            )
        for pid, childs in children_by_parent.items():
            merged = unary_union(childs)
            diff = geoms_by_level[level - 1][pid].symmetric_difference(merged)
            assert diff.area < 1e-9, pid


def test_exported_ring_winding(tmp_path, world):
    from shapely.geometry import shape

    out = tmp_path / "wwind"
    world.to_geojson(out)
    for level, name in enumerate(world.spec.level_names):
        fc = json.loads((out / f"level{level}_{name}.geojson").read_text())
        for f in fc["features"]:
            g = shape(f["geometry"])
            polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
            for p in polys:
                assert p.exterior.is_ccw, f["properties"]["id"]
                for hole in p.interiors:
                    assert not hole.is_ccw, f["properties"]["id"]


def test_invariants_across_spec_shapes():
    specs = [
        WorldSpec(levels=[12], n_landmasses=4, resolution=6000, seed=3),
        WorldSpec(levels=[6, 5], count_variance=0.5, resolution=6000, seed=4),
        WorldSpec(
            levels=[4, 3, 2], n_landmasses=1, spread=1.0,
            coast_ruggedness=1.0, resolution=6000, seed=5,
        ),
    ]
    for spec in specs:
        w = generate(spec)
        for u in w.units:
            assert u.geometry.is_valid and not u.geometry.is_empty, (spec.seed, u.id)
        if spec.count_variance == 0:
            expected = 1
            for level, c in enumerate(spec.levels):
                expected *= c
                assert len(w.units_at(level)) == expected, (spec.seed, level)
        leaves = w.units_at(len(spec.levels) - 1)
        assert sum(u.population for u in leaves) == spec.total_population
        assert len({u.landmass for u in w.units_at(0)}) == spec.n_landmasses


def test_low_level_units_contiguous():
    from scipy.sparse.csgraph import connected_components

    from mimesis_earth.partition import ISLET_MAX_ATOMS, count_sizeable_islands

    specs = [
        WorldSpec(levels=[4, 4, 3], n_landmasses=3, coast_ruggedness=0.8,
                  resolution=8000, seed=21),
        # archipelago-heavy: actually starves quota (4 quota-starved parents
        # at seed 7, measured), so the exemption branch below is exercised
        WorldSpec(levels=[6, 2, 2], n_landmasses=6, coast_ruggedness=1.0,
                  spread=1.0, resolution=10000, land_fraction=0.12, seed=7),
    ]
    for spec in specs:
        cap: dict = {}
        generate(spec, _capture=cap)
        mesh, level_nodes = cap["mesh"], cap["level_nodes"]
        counts_by_level = cap["counts_by_level"]
        probe_rng = np.random.default_rng(0)
        for level in range(1, len(spec.levels)):
            counts = counts_by_level[level]
            for node in level_nodes[level]:
                parent = level_nodes[level - 1][node["parent"]]
                m = count_sizeable_islands(mesh, parent["atoms"], probe_rng)
                if m > int(counts[node["parent"]]):
                    continue  # sanctioned: quota-starved archipelago parent
                atoms = node["atoms"]
                sub = mesh.adjacency[atoms][:, atoms]
                n_comp, comp = connected_components(sub, directed=False)
                sizes = np.bincount(comp)
                assert (sizes >= ISLET_MAX_ATOMS).sum() <= 1, (
                    spec.seed, level, len(atoms), sorted(sizes.tolist()),
                )


def test_exact_totals_at_any_variance_and_coupled_counts_vary():
    spec = WorldSpec(
        levels=[5, 4, 4], count_variance=0.8, count_coupling=1.0,
        resolution=8000, seed=23,
    )
    world = generate(spec)
    assert len(world.units_at(0)) == 5
    assert len(world.units_at(1)) == 20
    assert len(world.units_at(2)) == 80
    per_parent: dict = {}
    for u in world.units_at(1):
        per_parent[u.parent_id] = per_parent.get(u.parent_id, 0) + 1
    assert len(set(per_parent.values())) > 1


def test_border_meander_changes_borders_only_when_on():
    base = WorldSpec(levels=[4, 3], n_landmasses=2, resolution=6000, seed=31)
    w_on = generate(base)
    w_on2 = generate(base)
    w_off = generate(base.model_copy(update={"border_meander": 0.0}))
    j_on = json.dumps(w_on.geojson_dict(1), sort_keys=True)
    assert j_on == json.dumps(w_on2.geojson_dict(1), sort_keys=True)
    assert j_on != json.dumps(w_off.geojson_dict(1), sort_keys=True)


def test_elevation_exported_and_coherent(tmp_path):
    spec = WorldSpec(levels=[4, 3], n_landmasses=2, coast_ruggedness=0.8,
                     border_meander=1.0, resolution=8000, seed=41)
    cap: dict = {}
    world = generate(spec, _capture=cap)
    for u in world.units:
        assert isinstance(u.elevation_m, int)
        assert -100 <= u.elevation_m <= 5000
    assert max(u.elevation_m for u in world.units_at(1)) > 300
    out = tmp_path / "elev"
    world.to_geojson(out)
    fc = json.loads((out / "level0_country.geojson").read_text())
    assert all("elevation_m" in f["properties"] for f in fc["features"])
    world.to_csv(tmp_path / "units.csv")
    assert "elevation_m" in (tmp_path / "units.csv").read_text().splitlines()[0]
    # coherence: with meander=1, country borders sit on high ground
    mesh, nodes = cap["mesh"], cap["level_nodes"]
    elevation = cap["elevation"]
    label = np.full(len(mesh.points), -1)
    for i, node in enumerate(nodes[0]):
        label[node["atoms"]] = i
    e = mesh.edges
    border = (label[e[:, 0]] >= 0) & (label[e[:, 1]] >= 0) & (
        label[e[:, 0]] != label[e[:, 1]]
    )
    border_atoms = np.unique(np.concatenate([e[border, 0], e[border, 1]]))
    land_atoms = np.flatnonzero(label >= 0)
    assert elevation[border_atoms].mean() > elevation[land_atoms].mean() + 0.2

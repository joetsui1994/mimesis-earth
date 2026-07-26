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

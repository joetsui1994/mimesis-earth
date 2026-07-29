import math

import pytest
from pydantic import ValidationError

from mimesis_earth.spec import GENERATOR_VERSION, WorldSpec


def test_defaults_are_valid():
    spec = WorldSpec()
    assert spec.levels == [6, 5, 6]
    assert spec.level_names == ["country", "province", "district"]
    assert spec.seed == 0
    assert spec.generator_version == GENERATOR_VERSION


def test_level_names_default_matches_levels_length():
    spec = WorldSpec(levels=[4, 3])
    assert spec.level_names == ["country", "province"]


def test_rejects_bad_spread():
    with pytest.raises(ValidationError):
        WorldSpec(spread=1.5)


def test_rejects_fewer_countries_than_landmasses():
    with pytest.raises(ValidationError, match="n_landmasses"):
        WorldSpec(levels=[2, 3], n_landmasses=5)


def test_rejects_resolution_too_low_for_leaf_count():
    with pytest.raises(ValidationError, match="resolution"):
        WorldSpec(levels=[20, 20, 20], resolution=2000)


def test_rejects_mismatched_border_roughness_list():
    with pytest.raises(ValidationError, match="border_roughness"):
        WorldSpec(levels=[4, 4], border_roughness=[0.1, 0.2, 0.3])


def test_rejects_mismatched_level_names():
    with pytest.raises(ValidationError, match="level_names"):
        WorldSpec(levels=[4, 4], level_names=["only-one"])


def test_json_roundtrip():
    spec = WorldSpec(levels=[8, 6, 9], seed=42, spread=0.6)
    spec2 = WorldSpec.model_validate_json(spec.model_dump_json())
    assert spec2 == spec


def test_leaf_count_helper():
    assert WorldSpec(levels=[8, 6, 9]).max_leaf_count() == math.prod([8, 6, 9])


def test_rejects_zero_level_count():
    with pytest.raises(ValidationError, match="levels"):
        WorldSpec(levels=[0, 3])


def test_rejects_negative_border_roughness():
    with pytest.raises(ValidationError, match="border_roughness"):
        WorldSpec(border_roughness=-0.5)


def test_rejects_nan_border_roughness():
    with pytest.raises(ValidationError, match="border_roughness"):
        WorldSpec(border_roughness=float("nan"))


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        WorldSpec(bogus_field=1)


def test_new_realism_fields_defaults():
    spec = WorldSpec()
    assert spec.size_variance == 0.4
    assert spec.generator_version == "0.6.0"


def test_rejects_bad_size_variance():
    with pytest.raises(ValidationError):
        WorldSpec(size_variance=2.5)
    # extreme-but-legal skew is accepted
    assert WorldSpec(size_variance=2.0).size_variance == 2.0


def test_border_meander_field():
    spec = WorldSpec()
    assert spec.border_meander == 0.8
    assert spec.generator_version == "0.6.0"
    with pytest.raises(ValidationError):
        WorldSpec(border_meander=1.5)
    assert WorldSpec(border_meander=0.0).border_meander == 0.0


def test_spec_rejects_removed_and_list_fields():
    with pytest.raises(Exception):
        WorldSpec(count_coupling=0.5)          # removed field, extra=forbid
    with pytest.raises(Exception):
        WorldSpec(count_variance=0.5)          # removed field
    with pytest.raises(Exception):
        WorldSpec(border_roughness=[0.2, 0.5]) # now scalar-only


def test_spec_border_roughness_scalar_ok():
    s = WorldSpec(border_roughness=0.9)
    assert s.border_roughness == 0.9

"""World specification: the full parameter set that (with a seed) defines a world."""

import math
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

GENERATOR_VERSION = "0.4.0"

_DEFAULT_LEVEL_NAMES = ["country", "province", "district", "ward", "block"]

# Each leaf unit needs at least this many atoms to have a drawable polygon.
MIN_ATOMS_PER_LEAF = 8


class WorldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    levels: list[int] = Field(default=[6, 5, 6], min_length=1, max_length=5)
    level_names: Optional[list[str]] = None
    n_landmasses: int = Field(default=3, ge=1, le=64)
    spread: float = Field(default=0.7, ge=0.0, le=1.0)
    land_fraction: float = Field(default=0.3, gt=0.0, lt=0.8)
    coast_ruggedness: float = Field(default=0.5, ge=0.0, le=1.0)
    border_roughness: Union[float, list[float]] = 0.7
    count_variance: float = Field(default=0.2, ge=0.0, le=2.0)
    size_variance: float = Field(default=0.4, ge=0.0, le=2.0)
    count_coupling: float = Field(default=0.85, ge=0.0, le=1.0)
    border_meander: float = Field(default=0.8, ge=0.0, le=1.0)
    total_population: int = Field(default=50_000_000, gt=0)
    resolution: int = Field(default=20_000, ge=2_000, le=200_000)
    seed: int = 0
    generator_version: str = GENERATOR_VERSION

    def max_leaf_count(self) -> int:
        return math.prod(self.levels)

    def border_roughness_per_level(self) -> list[float]:
        if isinstance(self.border_roughness, list):
            return self.border_roughness
        return [self.border_roughness] * len(self.levels)

    @model_validator(mode="after")
    def _validate(self) -> "WorldSpec":
        if any(c < 1 for c in self.levels):
            raise ValueError(f"every entry in levels must be >= 1, got {self.levels}")
        if self.level_names is None:
            self.level_names = _DEFAULT_LEVEL_NAMES[: len(self.levels)]
        if len(self.level_names) != len(self.levels):
            raise ValueError(
                f"level_names has {len(self.level_names)} entries but levels has "
                f"{len(self.levels)}"
            )
        if self.levels[0] < self.n_landmasses:
            raise ValueError(
                f"levels[0]={self.levels[0]} must be >= n_landmasses="
                f"{self.n_landmasses} (each landmass needs at least one "
                f"top-level unit); lower n_landmasses or raise levels[0]"
            )
        if isinstance(self.border_roughness, list) and len(
            self.border_roughness
        ) != len(self.levels):
            raise ValueError("border_roughness list must match levels length")
        values = (
            self.border_roughness
            if isinstance(self.border_roughness, list)
            else [self.border_roughness]
        )
        if not all(0.0 <= v <= 2.0 for v in values):
            raise ValueError("border_roughness values must be between 0.0 and 2.0")
        expected_land_atoms = self.resolution * self.land_fraction
        if self.max_leaf_count() * MIN_ATOMS_PER_LEAF > expected_land_atoms:
            need = int(self.max_leaf_count() * MIN_ATOMS_PER_LEAF / self.land_fraction)
            raise ValueError(
                f"resolution={self.resolution} is too low for "
                f"~{self.max_leaf_count()} leaf units at land_fraction="
                f"{self.land_fraction}; raise resolution to >= {need} or reduce "
                f"level counts"
            )
        return self

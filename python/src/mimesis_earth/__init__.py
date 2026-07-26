"""mimesis-earth: rapid synthetic geography generator."""

from mimesis_earth.generate import generate
from mimesis_earth.spec import WorldSpec
from mimesis_earth.world import Unit, World

__version__ = "0.1.0"

__all__ = ["WorldSpec", "World", "Unit", "generate", "__version__"]

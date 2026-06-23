"""Persisted snowdb config entities: the resource-typed envelope + root config.

Every file the snowdb persists carries one opaque, versioned ``resource``
discriminator string (e.g. ``snowtool.snowdb/v1``) as its first field: the
``/vN`` is human-facing, but the whole string is an exact-match type tag. A new
schema version is a new type with its own model and migration chain -- no
entity's version constrains another's, and there is no global snowdb version
number. (The ``TypeAdapter`` union that routes a parsed file to its model grows
as entities are added; in this first cut the only persisted entity is the root
config.)

The root config is the system's single entry point: handed one path, the snowdb
reaches everything else (datasets, the AOI index, the AOI records) by the links
it holds. ``init`` writes a conventional tree and these links into it, but
nothing afterward *requires* the tree -- the code follows the links.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field

# The conventional filename ``snowdb init`` writes the root config to, and the
# name :meth:`SnowDb.open` looks for when handed a root *directory*. This is the
# one "magic path" the system has; everything else is reached by link from it.
CONFIG_FILENAME = 'snowdb_conf.json'


class ResourceModel(BaseModel):
    """Base for every persisted snowtool entity.

    Each concrete entity pins :attr:`resource` to a ``Literal`` so a parsed file
    routes to exactly one model. The string embeds the schema version; bumping it
    is a new type, never an in-place reinterpretation of the old one.
    """

    resource: str


class RootConfig(ResourceModel):
    """The snowdb root config (``snowtool.snowdb/v1``): links + creation stamp.

    Holds the links the system follows -- the registered dataset configs, the AOI
    index, and the AOI records directory -- plus when the root was created. A
    relative link resolves against this file's own directory (a relocatable
    tree); an absolute link points at a staged-elsewhere artifact. No datasets are
    registered by default: a dataset goes live by adding its link here.
    """

    resource: Literal['snowtool.snowdb/v1'] = 'snowtool.snowdb/v1'
    created_at: datetime
    datasets: list[str] = Field(default_factory=list)
    aoi_index: str = 'aois/index.geojson'
    aoi_records: str = 'aois/records'

    @classmethod
    def create(cls: type[Self]) -> Self:
        """A fresh root config stamped with the current UTC time, no datasets."""
        return cls(created_at=datetime.now(UTC))

    @classmethod
    def load(cls: type[Self], path: Path) -> Self:
        """Parse and validate a root config file (raises if it is not one)."""
        return cls.model_validate_json(Path(path).read_text())

    def save(self: Self, path: Path) -> None:
        """Write the config as indented JSON with a trailing newline."""
        Path(path).write_text(self.model_dump_json(indent=2) + '\n')

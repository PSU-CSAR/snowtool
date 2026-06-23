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
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, TypeAdapter

from snowtool.snowdb.variables import Reducer

# The conventional filename ``snowdb init`` writes the root config to, and the
# name :meth:`SnowDb.open` looks for when handed a root *directory*. This is the
# one "magic path" the system has; everything else is reached by link from it.
CONFIG_FILENAME = 'snowdb_conf.json'

# The conventional filename a dataset's config is written to under its own
# directory (``data/<name>/dataset.json``); a registered link may point anywhere,
# but ``dataset create`` stages here.
DATASET_CONFIG_FILENAME = 'dataset.json'


class ResourceModel(BaseModel):
    """Base for every persisted snowtool entity.

    Each concrete entity pins :attr:`resource` to a ``Literal`` so a parsed file
    routes to exactly one model. The string embeds the schema version; bumping it
    is a new type, never an in-place reinterpretation of the old one.
    """

    resource: str


class PathDatasetLink(BaseModel):
    """A dataset registration that points at a config file on the filesystem.

    ``path`` is relative to the root config's own directory (a relocatable tree)
    or absolute (a staged-elsewhere dataset). The ``type`` tag discriminates the
    link kind: today only ``path``, but the object form leaves room for other
    registration kinds (e.g. a remote or inline dataset) without reshaping the
    root config.
    """

    type: Literal['path'] = 'path'
    path: str


# A registered dataset link, discriminated on ``type`` so new link kinds slot in
# as additional union members.
DatasetLink = Annotated[PathDatasetLink, Field(discriminator='type')]


class RootConfig(ResourceModel):
    """The snowdb root config (``snowtool.snowdb/v1``): links + creation stamp.

    Holds the links the system follows -- the registered datasets (a map of
    dataset name to its :class:`DatasetLink`), the AOI index, and the AOI records
    directory -- plus when the root was created. No datasets are registered by
    default: a dataset goes live by adding its link here.
    """

    resource: Literal['snowtool.snowdb/v1'] = 'snowtool.snowdb/v1'
    created_at: datetime
    datasets: dict[str, DatasetLink] = Field(default_factory=dict)
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


class UnitConfig(BaseModel):
    """A variable's reporting unit, serialized inline as ``{name, scale_factor}``."""

    name: str
    scale_factor: float


class VariableConfig(BaseModel):
    """One requestable variable in a dataset config (the dict key is its ``key``).

    Mirrors :class:`~snowtool.snowdb.variables.DatasetVariable` minus the key:
    how to find its files (``glob``), read them (``dtype``/``nodata``), reduce
    them (``reducer``), and report them (``unit``).
    """

    unit: UnitConfig
    reducer: Reducer
    dtype: str
    nodata: float
    glob: str


class GridConfig(BaseModel):
    """A dataset's north-up tiled grid, mirroring
    :class:`~snowtool.snowdb.spec.GridParams` (``crs`` is an EPSG int or a WKT
    string)."""

    origin_x: float
    origin_y: float
    px_size: float
    cols: int
    rows: int
    tile_size: int
    crs: int | str = 4326


class DatasetConfig(ResourceModel):
    """A self-describing dataset definition (``snowtool.dataset/v1``).

    Everything a dataset *is*, independent of where it lives on disk: its
    ``grid``, its ``variables``, the registry ``ingester`` name that turns source
    data into its COGs (``None`` for a read-only/derived dataset), the elevation
    ``band_step_ft``, and an optional served ``footprint`` (a GeoJSON geometry
    mapping in the grid CRS; omitted means the whole grid extent). There is no
    runtime "kind" and no name -- the name comes from where the config is
    registered. :meth:`~snowtool.snowdb.spec.DatasetSpec.from_config` deserializes
    one into a spec (a trivial map, no merge).

    (``band_step_ft`` is top-level here for now; it moves under a per-dataset
    ``zones`` block in a later phase.)
    """

    resource: Literal['snowtool.dataset/v1'] = 'snowtool.dataset/v1'
    grid: GridConfig
    variables: dict[str, VariableConfig]
    ingester: str | None = None
    band_step_ft: int = 1000
    footprint: dict[str, Any] | None = None

    @classmethod
    def load(cls: type[Self], path: Path) -> Self:
        """Parse and validate a dataset config file (raises if it is not one)."""
        return cls.model_validate_json(Path(path).read_text())

    def save(self: Self, path: Path) -> None:
        """Write the config as indented JSON with a trailing newline."""
        Path(path).write_text(self.model_dump_json(indent=2) + '\n')


# The discriminated union over every persisted entity: a parsed file routes to
# exactly one model by its opaque ``resource`` string. Grown as entities are
# added; used where the entity type is not known up front (each model also has its
# own ``load`` for when it is).
Entity = Annotated[RootConfig | DatasetConfig, Field(discriminator='resource')]
ENTITY_ADAPTER: TypeAdapter[RootConfig | DatasetConfig] = TypeAdapter(Entity)


def load_entity(path: Path) -> RootConfig | DatasetConfig:
    """Parse any persisted snowtool entity, routing by its ``resource`` tag."""
    return ENTITY_ADAPTER.validate_json(Path(path).read_text())

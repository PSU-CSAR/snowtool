"""Persisted snowdb config entities: the resource-typed envelope + root config.

Every file the snowdb persists carries one opaque, versioned ``resource``
discriminator string (e.g. ``snowtool.snowdb/v1``) as its first field: the
``/vN`` is human-facing, but the whole string is an exact-match type tag. A new
schema version is a new type with its own model and migration chain -- no
entity's version constrains another's, and there is no global snowdb version
number. A ``TypeAdapter`` union routes a parsed file to its model.

The root config is the system's single entry point: handed one path, the snowdb
reaches everything else (datasets, the AOI index, the AOI records) from it. A
dataset may be *referenced* by a link to its own config file or defined *inline*
in the root config; anywhere a config can reference another config file it can
also embed it, so a whole snowdb can be built programmatically with no files at
all (e.g. in tests).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter

from snowtool.snowdb.variables import Reducer

# The conventional filename ``snowdb init`` writes the root config to, and the
# name :meth:`SnowDb.open` looks for when handed a root *directory*. This is the
# one "magic path" the system has; everything else is reached from it.
CONFIG_FILENAME = 'snowdb_conf.json'

# The conventional filename a dataset's config is written to under its own
# directory (``data/<name>/dataset.json``); a referenced link may point anywhere,
# but ``dataset create`` stages here.
DATASET_CONFIG_FILENAME = 'dataset.json'


class ResourceModel(BaseModel):
    """Base for every persisted snowtool entity.

    Each concrete entity pins :attr:`resource` to a ``Literal`` so a parsed file
    routes to exactly one model. The string embeds the schema version; bumping it
    is a new type, never an in-place reinterpretation of the old one.
    """

    resource: str


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


class PathDatasetLink(BaseModel):
    """A dataset registration that *references* a config file on the filesystem.

    ``path`` is relative to the root config's own directory (a relocatable tree)
    or absolute (a staged-elsewhere dataset). The dataset's data lives beside its
    config, wherever the path points.
    """

    type: Literal['path'] = 'path'
    path: str


class InlineDatasetLink(BaseModel):
    """A dataset registration that *embeds* its config in the root config.

    The inline counterpart of :class:`PathDatasetLink`: the whole
    :class:`DatasetConfig` is carried here rather than in a separate file, so a
    snowdb can be built up entirely in memory (no dataset files on disk). An inline
    dataset's data lives at the conventional ``data/<name>/`` under the root.
    """

    type: Literal['inline'] = 'inline'
    dataset: DatasetConfig


# A registered dataset link, discriminated on ``type``: a reference to a config
# file, or an inline definition. New link kinds slot in as further union members.
DatasetLink = Annotated[
    PathDatasetLink | InlineDatasetLink,
    Field(discriminator='type'),
]


class RootConfig(ResourceModel):
    """The snowdb root config (``snowtool.snowdb/v1``): datasets + AOI locations.

    Holds the registered datasets (a map of dataset name to its
    :class:`DatasetLink` -- referenced or inline), the AOI index and records
    locations, and when the root was created. No datasets are registered by
    default: a dataset goes live by adding it here.
    """

    resource: Literal['snowtool.snowdb/v1'] = 'snowtool.snowdb/v1'
    created_at: datetime
    datasets: dict[str, DatasetLink] = Field(default_factory=dict)
    aoi_index: str = 'aois/index.geojson'
    aoi_records: str = 'aois/records'

    # Where this config lives on disk: set when loaded/saved, ``None`` when built
    # in code. It is the base every relative link resolves against, so a config
    # without it cannot resolve relative paths (only absolute links / inline
    # datasets). A private attribute, not part of the serialized config or
    # equality -- it is provenance, not content.
    _path: Path | None = PrivateAttr(default=None)

    @property
    def path(self: Self) -> Path | None:
        """The config file's path (its root is the parent); ``None`` if in-code."""
        return self._path

    @path.setter
    def path(self: Self, value: Path | None) -> None:
        self._path = Path(value) if value is not None else None

    @classmethod
    def create(cls: type[Self]) -> Self:
        """A fresh root config stamped with the current UTC time, no datasets."""
        return cls(created_at=datetime.now(UTC))

    @classmethod
    def load(cls: type[Self], path: Path) -> Self:
        """Parse a root config file and remember where it was loaded from."""
        config = cls.model_validate_json(Path(path).read_text())
        config.path = Path(path)
        return config

    def save(self: Self, path: Path) -> None:
        """Write the config as indented JSON, remembering where it was written."""
        Path(path).write_text(self.model_dump_json(indent=2) + '\n')
        self.path = Path(path)


# The discriminated union over every persisted entity: a parsed file routes to
# exactly one model by its opaque ``resource`` string. Grown as entities are
# added; used where the entity type is not known up front (each model also has its
# own ``load`` for when it is).
Entity = Annotated[RootConfig | DatasetConfig, Field(discriminator='resource')]
ENTITY_ADAPTER: TypeAdapter[RootConfig | DatasetConfig] = TypeAdapter(Entity)


def load_entity(path: Path) -> RootConfig | DatasetConfig:
    """Parse any persisted snowtool entity, routing by its ``resource`` tag."""
    return ENTITY_ADAPTER.validate_json(Path(path).read_text())

"""Persisted snowdb config entities: the resource-typed envelope + root config.

Every file the snowdb persists carries one opaque, versioned ``resource``
discriminator string (e.g. ``snowtool.snowdb/v1``) as its first field: the
``/vN`` is human-facing, but the whole string is an exact-match type tag. A new
schema version is a new type with its own model and migration chain -- no
entity's version constrains another's, and there is no global snowdb version
number. A ``TypeAdapter`` union routes a parsed file to its model.

The root config is the system's single entry point: handed one path, the snowdb
reaches everything else (datasets, the pourpoint index and records) from it. A
dataset may be *referenced* by a link to its own config file or defined *inline*
in the root config; anywhere a config can reference another config file it can
also embed it, so a whole snowdb can be built programmatically with no files at
all (e.g. in tests).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from geojson_pydantic.geometries import Geometry
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_serializer,
)

from snowtool.snowdb.atomic import atomic_write_text
from snowtool.snowdb.grid import GridParams
from snowtool.snowdb.variables import DatasetVariable

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


class ZoneLayerParams(BaseModel):
    """The default query params for one zone layer, e.g. ``{"band_step_ft": 1000}``.

    One typed field per param a built-in scheme reads (a band width, a split
    threshold); ``None`` means unset (the scheme falls back to its own default).
    ``extra='forbid'`` so a typo'd/unknown param fails at config load rather than
    being silently ignored, and unset (``None``) fields are omitted from the
    serialized JSON so a stored block stays exactly ``{"band_step_ft": 1000}``.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    band_step_ft: int | None = None
    buckets: int | None = None
    threshold_pct: float | None = None
    entropy_threshold: float | None = None

    @model_serializer
    def _serialize(self: Self) -> dict[str, object]:
        return {name: value for name, value in self if value is not None}


class DatasetConfig(ResourceModel):
    """A self-describing dataset definition (``snowtool.dataset/v1``).

    Everything a dataset *is*, independent of where it lives on disk: its
    ``grid``, its ``variables``, the registry ``ingester`` name that turns source
    data into its COGs (``None`` for a read-only/derived dataset), its ``zones``,
    and an optional served ``footprint`` (a GeoJSON geometry mapping in the grid
    CRS; omitted means the whole grid extent). There is no runtime "kind" and no
    name -- the name comes from where the config is registered.
    :meth:`~snowtool.snowdb.spec.DatasetSpec.from_config` deserializes one into a
    spec (a trivial map, no merge).

    ``zones`` maps a zone-layer provider (``terrain``, ``landcover``) to its layers
    and each layer's default query params, e.g.
    ``{"terrain": {"elevation": {"band_step_ft": 1000}},
       "landcover": {"forest_cover": {"threshold_pct": 50}}}``. A provider's
    presence enables it for the dataset; absence means no such zone layer.
    """

    resource: Literal['snowtool.dataset/v1'] = 'snowtool.dataset/v1'
    grid: GridParams
    variables: dict[str, DatasetVariable]
    ingester: str | None = None
    zones: dict[str, dict[str, ZoneLayerParams]] = Field(default_factory=dict)
    # The region this dataset actually serves, as a GeoJSON geometry in the grid
    # CRS (e.g. a MODIS block minus a never-ingested tile); omitted means the whole
    # grid extent. Modeled with geojson-pydantic; the geometry math converts it to
    # shapely once, in DatasetSpec.coverage_domain.
    footprint: Geometry | None = None
    # Where this dataset's data lives (the dir holding cogs/, aoi-rasters/, ...).
    # Absolute -> anywhere (decoupled from the config's location); relative ->
    # against the config's own dir; omitted -> the convention (beside a referenced
    # config, ``data/<name>/`` for an inline one). ponytail: a plain path, not a
    # typed source union -- add that when a second (e.g. remote) backend is real.
    data_dir: Path | None = None

    @field_validator('variables', mode='before')
    @classmethod
    def _inject_variable_keys(cls: type[Self], value: object) -> object:
        """Inject each variable's ``key`` from its map key (the on-disk value omits
        it). A value already a :class:`DatasetVariable` must agree with its key."""
        if not isinstance(value, dict):
            return value
        injected: dict[object, object] = {}
        for key, var in value.items():
            if isinstance(var, DatasetVariable):
                if var.key != key:
                    raise ValueError(
                        f'variable key {var.key!r} does not match its map key {key!r}',
                    )
                injected[key] = var
            elif isinstance(var, dict):
                injected[key] = {**var, 'key': key}
            else:
                injected[key] = var
        return injected

    @classmethod
    def load(cls: type[Self], path: Path) -> Self:
        """Parse and validate a dataset config file (raises if it is not one)."""
        return cls.model_validate_json(Path(path).read_text())

    def save(self: Self, path: Path) -> None:
        """Write the config as indented JSON with a trailing newline (atomically)."""
        atomic_write_text(Path(path), self.model_dump_json(indent=2) + '\n')


class PathDatasetLink(BaseModel):
    """A dataset registration that *references* a config file on the filesystem.

    ``path`` is relative to the root config's own directory (a relocatable tree)
    or absolute (a staged-elsewhere dataset). The dataset's data lives beside its
    config, wherever the path points. ``active`` gates *visibility to readers*
    (query/API), not existence: an inactive dataset is still registered -- resolved
    by name for management (ingest, zone generation, health checks) -- but the
    read surface skips it. Defaults ``True`` so a bare hand-written link just
    works; ``dataset create``/``add`` register inactive and ``dataset activate``
    flips the flag.
    """

    type: Literal['path'] = 'path'
    path: Path
    active: bool = True


class InlineDatasetLink(BaseModel):
    """A dataset registration that *embeds* its config in the root config.

    The inline counterpart of :class:`PathDatasetLink`: the whole
    :class:`DatasetConfig` is carried here rather than in a separate file, so a
    snowdb can be built up entirely in memory (no dataset files on disk). An inline
    dataset's data lives at the conventional ``data/<name>/`` under the root.
    ``active`` behaves as on :class:`PathDatasetLink`.
    """

    type: Literal['inline'] = 'inline'
    dataset: DatasetConfig
    active: bool = True


# A registered dataset link, discriminated on ``type``: a reference to a config
# file, or an inline definition. New link kinds slot in as further union members.
DatasetLink = Annotated[
    PathDatasetLink | InlineDatasetLink,
    Field(discriminator='type'),
]


class RootConfig(ResourceModel):
    """The snowdb root config (``snowtool.snowdb/v1``): datasets + pourpoint locations.

    Holds the registered datasets (a map of dataset name to its
    :class:`DatasetLink` -- referenced or inline), the pourpoint index and records
    locations, and when the root was created. No datasets are registered by
    default: this map is the source of truth for what datasets *exist*, and each
    link's ``active`` flag for what readers *serve*.
    """

    resource: Literal['snowtool.snowdb/v1'] = 'snowtool.snowdb/v1'
    created_at: datetime
    datasets: dict[str, DatasetLink] = Field(default_factory=dict)
    pourpoint_index: Path = Path('pourpoints/index.geojson')
    pourpoint_records: Path = Path('pourpoints/records')
    # Per-provider generation source paths (provider name -> path; absolute, or
    # relative to this config). A provider absent here uses its default source
    # (3DEP for terrain, the MRLC bundle for land cover).
    sources: dict[str, Path] = Field(default_factory=dict)

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
        atomic_write_text(Path(path), self.model_dump_json(indent=2) + '\n')
        self.path = Path(path)


# The discriminated union over every persisted entity: a parsed file routes to
# exactly one model by its opaque ``resource`` string. Grown as entities are
# added; used where the entity type is not known up front (each model also has its
# own ``load`` for when it is).
Entity = Annotated[RootConfig | DatasetConfig, Field(discriminator='resource')]
ENTITY_ADAPTER: TypeAdapter[RootConfig | DatasetConfig] = TypeAdapter(Entity)


def load_entity(path: Path) -> RootConfig | DatasetConfig:
    """Parse any persisted snowtool entity, routing by its ``resource`` tag.

    The type-agnostic loader: use it when the entity kind is *not* known up front
    (a file off disk that could be either a root or a dataset config) and let the
    discriminated union resolve it. Product code that already knows the kind it
    wants calls the concrete ``RootConfig.load`` / ``DatasetConfig.load`` instead;
    this is the public entry point for the general "load whatever this file is"
    case (and the round-trip guarantee the persisted ``resource`` tags exist to
    provide).
    """
    return ENTITY_ADAPTER.validate_json(Path(path).read_text())

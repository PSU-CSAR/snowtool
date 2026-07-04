"""The zone-layer framework: a provider + registry abstraction over the static,
grid-co-registered layers that stratify a dataset for zonal statistics.

A *zone layer* is a raster derived once from an external source and laid down on
every dataset grid -- elevation and aspect (from a DEM), percent forest cover
(from NLCD), and so on. Each kind of zone layer is described by a
:class:`ZoneLayerProvider`: the layers it writes, the subdirectory they live in,
their shared provenance tag, the default source to read from, and how to run its
generation engine. A dataset holds one :class:`ZoneLayerSet` per provider (the
on-disk directory + the layers within it); :class:`ZoneLayerSource` is the common
base for the pluggable sources (a DEM source, an NLCD source).

This abstraction means a new zone layer is one provider plus one registry entry,
with no edits to ``Dataset``/``SnowDb``/CLI/diagnostics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import rasterio

from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.raster import TiledRaster

if TYPE_CHECKING:
    from collections.abc import Iterable
    from contextlib import AbstractContextManager
    from pathlib import Path

    import numpy

    from griffine.grid import TiledAffineGrid

    from snowtool.snowdb.grid import Bounds
    from snowtool.snowdb.zones.zoning import ZoneScheme


@dataclass(frozen=True)
class GenerationOptions:
    """Optional knobs for a zone-layer generation pass.

    A single typed value object threaded from the caller through the provider to
    its engine, in place of an untyped option bag. Only terrain consumes these
    today (block-level parallelism + the orientation-mean slope weighting); a
    provider ignores any field it does not use, and ``None`` defers to the engine's
    own default.
    """

    workers: int | None = None
    block_size: int | None = None
    # Terrain only: weight each cell's cos/sin aspect-orientation mean by
    # sin(slope), so steep pixels dominate the mean direction. ``False`` (the
    # default) counts every non-flat pixel once. A real, caller-settable option --
    # replaces the former hardcoded ``COSSIN_SLOPE_WEIGHTED`` module constant.
    cossin_slope_weighted: bool = False


@dataclass(frozen=True)
class ZoneLayer:
    """A single on-disk zone layer: its file, dtype, nodata, bands, and zoning.

    Shared by the generator (which writes it) and :class:`ZoneLayerSet` (which
    locates/reads it) so the two can never disagree on layout. ``key`` is the
    stable id a query references (e.g. ``'elevation'``); ``zoning`` is the
    :class:`~snowtool.snowdb.zones.zoning.ZoneScheme` that makes the layer a query-able
    zone (``None`` for a layer that is generated but not itself zone-able, e.g.
    the aspect components).
    """

    filename: str
    dtype: str
    nodata: float | int
    band_descriptions: tuple[str, ...]
    key: str
    zoning: ZoneScheme | None = None

    @property
    def count(self: Self) -> int:
        return len(self.band_descriptions)


@dataclass(frozen=True)
class ZoneLayerTarget:
    """A grid to bin a provider's layers into, and where to write them.

    The unit the generation engines consume: a named target grid plus the
    directory its layer set is written to. Unifies the byte-identical
    per-kind target classes the terrain and land-cover engines used.
    """

    name: str
    grid: TiledAffineGrid
    tile_size: int
    directory: Path


class ZoneLayerSet:
    """A dataset's directory of one provider's layers, and the ops on it.

    Filesystem-only: it locates the layer files, reports which exist, reads the
    shared provenance hash, and hands back a tiled reader for any layer.
    """

    def __init__(
        self: Self,
        directory: Path,
        layers: Iterable[ZoneLayer],
        hash_tag: str,
        format_version: int,
    ) -> None:
        self.directory = directory
        self.layers = tuple(layers)
        # The provenance tag this set's generation stamps on every layer (e.g.
        # SNOWTOOL_DEM_HASH); read back via provenance_hash().
        self.hash_tag = hash_tag
        # The current on-disk format version for this kind of layer; a built set
        # stamped with an older version is stale (see format_is_current()).
        self.format_version = format_version

    def layer_path(self: Self, layer: ZoneLayer) -> Path:
        return self.directory / layer.filename

    def present(self: Self) -> bool:
        """Whether every layer of a complete set exists on disk."""
        return all(self.layer_path(layer).is_file() for layer in self.layers)

    def missing_layers(self: Self) -> list[ZoneLayer]:
        """The layers that are not present on disk (the report selection)."""
        return [layer for layer in self.layers if not self.layer_path(layer).is_file()]

    def raster(self: Self, layer: ZoneLayer) -> TiledRaster[numpy.generic]:
        """A tiled COG reader for ``layer`` (for query-time reads)."""
        return TiledRaster(self.layer_path(layer))

    def provenance_hash(self: Self) -> str | None:
        """The set's provenance hash, or ``None`` if it isn't built.

        Reads only the first layer's tags (no array decode): the generation hash
        is stamped identically on every layer, so any present layer carries it.
        Returns ``None`` when the layer is absent or predates the tagging.
        """
        path = self.layer_path(self.layers[0])
        if not path.is_file():
            return None
        with rasterio.open(path) as ds:
            return ds.tags().get(self.hash_tag)

    def stored_format_version(self: Self) -> int | None:
        """The format version stamped on the built set, or ``None`` if unbuilt or
        the provenance tag is missing/untagged."""
        from snowtool.snowdb.provenance import parse_format_version

        return parse_format_version(self.provenance_hash())

    def format_is_current(self: Self) -> bool | None:
        """Whether a built set's stamped format version matches the current one.

        ``None`` when the set is not built (nothing to check); otherwise ``True``
        only if the stamped version equals :attr:`format_version`. A built set
        with a missing/legacy tag reads as ``False`` (its stored version is
        ``None``), so it is flagged for a rebuild.
        """
        if self.provenance_hash() is None:
            return None
        return self.stored_format_version() == self.format_version


class ZoneLayerSource(ABC):
    """A source of fine-resolution data, opened over a geographic extent.

    The common base for the pluggable sources a provider reads from (a DEM source,
    an NLCD source). The generation engine reprojects/streams whatever it is
    handed; the source just yields an opened dataset covering the requested bounds.
    """

    @abstractmethod
    def open(
        self: Self,
        bounds: Bounds,
    ) -> AbstractContextManager[rasterio.io.DatasetReader]:
        """Context manager yielding an opened dataset covering ``bounds``.

        ``bounds`` is ``(west, south, east, north)`` in EPSG:4326.
        """
        raise NotImplementedError


class ZoneLayerProvider(ABC):
    """A kind of zone layer: its layers, where they live, and how to build them.

    One provider per zone-layer kind (terrain, land cover, ...). Subclasses set
    :attr:`name` (the registry/query id), :attr:`subdir` (the dataset
    subdirectory its set lives in), :attr:`layers`, and :attr:`hash_tag`, and
    implement :meth:`default_source` and :meth:`generate`.
    """

    name: str
    subdir: str
    layers: tuple[ZoneLayer, ...]
    hash_tag: str
    format_version: int

    @abstractmethod
    def default_source(self: Self, root: Path) -> ZoneLayerSource:
        """The source this provider reads from when none is overridden.

        ``root`` is the snowdb root, so a source that caches a download can place
        its cache under the database.
        """
        raise NotImplementedError

    @abstractmethod
    def local_source(self: Self, path: Path) -> ZoneLayerSource:
        """A source reading the operator's own on-disk raster at ``path``.

        Backs the CLI override flags (``--source PROVIDER PATH``): each provider
        knows the concrete local-file source for its data kind.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self: Self,
        source: ZoneLayerSource,
        targets: list[ZoneLayerTarget],
        bounds: Bounds,
        *,
        force: bool = False,
        options: GenerationOptions | None = None,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> dict[str, str]:
        """Open ``source`` over ``bounds`` and bin its layers into every target.

        Owns the ``with source.open(bounds)`` and the engine call. The engine is
        injectable per provider (a test passes a fast stand-in via the constructor);
        an un-injected provider resolves the real engine lazily, since the engine
        module imports its provider and a module-level import would cycle. Returns
        the per-target provenance hash. ``options`` carries engine knobs (e.g.
        terrain's ``workers``/``block_size``); a provider ignores any it does not
        use, and ``None`` defers to the engine defaults. ``progress`` reports the
        long step (terrain's per-block reprojection, the NLCD download).
        """
        raise NotImplementedError

    def layer_set(self: Self, directory: Path) -> ZoneLayerSet:
        """The :class:`ZoneLayerSet` for this provider rooted at ``directory``."""
        return ZoneLayerSet(
            directory,
            self.layers,
            self.hash_tag,
            self.format_version,
        )


@dataclass(frozen=True)
class AvailableZone:
    """A query-able zone layer: its provider, layer, and zoning scheme.

    Keyed in :func:`available_zones` by ``'<provider>.<layer.key>'`` (e.g.
    ``'terrain.elevation'``) -- the stable id a query references.
    """

    provider: ZoneLayerProvider
    layer: ZoneLayer
    scheme: ZoneScheme


def available_zones(
    providers: Iterable[ZoneLayerProvider],
) -> dict[str, AvailableZone]:
    """Every query-able zone across ``providers``, keyed ``'<provider>.<layer.key>'``.

    Enumerates each provider's layers that declare a zoning scheme
    (``layer.zoning is not None``); layers that are generated but not themselves
    zone-able (e.g. the aspect components) never appear.
    """
    zones: dict[str, AvailableZone] = {}
    for provider in providers:
        for layer in provider.layers:
            if layer.zoning is not None:
                zones[f'{provider.name}.{layer.key}'] = AvailableZone(
                    provider,
                    layer,
                    layer.zoning,
                )
    return zones

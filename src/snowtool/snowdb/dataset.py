from __future__ import annotations

import shutil

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy
import numpy.typing
import rasterio

from rasterio.features import rasterize

from snowtool import types
from snowtool.exceptions import AOIRasterNotFoundError, SNODASError
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import AOI_HASH_TAG, AOI_MASK_NODATA, TILE_BBOX_TAG
from snowtool.snowdb.grid import bounding_tiles, grid_extent_4326, tile_base_origin
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.raster import AOIRaster
from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from affine import Affine
    from griffine.grid import AffineGrid, AffineGridTile, TiledAffineGrid
    from shapely import Geometry

    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.ingest import WritableRaster
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.variables import DatasetVariable
    from snowtool.snowdb.zone_layer import (
        GenerationOptions,
        ZoneLayerProvider,
        ZoneLayerSet,
        ZoneLayerSource,
        ZoneLayerTarget,
    )

# On-disk format version of the burned AOI raster (per-pixel cell area, 0 outside).
# The AOI raster has no ingester/provider -- the Dataset burns it generically -- so
# its version is owned here, by its writer, and stamped onto AOI_HASH_TAG via
# aoi_provenance. Bump on a material format change (e.g. the boolean-mask ->
# cell-area switch) so existing rasters read as stale and re-rasterize.
AOI_RASTER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DatasetArtifacts:
    """Which of a dataset's on-disk artifacts are present.

    A read-only snapshot used by the diagnostics/report commands.
    ``zone_layers`` maps each configured zone-layer provider's name
    (``terrain``, ``landcover``, ...) to whether its complete set is present on
    disk.
    """

    zone_layers: dict[str, bool]
    aoi_rasters: bool
    cogs: bool


def make_geometry_mask(
    geometry,
    *,
    out_shape: tuple[int, int],
    transform: Affine,
) -> numpy.typing.NDArray[numpy.bool_]:
    """Rasterize ``geometry`` to a boolean mask, True inside.

    ``geometry`` must already be in the grid/``transform`` CRS.
    """
    burned = rasterize(
        [geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8',
    )
    return burned.astype(bool)


def _window_cell_areas(
    base_grid: AffineGrid,
    start_row: int,
    height: int,
    width: int,
    cell_area: float | None,
) -> numpy.typing.NDArray[numpy.float32]:
    """Per-pixel cell area (m^2) for an AOI window, broadcast to ``(height, width)``.

    A projected grid passes its constant ``cell_area`` (every cell is identical).
    A geographic grid passes ``None``: geodesic cell area depends only on latitude
    (row), so one value per window row is computed from ``base_grid`` and
    broadcast across the columns.
    """
    if cell_area is not None:
        return numpy.broadcast_to(numpy.float32(cell_area), (height, width))
    row_areas = numpy.fromiter(
        (base_grid[start_row + i, 0].area for i in range(height)),
        dtype=numpy.float32,
        count=height,
    )
    return numpy.broadcast_to(row_areas[:, numpy.newaxis], (height, width))


def aoi_provenance(geometry_hash: str) -> str:
    """The versioned tag an AOI raster is stamped with and checked against.

    Combines the AOI's pure geometry digest with the burned-raster format version
    (see :func:`~snowtool.snowdb.provenance.versioned_hash`), so a format change
    invalidates every existing raster through the same equality check that catches
    a geometry change.
    """
    return versioned_hash(AOI_RASTER_FORMAT_VERSION, geometry_hash)


def write_aoi_raster(
    path: Path,
    geometry: Geometry,
    crs: rasterio.crs.CRS,
    start_tile: AffineGridTile,
    end_tile: AffineGridTile,
    tile_size: int,
    provenance: str,
    *,
    base_grid: AffineGrid,
    cell_area: float | None,
) -> None:
    """Burn ``geometry`` to a per-pixel cell-area AOI COG over its tile-bbox window.

    Each pixel whose centre falls inside the basin gets the area (m^2) it
    rasterizes to on this grid; every other pixel is ``0``. The single raster is
    therefore both the in/out-of-polygon membership signal and the area weights
    the zonal reduction needs -- there is no separate ``areas.tif``. ``cell_area``
    is the grid's constant cell area on a projected grid, or ``None`` on a
    geographic grid (per-row geodesic area is computed from ``base_grid``).

    ``provenance`` is the versioned AOI tag (see :func:`aoi_provenance`): the AOI
    geometry digest plus the burned-raster format version.

    It is deliberately decoupled from the DEM. Elevation (for banding) and any
    other terrain variable are read live from the dataset's terrain set at query
    time, so a terrain rebuild never invalidates an AOI raster: its only AOI-side
    provenance axis is the geometry (the ``SNOWTOOL_AOI_HASH`` tag); the cell
    areas are a pure function of the fixed dataset grid.
    """
    start = tile_base_origin(start_tile)
    end_origin = tile_base_origin(end_tile)
    end_row = end_origin.row + end_tile.rows
    end_col = end_origin.col + end_tile.cols
    height = end_row - start.row
    width = end_col - start.col

    # The tile's own affine is the upper-left transform of the AOI window, at
    # base (full) resolution.
    transform = start_tile.transform

    # ``geometry`` is already in the grid CRS (see Dataset.rasterize_aoi).
    aoi_mask = make_geometry_mask(
        geometry,
        out_shape=(height, width),
        transform=transform,
    )
    areas = _window_cell_areas(base_grid, start.row, height, width, cell_area)
    aoi_area = numpy.where(aoi_mask, areas, numpy.float32(0)).astype(numpy.float32)

    tags = {
        TILE_BBOX_TAG: (
            f'{start_tile.row} {start_tile.col} {end_tile.row} {end_tile.col}'
        ),
        # Records the geometry + format version this raster was burned from, so a
        # changed basin OR a format bump is detected (and re-rasterized) by a cheap
        # tag read.
        AOI_HASH_TAG: provenance,
    }

    write_cog(
        path,
        aoi_area,
        transform=transform,
        crs=crs,
        # 0 = outside the AOI (no real cell has 0 area), so it doubles as the
        # nodata sentinel.
        nodata=AOI_MASK_NODATA,
        tile_size=tile_size,
        tags=tags,
        compute_stats=False,
    )


class Dataset:
    """A :class:`DatasetSpec` bound to its ``data/<name>/`` directory.

    Owns the per-dataset filesystem layout (``aoi-rasters/``, the per-provider
    zone-layer subdirs, ``cogs/``) and the operations on it; grid/variables are
    reached through ``self.spec``.
    """

    def __init__(
        self: Self,
        spec: DatasetSpec,
        path: Path,
        providers: Iterable[ZoneLayerProvider] = DEFAULT_ZONE_LAYER_PROVIDERS,
    ) -> None:
        self.spec = spec
        self.path = path
        self._aoi_rasters = self.path / 'aoi-rasters'
        self._cogs = self.path / 'cogs'
        # One zone-layer set per provider this dataset *enables* (its config's
        # zones block), keyed by provider name (e.g. 'terrain', 'landcover'). A
        # registered provider the dataset does not enable is bound to neither
        # `providers` nor `zones`, so it is never generated, served, or reported
        # for this dataset. A new zone-layer kind is just a new provider in the
        # registry (enabled by a dataset's zones), with no edits here.
        self.providers = {
            provider.name: provider
            for provider in providers
            if spec.enables(provider.name)
        }
        self.zones: dict[str, ZoneLayerSet] = {
            provider.name: provider.layer_set(self.path / provider.subdir)
            for provider in self.providers.values()
        }

    @property
    def grid(self: Self) -> TiledAffineGrid:
        return self.spec.grid

    @property
    def coverage_domain(self: Self) -> CoverageDomain:
        """The static region this dataset can serve (for AOI coverage)."""
        return self.spec.coverage_domain

    @property
    def grid_crs(self: Self) -> rasterio.crs.CRS:
        # The rasterio view of the grid's CRS, used when writing COGs. Derived
        # from the spec's single parsed CRS (not re-parsed from grid_params) so
        # the pyproj and rasterio sides can never disagree.
        return rasterio.crs.CRS.from_user_input(self.spec.crs)

    def validate(self: Self) -> Self:
        if not self.path.exists():
            raise FileNotFoundError(f'Unable to read directory: {self.path}')

        if not self.path.is_dir():
            raise ValueError(f'Not a directory: {self.path}')

        return self

    @classmethod
    def create(
        cls: type[Self],
        spec: DatasetSpec,
        path: Path,
        force: bool = False,
    ) -> Self:
        """Create the dataset's directory skeleton.

        Zone layers (terrain, land cover, ...) are *not* built here: each needs a
        source and is generated separately by :meth:`generate_zone_layers` (so
        generation can share one source read across every dataset -- see
        :meth:`SnowDb.generate_zone_layers`).
        """
        self = cls(spec, path)

        try:
            # The dataset dir itself may already exist as an empty skeleton from
            # `snowdb init`, so tolerate it; whether the dataset is already
            # *populated* is enforced by the artifact guards below (the
            # aoi-rasters/cogs dirs), which still refuse to clobber existing data
            # without force.
            self.path.mkdir(parents=True, exist_ok=True)
            self._aoi_rasters.mkdir(exist_ok=force)
            self._cogs.mkdir(exist_ok=force)
        except FileExistsError as e:
            raise FileExistsError(
                f'Could not create {self.spec.name} dataset: {self.path} already '
                'exists. Remove and try again or use `force=True`.',
            ) from e

        return self

    def zone_target(self: Self, provider: ZoneLayerProvider) -> ZoneLayerTarget:
        """This dataset's grid as a target for ``provider``'s generation engine."""
        from snowtool.snowdb.zone_layer import ZoneLayerTarget

        return ZoneLayerTarget(
            name=self.spec.name,
            grid=self.grid,
            tile_size=self.spec.grid_params.tile_size,
            directory=self.zones[provider.name].directory,
        )

    def generate_zone_layers(
        self: Self,
        provider: ZoneLayerProvider,
        source: ZoneLayerSource,
        *,
        force: bool = False,
        options: GenerationOptions | None = None,
    ) -> str:
        """Generate this dataset's zone-layer set for ``provider`` from ``source``.

        A single-grid pass over the source (binning only into this grid); for the
        multi-grid shared-source pass, see :meth:`SnowDb.generate_zone_layers`.
        ``options`` carries engine knobs (e.g. terrain's ``workers``/
        ``block_size``). Returns the set's provenance hash.
        """
        bounds = grid_extent_4326(self.grid)
        hashes = provider.generate(
            source,
            [self.zone_target(provider)],
            bounds,
            force=force,
            options=options,
        )
        return hashes[self.spec.name]

    @staticmethod
    def _format_date(date: date) -> str:
        return date.strftime('%Y%m%d')

    def raster_paths_from_query(
        self: Self,
        query: types.DateQuery,
        variable: DatasetVariable,
    ) -> Iterator[tuple[date, Path]]:
        # Filter the dates that actually exist (one directory listing) rather than
        # probing every calendar day in the range; this also lets a query carry an
        # open-ended interval, since selection is over a finite set. Each date's
        # single COG (and the multiple-match guard) comes from variable_path.
        for date_ in query.select(self.available_dates()):
            path = self.variable_path(date_, variable)
            if path is not None:
                yield date_, path

    def aoi_raster_path_from_triplet(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> Path:
        return self._aoi_rasters / f'{types.triplet_to_stem(station_triplet)}.tif'

    def rasterize_aoi(self, aoi: Pourpoint, force: bool = False) -> AOIRaster:
        # A management (write) op may run against a dataset that has no data yet,
        # so create the aoi-rasters dir if it is missing (but never the base
        # snowdb dirs -- those are SnowDbManager.initialize's job).
        self._aoi_rasters.mkdir(parents=True, exist_ok=True)

        path = self.aoi_raster_path_from_triplet(aoi.station_triplet)
        if not force and path.exists():
            raise FileExistsError(
                f'Could not create AOI raster: {path} already exists. '
                'Remove and try again or use `force=True`.',
            )

        # The AOI is stored in WGS84; reproject it into this grid's CRS so its
        # tile extent and pixel mask are computed in the grid's own coordinates.
        geometry = aoi.geometry_in_crs(self.grid.crs)
        ul_tile, br_tile = bounding_tiles(self.grid, geometry.bounds)

        # A projected grid burns its constant cell area; a geographic grid burns
        # per-row geodesic area, computed from the base grid (cell_area=None).
        cell_area = None if self.spec.is_geographic else self.spec.cell_area

        write_aoi_raster(
            path,
            geometry,
            self.grid_crs,
            ul_tile,
            br_tile,
            tile_size=self.spec.grid_params.tile_size,
            provenance=aoi_provenance(aoi.geometry_hash),
            base_grid=self.grid.base_grid,
            cell_area=cell_area,
        )

        return AOIRaster.open(path, self.grid)

    def aoi_raster_hash(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> str | None:
        """The AOI-geometry hash an existing AOI raster was burned from.

        Reads only the COG's tags (no array decode); returns ``None`` if the
        raster does not exist or predates the ``AOI_HASH_TAG`` tagging.
        """
        path = self.aoi_raster_path_from_triplet(station_triplet)
        if not path.is_file():
            return None
        with rasterio.open(path) as ds:
            return ds.tags().get(AOI_HASH_TAG)

    def aoi_raster_is_current(self: Self, aoi: Pourpoint) -> bool:
        """Whether a burned AOI raster exists AND matches ``aoi``'s geometry AND
        the current burned-raster format version.

        ``False`` means missing or stale (changed geometry *or* an old format
        version) -- either way :meth:`rasterize_aoi` should (re)build it.
        """
        return self.aoi_raster_hash(aoi.station_triplet) == aoi_provenance(
            aoi.geometry_hash,
        )

    def rasterize_aoi_if_needed(
        self: Self,
        aoi: Pourpoint,
        *,
        rebuild: bool = False,
    ) -> bool:
        """Build the AOI raster when missing or stale; True if it was (re)built.

        ``rebuild=True`` forces a rebuild regardless of current state. The
        converge-by-default path (``rebuild=False``) skips a raster only when it
        is already current (a matching :attr:`Pourpoint.geometry_hash` tag).
        """
        if not rebuild and self.aoi_raster_is_current(aoi):
            return False
        self.rasterize_aoi(aoi, force=True)
        return True

    def remove_aoi_raster(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> bool:
        """Delete this dataset's burned AOI raster for ``triplet``; True if present."""
        path = self.aoi_raster_path_from_triplet(station_triplet)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def ingest(self: Self, source: Path, *, force: bool = False) -> list[date]:
        """Ingest a source artifact into per-date COGs, via this dataset's ingester.

        Delegates to ``spec.ingester`` (the dataset-kind-specific parser); raises
        if the dataset has no configured ingester. Returns the dates ingested.
        """
        ingester = self.spec.ingester
        if ingester is None:
            raise SNODASError(
                f"dataset '{self.spec.name}' has no configured ingester; "
                'nothing can be ingested into it.',
            )
        return ingester.ingest(source, self, force=force)

    def write_date_cogs(
        self: Self,
        date: date,
        rasters: Iterable[WritableRaster],
        *,
        force: bool = False,
    ) -> None:
        """Write a date's already-on-grid rasters into ``cogs/<YYYYMMDD>/``.

        The dataset-agnostic write side of ingest: it owns the date directory;
        the rasters (produced by an :class:`~snowtool.snowdb.ingest.Ingester`)
        know how to write themselves as COGs into it.
        """
        output_dir = self.date_dir(date)

        try:
            output_dir.mkdir(parents=True, exist_ok=force)
        except FileExistsError as e:
            raise FileExistsError(
                f'Could not create raster dir: {output_dir} already exists. '
                'Remove directory and try again, or use `force=True`.',
            ) from e

        for raster in rasters:
            raster.write_cog(output_dir=output_dir, force=force)

    def aoi_rasters(self: Self) -> Iterator[AOIRaster]:
        yield from (
            AOIRaster.open(path, self.grid) for path in self._aoi_rasters.glob('*.tif')
        )

    def load_aoi_raster(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> AOIRaster:
        """Open the burned AOI raster for ``triplet`` (the stats read input).

        Raises :class:`FileNotFoundError` (pointing at ``pourpoint rasterize``) when the
        raster has not been built for this dataset, so a stats query surfaces a
        clean missing-prerequisite error rather than a bare open failure.
        """
        path = self.aoi_raster_path_from_triplet(station_triplet)
        if not path.is_file():
            raise AOIRasterNotFoundError(
                f'No AOI raster for {station_triplet!r} in dataset '
                f'{self.spec.name!r}; run `pourpoint rasterize` first.',
            )
        return AOIRaster.open(path, self.grid)

    # --- read-only query helpers (drive the report/diagnostics commands) ------

    @staticmethod
    def _parse_date_dir(name: str) -> date | None:
        """The ``date`` a ``cogs/<name>/`` dir encodes, or ``None`` if it isn't one.

        ``name`` is a calendar label (``YYYYMMDD``); it is pinned to UTC purely
        to build a tz-aware value, which does not shift the date.
        """
        try:
            parsed = datetime.strptime(name, '%Y%m%d').replace(tzinfo=UTC)
        except ValueError:
            return None
        return parsed.date()

    def date_dir(self: Self, d: date) -> Path:
        """The ``cogs/<YYYYMMDD>/`` directory for date ``d`` (may not exist)."""
        return self._cogs / self._format_date(d)

    def available_dates(self: Self) -> list[date]:
        """Every date with an ingested ``cogs/<YYYYMMDD>/`` directory, ascending."""
        if not self._cogs.is_dir():
            return []
        dates = (
            self._parse_date_dir(child.name)
            for child in self._cogs.iterdir()
            if child.is_dir()
        )
        return sorted(d for d in dates if d is not None)

    def variable_path(
        self: Self,
        d: date,
        variable: DatasetVariable,
    ) -> Path | None:
        """The single COG for ``variable`` on date ``d``, or ``None`` if absent."""
        matching = list(self.date_dir(d).glob(variable.glob))
        if len(matching) > 1:
            raise RuntimeError(
                'Found multiple files matching date / variable '
                f"'{d}' / '{variable.key}': {matching}",
            )
        return matching[0] if matching else None

    def missing_variables(self: Self, d: date) -> set[DatasetVariable]:
        """Spec variables whose glob matches no file in date ``d``'s cogs dir.

        An absent date directory yields every variable (nothing is present).
        """
        return {
            variable
            for variable in self.spec.variables.values()
            if self.variable_path(d, variable) is None
        }

    def aoi_raster_paths(self: Self) -> list[Path]:
        """The burned ``aoi-rasters/*.tif`` files, sorted by path."""
        if not self._aoi_rasters.is_dir():
            return []
        return sorted(self._aoi_rasters.glob('*.tif'))

    def aoi_raster_triplets(self: Self) -> set[types.StationTriplet]:
        """Station triplets that have a burned ``aoi-rasters/<triplet>.tif``."""
        return {types.stem_to_triplet(path.stem) for path in self.aoi_raster_paths()}

    def artifact_status(self: Self) -> DatasetArtifacts:
        """Which of this dataset's on-disk artifacts currently exist."""
        return DatasetArtifacts(
            zone_layers={
                name: zone_set.present() for name, zone_set in self.zones.items()
            },
            aoi_rasters=self._aoi_rasters.is_dir(),
            cogs=self._cogs.is_dir(),
        )

    def dates_before(self: Self, before: date) -> list[date]:
        """Ingested dates strictly older than ``before`` (the prune selection)."""
        return [d for d in self.available_dates() if d < before]

    def remove_date(self: Self, d: date) -> bool:
        """Delete a date's ``cogs/<YYYYMMDD>/`` directory; True if it existed."""
        date_dir = self.date_dir(d)
        if not date_dir.is_dir():
            return False
        shutil.rmtree(date_dir)
        return True

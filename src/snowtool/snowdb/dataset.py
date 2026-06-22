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
from snowtool.exceptions import SNODASError
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import AOI_HASH_TAG, AOI_MASK_NODATA, TILE_BBOX_TAG
from snowtool.snowdb.grid import bounding_tiles, grid_extent_4326, tile_base_origin
from snowtool.snowdb.landcover import LandCoverSet
from snowtool.snowdb.raster import AOIRaster, AOIRasterWithArea, AreaRaster
from snowtool.snowdb.terrain import TerrainSet

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from affine import Affine
    from griffine.grid import AffineGridTile, TiledAffineGrid
    from shapely import Geometry

    from snowtool.snowdb.dem_source import DemSource
    from snowtool.snowdb.ingest import WritableRaster
    from snowtool.snowdb.landcover_generate import LandCoverTarget
    from snowtool.snowdb.landcover_source import LandCoverSource
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.terrain_generate import TerrainTarget
    from snowtool.snowdb.tiff_cache import TiffCache
    from snowtool.snowdb.variables import DatasetVariable


@dataclass(frozen=True)
class DatasetArtifacts:
    """Which of a dataset's on-disk artifacts are present.

    A read-only snapshot used by the diagnostics/report commands. ``area`` is
    ``None`` when an area raster is not applicable (a projected grid has a
    constant cell area and stores no ``areas.tif``); otherwise it reflects
    whether ``areas.tif`` exists. ``terrain`` is whether the complete terrain set
    (elevation + aspect layers) is present; ``landcover`` is whether the
    land-cover set (percent forest cover) is present.
    """

    terrain: bool
    landcover: bool
    aoi_rasters: bool
    cogs: bool
    area: bool | None


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


def write_aoi_raster(
    path: Path,
    geometry: Geometry,
    crs: rasterio.crs.CRS,
    start_tile: AffineGridTile,
    end_tile: AffineGridTile,
    tile_size: int,
    geometry_hash: str,
) -> None:
    """Burn ``geometry`` to a boolean AOI mask COG over its tile-bbox window.

    The AOI raster is a bare in/out-of-polygon mask (1 inside, 0 outside) -- it
    is deliberately decoupled from the DEM. Elevation (for banding) and any other
    terrain variable are read live from the dataset's terrain set at query time,
    so a terrain rebuild never invalidates an AOI raster: its only provenance axis
    is the AOI geometry (the ``SNOWTOOL_AOI_HASH`` tag).
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

    tags = {
        TILE_BBOX_TAG: (
            f'{start_tile.row} {start_tile.col} {end_tile.row} {end_tile.col}'
        ),
        # Records which AOI geometry this raster was burned from, so a changed
        # basin can be detected (and re-rasterized) by a cheap tag read.
        AOI_HASH_TAG: geometry_hash,
    }

    write_cog(
        path,
        aoi_mask.astype('uint8'),
        transform=transform,
        crs=crs,
        # 0 = outside the AOI; the mask carries no other information so stats are
        # pointless.
        nodata=AOI_MASK_NODATA,
        tile_size=tile_size,
        predictor=2,
        tags=tags,
        compute_stats=False,
    )


class Dataset:
    """A :class:`DatasetSpec` bound to its ``data/<name>/`` directory.

    Owns the per-dataset filesystem layout (``aoi-rasters/``, ``areas.tif``,
    ``terrain/``, ``cogs/``) and the operations on it; grid/variables are reached
    through ``self.spec``.
    """

    def __init__(self: Self, spec: DatasetSpec, path: Path) -> None:
        self.spec = spec
        self.path = path
        self._aoi_rasters = self.path / 'aoi-rasters'
        self._cogs = self.path / 'cogs'
        self._area_raster = self.path / 'areas.tif'
        self.terrain = TerrainSet(self.path / 'terrain')
        self.landcover = LandCoverSet(self.path / 'landcover')

    @property
    def grid(self: Self) -> TiledAffineGrid:
        return self.spec.grid

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
        """Create the dataset's directory skeleton and area raster.

        Terrain (the DEM-derived elevation + aspect layers) is *not* built here:
        it needs a DEM source and is generated separately by
        :meth:`generate_terrain` (so generation can share one source read across
        every dataset -- see :meth:`SnowDb.generate_terrain`).
        """
        self = cls(spec, path)

        try:
            # The dataset dir itself may already exist as an empty skeleton from
            # `snowdb init`, so tolerate it; whether the dataset is already
            # *populated* is enforced by the artifact guards below (the
            # aoi-rasters/cogs dirs and the area raster), which still refuse to
            # clobber existing data without force.
            self.path.mkdir(parents=True, exist_ok=True)
            self._aoi_rasters.mkdir(exist_ok=force)
            self._cogs.mkdir(exist_ok=force)
            # A per-pixel area raster only carries information on a geographic
            # grid (geodesic area varies by latitude); a projected grid has a
            # constant cell area, supplied at read time from spec.cell_area.
            if self.spec.is_geographic:
                self.make_area_raster(force=force)
        except FileExistsError as e:
            raise FileExistsError(
                f'Could not create {self.spec.name} dataset: {self.path} already '
                'exists. Remove and try again or use `force=True`.',
            ) from e

        return self

    def area_raster(self: Self) -> AreaRaster:
        return AreaRaster(self._area_raster)

    async def load_aoi_with_area(
        self: Self,
        aoi_raster: AOIRaster,
        cache: TiffCache,
    ) -> AOIRasterWithArea:
        """Attach per-pixel area to an AOI raster, however this grid stores it.

        Geographic grids read the per-row geodesic ``areas.tif``; projected grids
        have no area raster and use the constant ``spec.cell_area`` instead (see
        :meth:`AOIRasterWithArea.with_constant_area`).
        """
        if self.spec.is_geographic:
            return await AOIRasterWithArea.from_aoi_raster(
                aoi_raster,
                self.area_raster(),
                cache,
            )
        return AOIRasterWithArea.with_constant_area(aoi_raster, self.spec.cell_area)

    def ensure_area_raster(self: Self) -> bool:
        """Build the area raster if it is needed and missing; True if it was built.

        Idempotent (drives ``snowdb init``): a projected grid needs none, and an
        existing one is left untouched.
        """
        if not self.spec.is_geographic or self._area_raster.exists():
            return False
        self.make_area_raster()
        return True

    def make_area_raster(self: Self, force: bool = False) -> None:
        if not force and self._area_raster.exists():
            raise FileExistsError(
                'Could not create area raster: '
                f'{self._area_raster} already exists. '
                'Remove and try again or use `force=True`.',
            )

        base = self.grid.base_grid
        rows, cols = base.rows, base.cols

        # Geodesic pixel area depends only on latitude (row), so compute one
        # value per row and broadcast across columns.
        row_areas = numpy.fromiter(
            (base[row, 0].area for row in range(rows)),
            dtype=numpy.float32,
            count=rows,
        )
        area_array = numpy.broadcast_to(
            row_areas[:, numpy.newaxis],
            (rows, cols),
        )

        write_cog(
            self._area_raster,
            area_array,
            transform=base.transform,
            # Match the DEM/COGs: the area raster shares the grid's CRS, not a
            # hardcoded WGS84 (identical for a 4326 grid, correct for any other
            # geographic CRS).
            crs=self.grid_crs,
            tile_size=self.spec.grid_params.tile_size,
            predictor=3,
        )

    def terrain_target(self: Self) -> TerrainTarget:
        """This dataset's grid as a target for the terrain-generation engine."""
        from snowtool.snowdb.terrain_generate import TerrainTarget

        return TerrainTarget(
            name=self.spec.name,
            grid=self.grid,
            tile_size=self.spec.grid_params.tile_size,
            directory=self.terrain.directory,
        )

    def generate_terrain(
        self: Self,
        source: DemSource,
        *,
        force: bool = False,
    ) -> str:
        """Generate this dataset's terrain set from ``source``.

        A single-grid pass over the source (binning only into this grid); for the
        multi-grid shared-source pass, see :meth:`SnowDb.generate_terrain`.
        Returns the terrain set's provenance hash.
        """
        from snowtool.snowdb.terrain_generate import generate_terrain

        bounds = grid_extent_4326(self.grid)
        with source.open(bounds) as src:
            hashes = generate_terrain(
                src,
                [self.terrain_target()],
                work_crs=source.work_crs,
                work_resolution=source.work_resolution,
                force=force,
            )
        return hashes[self.spec.name]

    def landcover_target(self: Self) -> LandCoverTarget:
        """This dataset's grid as a target for the land-cover-generation engine."""
        from snowtool.snowdb.landcover_generate import LandCoverTarget

        return LandCoverTarget(
            name=self.spec.name,
            grid=self.grid,
            tile_size=self.spec.grid_params.tile_size,
            directory=self.landcover.directory,
        )

    def generate_landcover(
        self: Self,
        source: LandCoverSource,
        *,
        force: bool = False,
    ) -> str:
        """Generate this dataset's land-cover set from ``source``.

        A single-grid pass over the source (binning only into this grid); for the
        multi-grid shared-source pass, see :meth:`SnowDb.generate_landcover`.
        Returns the land-cover set's provenance hash.
        """
        from snowtool.snowdb.landcover_generate import generate_landcover

        bounds = grid_extent_4326(self.grid)
        with source.open(bounds) as src:
            hashes = generate_landcover(
                src,
                [self.landcover_target()],
                force=force,
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
        for date_ in query.generate_sequence():
            matching_files = list(
                (self._cogs / self._format_date(date_)).glob(variable.glob),
            )

            if len(matching_files) < 1:
                continue
            if len(matching_files) > 1:
                raise RuntimeError(
                    'Found multiple files matching date / variable '
                    f"'{date_}' / '{variable.key}': {matching_files}",
                )

            yield date_, matching_files[0]

    def aoi_raster_path_from_triplet(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> Path:
        return self._aoi_rasters / f'{types.triplet_to_stem(station_triplet)}.tif'

    def rasterize_aoi(self, aoi: AOI, force: bool = False) -> AOIRaster:
        # A management (write) op may run against a dataset that has no data yet,
        # so create the aoi-rasters dir if it is missing (but never the base
        # snowdb dirs -- those are SnowDb.initialize's job).
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

        write_aoi_raster(
            path,
            geometry,
            self.grid_crs,
            ul_tile,
            br_tile,
            tile_size=self.spec.grid_params.tile_size,
            geometry_hash=aoi.geometry_hash,
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

    def aoi_raster_is_current(self: Self, aoi: AOI) -> bool:
        """Whether a burned AOI raster exists AND matches ``aoi``'s geometry.

        ``False`` means missing or stale -- either way :meth:`rasterize_aoi`
        should (re)build it.
        """
        return self.aoi_raster_hash(aoi.station_triplet) == aoi.geometry_hash

    def rasterize_aoi_if_needed(
        self: Self,
        aoi: AOI,
        *,
        rebuild: bool = False,
    ) -> bool:
        """Build the AOI raster when missing or stale; True if it was (re)built.

        ``rebuild=True`` forces a rebuild regardless of current state. The
        converge-by-default path (``rebuild=False``) skips a raster only when it
        is already current (a matching :attr:`AOI.geometry_hash` tag).
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
            AOIRaster.open(path, self.grid)
            for path in self._aoi_rasters.glob('*.tif')
        )

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
        return {
            types.stem_to_triplet(path.stem) for path in self.aoi_raster_paths()
        }

    def artifact_status(self: Self) -> DatasetArtifacts:
        """Which of this dataset's on-disk artifacts currently exist."""
        return DatasetArtifacts(
            terrain=self.terrain.present(),
            landcover=self.landcover.present(),
            aoi_rasters=self._aoi_rasters.is_dir(),
            cogs=self._cogs.is_dir(),
            area=self._area_raster.is_file() if self.spec.is_geographic else None,
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

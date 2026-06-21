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
from rasterio.warp import Resampling, reproject

from snowtool import types
from snowtool.exceptions import SNODASError
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.cog import WGS84, write_cog
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.grid import bounding_tiles, tile_base_origin
from snowtool.snowdb.raster import DEM, AOIRaster, AOIRasterWithArea, AreaRaster

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from affine import Affine
    from griffine.grid import AffineGridTile, TiledAffineGrid
    from shapely import Geometry

    from snowtool.snowdb.ingest import WritableRaster
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.tiff_cache import TiffCache
    from snowtool.snowdb.variables import DatasetVariable


@dataclass(frozen=True)
class DatasetArtifacts:
    """Which of a dataset's on-disk artifacts are present.

    A read-only snapshot used by the diagnostics/report commands. ``area`` is
    ``None`` when an area raster is not applicable (a projected grid has a
    constant cell area and stores no ``areas.tif``); otherwise it reflects
    whether ``areas.tif`` exists.
    """

    dem: bool
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
    dem: DEM,
    start_tile: AffineGridTile,
    end_tile: AffineGridTile,
    tile_size: int,
) -> None:
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

    masked = numpy.where(
        aoi_mask,
        dem.array[start.row : end_row, start.col : end_col],
        dem.nodata,
    )

    # floating points stuff messes up nodata
    masked = numpy.where(
        masked > -10000,
        masked,
        dem.nodata,
    )

    tags = {
        TILE_BBOX_TAG: (
            f'{start_tile.row} {start_tile.col} {end_tile.row} {end_tile.col}'
        ),
    }

    write_cog(
        path,
        masked.astype(dem.dtype),
        transform=transform,
        crs=dem.crs or WGS84,
        nodata=dem.nodata,
        tile_size=tile_size,
        predictor=3,
        tags=tags,
    )


class Dataset:
    """A :class:`DatasetSpec` bound to its ``data/<name>/`` directory.

    Owns the per-dataset filesystem layout (``aoi-rasters/``, ``areas.tif``,
    ``dem.tif``, ``cogs/``) and the operations on it; grid/variables are reached
    through ``self.spec``.
    """

    def __init__(self: Self, spec: DatasetSpec, path: Path) -> None:
        self.spec = spec
        self.path = path
        self._aoi_rasters = self.path / 'aoi-rasters'
        self._cogs = self.path / 'cogs'
        self._area_raster = self.path / 'areas.tif'
        self._dem = self.path / 'dem.tif'

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
        input_dem_path: Path,
        force: bool = False,
    ) -> Self:
        self = cls(spec, path)

        try:
            # The dataset dir itself may already exist as an empty skeleton from
            # `snowdb init`, so tolerate it; whether the dataset is already
            # *populated* is enforced by the artifact guards below (the
            # aoi-rasters/cogs dirs and the area/DEM rasters), which still refuse
            # to clobber existing data without force.
            self.path.mkdir(parents=True, exist_ok=True)
            self._aoi_rasters.mkdir(exist_ok=force)
            self._cogs.mkdir(exist_ok=force)
            # A per-pixel area raster only carries information on a geographic
            # grid (geodesic area varies by latitude); a projected grid has a
            # constant cell area, supplied at read time from spec.cell_area.
            if self.spec.is_geographic:
                self.make_area_raster(force=force)
            self.create_resampled_dem(input_dem_path, force=force)
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

    def create_resampled_dem(
        self: Self,
        input_dem_path: Path,
        force: bool = False,
    ) -> None:
        self.resample_to_grid(input_dem_path, self._dem, force=force)

    def resample_to_grid(
        self: Self,
        input_raster_path: Path,
        output_raster_path: Path,
        force: bool = False,
    ) -> None:
        if not force and output_raster_path.exists():
            raise FileExistsError(
                'Could not create output raster: '
                f'{output_raster_path} already exists. '
                'Remove and try again or use `force=True`.',
            )

        base = self.grid.base_grid
        dst_transform = base.transform
        dst_shape = (base.rows, base.cols)
        dst_crs = self.grid_crs

        with rasterio.open(input_raster_path) as src:
            dtype = src.dtypes[0]
            src_nodata = src.nodata
            if src_nodata is None:
                raise SNODASError(
                    f'Cannot resample {input_raster_path}: the source has no '
                    'nodata value. Resampling to the dataset grid can leave '
                    'cells uncovered by the source, and without a nodata value '
                    'there is no safe way to mark them. Define a nodata value '
                    'on the source raster and try again.',
                )
            # Pre-fill with nodata so any destination cell the reprojection does
            # not cover is marked nodata rather than left as uninitialized
            # memory.
            destination = numpy.full(dst_shape, src_nodata, dtype=dtype)
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=src_nodata,
                resampling=Resampling.average,
            )

        write_cog(
            output_raster_path,
            destination,
            transform=dst_transform,
            crs=dst_crs,
            nodata=src_nodata,
            tile_size=self.spec.grid_params.tile_size,
        )

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
        return self._aoi_rasters / f'{station_triplet.replace(":", "_")}.tif'

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
            DEM.open(self._dem),
            ul_tile,
            br_tile,
            tile_size=self.spec.grid_params.tile_size,
        )

        return AOIRaster.open(path, self.grid)

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
            types.StationTriplet(path.stem.replace('_', ':'))
            for path in self.aoi_raster_paths()
        }

    def artifact_status(self: Self) -> DatasetArtifacts:
        """Which of this dataset's on-disk artifacts currently exist."""
        return DatasetArtifacts(
            dem=self._dem.is_file(),
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

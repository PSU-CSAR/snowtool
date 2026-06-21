from __future__ import annotations

from datetime import date
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
from snowtool.snowdb.input_rasters import SNODASInputRasterSet
from snowtool.snowdb.raster import DEM, AOIRaster, AOIRasterWithArea, AreaRaster

if TYPE_CHECKING:
    from collections.abc import Iterator

    from affine import Affine
    from griffine.grid import AffineGridTile, TiledAffineGrid
    from shapely import Geometry

    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.tiff_cache import TiffCache
    from snowtool.snowdb.variables import DatasetVariable


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
        return rasterio.crs.CRS.from_user_input(self.spec.grid_params.crs)

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
            self.path.mkdir(exist_ok=force)
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
            crs=WGS84,
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

    def import_snodas_rasters(
        self: Self,
        rasters: SNODASInputRasterSet,
        force: bool = False,
    ) -> None:
        output_dir = self._cogs / self._format_date(rasters.date)

        try:
            output_dir.mkdir(exist_ok=force)
        except FileExistsError as e:
            raise FileExistsError(
                'Could not create SNODAS raster dir: '
                f'{output_dir} already  exists. '
                'Remove directory and try again, or use `force=True`.',
            ) from e

        for raster in rasters:
            raster.write_cog(output_dir=output_dir, force=force)

    def aoi_rasters(self: Self) -> Iterator[AOIRaster]:
        yield from (
            AOIRaster.open(path, self.grid)
            for path in self._aoi_rasters.glob('*.tif')
        )

from __future__ import annotations

import asyncio

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy
import numpy.typing
import rasterio

from snowtool import types
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.grid import (
    PixelCoord,
    tile_base_origin,
    tiles_in_bbox,
)

if TYPE_CHECKING:
    from datetime import date

    from affine import Affine
    from griffine.grid import AffineGridTile, TiledAffineGrid
    from rasterio.crs import CRS

    from snowtool.snowdb.tiff_cache import TiffCache


def _decode_to_array(
    decoded: Any,
    tile: AffineGridTile,
) -> numpy.typing.NDArray[Any]:
    array = numpy.asarray(decoded)
    if array.ndim == 3:
        array = array[..., 0]
    # Edge blocks are stored padded to the full tile size; trim to the tile's
    # actual extent.
    return array[: tile.rows, : tile.cols]


async def load_tile(
    path: Path,
    tile: AffineGridTile,
    cache: TiffCache,
) -> numpy.typing.NDArray[Any]:
    """Read the single COG block backing ``tile`` (full resolution)."""
    tiff = await cache.get(path)
    # async-tiff addresses blocks as (x=col, y=row, z=overview); z=0 is full res.
    fetched = await tiff.fetch_tile(tile.col, tile.row, 0)
    return _decode_to_array(await fetched.decode(), tile)


class TiledRaster[T: numpy.generic]:
    def __init__(self: Self, path: Path) -> None:
        self.path: Path = Path(path)

        if not self.path.is_file():
            raise TypeError('not a file')

    async def load_tiles(
        self: Self,
        tiles: list[AffineGridTile],
        cache: TiffCache,
    ) -> list[numpy.typing.NDArray[T]]:
        """Read several COG blocks in one batched, coalesced fetch.

        The blocks are handed to async-tiff together so it can coalesce the
        byte-range reads; the decodes then run concurrently.
        """
        if not tiles:
            return []
        tiff = await cache.get(self.path)
        fetched = await tiff.fetch_tiles([(t.col, t.row) for t in tiles], 0)
        decoded = await asyncio.gather(*(tile.decode() for tile in fetched))
        return [
            _decode_to_array(data, tile)
            for data, tile in zip(decoded, tiles, strict=True)
        ]


class AreaRaster(TiledRaster[numpy.float32]):
    pass


class DataRaster(TiledRaster[numpy.generic]):
    """A dated data COG for one variable on one date.

    The read path is dataset-agnostic: the date comes from the ``cogs/<date>/``
    directory the file was found in, not from parsing its name, and the read
    dtype comes from the requesting variable.
    """

    def __init__(self: Self, path: Path, date: date) -> None:
        super().__init__(path)
        self.date = date


def tiles_from_tags(
    grid: TiledAffineGrid,
    tags: dict[str, str],
) -> tuple[PixelCoord, list[AffineGridTile]]:
    """Resolve an AOI window's origin and tiles from a COG's metadata.

    AOI rasters store a ``ul_row ul_col br_row br_col`` tile bounding box in
    ``SNOWTOOL_TILE_BBOX``. The upper-left tile is the window origin and every
    tile in the box is read (the AOI mask nulls non-AOI pixels). Legacy snodas
    quadkey tags are not read here; migrate them first with
    ``snowtool migration aoi-tags``.
    """
    try:
        bbox = tags[TILE_BBOX_TAG]
    except KeyError as e:
        raise ValueError('AOI raster is missing tile metadata') from e

    ul_row, ul_col, br_row, br_col = (int(v) for v in bbox.split())
    origin = tile_base_origin(grid[ul_row, ul_col])
    tiles = tiles_in_bbox(grid, ul_row, ul_col, br_row, br_col)
    return origin, tiles


@dataclass
class AOIRaster:
    """A burned AOI: a boolean in/out-of-polygon mask over its tile-bbox window.

    Decoupled from the DEM -- ``array`` is a ``uint8`` mask (1 inside the basin,
    0 outside), not elevation. Elevation banding and any other terrain variable
    are read live from the dataset's terrain set at query time.
    """

    path: Path
    array: numpy.typing.NDArray[numpy.uint8]
    tiles: list[AffineGridTile]
    origin: PixelCoord

    @property
    def station_triplet(self: Self) -> types.StationTriplet:
        return types.StationTriplet(self.path.stem.replace('_', ':'))

    @classmethod
    def open(
        cls: type[Self],
        path: Path,
        grid: TiledAffineGrid,
    ) -> Self:
        with rasterio.open(path) as ds:
            tags = ds.tags()
            origin, tiles = tiles_from_tags(grid, tags)
            array: numpy.typing.NDArray[numpy.uint8] = ds.read(1)

        return cls(
            path=path,
            array=array,
            tiles=tiles,
            origin=origin,
        )

    async def load_raster_tiles_into_array(
        self: Self,
        raster: TiledRaster,
        array: numpy.typing.NDArray[Any],
        cache: TiffCache,
    ) -> None:
        # One coalesced fetch per source COG, then place each block.
        blocks = await raster.load_tiles(self.tiles, cache)
        for tile, block in zip(self.tiles, blocks, strict=True):
            tile_origin = tile_base_origin(tile)
            offset_row = tile_origin.row - self.origin.row
            offset_col = tile_origin.col - self.origin.col
            array[
                offset_row : offset_row + tile.rows,
                offset_col : offset_col + tile.cols,
            ] = block


@dataclass
class AOIRasterWithArea(AOIRaster):
    area: numpy.typing.NDArray[numpy.float32]

    @classmethod
    def _with_area(
        cls: type[Self],
        aoi_raster: AOIRaster,
        area: numpy.typing.NDArray[numpy.float32],
    ) -> Self:
        """Wrap an AOIRaster with its per-pixel ``area``, forwarding every base
        field so a new AOIRaster field needs no change at the call sites below.
        """
        base = {f.name: getattr(aoi_raster, f.name) for f in fields(AOIRaster)}
        return cls(area=area, **base)

    @classmethod
    def with_constant_area(
        cls: type[Self],
        aoi_raster: AOIRaster,
        cell_area: float,
    ) -> Self:
        """Attach a constant per-pixel area (projected grids have no area raster).

        On a projected/linear grid every cell has identical planar area, so the
        per-pixel area is just ``spec.cell_area`` -- no ``areas.tif`` is read.
        It is exposed as a zero-copy ``broadcast_to`` view (stride 0) rather than
        a materialized array: the zonal-stats reduction only ever reads
        ``area`` through boolean masks, so a full N-pixel copy of one constant
        would waste memory on the long-lived AOI for no information.
        """
        area = numpy.broadcast_to(
            numpy.float32(cell_area),
            aoi_raster.array.shape,
        )
        return cls._with_area(aoi_raster, area)

    @classmethod
    async def from_aoi_raster(
        cls: type[Self],
        aoi_raster: AOIRaster,
        area_raster: AreaRaster,
        cache: TiffCache,
    ) -> Self:
        area = numpy.zeros_like(
            aoi_raster.array,
            dtype=numpy.float32,
        )
        await aoi_raster.load_raster_tiles_into_array(area_raster, area, cache)
        return cls._with_area(aoi_raster, area)


@dataclass
class DEM:
    array: numpy.typing.NDArray[numpy.float32]
    transform: Affine
    crs: CRS | None
    nodata: float

    @property
    def dtype(self: Self) -> numpy.dtype:
        return self.array.dtype

    @classmethod
    def open(cls: type[Self], path: Path) -> Self:
        with rasterio.open(str(path)) as ds:
            array: numpy.typing.NDArray[numpy.float32] = ds.read(1)
            transform = ds.transform
            crs = ds.crs
            nodata = ds.nodata

        return cls(
            array=array,
            transform=transform,
            crs=crs,
            nodata=nodata,
        )

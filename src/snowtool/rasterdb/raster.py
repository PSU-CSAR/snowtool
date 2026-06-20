from __future__ import annotations

import asyncio

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy
import numpy.typing
import rasterio

from snowtool import types
from snowtool.exceptions import SNODASError
from snowtool.rasterdb import constants
from snowtool.rasterdb.grid import (
    PixelCoord,
    tile_base_origin,
    tile_from_quadkey,
    tiles_in_bbox,
)

if TYPE_CHECKING:
    from affine import Affine
    from griffine.grid import AffineGridTile, TiledAffineGrid
    from rasterio.crs import CRS

    from snowtool.rasterdb.fileinfo import SNODASFileInfo
    from snowtool.rasterdb.tiff_cache import TiffCache


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


class SNODASRaster(TiledRaster[numpy.int16]):
    def __init__(self: Self, fileinfo: SNODASFileInfo) -> None:
        super().__init__(fileinfo.path)
        self.fileinfo = fileinfo


def _tiles_from_tags(
    grid: TiledAffineGrid,
    tags: dict[str, str],
) -> tuple[PixelCoord, list[AffineGridTile]]:
    """Resolve the AOI window origin and the tiles it spans, from metadata.

    Current files store a ``ul_row ul_col br_row br_col`` tile bounding box;
    legacy (snodas) files store an origin-tile quadkey plus a per-tile
    intersected set of quadkeys. The tag names distinguish the two, so old COGs
    still read.
    """
    if constants.TILE_BBOX_TAG in tags:
        ul_row, ul_col, br_row, br_col = (
            int(v) for v in tags[constants.TILE_BBOX_TAG].split()
        )
        origin = tile_base_origin(grid[ul_row, ul_col])
        tiles = tiles_in_bbox(grid, ul_row, ul_col, br_row, br_col)
        return origin, tiles

    if constants.LEGACY_ORIGIN_TILE_TAG in tags:
        origin = tile_base_origin(
            tile_from_quadkey(grid, tags[constants.LEGACY_ORIGIN_TILE_TAG]),
        )
        tiles = [
            tile_from_quadkey(grid, val)
            for key, val in tags.items()
            if key.startswith(constants.LEGACY_TILE_TAG_PREFIX)
        ]
        return origin, tiles

    raise ValueError('AOI raster is missing tile metadata')


@dataclass
class AOIRaster:
    path: Path
    array: numpy.typing.NDArray[numpy.float32]
    tiles: list[AffineGridTile]
    origin: PixelCoord
    min_elevation: float
    max_elevation: float

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
            origin, tiles = _tiles_from_tags(grid, tags)

            band_tags = ds.tags(1)
            try:
                min_ = float(band_tags['STATISTICS_MINIMUM'])
                max_ = float(band_tags['STATISTICS_MAXIMUM'])
            except KeyError as e:
                # write_cog omits the STATISTICS_* tags when a band is entirely
                # nodata, which for an AOI raster means the AOI polygon does not
                # overlap any valid DEM pixel.
                raise SNODASError(
                    f'AOI raster {path} has no elevation statistics; the AOI '
                    'does not overlap any valid DEM data.',
                ) from e
            array: numpy.typing.NDArray[numpy.float32] = ds.read(1)

        return cls(
            path=path,
            array=array,
            tiles=tiles,
            origin=origin,
            min_elevation=min_,
            max_elevation=max_,
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
        return cls(
            area=area,
            path=aoi_raster.path,
            array=aoi_raster.array,
            tiles=aoi_raster.tiles,
            origin=aoi_raster.origin,
            min_elevation=aoi_raster.min_elevation,
            max_elevation=aoi_raster.max_elevation,
        )


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

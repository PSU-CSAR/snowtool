from __future__ import annotations

import asyncio

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy
import numpy.typing

if TYPE_CHECKING:
    from datetime import date

    from griffine.grid import AffineGridTile

    from snowtool.snowdb.raster.tiff_cache import TiffCache


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


class TiledRaster:
    def __init__(self: Self, path: Path) -> None:
        self.path: Path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f'No such raster file: {self.path}')

    async def load_tiles(
        self: Self,
        tiles: list[AffineGridTile],
        cache: TiffCache,
    ) -> list[numpy.typing.NDArray[numpy.generic]]:
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


class DataRaster(TiledRaster):
    """A dated data COG for one variable on one date.

    The read path is dataset-agnostic: the date comes from the ``cogs/<date>/``
    directory the file was found in, not from parsing its name, and the read
    dtype comes from the requesting variable.
    """

    def __init__(self: Self, path: Path, date: date) -> None:
        super().__init__(path)
        self.date = date

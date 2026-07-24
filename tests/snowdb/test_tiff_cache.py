"""Tests for the async TIFF handle cache and the async tile read path."""

import asyncio

import numpy
import rasterio

from snowtool.snowdb.raster import TiledRaster
from snowtool.snowdb.raster.tiff_cache import TiffCache

from ..conftest import synthetic_grid

TILE = 256


def _write_block_indexed_cog(path, grid):
    """COG whose every pixel encodes its own tile so reads are verifiable."""
    rows, cols = grid.base_grid.rows, grid.base_grid.cols
    array = numpy.zeros((rows, cols), dtype=numpy.int16)
    for trow in range(grid.size[0]):
        for tcol in range(grid.size[1]):
            value = trow * 10 + tcol
            array[
                trow * TILE : (trow + 1) * TILE,
                tcol * TILE : (tcol + 1) * TILE,
            ] = value
    with rasterio.open(
        path,
        'w',
        driver='COG',
        height=rows,
        width=cols,
        count=1,
        dtype='int16',
        crs=rasterio.CRS.from_epsg(4326),
        transform=grid.base_grid.transform,
        blocksize=TILE,
    ) as dst:
        dst.write(array, 1)
    return array


def test_load_tiles_batched_preserves_order(tmp_path):
    grid = synthetic_grid()
    path = tmp_path / 'blocks.tif'
    full = _write_block_indexed_cog(path, grid)
    tiles = [grid[r, c] for r in range(2) for c in range(2)]

    async def run():
        cache = TiffCache(maxsize=8)
        return await TiledRaster(path).load_tiles(tiles, cache)

    blocks = asyncio.run(run())
    assert len(blocks) == len(tiles)
    # one batched fetch_tiles call returns blocks in the requested order, each
    # decoded to its own tile's extent and values
    for tile, block in zip(tiles, blocks, strict=True):
        assert block.shape == (TILE, TILE)
        expected = full[
            tile.row * TILE : (tile.row + 1) * TILE,
            tile.col * TILE : (tile.col + 1) * TILE,
        ]
        assert numpy.array_equal(block, expected)
        assert (block == tile.row * 10 + tile.col).all()


def test_cache_reuses_handle(tmp_path):
    grid = synthetic_grid()
    path = tmp_path / 'blocks.tif'
    _write_block_indexed_cog(path, grid)

    async def run():
        cache = TiffCache(maxsize=8)
        first = await cache.get(path)
        second = await cache.get(path)
        return first is second, len(cache)

    same, size = asyncio.run(run())
    assert same
    assert size == 1


def test_cache_is_bounded(tmp_path):
    grid = synthetic_grid()
    paths = []
    for i in range(4):
        p = tmp_path / f'blocks_{i}.tif'
        _write_block_indexed_cog(p, grid)
        paths.append(p)

    async def run():
        cache = TiffCache(maxsize=2)
        handles = [await cache.get(p) for p in paths]
        bounded = len(cache) == 2
        # LRU: the two most-recently opened are retained (hit -> same handle),
        # while the oldest was evicted (miss -> a freshly opened handle).
        kept = (await cache.get(paths[3])) is handles[3]
        evicted = (await cache.get(paths[0])) is not handles[0]
        return bounded, kept, evicted

    bounded, kept, evicted = asyncio.run(run())
    assert bounded
    assert kept
    assert evicted


def test_concurrent_gets_open_once(tmp_path, monkeypatch):
    grid = synthetic_grid()
    path = tmp_path / 'blocks.tif'
    _write_block_indexed_cog(path, grid)

    opens = 0
    original = TiffCache._open

    async def counting_open(p):
        nonlocal opens
        opens += 1
        return await original(p)

    monkeypatch.setattr(TiffCache, '_open', staticmethod(counting_open))

    async def run():
        cache = TiffCache(maxsize=8)
        return await asyncio.gather(*(cache.get(path) for _ in range(20)))

    handles = asyncio.run(run())
    assert opens == 1
    assert all(h is handles[0] for h in handles)

"""Opt-in smoke test of ThreeDEP against the real public 3DEP S3 bucket.

Deselected by default (``-m "not network"`` in addopts); run with ``pytest -m
network``. It discovers the two real 1-degree tiles straddling a tile boundary,
builds the VRT mosaic, and reads only a tiny window spanning the seam -- COG range
reads, so a few MB at most, not the ~400 MB tiles. This is the one thing that
can't be checked offline: that discovery reads real headers and the hand-built
VRT stitches real, *overlapping* 3DEP tiles with no gap at the boundary.
"""

import math

import numpy
import pytest
import rasterio

from rasterio.warp import transform_bounds
from rasterio.windows import Window

from snowtool.snowdb.dem_source import ThreeDEP, candidate_tiles, discover_tiles
from snowtool.snowdb.grid import grid_extent_4326, make_grid
from snowtool.snowdb.terrain import TerrainProvider
from snowtool.snowdb.terrain_generate import generate_terrain
from snowtool.snowdb.zone_layer import ZoneLayerTarget

pytestmark = pytest.mark.network

# A small box straddling the lon -106 boundary between tiles n40w107 and n40w106
# (Colorado Front Range -- 3DEP has full coverage here).
BOUNDS = (-106.05, 39.50, -105.95, 39.60)


def _skip_if_offline(call):
    # A missing key is FileNotFoundError (OSError); every other S3 failure is
    # async-tiff's own exception, which isn't importable -- match it by name.
    try:
        return call()
    except OSError as e:  # pragma: no cover - env
        pytest.skip(f'3DEP S3 not reachable: {e}')
    except Exception as e:  # pragma: no cover - env
        if type(e).__name__ == 'AsyncTiffException':
            pytest.skip(f'3DEP S3 not reachable: {e}')
        raise


def test_candidate_and_discovered_tiles_straddle_the_boundary():
    assert candidate_tiles(BOUNDS) == ['n40w106', 'n40w107']

    tiles = _skip_if_offline(lambda: discover_tiles(BOUNDS))
    names = {t.uri.split('/')[-2] for t in tiles}
    assert {'n40w106', 'n40w107'} <= names


def test_vrt_mosaic_has_no_seam_across_the_tile_boundary():
    def _read():
        with ThreeDEP().open(BOUNDS) as src:
            nodata = src.nodata
            # A ~tens-of-pixels window centred on the lon -106 seam.
            top_row, left_col = src.index(-106.01, 39.56)
            bot_row, right_col = src.index(-105.99, 39.54)
            window = Window(
                left_col,
                top_row,
                right_col - left_col,
                bot_row - top_row,
            )
            return src.read(1, window=window), nodata

    data, nodata = _skip_if_offline(_read)

    assert data.size > 0
    # No nodata stripe at the seam, and plausible Front Range elevations (m).
    if nodata is not None:
        assert not numpy.any(data == nodata)
    assert numpy.isfinite(data).all()
    assert data.min() > 0
    assert data.max() < 5000


# A ~6 km box straddling the lon -106 tile seam (n40w106 | n40w107), inland Colorado
# with full 3DEP coverage. The engine clips its work grid to this footprint, so only
# the seam-straddling window of the source tiles is range-read.
SEAM_BOX = (-106.03, 39.52, -105.97, 39.58)


def test_parallel_engine_reproduces_serial_on_real_3dep(tmp_path):
    """Drive the engine like any caller against real 3DEP: grid + CRS in, output out.

    The offline determinism test uses a synthetic local GTiff; the *default* source
    is a hand-built S3 VRT mosaic, whose concurrent-read behaviour -- and our read
    lock's fix for it -- only surfaces against the real GDAL driver. We just hand the
    engine a small target grid straddling the n40w106/n40w107 tile seam; because it
    clips its work grid to the target footprint, only that window of the source COG
    tiles is range-read (tens of seconds, a few MB) rather than the whole tiles. The
    grid spans the seam, so this also covers cross-tile stitching. Serial vs. parallel
    must match exactly, including the generation hash.
    """

    source = ThreeDEP()
    west, south, east, north = transform_bounds('EPSG:4326', 'EPSG:5070', *SEAM_BOX)
    px = 50.0
    grid = make_grid(
        origin_x=west,
        origin_y=north,
        px_size=px,
        cols=max(128, math.ceil((east - west) / px)),
        rows=max(128, math.ceil((north - south) / px)),
        tile_size=128,
        crs=5070,
    )
    bounds = grid_extent_4326(grid)

    def _generate(directory, workers):
        with source.open(bounds) as src:
            target = ZoneLayerTarget(
                name='t',
                grid=grid,
                tile_size=128,
                directory=directory / 'terrain',
            )
            return generate_terrain(
                src,
                [target],
                work_crs=source.work_crs,
                work_resolution=source.work_resolution,
                workers=workers,
                block_size=256,
                force=True,
            )

    serial = _skip_if_offline(lambda: _generate(tmp_path / 's', 1))
    parallel = _skip_if_offline(lambda: _generate(tmp_path / 'p', 4))

    assert serial['t'] == parallel['t']
    provider = TerrainProvider()
    serial_set = provider.layer_set(tmp_path / 's' / 'terrain')
    parallel_set = provider.layer_set(tmp_path / 'p' / 'terrain')
    for layer in provider.layers:
        with (
            rasterio.open(serial_set.layer_path(layer)) as a,
            rasterio.open(parallel_set.layer_path(layer)) as b,
        ):
            numpy.testing.assert_array_equal(a.read(), b.read())

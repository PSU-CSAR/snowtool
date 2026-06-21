"""Opt-in smoke test of ThreeDEP against the real public 3DEP S3 bucket.

Deselected by default (``-m "not network"`` in addopts); run with ``pytest -m
network``. It enumerates the two real 1-degree tiles straddling a tile boundary,
builds the VRT mosaic, and reads only a tiny window spanning the seam -- COG range
reads, so a few MB at most, not the ~400 MB tiles. This is the one thing that
can't be checked offline: that the hand-built VRT stitches real, *overlapping*
3DEP tiles with no gap at the boundary.
"""

import numpy
import pytest

from rasterio.windows import Window

from snowtool.snowdb.dem_source import ThreeDEP, candidate_tiles, existing_tile_keys

pytestmark = pytest.mark.network

# A small box straddling the lon -106 boundary between tiles n40w107 and n40w106
# (Colorado Front Range -- 3DEP has full coverage here).
BOUNDS = (-106.05, 39.50, -105.95, 39.60)


def _skip_if_offline(call):
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        return call()
    except (BotoCoreError, ClientError, OSError) as e:  # pragma: no cover - env
        pytest.skip(f'3DEP S3 not reachable: {e}')


def test_candidate_and_existing_tiles_straddle_the_boundary():
    assert candidate_tiles(BOUNDS) == ['n40w106', 'n40w107']

    keys = _skip_if_offline(lambda: existing_tile_keys(BOUNDS))
    names = {k.split('/')[-2] for k in keys}
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

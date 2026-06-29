"""The land-cover-generation engine on a tiny, hand-checkable NLCD source.

A categorical source split forest (class 42) / non-forest (class 81) bins to a
percent-forest layer whose per-cell value is just the fraction of forest pixels.
Working in EPSG:5070 (so source and target share a CRS) keeps the point-in-cell
binning a near-identity, so the expected percentages are exact.
"""

import hashlib

import numpy
import pytest
import rasterio

from rasterio.crs import CRS

from snowtool.snowdb.constants import FOREST_PCT_NODATA, NLCD_HASH_TAG
from snowtool.snowdb.grid import make_grid
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.zones.landcover import (
    FOREST_COVER,
    LANDCOVER_FORMAT_VERSION,
    LANDCOVER_LAYERS,
    LandCoverProvider,
)
from snowtool.snowdb.zones.landcover_generate import generate_landcover
from snowtool.snowdb.zones.zone_layer import ZoneLayerTarget


def _landcover_set(directory):
    """The land-cover ZoneLayerSet rooted at ``directory`` (test reader)."""
    return LandCoverProvider().layer_set(directory)


WORK_EPSG = 5070
ORIGIN_X = -500_000.0
ORIGIN_Y = 2_000_000.0
SRC_PX = 10.0
SRC_N = 512
FOREST = 42  # evergreen forest (in FOREST_CLASSES)
NONFOREST = 81  # pasture/hay
NODATA = 0  # NLCD background/unclassified
# 128 cells x 40 m == 5120 m == the 512 x 10 m source extent (4 fine px / cell).
TARGET_N = 128
TARGET_PX = 40.0
TARGET_TILE = 128


def _source_nlcd(path, array):
    transform = rasterio.transform.from_origin(ORIGIN_X, ORIGIN_Y, SRC_PX, SRC_PX)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SRC_N,
        width=SRC_N,
        count=1,
        dtype='uint8',
        crs=CRS.from_epsg(WORK_EPSG),
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(array, 1)
    return path


def _target(tmp_path, name='t', *, px=TARGET_PX, n=TARGET_N, tile=TARGET_TILE):
    grid = make_grid(
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        px_size=px,
        cols=n,
        rows=n,
        tile_size=tile,
        crs=WORK_EPSG,
    )
    return ZoneLayerTarget(
        name=name,
        grid=grid,
        tile_size=tile,
        directory=tmp_path / name / 'landcover',
    )


def test_generate_writes_forest_layer_with_expected_percentages(tmp_path):
    # West half forest, east half non-forest; a fully-nodata block in the corner.
    array = numpy.full((SRC_N, SRC_N), NONFOREST, dtype='uint8')
    array[:, : SRC_N // 2] = FOREST
    array[:8, :8] = NODATA  # covers target cells [0,0],[0,1],[1,0],[1,1] entirely

    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        hashes = generate_landcover(src, [target], force=True)

    landcover = _landcover_set(target.directory)
    assert landcover.present()
    assert set(hashes) == {'t'}

    with rasterio.open(landcover.layer_path(FOREST_COVER)) as ds:
        forest_pct = ds.read(1)
        assert ds.dtypes[0] == 'uint8'
        assert ds.nodata == FOREST_PCT_NODATA

    # West-half cells are all-forest (100%), east-half all non-forest (0%).
    assert forest_pct[64, 20] == 100
    assert forest_pct[64, 100] == 0
    # The fully-nodata corner cell caught no valid pixels -> nodata, not 0%.
    assert forest_pct[0, 0] == FOREST_PCT_NODATA


def test_generate_computes_fractional_percent(tmp_path):
    # Every other source column is forest, so each 4-wide cell is 2/4 = 50% forest.
    array = numpy.full((SRC_N, SRC_N), NONFOREST, dtype='uint8')
    array[:, ::2] = FOREST

    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        generate_landcover(src, [target], force=True)

    with rasterio.open(_landcover_set(target.directory).layer_path(FOREST_COVER)) as ds:
        forest_pct = ds.read(1)

    assert (forest_pct[20:108, 20:108] == 50).all()


def test_generate_hash_is_one_stable_generation_id(tmp_path):
    array = numpy.full((SRC_N, SRC_N), FOREST, dtype='uint8')
    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        first = generate_landcover(src, [target], force=True)

    landcover = _landcover_set(target.directory)
    with rasterio.open(landcover.layer_path(FOREST_COVER)) as ds:
        forest_pct = ds.read(1)
    expected = versioned_hash(
        LANDCOVER_FORMAT_VERSION,
        hashlib.sha256(b't' + forest_pct.tobytes()).hexdigest(),
    )
    assert first['t'] == expected
    assert landcover.provenance_hash() == expected

    for layer in LANDCOVER_LAYERS:
        with rasterio.open(landcover.layer_path(layer)) as ds:
            assert ds.tags()[NLCD_HASH_TAG] == expected

    # Regenerating the same source is deterministic -> identical id.
    with rasterio.open(src_path) as src:
        second = generate_landcover(src, [target], force=True)
    assert second['t'] == expected


def test_generate_bins_into_multiple_grids_in_one_pass(tmp_path):
    array = numpy.full((SRC_N, SRC_N), NONFOREST, dtype='uint8')
    array[:, : SRC_N // 2] = FOREST
    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)

    fine = _target(tmp_path, 'fine')
    coarse = _target(tmp_path, 'coarse', px=80.0, n=64, tile=64)

    with rasterio.open(src_path) as src:
        hashes = generate_landcover(src, [fine, coarse], force=True)

    assert set(hashes) == {'fine', 'coarse'}
    for target in (fine, coarse):
        landcover = _landcover_set(target.directory)
        assert landcover.present()
        with rasterio.open(landcover.layer_path(FOREST_COVER)) as ds:
            forest_pct = ds.read(1)
        assert forest_pct[10, 5] == 100  # west half
        assert forest_pct[10, -5] == 0  # east half


def test_generate_refuses_to_overwrite_without_force(tmp_path):
    array = numpy.full((SRC_N, SRC_N), FOREST, dtype='uint8')
    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        generate_landcover(src, [target], force=True)
        with pytest.raises(FileExistsError, match='already has'):
            generate_landcover(src, [target], force=False)

"""The terrain-generation engine on a tiny, hand-checkable projected DEM.

A source that tilts uniformly up toward the east has a west-facing aspect
everywhere (water runs downhill to the west), so the majority class is W and the
orientation is purely eastness = -1. Working in EPSG:5070 (the engine's work CRS)
keeps the reprojection a near-identity, so the geometry is exact.
"""

import hashlib

import numpy
import pytest
import rasterio

from rasterio.crs import CRS

from snowtool.exceptions import SNODASWarning
from snowtool.snowdb.grid import make_grid
from snowtool.snowdb.terrain import (
    ASPECT_W,
    ELEVATION_NODATA,
    TerrainSet,
)
from snowtool.snowdb.terrain_generate import TerrainTarget, generate_terrain

WORK_EPSG = 5070
ORIGIN_X = -500_000.0
ORIGIN_Y = 2_000_000.0
SRC_PX = 10.0
SRC_N = 512
NODATA = -9999.0
# 128 cells x 40 m == 5120 m == the 512 x 10 m source extent (4 fine px / cell).
TARGET_N = 128
TARGET_PX = 40.0
TARGET_TILE = 128


def _source_dem(path, *, nodata=NODATA):
    """A DEM tilting up toward the east (elevation grows with column)."""
    cols = numpy.arange(SRC_N, dtype='float32')
    elevation = numpy.broadcast_to(cols * SRC_PX, (SRC_N, SRC_N)).copy()
    transform = rasterio.transform.from_origin(ORIGIN_X, ORIGIN_Y, SRC_PX, SRC_PX)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SRC_N,
        width=SRC_N,
        count=1,
        dtype='float32',
        crs=CRS.from_epsg(WORK_EPSG),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(elevation, 1)
    return path


def _int_source_dem(path, *, nodata=-32768):
    """An int16 DEM tilting east, with its western half set to nodata fill.

    Integer elevation rasters (e.g. SRTM/ASTER) are common; the western half is
    nodata so a regression can assert that fill is masked rather than aggregated
    as real elevation.
    """
    cols = numpy.arange(SRC_N, dtype='int16')
    elevation = numpy.broadcast_to(cols * int(SRC_PX), (SRC_N, SRC_N)).astype('int16')
    elevation[:, : SRC_N // 2] = nodata
    transform = rasterio.transform.from_origin(ORIGIN_X, ORIGIN_Y, SRC_PX, SRC_PX)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SRC_N,
        width=SRC_N,
        count=1,
        dtype='int16',
        crs=CRS.from_epsg(WORK_EPSG),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(elevation, 1)
    return path


def _target(tmp_path):
    grid = make_grid(
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        px_size=TARGET_PX,
        cols=TARGET_N,
        rows=TARGET_N,
        tile_size=TARGET_TILE,
        crs=WORK_EPSG,
    )
    return TerrainTarget(
        name='t',
        grid=grid,
        tile_size=TARGET_TILE,
        directory=tmp_path / 'terrain',
    )


def test_generate_writes_terrain_set_with_expected_orientation(tmp_path):
    src_path = _source_dem(tmp_path / 'src.tif')
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        hashes = generate_terrain(src, [target], force=True)

    terrain = TerrainSet(target.directory)
    assert terrain.present()
    assert set(hashes) == {'t'}

    with rasterio.open(terrain.elevation_path) as ds:
        elevation = ds.read(1)
    with rasterio.open(terrain.aspect_majority_path) as ds:
        majority = ds.read(1)
    with rasterio.open(terrain.aspect_components_path) as ds:
        northness = ds.read(1)
        eastness = ds.read(2)

    # Interior cells avoid the Horn/edge border that has no defined aspect.
    interior = (slice(20, 108), slice(20, 108))

    # Elevation grows west -> east.
    assert elevation[64, 20] < elevation[64, 100]

    # A surface rising to the east faces west everywhere inside.
    assert (majority[interior] == ASPECT_W).all()

    # West is aspect 270 deg: cos == 0 (no northness), sin == -1 (full eastness).
    assert numpy.abs(northness[interior]).max() < 0.05
    assert eastness[interior].max() < -0.95


def test_generate_hash_is_one_stable_generation_id(tmp_path):
    from snowtool.snowdb.constants import DEM_HASH_TAG
    from snowtool.snowdb.terrain import TERRAIN_LAYERS

    src_path = _source_dem(tmp_path / 'src.tif')
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        first = generate_terrain(src, [target], force=True)

    terrain = TerrainSet(target.directory)
    with rasterio.open(terrain.elevation_path) as ds:
        elevation = ds.read(1)
    # The id is a digest of name + finalized elevation (one per pass).
    expected = hashlib.sha256(b't' + elevation.tobytes()).hexdigest()
    assert first['t'] == expected
    assert terrain.dem_hash() == expected

    # Every layer of the set carries the same generation id.
    for layer in TERRAIN_LAYERS:
        with rasterio.open(terrain.layer_path(layer)) as ds:
            assert ds.tags()[DEM_HASH_TAG] == expected

    # Regenerating the same source is deterministic -> identical id.
    with rasterio.open(src_path) as src:
        second = generate_terrain(src, [target], force=True)
    assert second['t'] == expected


def test_generate_masks_nodata_from_integer_source(tmp_path):
    # Integer DEMs (SRTM/ASTER-style int16) must generate correctly end to end:
    # the warp's NaN nodata works on a float band even though the source is int,
    # so declared fill is masked, not aggregated as real elevation.
    src_path = _int_source_dem(tmp_path / 'src.tif')
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        generate_terrain(src, [target], force=True)

    with rasterio.open(TerrainSet(target.directory).elevation_path) as ds:
        elevation = ds.read(1)

    # West half was nodata fill -> those cells are nodata, not aggregated as 0.
    assert elevation[64, 20] == ELEVATION_NODATA
    # East half is real, finite, positive elevation.
    assert elevation[64, 110] != ELEVATION_NODATA
    assert numpy.isfinite(elevation[64, 110])
    assert elevation[64, 110] > 0


def test_generate_warns_when_source_declares_no_nodata(tmp_path):
    # A source with no declared nodata is trusted (all pixels valid) but warned
    # about, since an undeclared fill value would be aggregated as real data.
    src_path = _source_dem(tmp_path / 'src.tif', nodata=None)
    target = _target(tmp_path)

    with rasterio.open(src_path) as src, pytest.warns(SNODASWarning):
        generate_terrain(src, [target], force=True)


def test_generate_bins_into_multiple_grids_in_one_pass(tmp_path):
    src_path = _source_dem(tmp_path / 'src.tif')
    fine = _target(tmp_path / 'fine')
    coarse = TerrainTarget(
        name='coarse',
        grid=make_grid(
            origin_x=ORIGIN_X,
            origin_y=ORIGIN_Y,
            px_size=80.0,
            cols=64,
            rows=64,
            tile_size=64,
            crs=WORK_EPSG,
        ),
        tile_size=64,
        directory=tmp_path / 'coarse' / 'terrain',
    )

    with rasterio.open(src_path) as src:
        hashes = generate_terrain(src, [fine, coarse], force=True)

    assert set(hashes) == {'t', 'coarse'}
    # Both grids generated together share one generation id.
    assert hashes['t'] == hashes['coarse']
    for target in (fine, coarse):
        terrain = TerrainSet(target.directory)
        assert terrain.present()
        assert terrain.dem_hash() == hashes['t']
        with rasterio.open(terrain.aspect_majority_path) as ds:
            majority = ds.read(1)
        assert (majority[20:40, 20:40] == ASPECT_W).all()


def test_generate_auto_derives_resolution_when_none(tmp_path):
    # work_resolution=None lets GDAL derive the source's native resolution; the
    # 10 m 5070 source then yields the same west-facing terrain as the pinned 10 m.
    src_path = _source_dem(tmp_path / 'src.tif')
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        generate_terrain(src, [target], work_resolution=None, force=True)

    terrain = TerrainSet(target.directory)
    assert terrain.present()
    with rasterio.open(terrain.aspect_majority_path) as ds:
        majority = ds.read(1)
    assert (majority[20:108, 20:108] == ASPECT_W).all()


def test_generate_refuses_to_overwrite_without_force(tmp_path):
    import pytest

    src_path = _source_dem(tmp_path / 'src.tif')
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        generate_terrain(src, [target], force=True)
        with pytest.raises(FileExistsError, match='already has'):
            generate_terrain(src, [target], force=False)

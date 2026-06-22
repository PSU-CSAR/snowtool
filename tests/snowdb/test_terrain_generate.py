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
    ASPECT_COMPONENTS,
    ASPECT_MAJORITY,
    ASPECT_W,
    ELEVATION,
    ELEVATION_NODATA,
    TerrainProvider,
)
from snowtool.snowdb.terrain_generate import generate_terrain
from snowtool.snowdb.zone_layer import ZoneLayerTarget


def _terrain_set(directory):
    """The terrain ZoneLayerSet rooted at ``directory`` (test reader)."""
    return TerrainProvider().layer_set(directory)

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
    return ZoneLayerTarget(
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

    terrain = _terrain_set(target.directory)
    assert terrain.present()
    assert set(hashes) == {'t'}

    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        elevation = ds.read(1)
    with rasterio.open(terrain.layer_path(ASPECT_MAJORITY)) as ds:
        majority = ds.read(1)
    with rasterio.open(terrain.layer_path(ASPECT_COMPONENTS)) as ds:
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

    terrain = _terrain_set(target.directory)
    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        elevation = ds.read(1)
    # The id is a digest of name + finalized elevation (one per pass).
    expected = hashlib.sha256(b't' + elevation.tobytes()).hexdigest()
    assert first['t'] == expected
    assert terrain.provenance_hash() == expected

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

    with rasterio.open(_terrain_set(target.directory).layer_path(ELEVATION)) as ds:
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
    coarse = ZoneLayerTarget(
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
        terrain = _terrain_set(target.directory)
        assert terrain.present()
        assert terrain.provenance_hash() == hashes['t']
        with rasterio.open(terrain.layer_path(ASPECT_MAJORITY)) as ds:
            majority = ds.read(1)
        assert (majority[20:40, 20:40] == ASPECT_W).all()


def test_generate_auto_derives_resolution_when_none(tmp_path):
    # work_resolution=None lets GDAL derive the source's native resolution; the
    # 10 m 5070 source then yields the same west-facing terrain as the pinned 10 m.
    src_path = _source_dem(tmp_path / 'src.tif')
    target = _target(tmp_path)

    with rasterio.open(src_path) as src:
        generate_terrain(src, [target], work_resolution=None, force=True)

    terrain = _terrain_set(target.directory)
    assert terrain.present()
    with rasterio.open(terrain.layer_path(ASPECT_MAJORITY)) as ds:
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


def _read_layers(directory):
    terrain = _terrain_set(directory)
    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        elevation = ds.read(1)
    with rasterio.open(terrain.layer_path(ASPECT_MAJORITY)) as ds:
        majority = ds.read(1)
    with rasterio.open(terrain.layer_path(ASPECT_COMPONENTS)) as ds:
        components = ds.read()
    return elevation, majority, components


@pytest.mark.parametrize('workers', [2, 4, 8])
@pytest.mark.parametrize('block_size', [64, 100])
def test_parallel_matches_serial_bit_for_bit(tmp_path, workers, block_size):
    # The determinism guard: binning runs serially in block order regardless of
    # worker count, so a parallel pass must reproduce the serial pass exactly --
    # including the dem_hash, which digests the finalized elevation bytes. A small
    # block size makes this tiny source span many blocks so the parallel pipeline's
    # windowing/out-of-order completion is genuinely exercised; 100 is deliberately
    # a non-divisor of the work grid, exercising ragged edge blocks.
    src_path = _source_dem(tmp_path / 'src.tif')
    serial_t = _target(tmp_path / 'serial')
    parallel_t = _target(tmp_path / 'parallel')

    with rasterio.open(src_path) as src:
        serial_hash = generate_terrain(
            src, [serial_t], workers=1, block_size=block_size, force=True,
        )
    with rasterio.open(src_path) as src:
        parallel_hash = generate_terrain(
            src, [parallel_t], workers=workers, block_size=block_size, force=True,
        )

    assert serial_hash['t'] == parallel_hash['t']

    serial = _read_layers(serial_t.directory)
    parallel = _read_layers(parallel_t.directory)
    for s, p in zip(serial, parallel, strict=True):
        # assert_array_equal treats matching NaNs (the components nodata) as equal.
        numpy.testing.assert_array_equal(s, p)


def test_parallel_matches_serial_for_multiple_targets(tmp_path):
    # Multi-target pass: each block runs one transform per target, and the serial
    # reducer bins every target in the same fixed order. The shared generation
    # hash (over all targets, sorted by name) must still match the serial pass.
    block_size = 64
    src_path = _source_dem(tmp_path / 'src.tif')

    def _targets(root):
        fine = _target(root / 'fine')
        coarse = ZoneLayerTarget(
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
            directory=root / 'coarse' / 'terrain',
        )
        return fine, coarse

    serial_fine, serial_coarse = _targets(tmp_path / 'serial')
    parallel_fine, parallel_coarse = _targets(tmp_path / 'parallel')

    with rasterio.open(src_path) as src:
        serial_hash = generate_terrain(
            src, [serial_fine, serial_coarse], workers=1, block_size=block_size,
            force=True,
        )
    with rasterio.open(src_path) as src:
        parallel_hash = generate_terrain(
            src, [parallel_fine, parallel_coarse], workers=4, block_size=block_size,
            force=True,
        )

    assert serial_hash == parallel_hash
    for serial_t, parallel_t in (
        (serial_fine, parallel_fine),
        (serial_coarse, parallel_coarse),
    ):
        for s, p in zip(
            _read_layers(serial_t.directory),
            _read_layers(parallel_t.directory),
            strict=True,
        ):
            numpy.testing.assert_array_equal(s, p)


def test_effective_workers_defaults_and_honors_explicit_request():
    import os

    from snowtool.snowdb.terrain_generate import MAX_AUTO_WORKERS, _effective_workers

    # An explicit request is honoured as-is -- no silent memory override.
    assert _effective_workers(4) == 4
    assert _effective_workers(64) == 64
    # <= 1 is the serial path.
    assert _effective_workers(1) == 1
    assert _effective_workers(0) == 1
    # None means auto: one thread per CPU, but never more than the cap.
    auto = _effective_workers(None)
    assert auto == min(os.cpu_count() or 1, MAX_AUTO_WORKERS)
    assert 1 <= auto <= MAX_AUTO_WORKERS


def test_clip_grid_to_bounds_shrinks_to_footprint_on_lattice():
    from rasterio.transform import from_origin

    from snowtool.snowdb.terrain_generate import _clip_grid_to_bounds

    # A big 10000x10000 @ 10 m work grid; clip to a 1 km box in the middle.
    full = from_origin(0.0, 100_000.0, 10.0, 10.0)
    transform, width, height = _clip_grid_to_bounds(
        full, 10_000, 10_000, (40_000.0, 50_000.0, 41_000.0, 51_000.0),
    )

    # Same resolution, and the origin shifts by a whole number of pixels (the clip
    # never rephases the lattice -- that is what keeps the output identical).
    assert (transform.a, transform.e) == (full.a, full.e)
    col_off = (transform.c - full.c) / full.a
    row_off = (transform.f - full.f) / full.e
    assert abs(col_off - round(col_off)) < 1e-9
    assert abs(row_off - round(row_off)) < 1e-9

    # Far smaller than the full grid, but still covers the 1 km (= 100 px) box.
    assert 100 <= width < 200
    assert 100 <= height < 200
    inv = ~transform
    for x, y in ((40_000.0, 51_000.0), (41_000.0, 50_000.0)):
        c, r = inv * (x, y)
        assert 0 <= c <= width
        assert 0 <= r <= height


def test_clip_grid_to_bounds_none_when_disjoint():
    from rasterio.transform import from_origin

    from snowtool.snowdb.terrain_generate import _clip_grid_to_bounds

    full = from_origin(0.0, 100_000.0, 10.0, 10.0)
    # A box nowhere near the grid -> nothing to stream.
    assert _clip_grid_to_bounds(full, 100, 100, (1e6, 1e6, 1.1e6, 1.1e6)) is None


def test_generate_clips_to_subregion_target_of_larger_source(tmp_path):
    # A target covering only a sub-region of a larger source still gets correct,
    # fully-populated terrain there: the engine clips its work grid to the target
    # footprint (so only that part of the source is read), not the whole source.
    src_path = _source_dem(tmp_path / 'src.tif')  # 512 x 512 @ 10 m, tilts east
    # A 64-cell @ 40 m grid over the interior-east of the source (well inside it).
    grid = make_grid(
        origin_x=ORIGIN_X + 256 * SRC_PX,
        origin_y=ORIGIN_Y - 128 * SRC_PX,
        px_size=40.0,
        cols=64,
        rows=64,
        tile_size=64,
        crs=WORK_EPSG,
    )
    target = ZoneLayerTarget(
        name='sub', grid=grid, tile_size=64, directory=tmp_path / 'sub' / 'terrain',
    )

    with rasterio.open(src_path) as src:
        generate_terrain(src, [target], force=True)

    terrain = _terrain_set(target.directory)
    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        elevation = ds.read(1)
    with rasterio.open(terrain.layer_path(ASPECT_MAJORITY)) as ds:
        majority = ds.read(1)

    # Fully populated (no nodata) and west-facing throughout the interior.
    interior = (slice(10, 54), slice(10, 54))
    assert (elevation[interior] != ELEVATION_NODATA).all()
    assert numpy.isfinite(elevation[interior]).all()
    assert (majority[interior] == ASPECT_W).all()


def test_generate_into_modis_sinusoidal_target(tmp_path):
    # Every other test bins into EPSG:5070; this exercises a curved, WKT-only
    # projected target CRS (the instarr MODIS-Sinusoidal grid) end to end -- the
    # work-CRS -> target-CRS transform, the clip footprint, and the COG write -- and
    # confirms parallel still matches serial there.
    from rasterio.warp import transform_bounds

    from snowtool.snowdb.datasets.instarr import MODIS_SINUSOIDAL_WKT
    from snowtool.snowdb.grid import grid_extent_4326

    # A small sinusoidal target inside the instarr coverage (~Colorado).
    sin_grid = make_grid(
        origin_x=-9_200_000.0,
        origin_y=4_420_000.0,
        px_size=463.3127,
        cols=96,
        rows=96,
        tile_size=96,
        crs=MODIS_SINUSOIDAL_WKT,
    )
    w, s, e, n = grid_extent_4326(sin_grid)

    # A 5070 DEM covering that area (+margin), tilting east -> west-facing aspect.
    sw, ss, se, sn = transform_bounds('EPSG:4326', 'EPSG:5070', w, s, e, n)
    pad, res = 3000.0, 30.0
    sw, ss, se, sn = sw - pad, ss - pad, se + pad, sn + pad
    width, height = int((se - sw) / res), int((sn - ss) / res)
    elevation = numpy.broadcast_to(
        numpy.arange(width, dtype='float32') * res, (height, width),
    ).copy()
    src_path = tmp_path / 'src.tif'
    with rasterio.open(
        src_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=CRS.from_epsg(5070),
        transform=rasterio.transform.from_origin(sw, sn, res, res),
        nodata=NODATA,
    ) as dst:
        dst.write(elevation, 1)

    def _run(directory, workers):
        target = ZoneLayerTarget(
            name='sin', grid=sin_grid, tile_size=96, directory=directory / 'terrain',
        )
        with rasterio.open(src_path) as src:
            return generate_terrain(
                src, [target], work_resolution=res, workers=workers, force=True,
            )

    serial = _run(tmp_path / 's', 1)
    parallel = _run(tmp_path / 'p', 4)
    assert serial['sin'] == parallel['sin']  # determinism holds for a WKT CRS too

    terrain = _terrain_set(tmp_path / 's' / 'terrain')
    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        assert 'Sinusoidal' in ds.crs.to_wkt()  # written with the target's WKT CRS
        elevation_out = ds.read(1)
    with rasterio.open(terrain.layer_path(ASPECT_MAJORITY)) as ds:
        majority = ds.read(1)

    interior = (slice(30, 66), slice(30, 66))
    assert numpy.isfinite(elevation_out[interior]).all()
    assert (elevation_out[interior] != ELEVATION_NODATA).all()
    assert (majority[interior] == ASPECT_W).all()


def test_generate_writes_nodata_when_target_disjoint_from_source(tmp_path):
    # A target that doesn't overlap the source exercises the clip's no-overlap path
    # (_clip_grid_to_bounds -> None): the engine skips streaming, finalizes empty
    # accumulators, and still writes a valid all-nodata terrain set (no error).
    from snowtool.snowdb.terrain import ASPECT_MAJORITY_NODATA

    src_path = _source_dem(tmp_path / 'src.tif')
    far = make_grid(
        origin_x=ORIGIN_X + 10_000_000.0,
        origin_y=ORIGIN_Y,
        px_size=TARGET_PX,
        cols=TARGET_N,
        rows=TARGET_N,
        tile_size=TARGET_TILE,
        crs=WORK_EPSG,
    )
    target = ZoneLayerTarget(
        name='far',
        grid=far,
        tile_size=TARGET_TILE,
        directory=tmp_path / 'far' / 'terrain',
    )

    with rasterio.open(src_path) as src:
        hashes = generate_terrain(src, [target], force=True)

    assert set(hashes) == {'far'}
    terrain = _terrain_set(target.directory)
    assert terrain.present()
    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        elevation = ds.read(1)
    with rasterio.open(terrain.layer_path(ASPECT_MAJORITY)) as ds:
        majority = ds.read(1)
    # No source pixel fell in any cell -> everything nodata.
    assert (elevation == ELEVATION_NODATA).all()
    assert (majority == ASPECT_MAJORITY_NODATA).all()

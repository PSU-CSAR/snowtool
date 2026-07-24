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

from snowtool.exceptions import ArtifactExistsError
from snowtool.snowdb.constants import FOREST_PCT_NODATA, NLCD_HASH_TAG
from snowtool.snowdb.grid import grid_extent_4326
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.zones.landcover import LandCoverProvider
from snowtool.snowdb.zones.landcover_generate import generate_landcover
from snowtool.snowdb.zones.landcover_layers import (
    FOREST_COVER,
    LANDCOVER_FORMAT_VERSION,
    LANDCOVER_LAYERS,
)
from snowtool.snowdb.zones.landcover_source import LocalFile

from ._engine_harness import (
    ORIGIN_X,
    ORIGIN_Y,
    SRC_N,
    SRC_PX,
    WORK_EPSG,
    make_target,
    run_serial_vs_parallel,
)


def _landcover_set(directory):
    """The land-cover ZoneLayerSet rooted at ``directory`` (test reader)."""
    return LandCoverProvider().layer_set(directory)


FOREST = 42  # evergreen forest (in FOREST_CLASSES)
NONFOREST = 81  # pasture/hay
NODATA = 0  # NLCD background/unclassified


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


def _target(tmp_path, name='t', **kwargs):
    return make_target(tmp_path / name / 'landcover', name=name, **kwargs)


def _bounds(*targets):
    """The combined EPSG:4326 extent of ``targets`` -- the ``bounds`` the engine
    passes to ``source.open`` (a ``LocalFile`` ignores it and reads the whole file,
    but the engine still reads only the window over the target footprints)."""
    boxes = [grid_extent_4326(t.grid) for t in targets]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def test_generate_writes_forest_layer_with_expected_percentages(tmp_path):
    # West half forest, east half non-forest; a fully-nodata block in the corner.
    array = numpy.full((SRC_N, SRC_N), NONFOREST, dtype='uint8')
    array[:, : SRC_N // 2] = FOREST
    array[:8, :8] = NODATA  # covers target cells [0,0],[0,1],[1,0],[1,1] entirely

    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    target = _target(tmp_path)

    hashes = generate_landcover(
        LocalFile(src_path),
        [target],
        _bounds(target),
        force=True,
    )

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

    generate_landcover(LocalFile(src_path), [target], _bounds(target), force=True)

    with rasterio.open(_landcover_set(target.directory).layer_path(FOREST_COVER)) as ds:
        forest_pct = ds.read(1)

    assert (forest_pct[20:108, 20:108] == 50).all()


def test_generate_hash_is_one_stable_generation_id(tmp_path):
    array = numpy.full((SRC_N, SRC_N), FOREST, dtype='uint8')
    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    target = _target(tmp_path)

    first = generate_landcover(
        LocalFile(src_path),
        [target],
        _bounds(target),
        force=True,
    )

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
    second = generate_landcover(
        LocalFile(src_path),
        [target],
        _bounds(target),
        force=True,
    )
    assert second['t'] == expected


def test_generate_bins_into_multiple_grids_in_one_pass(tmp_path):
    array = numpy.full((SRC_N, SRC_N), NONFOREST, dtype='uint8')
    array[:, : SRC_N // 2] = FOREST
    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)

    fine = _target(tmp_path, 'fine')
    coarse = _target(tmp_path, 'coarse', px=80.0, n=64, tile=64)

    hashes = generate_landcover(
        LocalFile(src_path),
        [fine, coarse],
        _bounds(fine, coarse),
        force=True,
    )

    assert set(hashes) == {'fine', 'coarse'}
    for target in (fine, coarse):
        landcover = _landcover_set(target.directory)
        assert landcover.present()
        with rasterio.open(landcover.layer_path(FOREST_COVER)) as ds:
            forest_pct = ds.read(1)
        assert forest_pct[10, 5] == 100  # west half
        assert forest_pct[10, -5] == 0  # east half


@pytest.mark.parametrize('workers', [2, 4])
@pytest.mark.parametrize('block_size', [64, 96])
def test_landcover_parallel_matches_serial_bit_for_bit(tmp_path, workers, block_size):
    # The determinism guard: binning runs serially in block order regardless of
    # worker count, so a parallel pass must reproduce the serial pass exactly --
    # including the nlcd_hash, which digests the finalized forest-pct bytes. A
    # small block size makes this source span many blocks so the parallel
    # pipeline's windowing/out-of-order completion is genuinely exercised. 64
    # divides the 512-px source evenly; 96 does not (512 = 5*96 + 32), so its
    # trailing blocks are ragged and exercise iter_blocks' clamp path -- which
    # must still be bit-identical between serial and parallel.
    array = numpy.full((SRC_N, SRC_N), NONFOREST, dtype='uint8')
    array[:, : SRC_N // 2] = FOREST
    array[:8, :8] = NODATA
    array[::2, ::3] = FOREST  # scatter so cells hold mixed fractional values
    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    serial_t = _target(tmp_path / 'serial')
    parallel_t = _target(tmp_path / 'parallel')

    def _read_forest(directory):
        with rasterio.open(
            _landcover_set(directory).layer_path(FOREST_COVER),
        ) as ds:
            return (ds.read(1),)

    run_serial_vs_parallel(
        generate_landcover,
        LocalFile(src_path),
        serial_t,
        parallel_t,
        _read_forest,
        workers=workers,
        block_size=block_size,
    )


def test_generate_refuses_to_overwrite_without_force(tmp_path):
    array = numpy.full((SRC_N, SRC_N), FOREST, dtype='uint8')
    src_path = _source_nlcd(tmp_path / 'nlcd.tif', array)
    target = _target(tmp_path)

    generate_landcover(LocalFile(src_path), [target], _bounds(target), force=True)
    with pytest.raises(ArtifactExistsError, match='already has'):
        generate_landcover(LocalFile(src_path), [target], _bounds(target), force=False)

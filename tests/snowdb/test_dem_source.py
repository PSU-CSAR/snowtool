"""DEM sources: LocalFile, the osgeo-free VRT mosaic, and 3DEP tile enumeration."""

import numpy
import pytest
import rasterio

from rasterio.crs import CRS

from snowtool.snowdb.dem_source import (
    LocalFile,
    ThreeDEP,
    build_mosaic_vrt,
    candidate_tiles,
    existing_tile_keys,
)


def _tile(path, x0, y0, value, n=4, px=1.0):
    transform = rasterio.transform.from_origin(x0, y0, px, px)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=n,
        width=n,
        count=1,
        dtype='float32',
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(numpy.full((n, n), value, dtype='float32'), 1)
    return path


def test_local_file_open_yields_the_dataset(tmp_path):
    path = _tile(tmp_path / 'dem.tif', 0.0, 4.0, 7.0)
    with LocalFile(path).open((0, 0, 1, 1)) as src:
        assert src.read(1)[0, 0] == 7.0


def test_work_grid_defaults_and_overrides(tmp_path):
    from snowtool.snowdb.terrain_generate import (
        DEFAULT_WORK_CRS,
        DEFAULT_WORK_RESOLUTION,
    )

    path = _tile(tmp_path / 'dem.tif', 0.0, 4.0, 7.0)

    # 3DEP pins its native 10 m CONUS-Albers grid.
    threedep = ThreeDEP()
    assert threedep.work_crs == DEFAULT_WORK_CRS
    assert threedep.work_resolution == DEFAULT_WORK_RESOLUTION

    # A local DEM defaults to its own native resolution (None -> GDAL derives it).
    local = LocalFile(path)
    assert local.work_resolution is None

    # Both are overridable per source (e.g. 1 m lidar in a non-CONUS CRS).
    lidar = LocalFile(path, work_crs='EPSG:32611', work_resolution=1.0)
    assert (lidar.work_crs, lidar.work_resolution) == ('EPSG:32611', 1.0)


def test_candidate_tiles_names_nw_corner_tiles():
    # west of -107..-105, north of 39..41 -> the four 1-degree tiles between.
    tiles = candidate_tiles((-106.5, 39.2, -105.1, 40.3))
    assert tiles == ['n40w106', 'n40w107', 'n41w106', 'n41w107']


def test_build_mosaic_vrt_stitches_adjacent_tiles(tmp_path):
    left = _tile(tmp_path / 'a.tif', 0.0, 4.0, 1.0)
    right = _tile(tmp_path / 'b.tif', 4.0, 4.0, 2.0)

    vrt = build_mosaic_vrt([left, right], tmp_path / 'm.vrt')

    with rasterio.open(vrt) as ds:
        assert (ds.width, ds.height) == (8, 4)
        data = ds.read(1)
    assert (data[:, :4] == 1.0).all()
    assert (data[:, 4:] == 2.0).all()


def test_existing_tile_keys_keeps_only_present_tiles(monkeypatch):
    class _FakeS3:
        def head_object(self, Bucket, Key):  # noqa: N803 - boto3 kwarg name
            if 'n40w107' not in Key:
                raise RuntimeError('NoSuchKey')

    monkeypatch.setattr('boto3.client', lambda *a, **k: _FakeS3())

    keys = existing_tile_keys((-106.5, 39.2, -105.1, 40.3))

    assert len(keys) == 1
    assert keys[0].endswith('n40w107/USGS_13_n40w107.tif')


def test_threedep_open_builds_a_vrt_over_existing_tiles(tmp_path, monkeypatch):
    # Two real local tiles stand in for the remote COGs the enumeration returns.
    left = _tile(tmp_path / 'a.tif', 0.0, 4.0, 1.0)
    right = _tile(tmp_path / 'b.tif', 4.0, 4.0, 2.0)
    # The enumeration's keys are turned into source URIs verbatim here.
    monkeypatch.setattr(
        ThreeDEP,
        '_tile_uris',
        lambda self, bounds: [str(left), str(right)],
    )

    with ThreeDEP().open((0.0, 0.0, 8.0, 4.0)) as src:
        assert (src.width, src.height) == (8, 4)
        assert src.read(1)[0, 0] == 1.0


def test_threedep_open_errors_when_no_tiles(monkeypatch):
    monkeypatch.setattr(ThreeDEP, '_tile_uris', lambda self, bounds: [])
    with pytest.raises(RuntimeError, match='No 3DEP tiles'), ThreeDEP().open(
        (0.0, 0.0, 1.0, 1.0),
    ):
        pass

"""DEM sources: LocalFile, the osgeo-free VRT mosaic, and 3DEP tile discovery."""

import asyncio
import types

import numpy
import pytest
import rasterio

from async_tiff.enums import SampleFormat
from async_tiff.store import LocalStore
from rasterio.crs import CRS

from snowtool.exceptions import RemoteSourceError
from snowtool.snowdb.zones import terrain_source
from snowtool.snowdb.zones.terrain_generate import (
    DEFAULT_WORK_CRS,
    DEFAULT_WORK_RESOLUTION,
)
from snowtool.snowdb.zones.terrain_source import (
    TIFF,
    LocalFile,
    MosaicTile,
    ThreeDEP,
    _parse_geo_header,
    build_mosaic_vrt,
    candidate_tiles,
    discover_tiles,
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


def _parse_local(path) -> MosaicTile:
    """Parse a real local GeoTIFF through the production header-parse path."""

    async def _run():
        store = LocalStore(prefix=str(path.parent))
        tiff = await TIFF.open(path.name, store=store)
        return _parse_geo_header(tiff.ifd(0), str(path))

    return asyncio.run(_run())


def test_local_file_open_yields_the_dataset(tmp_path):
    path = _tile(tmp_path / 'dem.tif', 0.0, 4.0, 7.0)
    with LocalFile(path).open((0, 0, 1, 1)) as src:
        assert src.read(1)[0, 0] == 7.0


def test_work_grid_defaults_and_overrides(tmp_path):
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


def test_parse_geo_header_reads_the_tile_box(tmp_path):
    # The geo-header is read straight from the GeoTIFF's IFD -- no pixel reads.
    path = _tile(tmp_path / 'a.tif', 0.0, 4.0, 1.0)

    tile = _parse_local(path)

    assert (tile.origin_x, tile.origin_y) == (0.0, 4.0)
    assert (tile.px, tile.py) == (1.0, -1.0)
    assert (tile.width, tile.height) == (4, 4)
    assert tile.dtype == 'float32'
    assert tile.nodata == -9999.0
    assert CRS.from_wkt(tile.crs_wkt) == CRS.from_epsg(4326)


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        ({'model_pixel_scale': None}, 'north-up GeoTIFF'),
        ({'model_tiepoint': None}, 'north-up GeoTIFF'),
        ({'geo_key_directory': None}, 'north-up GeoTIFF'),
        (
            {
                'geo_key_directory': types.SimpleNamespace(
                    projected_type=None,
                    geographic_type=None,
                ),
            },
            'no projected or geographic CRS',
        ),
    ],
)
def test_parse_geo_header_rejects_missing_georeferencing(mutate, message):
    # The IFD georeferencing tags are all optional; a tile missing one is a clean
    # RemoteSourceError (the operator-facing remote-data taxonomy), not an opaque
    # None deref while building the transform/CRS.
    ifd = _fake_ifd()
    for key, value in mutate.items():
        setattr(ifd, key, value)

    with pytest.raises(RemoteSourceError, match=message):
        _parse_geo_header(ifd, '/vsis3/bucket/broken.tif')


def test_build_mosaic_vrt_stitches_adjacent_tiles(tmp_path):
    # Real tiles parsed through discovery's header path, then assembled with no
    # further reads -- the VRT places each by its derived geometry.
    left = _parse_local(_tile(tmp_path / 'a.tif', 0.0, 4.0, 1.0))
    right = _parse_local(_tile(tmp_path / 'b.tif', 4.0, 4.0, 2.0))

    vrt = build_mosaic_vrt([left, right], tmp_path / 'm.vrt')

    with rasterio.open(vrt) as ds:
        assert (ds.width, ds.height) == (8, 4)
        data = ds.read(1)
    assert (data[:, :4] == 1.0).all()
    assert (data[:, 4:] == 2.0).all()


def test_build_mosaic_vrt_refuses_heterogeneous_tiles(tmp_path):
    # The VRT header is taken wholesale from tile 0, so a tile that disagrees on
    # resolution (here a coarser pixel size) would be silently mis-placed. Refuse
    # loudly instead, naming which tile disagrees with tile 0.
    tile0 = _parse_local(_tile(tmp_path / 'a.tif', 0.0, 4.0, 1.0, px=1.0))
    tile1 = _parse_local(_tile(tmp_path / 'b.tif', 4.0, 4.0, 2.0, px=2.0))

    with pytest.raises(RemoteSourceError, match='heterogeneous tiles'):
        build_mosaic_vrt([tile0, tile1], tmp_path / 'm.vrt')


def test_threedep_open_builds_a_vrt_over_discovered_tiles(tmp_path, monkeypatch):
    # Two real local tiles stand in for the remote COGs discovery returns.
    left = _parse_local(_tile(tmp_path / 'a.tif', 0.0, 4.0, 1.0))
    right = _parse_local(_tile(tmp_path / 'b.tif', 4.0, 4.0, 2.0))
    monkeypatch.setattr(
        terrain_source,
        'discover_tiles',
        lambda bounds: [left, right],
    )

    with ThreeDEP().open((0.0, 0.0, 8.0, 4.0)) as src:
        assert (src.width, src.height) == (8, 4)
        assert src.read(1)[0, 0] == 1.0


# --- Discovery: existence + the review-#2 error-taxonomy guards ---------------


class _TransientError(Exception):
    """Stand-in for a non-not-found S3 failure (throttling / 5xx / network)."""


def _fake_ifd():
    return types.SimpleNamespace(
        model_pixel_scale=[1.0, 1.0, 0.0],
        model_tiepoint=[0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
        sample_format=[SampleFormat.Float],
        bits_per_sample=[32],
        gdal_nodata='-9999',
        geo_key_directory=types.SimpleNamespace(
            projected_type=None,
            geographic_type=4326,
        ),
        image_width=4,
        image_height=4,
    )


def _fake_tiff(open_fn):
    return types.SimpleNamespace(open=open_fn)


def test_discover_tiles_keeps_only_present_tiles(monkeypatch):
    # A genuine 404 (missing key -> FileNotFoundError) means "not published" and
    # is dropped; the existing tile is kept.
    async def _open(key, store):
        if 'n40w107' in key:
            return types.SimpleNamespace(ifd=lambda i: _fake_ifd())
        raise FileNotFoundError(key)

    monkeypatch.setattr(terrain_source, 'TIFF', _fake_tiff(_open))

    tiles = discover_tiles((-106.5, 39.2, -105.1, 40.3))

    assert len(tiles) == 1
    assert tiles[0].uri.endswith('n40w107/USGS_13_n40w107.tif')


def test_discover_tiles_surfaces_non_not_found_errors(monkeypatch):
    # A transient failure must surface, not be swallowed as "tile absent" --
    # otherwise the mosaic silently loses a real tile.
    async def _open(key, store):
        raise _TransientError('503')

    monkeypatch.setattr(terrain_source, 'TIFF', _fake_tiff(_open))

    with pytest.raises(_TransientError):
        discover_tiles((-106.5, 39.2, -105.1, 40.3))


def test_discover_tiles_probes_concurrently_and_sorts(monkeypatch):
    # All probes are in flight at once (one concurrent pass), and the result is
    # sorted by URI regardless of completion order.
    state = {'in_flight': 0, 'peak': 0}

    async def _open(key, store):
        state['in_flight'] += 1
        state['peak'] = max(state['peak'], state['in_flight'])
        await asyncio.sleep(0.01)
        state['in_flight'] -= 1
        return types.SimpleNamespace(ifd=lambda i: _fake_ifd())

    monkeypatch.setattr(terrain_source, 'TIFF', _fake_tiff(_open))

    # Four candidate tiles for this extent.
    tiles = discover_tiles((-106.5, 39.2, -105.1, 40.3))

    assert state['peak'] == 4
    uris = [t.uri for t in tiles]
    assert uris == sorted(uris)


def test_threedep_open_errors_when_no_tiles(monkeypatch):
    # Every candidate 404s -> discovery is empty -> ThreeDEP.open refuses.
    async def _open(key, store):
        raise FileNotFoundError(key)

    monkeypatch.setattr(terrain_source, 'TIFF', _fake_tiff(_open))

    with (
        pytest.raises(RemoteSourceError, match='No 3DEP tiles'),
        ThreeDEP().open(
            (0.0, 0.0, 1.0, 1.0),
        ),
    ):
        pass

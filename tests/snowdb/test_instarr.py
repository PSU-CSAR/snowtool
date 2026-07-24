"""INSTARR (SPIRES NRT) dataset definition + native-sinusoidal mosaic ingest.

The mosaic write path is covered with small synthetic GeoTIFF "tiles" (no NetCDF
fixture needed -- InstarrMosaicRaster takes GDAL source URIs, the ingester builds
the ``netcdf:`` ones); the ingester's tile-grouping is covered by driving
``spec.ingester.plan`` directly and asserting on the yielded ``DateIngest`` / built
rasters. The real multi-tile NetCDF mosaic is verified bit-exact against actual
SPIRES tiles during development.
"""

from datetime import date

import numpy
import pytest
import rasterio

from rasterio.transform import from_origin

from snowtool.exceptions import IngestSourceError, SnowtoolError
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS
from snowtool.snowdb.datasets.instarr import (
    INSTARR_SPEC,
    INSTARR_VARIABLES,
    InstarrMosaicRaster,
)
from snowtool.snowdb.spec import GridParams


def test_spec_is_native_modis_sinusoidal():
    spec = INSTARR_SPEC
    assert spec.name == 'instarr'
    # h08-h10 (3 wide) x v04-v05 (2 tall), 2400 px per tile.
    assert spec.grid_params.cols == 7200
    assert spec.grid_params.rows == 4800
    # 512-cell tiles (vs 256 on the ~925 m geographic grids) so the ~463 m cells
    # give a comparable ground footprint per read window.
    assert spec.grid_params.tile_size == 512
    # Projected (sinusoidal, no EPSG) -> constant cell area, burned into AOIs.
    assert spec.crs.to_epsg() is None
    # Identify the projection without to_dict()/to_proj4() (lossy, and warns).
    assert spec.crs.coordinate_operation.method_name == 'Sinusoidal'
    assert spec.is_geographic is False
    assert spec.cell_area == pytest.approx(463.3127165693847**2)
    assert spec.ingester is not None


def test_all_nine_variables():
    by_key = {v.key: v for v in INSTARR_VARIABLES}
    assert set(by_key) == {
        'snow_fraction',
        'viewable_snow_fraction',
        'albedo_dirty_flat',
        'albedo_dirty_terrain_corrected',
        'deltavis',
        'grain_size',
        'dust_concentration',
        'snow_cover_duration',
        'radiative_forcing',
    }
    for variable in INSTARR_VARIABLES:
        assert variable.reducer.value == 'mean'
        assert variable.glob == f'*__{variable.key}.tif'
    # uint8 (%, nodata 255) vs uint16 (nodata 65535) split.
    assert by_key['snow_fraction'].dtype == 'uint8'
    assert by_key['snow_fraction'].nodata == 255.0
    assert by_key['radiative_forcing'].dtype == 'uint16'
    assert by_key['radiative_forcing'].nodata == 65535.0


def test_registered_in_default_specs():
    assert 'instarr' in {s.name for s in DEFAULT_DATASET_SPECS}


def test_ingest_empty_source_raises(tmp_path):
    ds = Dataset(INSTARR_SPEC, tmp_path)
    with pytest.raises(SnowtoolError, match='No SPIRES NRT tiles'):
        ds.ingest(tmp_path)


def test_ingest_refuses_regex_failing_tile(tmp_path):
    # A file the glob claims (SPIRES_NRT_*.nc) but the regex cannot parse must
    # raise -- silently dropping it would let a malformed tile go unnoticed.
    valid = tmp_path / 'h08v04/2026/06/SPIRES_NRT_h08v04_MOD09GA061_20260613_V1.0.nc'
    malformed = tmp_path / 'h08v04/2026/06/SPIRES_NRT_garbage.nc'
    for path in (valid, malformed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    ds = Dataset(INSTARR_SPEC, tmp_path)
    with pytest.raises(IngestSourceError) as exc:
        ds.ingest(tmp_path)
    assert 'SPIRES_NRT_garbage.nc' in str(exc.value)


def test_ingest_refuses_out_of_block_tile(tmp_path):
    # h06 is outside the configured h08-h10 block. Placement is by computed
    # offset, so an out-of-block tile would wrap to a negative slice and land
    # in the wrong grid slot; the guard makes read_array's "on-grid by
    # construction" claim true.
    out_of_block = (
        tmp_path / 'h06v04/2026/06/SPIRES_NRT_h06v04_MOD09GA061_20260615_V1.0.nc'
    )
    out_of_block.parent.mkdir(parents=True, exist_ok=True)
    out_of_block.touch()

    ds = Dataset(INSTARR_SPEC, tmp_path)
    with pytest.raises(IngestSourceError, match='outside the configured grid block'):
        ds.ingest(tmp_path)


def test_ingest_groups_tiles_by_date(tmp_path):
    # Two dates, with two and one tiles respectively; plan groups them and yields one
    # DateIngest per date, whose build_rasters produces one per-variable raster
    # (source never opened). Drive the sanctioned plan seam directly.
    layout = {
        'h08v04/2026/06/SPIRES_NRT_h08v04_MOD09GA061_20260613_V1.0.nc',
        'h09v04/2026/06/SPIRES_NRT_h09v04_MOD09GA061_20260613_V1.0.nc',
        'h08v04/2026/06/SPIRES_NRT_h08v04_MOD09GA061_20260614_V1.0.nc',
    }
    for rel in layout:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    ds = Dataset(INSTARR_SPEC, tmp_path)
    items = list(INSTARR_SPEC.ingester.plan(tmp_path, ds))
    assert [item.date for item in items] == [date(2026, 6, 13), date(2026, 6, 14)]

    by_date = {item.date: item for item in items}
    # build_rasters is deferred and hash-bound; invoke as the driver would. The
    # declared out_names match what build_rasters produces.
    day13 = list(by_date[date(2026, 6, 13)].build_rasters('v1:deadbeef'))
    assert set(by_date[date(2026, 6, 13)].out_names) == {r.out_name for r in day13}
    day14 = list(by_date[date(2026, 6, 14)].build_rasters('v1:deadbeef'))

    # One raster per variable, each over the right number of tile source URIs.
    assert len(day13) == len(INSTARR_VARIABLES)
    swe_like = next(r for r in day13 if r.variable.key == 'snow_fraction')
    assert len(swe_like.source_uris) == 2
    assert all(uri.endswith(':snow_fraction') for uri in swe_like.source_uris)
    assert len(day14[0].source_uris) == 1
    # Distilled provenance name (per-tile h##v## dropped) + tags with the tiles.
    assert swe_like.out_name == 'SPIRES_NRT_MOD09GA061_20260613_V1.0__snow_fraction.tif'
    assert swe_like.tags['SOURCE_FILES'] == ' '.join(
        sorted(
            [
                'SPIRES_NRT_h08v04_MOD09GA061_20260613_V1.0.nc',
                'SPIRES_NRT_h09v04_MOD09GA061_20260613_V1.0.nc',
            ],
        ),
    )
    assert swe_like.tags['SOURCE_VERSION'] == 'V1.0'


def _sinusoidal_tile(path, value, *, left, top, size, px):
    array = numpy.full((size, size), value, dtype='uint8')
    transform = from_origin(left, top, px, px)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=1,
        dtype='uint8',
        crs=rasterio.crs.CRS.from_wkt(INSTARR_SPEC.crs.to_wkt()),
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(array, 1)


def test_mosaic_places_tiles_by_origin_and_leaves_gaps_nodata(tmp_path):
    # A 2x2 tile grid (tile=4px) with only two diagonal tiles present; each lands
    # in its slot by geographic origin, the other two quadrants stay nodata.
    px = 463.3127165693847
    tile_px = 16  # COG blocksize must be a multiple of 16
    origin_x, origin_y = 0.0, 0.0
    grid = GridParams(
        origin_x=origin_x,
        origin_y=origin_y,
        px_size=px,
        cols=2 * tile_px,
        rows=2 * tile_px,
        tile_size=tile_px,
        crs=INSTARR_SPEC.grid_params.crs,
    )

    # top-left tile (value 10) and bottom-right tile (value 40)
    tl = tmp_path / 'tl.tif'
    br = tmp_path / 'br.tif'
    _sinusoidal_tile(tl, 10, left=origin_x, top=origin_y, size=tile_px, px=px)
    _sinusoidal_tile(
        br,
        40,
        left=origin_x + tile_px * px,
        top=origin_y - tile_px * px,
        size=tile_px,
        px=px,
    )

    variable = INSTARR_SPEC.variables['snow_fraction']
    raster = InstarrMosaicRaster(
        variable,
        [str(tl), str(br)],
        grid,
        out_name='snow_fraction.tif',
        transform=rasterio.transform.from_origin(origin_x, origin_y, px, px),
        crs=rasterio.crs.CRS.from_wkt(INSTARR_SPEC.crs.to_wkt()),
        tags={'SOURCE_DATASET': 'instarr', 'SOURCE_VARIABLE': 'snow_fraction'},
    )
    out_dir = tmp_path / 'cogs'
    out_dir.mkdir()
    raster.write_cog(out_dir)

    with rasterio.open(out_dir / 'snow_fraction.tif') as cog:
        mosaic = cog.read(1)
        assert cog.nodata == 255.0
        assert cog.tags()['SOURCE_DATASET'] == 'instarr'
        assert cog.tags()['SOURCE_VARIABLE'] == 'snow_fraction'
    # top-left quadrant == 10, bottom-right == 40, the other two all nodata.
    assert (mosaic[:tile_px, :tile_px] == 10).all()
    assert (mosaic[tile_px:, tile_px:] == 40).all()
    assert (mosaic[:tile_px, tile_px:] == 255).all()
    assert (mosaic[tile_px:, :tile_px] == 255).all()

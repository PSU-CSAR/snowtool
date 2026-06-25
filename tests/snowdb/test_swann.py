"""SWANN 800m dataset definition + ingest.

The grid/variable literals are checked against the values read from a real product
file; the ingest is covered in two halves that need no NetCDF fixture (the bundled
GDAL is read-only for NetCDF):

  * the ingester's orchestration -- date parsing and one grid-aligned raster per
    variable -- via a captured ``write_date_cogs`` (the source is never opened), and
  * the raster's write side -- read a band, write it as a grid COG -- against a
    plain GeoTIFF source (SwannRaster takes any GDAL URI, so the NetCDF scheme is
    not needed to exercise the write path).

The end-to-end NetCDF read (GDAL returning the array north-up) is verified against
a real product file during development; it is GDAL's contract, not our code.
"""

from datetime import date

import numpy
import pytest
import rasterio

from rasterio.transform import from_origin

from snowtool.exceptions import SNODASError
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS
from snowtool.snowdb.datasets.swann import (
    SWANN_800M_SPEC,
    SWANN_800M_VARIABLES,
    SwannRaster,
)


def test_spec_grid_matches_product_file():
    spec = SWANN_800M_SPEC
    assert spec.name == 'swann-800m'
    assert spec.grid_params.cols == 7025
    assert spec.grid_params.rows == 3105
    assert spec.crs.to_epsg() == 4269
    # NAD83 is geographic -> AOI rasters burn per-row geodesic cell area.
    assert spec.is_geographic is True
    assert spec.ingester is not None
    # Distinct response-model prefix -> no OpenAPI schema collision with snodas.
    assert spec.model_prefix == 'Swann800m'


def test_variables():
    by_key = {v.key: v for v in SWANN_800M_VARIABLES}
    assert set(by_key) == {'swe', 'depth'}
    for variable in SWANN_800M_VARIABLES:
        assert variable.dtype == 'int16'
        assert variable.nodata == -999.0
        assert variable.reducer.value == 'mean'
        assert variable.unit.name == 'mm'
    # glob is the literal COG filename ingest writes for each variable.
    assert by_key['swe'].glob == 'swe.tif'
    assert by_key['depth'].glob == 'depth.tif'


def test_registered_in_default_specs():
    assert {'snodas', 'swann-800m'} <= {s.name for s in DEFAULT_DATASET_SPECS}


def test_ingest_rejects_unrecognized_filename(tmp_path):
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    with pytest.raises(SNODASError, match='Not a SWANN 800m file'):
        ds.ingest(tmp_path / 'some_other_file.nc')


def test_ingest_accepts_the_early_stage(tmp_path, monkeypatch):
    # Ingest is pinned to the `_early` revision (fastest available); it is the
    # one stage accepted.
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    monkeypatch.setattr(ds, 'write_date_cogs', lambda *a, **k: None)
    source = tmp_path / 'UA_SWE_Depth_800m_v1_20260613_early.nc'
    assert ds.ingest(source) == [date(2026, 6, 13)]


@pytest.mark.parametrize('variant', ['provisional', 'stable'])
def test_ingest_refuses_non_early_stages(tmp_path, monkeypatch, variant):
    # The regex still recognizes provisional/stable (so the error is precise),
    # but the stage pin refuses them to keep a single consistent revision.
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    monkeypatch.setattr(ds, 'write_date_cogs', lambda *a, **k: None)
    source = tmp_path / f'UA_SWE_Depth_800m_v1_20260613_{variant}.nc'
    with pytest.raises(SNODASError, match=f"Refusing to ingest '{variant}'-stage"):
        ds.ingest(source)


def test_ingest_builds_one_grid_aligned_raster_per_variable(tmp_path, monkeypatch):
    # ingest parses the date from the filename and hands write_date_cogs a raster
    # per variable, each carrying the grid's transform/CRS -- all without opening
    # the (here non-existent) source file.
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    captured: dict = {}

    def fake_write(d, rasters, *, force=False):
        captured.update(date=d, rasters=list(rasters), force=force)

    monkeypatch.setattr(ds, 'write_date_cogs', fake_write)

    source = tmp_path / 'UA_SWE_Depth_800m_v1_20240115_early.nc'
    assert ds.ingest(source, force=True) == [date(2024, 1, 15)]
    assert captured['date'] == date(2024, 1, 15)
    assert captured['force'] is True

    rasters = {r.out_name: r for r in captured['rasters']}
    assert set(rasters) == {'swe.tif', 'depth.tif'}
    assert rasters['swe.tif'].source_uri == f'netcdf:{source}:SWE'
    assert rasters['depth.tif'].source_uri == f'netcdf:{source}:DEPTH'

    grid_transform = tuple(ds.grid.base_grid.transform)[:6]
    for raster in rasters.values():
        assert raster.crs.to_epsg() == 4269
        assert raster.nodata == -999.0
        assert raster.tile_size == 256
        assert tuple(raster.transform)[:6] == grid_transform


def _geotiff_source(path, array, transform, crs):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=-999,
    ) as dst:
        dst.write(array, 1)


def test_swann_raster_writes_grid_aligned_cog(tmp_path):
    # SwannRaster writes any GDAL-readable band as a COG on the supplied grid
    # transform/CRS, byte-for-byte and correctly georeferenced.
    array = numpy.arange(32 * 32, dtype='int16').reshape(32, 32)
    transform = from_origin(-125.0208, 49.9375, 0.0083333, 0.0083333)
    crs = rasterio.crs.CRS.from_epsg(4269)
    src = tmp_path / 'src.tif'
    _geotiff_source(src, array, transform, crs)

    raster = SwannRaster(
        str(src),
        'swe.tif',
        transform=transform,
        crs=crs,
        tile_size=16,
        nodata=-999.0,
    )
    out_dir = tmp_path / 'cogs' / '20240115'
    out_dir.mkdir(parents=True)
    raster.write_cog(out_dir)

    with rasterio.open(out_dir / 'swe.tif') as cog:
        assert cog.crs.to_epsg() == 4269
        assert cog.nodata == -999.0
        assert cog.is_tiled
        assert tuple(cog.transform)[:6] == tuple(transform)[:6]
        assert numpy.array_equal(cog.read(1), array)


def test_swann_raster_refuses_overwrite_without_force(tmp_path):
    array = numpy.zeros((32, 32), dtype='int16')
    transform = from_origin(-125.0, 50.0, 0.01, 0.01)
    crs = rasterio.crs.CRS.from_epsg(4269)
    src = tmp_path / 'src.tif'
    _geotiff_source(src, array, transform, crs)

    raster = SwannRaster(
        str(src),
        'swe.tif',
        transform=transform,
        crs=crs,
        tile_size=16,
        nodata=-999.0,
    )
    out_dir = tmp_path / 'd'
    out_dir.mkdir()
    raster.write_cog(out_dir)

    with pytest.raises(FileExistsError, match='already exists'):
        raster.write_cog(out_dir)
    # force overwrites
    raster.write_cog(out_dir, force=True)

"""SWANN 800m dataset definition + ingest.

The grid/variable literals are checked against the values read from a real product
file; the ingest is covered in two halves that need no NetCDF fixture (the bundled
GDAL is read-only for NetCDF):

  * the ingester's orchestration -- date parsing and one grid-aligned raster per
    variable -- by driving ``spec.ingester.plan`` directly and asserting on the
    yielded ``DateIngest`` / built rasters (the source is never opened), and
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

from snowtool.exceptions import IngestSourceError, SnowtoolError
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS
from snowtool.snowdb.datasets.swann import (
    SWANN_800M_SPEC,
    SWANN_800M_VARIABLES,
    SwannRaster,
)
from snowtool.snowdb.ingest import INGEST_FORMAT_VERSION, GridGeometry
from snowtool.snowdb.provenance import hash_files, versioned_hash

from ..conftest import write_geotiff


def test_spec_grid_matches_product_file():
    spec = SWANN_800M_SPEC
    assert spec.name == 'swann-800m'
    assert spec.grid_params.cols == 7025
    assert spec.grid_params.rows == 3105
    assert spec.crs.to_epsg() == 4269
    # NAD83 is geographic -> AOI rasters burn per-row geodesic cell area.
    assert spec.is_geographic is True
    assert spec.ingester is not None


def test_variables():
    by_key = {v.key: v for v in SWANN_800M_VARIABLES}
    assert set(by_key) == {'swe', 'depth'}
    for variable in SWANN_800M_VARIABLES:
        assert variable.dtype == 'int16'
        assert variable.nodata == -999.0
        assert variable.reducer.value == 'mean'
        assert variable.unit.name == 'mm'
    # glob matches the provenance filename on its __<key> suffix.
    assert by_key['swe'].glob == '*__swe.tif'
    assert by_key['depth'].glob == '*__depth.tif'


def test_registered_in_default_specs():
    assert {'snodas', 'swann-800m'} <= {s.name for s in DEFAULT_DATASET_SPECS}


def test_ingest_rejects_unrecognized_filename(tmp_path):
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    with pytest.raises(SnowtoolError, match='Not a SWANN 800m file'):
        ds.ingest(tmp_path / 'some_other_file.nc')


def test_ingest_rejects_a_directory_source(tmp_path):
    # One file == one date: a directory earns a precise typed error. Named to
    # match the filename regex to prove the directory guard fires first.
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    source = tmp_path / 'UA_SWE_Depth_800m_v1_20260613_early.nc'
    source.mkdir()
    with pytest.raises(SnowtoolError, match='got a directory'):
        ds.ingest(source)


def test_ingest_accepts_the_early_stage(tmp_path):
    # Ingest is pinned to the `_early` revision (fastest available); it is the
    # one stage accepted. Drive the ingester's sanctioned plan seam directly and
    # assert on the single DateIngest it yields.
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    source = tmp_path / 'UA_SWE_Depth_800m_v1_20260613_early.nc'
    (item,) = SWANN_800M_SPEC.ingester.plan(source, ds)
    assert item.date == date(2026, 6, 13)


@pytest.mark.parametrize('variant', ['provisional', 'stable'])
def test_ingest_refuses_non_early_stages(tmp_path, variant):
    # The regex still recognizes provisional/stable (so the error is precise),
    # but the stage pin refuses them to keep a single consistent revision. The
    # refusal happens before any rasters are built or write_date_cogs is
    # reached, so this needs no seam at all -- fully real code.
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    source = tmp_path / f'UA_SWE_Depth_800m_v1_20260613_{variant}.nc'
    with pytest.raises(SnowtoolError, match=f"Refusing to ingest '{variant}'-stage"):
        ds.ingest(source)


def test_ingest_builds_one_grid_aligned_raster_per_variable(tmp_path):
    # plan parses the date from the filename and yields a single DateIngest whose
    # build_rasters produces one grid-aligned raster per variable, each carrying the
    # grid's transform/CRS -- all without opening the (here non-existent) source
    # file. Drive the sanctioned plan seam directly (no write_date_cogs patching).
    ds = Dataset(SWANN_800M_SPEC, tmp_path)
    source = tmp_path / 'UA_SWE_Depth_800m_v1_20240115_early.nc'
    stem = 'UA_SWE_Depth_800m_v1_20240115_early'

    (item,) = SWANN_800M_SPEC.ingester.plan(source, ds)
    assert item.date == date(2024, 1, 15)
    assert item.source_files == [source]
    # The declared out_names (read by the skip check before any build) match what
    # build_rasters produces below.
    assert set(item.out_names) == {f'{stem}__swe.tif', f'{stem}__depth.tif'}

    # build_rasters is deferred and driver-hash-bound: invoke it as the driver would,
    # with a versioned source hash over the source file.
    source.write_bytes(b'fake swann netcdf')  # only needed for the hash computation
    expected_hash = versioned_hash(INGEST_FORMAT_VERSION, hash_files([source]))
    rasters = {r.out_name: r for r in item.build_rasters(expected_hash)}

    assert set(rasters) == {f'{stem}__swe.tif', f'{stem}__depth.tif'}
    assert rasters[f'{stem}__swe.tif'].source_uri == f'netcdf:{source}:SWE'
    assert rasters[f'{stem}__depth.tif'].source_uri == f'netcdf:{source}:DEPTH'
    # Source provenance (incl. the source hash) is carried into the COG tags.
    assert rasters[f'{stem}__swe.tif'].tags == {
        'SOURCE_DATASET': 'swann-800m',
        'SOURCE_DATE': '2024-01-15',
        'SOURCE_VARIABLE': 'swe',
        'SOURCE_FILES': source.name,
        'SOURCE_HASH': expected_hash,
        'SOURCE_STAGE': 'early',
    }

    grid_transform = tuple(ds.grid.base_grid.transform)[:6]
    grid_shape = (ds.spec.grid_params.rows, ds.spec.grid_params.cols)
    for raster in rasters.values():
        # The grid geometry (transform/CRS/tile_size/shape) is now one value.
        assert raster.geometry.crs.to_epsg() == 4269
        assert raster.nodata == -999.0
        assert raster.geometry.tile_size == 256
        assert raster.geometry.shape == grid_shape
        assert tuple(raster.geometry.transform)[:6] == grid_transform


def _geotiff_source(path, array, transform, crs):
    write_geotiff(path, array, transform=transform, crs=crs, nodata=-999)


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
        GridGeometry(transform=transform, crs=crs, tile_size=16, shape=(32, 32)),
        nodata=-999.0,
        tags={'SOURCE_DATASET': 'swann-800m', 'SOURCE_VARIABLE': 'swe'},
    )
    out_dir = tmp_path / 'cogs' / '20240115'
    out_dir.mkdir(parents=True)
    raster.write_cog(out_dir)

    with rasterio.open(out_dir / 'swe.tif') as cog:
        assert cog.crs.to_epsg() == 4269
        assert cog.nodata == -999.0
        assert cog.block_shapes[0] == (16, 16)
        assert tuple(cog.transform)[:6] == tuple(transform)[:6]
        assert numpy.array_equal(cog.read(1), array)
        # tags round-trip into the written COG
        assert cog.tags()['SOURCE_DATASET'] == 'swann-800m'
        assert cog.tags()['SOURCE_VARIABLE'] == 'swe'


def test_swann_raster_refuses_shape_mismatch(tmp_path):
    # A source band whose shape differs from the dataset grid (a truncated/regridded
    # UA file) would land mis-aligned under the spec transform, so it must raise
    # rather than write a silently wrong COG.
    array = numpy.zeros((32, 32), dtype='int16')
    transform = from_origin(-125.0208, 49.9375, 0.0083333, 0.0083333)
    crs = rasterio.crs.CRS.from_epsg(4269)
    src = tmp_path / 'src.tif'
    _geotiff_source(src, array, transform, crs)

    raster = SwannRaster(
        str(src),
        'swe.tif',
        # grid expects a bigger array than the source has
        GridGeometry(transform=transform, crs=crs, tile_size=16, shape=(64, 64)),
        nodata=-999.0,
    )
    out_dir = tmp_path / 'cogs'
    out_dir.mkdir()
    with pytest.raises(
        IngestSourceError,
        match=r'produced an array of shape \(32, 32\), expected',
    ):
        raster.write_cog(out_dir)

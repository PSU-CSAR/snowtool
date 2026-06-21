"""Shared fixtures, including the synthetic-grid pipeline fixtures.

Everything runs on a tiny 512x512 (2x2 tile) grid so the full pipeline —
resample, area raster, AOI rasterize, zonal stats — exercises real rasterio /
griffine code on hand-computable data, with no system GDAL and no large inputs.
These live at the top level so both the ``snowdb`` and ``cli`` suites reuse them.
"""

import hashlib
import json

import numpy
import pytest
import rasterio

from rasterio.crs import CRS

from snowtool.settings import Settings
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import DEM_HASH_TAG
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import SNODAS_VARIABLES
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.terrain import (
    ASPECT_COMPONENTS,
    ASPECT_FLAT,
    ASPECT_MAJORITY,
    ELEVATION,
)

# Small synthetic grid parameters.
ORIGIN_X = -120.0
ORIGIN_Y = 45.0
PX = 0.01
SIZE = 512
TILE = 256

DEM_ELEVATION_M = 1000.0  # uniform; 1000 m -> ~3280 ft -> band (3000, 4000) ft
DEM_NODATA = -9999.0
SWE_VALUE = 50  # uniform int16 SWE value


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(snowdb_path=tmp_path)


@pytest.fixture
def spec():
    """A tiny synthetic DatasetSpec (2x2 tile geographic grid)."""
    return DatasetSpec(
        name='test',
        grid_params=GridParams(
            origin_x=ORIGIN_X,
            origin_y=ORIGIN_Y,
            px_size=PX,
            cols=SIZE,
            rows=SIZE,
            tile_size=TILE,
        ),
        variables=SNODAS_VARIABLES,
    )


@pytest.fixture
def grid(spec):
    return spec.grid


@pytest.fixture
def source_dem(tmp_path, grid):
    """A uniform-elevation source DEM on the grid extent."""
    path = tmp_path / 'source_dem.tif'
    array = numpy.full((SIZE, SIZE), DEM_ELEVATION_M, dtype=numpy.float32)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SIZE,
        width=SIZE,
        count=1,
        dtype='float32',
        crs=CRS.from_epsg(4326),
        transform=grid.base_grid.transform,
        nodata=DEM_NODATA,
    ) as dst:
        dst.write(array, 1)
    return path


def write_terrain(dataset, elevation_value: float = DEM_ELEVATION_M) -> str:
    """Write a uniform terrain set onto a dataset's grid (no engine run).

    Mirrors what :func:`generate_terrain` produces -- uniform elevation, all-flat
    aspect -- so tests that just need terrain present (e.g. elevation banding) get
    deterministic, hand-computable values without the streaming generation pass.
    Returns the provenance hash stamped on every layer.
    """
    directory = dataset.terrain.directory
    directory.mkdir(parents=True, exist_ok=True)
    base = dataset.grid.base_grid
    shape = (base.rows, base.cols)
    transform = base.transform
    crs = dataset.grid_crs
    tile = dataset.spec.grid_params.tile_size

    elevation = numpy.full(shape, elevation_value, dtype='float32')
    dem_hash = hashlib.sha256(elevation.tobytes()).hexdigest()
    tags = {DEM_HASH_TAG: dem_hash}

    write_cog(
        directory / ELEVATION.filename,
        elevation,
        transform=transform,
        crs=crs,
        tile_size=tile,
        nodata=ELEVATION.nodata,
        tags=tags,
        band_descriptions=ELEVATION.band_descriptions,
    )
    write_cog(
        directory / ASPECT_MAJORITY.filename,
        numpy.full(shape, ASPECT_FLAT, dtype='uint8'),
        transform=transform,
        crs=crs,
        tile_size=tile,
        nodata=ASPECT_MAJORITY.nodata,
        tags=tags,
        band_descriptions=ASPECT_MAJORITY.band_descriptions,
    )
    write_cog(
        directory / ASPECT_COMPONENTS.filename,
        numpy.full((2, *shape), numpy.nan, dtype='float32'),
        transform=transform,
        crs=crs,
        tile_size=tile,
        nodata=ASPECT_COMPONENTS.nodata,
        tags=tags,
        compute_stats=False,
        band_descriptions=ASPECT_COMPONENTS.band_descriptions,
    )
    return dem_hash


@pytest.fixture
def dataset(tmp_path, spec):
    """A fully created Dataset: directory, area raster, and a uniform terrain set."""
    ds = Dataset.create(spec, tmp_path / 'db')
    write_terrain(ds)
    return ds


@pytest.fixture
def aoi_geojson(tmp_path):
    """A pourpoint with a polygon inside tile (0, 0)."""
    # lon -119.9..-119.0, lat 44.9..44.0 -> well inside the first tile.
    polygon = {
        'type': 'Polygon',
        'coordinates': [
            [
                [-119.9, 44.9],
                [-119.0, 44.9],
                [-119.0, 44.0],
                [-119.9, 44.0],
                [-119.9, 44.9],
            ],
        ],
    }
    point = {'type': 'Point', 'coordinates': [-119.45, 44.45]}
    feature = {
        'type': 'GeometryCollection',
        'id': '12345:MT:USGS',
        'geometries': [point, polygon],
        'properties': {'name': 'Test Basin', 'source': 'test'},
    }
    path = tmp_path / 'pourpoint.geojson'
    path.write_text(json.dumps(feature))
    return path


def snodas_swe_name(date_str: str = '20180427') -> str:
    """A filename matching the SNODAS SWE regex + product glob."""
    # region=us model=ssm datatype=v1 code=1034 scaled=S vcode=lL00
    # T timecode=0001 TTNATS <date> hour=05 interval=H offset=P001
    return f'us_ssmv11034SlL00T0001TTNATS{date_str}05HP001'


@pytest.fixture
def swe_cog(dataset, grid):
    """Write a uniform SWE COG for 2018-04-27 into the db's cogs dir."""
    date_str = '20180427'
    out_dir = dataset._cogs / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{snodas_swe_name(date_str)}.tif'
    array = numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16)
    write_cog(
        path,
        array,
        transform=grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )
    return path

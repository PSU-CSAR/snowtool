"""Synthetic-grid fixtures for the rasterdb pipeline tests.

Everything runs on a tiny 512x512 (2x2 tile) grid so the full pipeline —
resample, area raster, AOI rasterize, zonal stats — exercises real rasterio /
griffine code on hand-computable data, with no system GDAL and no large inputs.
"""

import json

import numpy
import pytest
import rasterio

from rasterio.crs import CRS

from snowtool.rasterdb.cog import write_cog
from snowtool.rasterdb.db import RasterDatabase
from snowtool.rasterdb.grid import make_snodas_grid

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
def grid():
    return make_snodas_grid(
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        px_size=PX,
        cols=SIZE,
        rows=SIZE,
        tile_size=TILE,
    )


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


@pytest.fixture
def rasterdb(tmp_path, grid, source_dem):
    """A fully created RasterDatabase (area raster + resampled DEM)."""
    return RasterDatabase.create(
        tmp_path / 'db',
        source_dem,
        grid=grid,
    )


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
def swe_cog(rasterdb, grid):
    """Write a uniform SWE COG for 2018-04-27 into the db's cogs dir."""
    date_str = '20180427'
    out_dir = rasterdb._cogs / date_str
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

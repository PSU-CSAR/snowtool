"""The projected-grid (planar) area path: no areas.tif, constant cell_area.

On a geographic grid per-pixel area varies by latitude and is stored in
``areas.tif`` (see test_pipeline). On a projected grid every cell has the same
planar area, so no area raster is written or read -- ``spec.cell_area`` is
broadcast over the AOI window instead (decision #7).
"""

import asyncio
import json

import numpy
import pytest
import rasterio

from pyproj import Transformer
from rasterio.crs import CRS

from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.constants import AOI_MASK_INSIDE
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.grid import PixelCoord
from snowtool.snowdb.raster import AOIRaster
from snowtool.snowdb.spec import DatasetSpec, GridParams

from .conftest import write_terrain

PX = 1000.0  # 1 km square pixels -> constant 1e6 m^2 cells
SIZE = 128
TILE = 64
EPSG = 32611  # UTM zone 11N (metres)
DEM_VALUE = 500.0
NODATA = -9999.0


@pytest.fixture
def spec():
    return DatasetSpec(
        name='utm',
        grid_params=GridParams(
            origin_x=500_000.0,
            origin_y=4_000_000.0,
            px_size=PX,
            cols=SIZE,
            rows=SIZE,
            tile_size=TILE,
            crs=EPSG,
        ),
    )


@pytest.fixture
def dataset(tmp_path, spec):
    ds = Dataset.create(spec, tmp_path / 'db')
    # Write a uniform terrain set directly (the streaming engine is tested
    # separately; running it here on a reprojected 10 m grid would be huge).
    write_terrain(ds, DEM_VALUE)
    return ds


def test_create_skips_area_raster_on_projected_grid(dataset):
    assert dataset.spec.is_geographic is False
    # No areas.tif on a projected grid...
    assert not dataset._area_raster.exists()
    # ...and terrain is written in the grid's own CRS.
    from snowtool.snowdb.terrain import ELEVATION

    terrain = dataset.zones['terrain']
    assert terrain.present()
    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        assert ds.crs == CRS.from_epsg(EPSG)
        assert numpy.allclose(ds.read(1), DEM_VALUE)


def test_rasterize_aoi_reprojects_wgs84_geometry_onto_projected_grid(dataset, tmp_path):
    """A WGS84 AOI is reprojected into the grid's UTM CRS before tiling/burning.

    The polygon is built in the grid's CRS (a bbox well inside the grid extent)
    then expressed as WGS84 geojson -- exactly how a real global AOI arrives. If
    the lon/lat coordinates were used directly against the metre-based grid (the
    old bug), they'd fall far outside it; a valid masked window with inside
    pixels proves the reprojection happened.
    """
    east0, east1 = 540_000.0, 560_000.0
    north0, north1 = 3_900_000.0, 3_940_000.0
    to_wgs84 = Transformer.from_crs(EPSG, 4326, always_xy=True)

    ring = [
        list(to_wgs84.transform(x, y))
        for x, y in [
            (east0, north1),
            (east1, north1),
            (east1, north0),
            (east0, north0),
            (east0, north1),
        ]
    ]
    centroid = list(to_wgs84.transform((east0 + east1) / 2, (north0 + north1) / 2))

    feature = {
        'type': 'GeometryCollection',
        'id': '99:NV:USGS',
        'geometries': [
            {'type': 'Point', 'coordinates': centroid},
            {'type': 'Polygon', 'coordinates': [ring]},
        ],
        'properties': {'name': 'Projected Basin', 'source': 'test'},
    }
    geojson = tmp_path / 'projected_pourpoint.geojson'
    geojson.write_text(json.dumps(feature))

    aoi_raster = dataset.rasterize_aoi(AOI.from_geojson(geojson))

    inside = aoi_raster.array == AOI_MASK_INSIDE
    assert 0 < inside.sum() < aoi_raster.array.size
    # Every selected tile is within the 2x2-tile grid.
    for tile in aoi_raster.tiles:
        assert 0 <= tile.row < SIZE // TILE
        assert 0 <= tile.col < SIZE // TILE


def test_cell_area_converts_non_metre_units_to_m2():
    # A projected CRS measured in US survey feet: griffine's planar cell area is
    # ft^2, which cell_area must convert to m^2 so all area output is metres.
    us_foot_to_metre = 0.30480060960121924  # EPSG:2225 linear unit
    px_feet = 1000.0
    spec = DatasetSpec(
        name='stateplane',
        grid_params=GridParams(
            origin_x=2_000_000.0,
            origin_y=500_000.0,
            px_size=px_feet,
            cols=SIZE,
            rows=SIZE,
            tile_size=TILE,
            crs=2225,  # NAD83 / California zone 2 (US survey feet)
        ),
    )

    assert spec.is_geographic is False
    assert spec.cell_area == pytest.approx((px_feet * us_foot_to_metre) ** 2)
    # ...and emphatically not the raw planar ft^2 value.
    assert spec.cell_area != pytest.approx(px_feet * px_feet)


def test_load_aoi_with_area_uses_constant_cell_area(dataset):
    aoi_raster = AOIRaster(
        path=dataset.path / 'fake.tif',
        array=numpy.full((4, 4), AOI_MASK_INSIDE, dtype=numpy.uint8),
        tiles=[],
        origin=PixelCoord(0, 0),
    )

    # cache=None proves the projected branch never reads an area raster.
    result = asyncio.run(dataset.load_aoi_with_area(aoi_raster, cache=None))

    assert result.area.shape == aoi_raster.array.shape
    assert (result.area == numpy.float32(dataset.spec.cell_area)).all()
    assert dataset.spec.cell_area == pytest.approx(PX * PX)
    # The constant area is a zero-copy broadcast view, not a materialized
    # N-pixel array, so it carries no per-pixel storage.
    assert result.area.flags['OWNDATA'] is False

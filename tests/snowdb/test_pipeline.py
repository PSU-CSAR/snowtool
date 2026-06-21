"""End-to-end pipeline tests on the synthetic grid."""

import asyncio

import numpy
import pytest
import rasterio

from snowtool.exceptions import SNODASError
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.raster import AOIRaster, AOIRasterWithArea
from snowtool.snowdb.raster_collection import RasterCollection
from snowtool.snowdb.tiff_cache import TiffCache
from snowtool.snowdb.zonal_stats import ZonalStats

from .conftest import DEM_ELEVATION_M, SIZE, SWE_VALUE, TILE


def test_resampled_dem(dataset, grid):
    with rasterio.open(dataset._dem) as ds:
        assert ds.shape == (SIZE, SIZE)
        assert ds.crs == rasterio.CRS.from_epsg(4326)
        assert ds.transform == grid.base_grid.transform
        data = ds.read(1)
    # uniform input -> uniform resample
    assert numpy.allclose(data, DEM_ELEVATION_M)


def test_area_raster(dataset, grid):
    with rasterio.open(dataset._area_raster) as ds:
        assert ds.shape == (SIZE, SIZE)
        data = ds.read(1)
    assert (data > 0).all()
    # area depends only on row; matches per-row geodesic area
    assert data[0, 0] == data[0, -1]
    assert data[10, 5] != data[400, 5]
    expected_row0 = grid.base_grid[0, 0].area
    assert data[0, 0] == numpy.float32(expected_row0)


def test_rasterize_aoi(dataset, aoi_geojson):
    aoi = AOI.from_geojson(aoi_geojson)
    aoi_raster = dataset.rasterize_aoi(aoi)

    # polygon sits inside a single tile -> one-tile AOI window
    assert aoi_raster.array.shape == (TILE, TILE)

    # metadata: a dataset-agnostic tile bounding box "ul_row ul_col br_row br_col"
    with rasterio.open(aoi_raster.path) as ds:
        tags = ds.tags()
    assert TILE_BBOX_TAG in tags
    bbox = [int(x) for x in tags[TILE_BBOX_TAG].split()]
    assert len(bbox) == 4
    assert len(aoi_raster.tiles) >= 1

    # inside-polygon pixels carry the DEM elevation; min/max == uniform value
    assert aoi_raster.min_elevation == DEM_ELEVATION_M
    assert aoi_raster.max_elevation == DEM_ELEVATION_M

    # the masked-in pixel count is positive and less than the full window
    inside = aoi_raster.array == DEM_ELEVATION_M
    assert 0 < inside.sum() < TILE * TILE


def test_aoi_raster_reopen_roundtrips_tiles(dataset, aoi_geojson):
    aoi = AOI.from_geojson(aoi_geojson)
    written = dataset.rasterize_aoi(aoi, force=True)
    reopened = AOIRaster.open(written.path, dataset.grid)
    assert {(t.row, t.col) for t in reopened.tiles} == {
        (t.row, t.col) for t in written.tiles
    }
    assert reopened.origin == written.origin


def test_zonal_stats(dataset, aoi_geojson, swe_cog):
    aoi = AOI.from_geojson(aoi_geojson)
    aoi_raster = dataset.rasterize_aoi(aoi)

    swe = dataset.spec.variables['swe']
    collection = RasterCollection.from_variables_query(
        query=_SingleDateQuery(),
        variables={swe},
        dataset=dataset,
    )

    async def run():
        # one cache, one event loop for the whole read path
        cache = TiffCache(maxsize=8)
        aoi_with_area = await AOIRasterWithArea.from_aoi_raster(
            aoi_raster,
            dataset.area_raster(),
            cache,
        )
        stats = await ZonalStats.calculate(
            aoi_with_area,
            collection,
            cache,
            dataset.spec,
        )
        return aoi_with_area, stats

    aoi_with_area, stats = asyncio.run(run())
    dumped = stats.dump()
    assert len(dumped) == 1
    zones = dumped[0].zones

    # Exactly one band (the one containing 1000 m == ~3280 ft) has data.
    with_data = [z for z in zones if z.area_m2 > 0]
    assert len(with_data) == 1
    band = with_data[0]
    assert band.min_elevation_ft == 3000
    assert band.max_elevation_ft == 4000
    assert band.mean_swe_mm == SWE_VALUE

    # area equals the summed geodesic area of the in-AOI pixels
    inside = aoi_with_area.array == DEM_ELEVATION_M
    expected_area = float(aoi_with_area.area[inside].sum())
    assert band.area_m2 == expected_area

    # all other bands are empty (area 0; mean is nan in-model, serialized to
    # None by the model's field serializer)
    import math

    for zone in zones:
        if zone is band:
            continue
        assert zone.area_m2 == 0
        assert math.isnan(zone.mean_swe_mm)


def test_aoi_raster_open_raises_clear_error_when_all_nodata(tmp_path, grid):
    """An AOI not overlapping valid DEM has no STATISTICS_* tags on reopen."""
    nodata = -9999.0
    path = tmp_path / 'empty_aoi.tif'
    # Entire window is nodata, so write_cog embeds no band statistics.
    write_cog(
        path,
        numpy.full((TILE, TILE), nodata, dtype=numpy.float32),
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=nodata,
        tags={TILE_BBOX_TAG: '0 0 0 0'},
    )

    with pytest.raises(SNODASError, match='does not overlap any valid DEM'):
        AOIRaster.open(path, grid)


class _SingleDateQuery:
    """Minimal DateQuery for 2018-04-27 only."""

    def generate_sequence(self):
        from datetime import date

        yield date(2018, 4, 27)

    def csv_name(self, pourpoint_name, zone_size=0):  # pragma: no cover
        return 'test.csv'

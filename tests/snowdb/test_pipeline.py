"""End-to-end pipeline tests on the synthetic grid."""

import asyncio

import numpy
import rasterio

from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.constants import AOI_MASK_INSIDE, TILE_BBOX_TAG
from snowtool.snowdb.raster import AOIRaster, AOIRasterWithArea
from snowtool.snowdb.raster_collection import RasterCollection
from snowtool.snowdb.tiff_cache import TiffCache
from snowtool.snowdb.zonal_stats import ZonalStats

from .conftest import DEM_ELEVATION_M, SIZE, SWE_VALUE, TILE


def test_terrain_elevation(dataset, grid):
    with rasterio.open(dataset.terrain.elevation_path) as ds:
        assert ds.shape == (SIZE, SIZE)
        assert ds.crs == rasterio.CRS.from_epsg(4326)
        assert ds.transform == grid.base_grid.transform
        data = ds.read(1)
    # uniform terrain fixture -> uniform elevation
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


def test_area_raster_uses_grid_crs_not_hardcoded_wgs84(tmp_path):
    # A non-4326 geographic CRS (NAD83) proves the area raster is written in the
    # grid's own CRS, not a hardcoded WGS84.
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.spec import DatasetSpec, GridParams

    spec = DatasetSpec(
        name='nad83',
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=128,
            rows=128,
            tile_size=128,
            crs=4269,  # NAD83 geographic
        ),
    )
    path = tmp_path / 'db'
    path.mkdir()
    dataset = Dataset(spec, path)

    dataset.make_area_raster()

    with rasterio.open(dataset._area_raster) as ds:
        assert ds.crs.to_epsg() == dataset.grid_crs.to_epsg() == 4269
        assert ds.crs.to_epsg() != 4326


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

    # The AOI raster is a bare boolean mask (decoupled from the DEM): inside
    # pixels are AOI_MASK_INSIDE, and there are some but not the whole window.
    inside = aoi_raster.array == AOI_MASK_INSIDE
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
            dataset.terrain.elevation_raster(),
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
    inside = aoi_with_area.array == AOI_MASK_INSIDE
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


def test_aoi_raster_open_reads_mask_without_dem(tmp_path, grid):
    """A bare mask reopens cleanly -- AOI rasters no longer depend on a DEM."""
    from snowtool.snowdb.cog import write_cog

    path = tmp_path / 'mask_aoi.tif'
    mask = numpy.zeros((TILE, TILE), dtype=numpy.uint8)
    mask[10:20, 10:20] = AOI_MASK_INSIDE
    write_cog(
        path,
        mask,
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=0,
        tags={TILE_BBOX_TAG: '0 0 0 0'},
        compute_stats=False,
    )

    aoi_raster = AOIRaster.open(path, grid)
    assert aoi_raster.array.shape == (TILE, TILE)
    assert (aoi_raster.array == AOI_MASK_INSIDE).sum() == 100


class _SingleDateQuery:
    """Minimal DateQuery for 2018-04-27 only."""

    def generate_sequence(self):
        from datetime import date

        yield date(2018, 4, 27)

    def csv_name(self, pourpoint_name, zone_size=0):  # pragma: no cover
        return 'test.csv'

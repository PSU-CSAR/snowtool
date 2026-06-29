"""End-to-end pipeline tests on the synthetic grid."""

import asyncio

from datetime import date

import numpy
import pytest
import rasterio

from snowtool import types
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.raster import AOIRaster
from snowtool.snowdb.raster_collection import RasterCollection
from snowtool.snowdb.terrain import ELEVATION
from snowtool.snowdb.tiff_cache import TiffCache
from snowtool.snowdb.zonal_stats import ZonalStats, ZoneSelection

from .conftest import DEM_ELEVATION_M, SIZE, SWE_VALUE, TILE

# The synthetic SWE COG is ingested for this date; a closed one-day range selects it.
QUERY = types.DateRangeQuery(start_date=date(2018, 4, 27), end_date=date(2018, 4, 27))


def test_terrain_elevation(dataset, grid):

    terrain = dataset.zones['terrain']
    with rasterio.open(terrain.layer_path(ELEVATION)) as ds:
        assert ds.shape == (SIZE, SIZE)
        assert ds.crs == rasterio.CRS.from_epsg(4326)
        assert ds.transform == grid.base_grid.transform
        data = ds.read(1)
    # uniform terrain fixture -> uniform elevation
    assert numpy.allclose(data, DEM_ELEVATION_M)


def test_rasterize_aoi(dataset, pourpoint_geojson, grid):
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
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

    # The AOI raster burns per-pixel cell area (decoupled from the DEM): inside
    # pixels carry their geodesic cell area, 0 outside, over some but not the whole
    # window.
    inside = aoi_raster.array > 0
    assert 0 < inside.sum() < TILE * TILE
    # The burned value is the grid's geodesic cell area for that row.
    ul = bbox[0] * TILE  # the window's upper-left base row
    inside_rows = numpy.where(inside.any(axis=1))[0]
    sample_row = int(inside_rows[0])
    assert aoi_raster.array[sample_row][inside[sample_row]][0] == numpy.float32(
        grid.base_grid[ul + sample_row, 0].area,
    )


def test_aoi_raster_reopen_roundtrips_tiles(dataset, pourpoint_geojson):
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    written = dataset.rasterize_aoi(aoi, force=True)
    reopened = AOIRaster.open(written.path, dataset.grid)
    assert {(t.row, t.col) for t in reopened.tiles} == {
        (t.row, t.col) for t in written.tiles
    }
    assert reopened.origin == written.origin


def test_zonal_stats(dataset, pourpoint_geojson, swe_cog):
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    aoi_raster = dataset.rasterize_aoi(aoi)

    swe = dataset.spec.variables['swe']
    collection = RasterCollection.from_variables_query(
        query=QUERY,
        variables={swe},
        dataset=dataset,
    )

    async def run():
        # one cache, one event loop for the whole read path
        cache = TiffCache(maxsize=8)
        # No zone selection -> a whole-basin reduction: a single cell per date
        # whose zone tuple is empty (the K=0 crossed-index case).
        return await ZonalStats.calculate(
            aoi_raster,
            collection,
            cache,
            dataset,
        )

    stats = asyncio.run(run())
    dumped = stats.dump()
    assert len(dumped) == 1
    # No stratification -> no zone axes, exactly one whole-basin cell.
    assert dumped[0].zone_layers == []
    (cell,) = dumped[0].zones
    assert cell.zone == []
    assert cell.mean_swe_mm == SWE_VALUE

    # area equals the summed geodesic area of the in-AOI pixels -- which the AOI
    # raster now carries directly (cell area inside, 0 outside), so its full sum.
    assert cell.area_m2 == float(aoi_raster.array.sum())


def _crossed_stats(dataset, pourpoint_geojson, selections, *, max_zone_cells=10_000):
    """Run ZonalStats over the synthetic SWE COG crossed by ``selections``."""
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    aoi_raster = dataset.rasterize_aoi(aoi)
    swe = dataset.spec.variables['swe']
    collection = RasterCollection.from_variables_query(
        query=QUERY,
        variables={swe},
        dataset=dataset,
    )

    async def run():
        cache = TiffCache(maxsize=8)
        stats = await ZonalStats.calculate(
            aoi_raster,
            collection,
            cache,
            dataset,
            zone_selections=selections,
            max_zone_cells=max_zone_cells,
        )
        return aoi_raster, stats

    return asyncio.run(run())


@pytest.mark.parametrize(
    ('selections', 'kwargs', 'match'),
    [
        # 16 elevation bands alone exceed a deliberately tiny cap; the guard fires
        # before any raster read.
        ([ZoneSelection('terrain.elevation')], {'max_zone_cells': 4}, 'max_zone_cells'),
        ([ZoneSelection('terrain.nope')], {}, 'Unknown zone layer'),
    ],
)
def test_calculate_rejects_bad_crossed_request(
    dataset,
    pourpoint_geojson,
    swe_cog,
    selections,
    kwargs,
    match,
):
    with pytest.raises(ValueError, match=match):
        _crossed_stats(dataset, pourpoint_geojson, selections, **kwargs)


def test_zonal_stats_crosses_elevation_and_forest_cover(
    dataset,
    pourpoint_geojson,
    swe_cog,
):
    # The synthetic dataset is uniform: elevation 1000 m (-> 3000-4000 ft band) and
    # forest 100% (>= the 50% default threshold -> "forested"), so crossing the two
    # axes yields exactly one populated cell -- the SWE value over the in-AOI area.
    aoi_raster, stats = _crossed_stats(
        dataset,
        pourpoint_geojson,
        [
            ZoneSelection('terrain.elevation'),
            ZoneSelection('landcover.forest_cover'),
        ],
    )

    dumped = stats.dump()
    assert dumped[0].zone_layers == ['terrain.elevation', 'landcover.forest_cover']
    # 16 elevation bands x 2 forest classes (forested/unforested) = 32 cells.
    cells = dumped[0].zones
    assert len(cells) == 32
    with_data = [c for c in cells if c.area_m2 > 0]
    assert len(with_data) == 1
    cell = with_data[0]

    elev_ref, forest_ref = cell.zone
    assert (elev_ref.layer, elev_ref.min, elev_ref.max) == (
        'terrain.elevation',
        3000,
        4000,
    )
    # Forest cover is a threshold split: a structured threshold + side, not bands.
    assert forest_ref.layer == 'landcover.forest_cover'
    assert forest_ref.threshold == 50
    assert forest_ref.unit == '%'
    assert forest_ref.side == 'above'
    assert forest_ref.label == 'forested'
    assert cell.mean_swe_mm == SWE_VALUE

    assert cell.area_m2 == float(aoi_raster.array.sum())


def test_zonal_stats_forest_threshold_is_overridable(
    dataset,
    pourpoint_geojson,
    swe_cog,
):
    # The synthetic forest layer is 100%; a threshold above 100 flips the whole AOI
    # from "forested" to "unforested", proving the per-query threshold knob works.
    _, stats = _crossed_stats(
        dataset,
        pourpoint_geojson,
        [ZoneSelection('landcover.forest_cover', override=100.5)],
    )
    (cell,) = [c for c in stats.dump()[0].zones if c.area_m2 > 0]
    (forest_ref,) = cell.zone
    assert forest_ref.side == 'below'
    assert forest_ref.label == 'unforested'
    assert forest_ref.threshold == 100.5


def test_zonal_stats_crosses_elevation_and_categorical_aspect(
    dataset,
    pourpoint_geojson,
    swe_cog,
):
    # The synthetic terrain is all-flat aspect, so crossing elevation x aspect puts
    # all data in the (3000-4000 ft) x (flat) cell, shown as a class-labelled ref.
    _, stats = _crossed_stats(
        dataset,
        pourpoint_geojson,
        [ZoneSelection('terrain.elevation'), ZoneSelection('terrain.aspect')],
    )

    dumped = stats.dump()
    assert dumped[0].zone_layers == ['terrain.elevation', 'terrain.aspect']
    with_data = [c for c in dumped[0].zones if c.area_m2 > 0]
    assert len(with_data) == 1
    elev_ref, aspect_ref = with_data[0].zone
    assert elev_ref.min == 3000
    # The aspect axis is categorical: a class ref carrying its code + label.
    assert aspect_ref.layer == 'terrain.aspect'
    assert aspect_ref.code == 4
    assert aspect_ref.label == 'flat'


def test_aoi_raster_open_reads_area_without_dem(tmp_path, grid):
    """An area-valued AOI raster reopens cleanly -- no dependence on a DEM."""

    path = tmp_path / 'area_aoi.tif'
    area = numpy.zeros((TILE, TILE), dtype=numpy.float32)
    area[10:20, 10:20] = 123.5  # in-basin cell area
    write_cog(
        path,
        area,
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=0,
        tags={TILE_BBOX_TAG: '0 0 0 0'},
        compute_stats=False,
    )

    aoi_raster = AOIRaster.open(path, grid)
    assert aoi_raster.array.shape == (TILE, TILE)
    assert aoi_raster.array.dtype == numpy.float32
    assert (aoi_raster.array > 0).sum() == 100

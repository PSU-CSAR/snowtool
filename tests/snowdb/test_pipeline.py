"""End-to-end pipeline tests on the synthetic grid."""

import asyncio

from datetime import date

import numpy
import pytest
import rasterio

from snowtool.snowdb.aoi_raster import AOIRaster
from snowtool.snowdb.config import BandStepParams
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import SNODAS_VARIABLES
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.query import DateRangeQuery
from snowtool.snowdb.raster.cog import write_cog
from snowtool.snowdb.raster.collection import RasterCollection
from snowtool.snowdb.raster.tiff_cache import TiffCache
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.zonal_stats import ZonalStats, ZoneSelection
from snowtool.snowdb.zones.terrain import ELEVATION

from ..conftest import (
    DEM_ELEVATION_M,
    ORIGIN_X,
    ORIGIN_Y,
    PX,
    SIZE,
    SWE_VALUE,
    TILE,
    snodas_swe_name,
    write_terrain,
)

# The synthetic SWE COG is ingested for this date; a closed one-day range selects it.
QUERY = DateRangeQuery(start_date=date(2018, 4, 27), end_date=date(2018, 4, 27))


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
    written = dataset.rasterize_aoi(aoi, rebuild=True)
    reopened = AOIRaster.open(written.path, dataset.grid)
    assert {(t.row, t.col) for t in reopened.tiles} == {
        (t.row, t.col) for t in written.tiles
    }
    assert reopened.origin == written.origin


def test_dump_compact_whole_basin(dataset, pourpoint_geojson, swe_cog):
    aoi_raster, stats = _crossed_stats(dataset, pourpoint_geojson, [])
    compact = stats.dump_compact()

    assert compact.zone_layers == []
    assert compact.variables == ['mean_swe_mm']
    # Whole basin: one cell with an empty zone ref list and the basin area.
    (zone,) = compact.zones
    assert zone.zone == []
    assert zone.area_m2 == pytest.approx(float(aoi_raster.array.sum()))
    # One ingested date -> one matrix row (one zone) x one variable, = SWE_VALUE.
    (matrix,) = compact.results.values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]


def test_dump_compact_empty_zone_is_null(dataset, pourpoint_geojson, swe_cog):
    _, stats = _crossed_stats(
        dataset,
        pourpoint_geojson,
        [ZoneSelection('terrain.elevation')],
    )
    compact = stats.dump_compact(include_empty_zones=True)
    # 16 elevation bands, only 3000-4000 ft populated; the empty bands are 0-area
    # with a null variable value.
    assert len(compact.zones) == 16
    populated = [i for i, z in enumerate(compact.zones) if z.area_m2 > 0]
    assert len(populated) == 1
    (matrix,) = compact.results.values()
    for i, row in enumerate(matrix):
        if i in populated:
            assert row == [pytest.approx(SWE_VALUE)]
        else:
            assert row == [None]


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
            zones=selections,
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

    # 16 elevation bands x 2 forest classes (forested/unforested) = 32 crossed cells,
    # but the uniform synthetic AOI populates just one. By default the 31 empty
    # (0-area) combinations are dropped; include_empty_zones restores the full product.
    full = stats.dump_compact(include_empty_zones=True)
    assert len(full.zones) == 32

    compact = stats.dump_compact()
    assert compact.zone_layers == ['terrain.elevation', 'landcover.forest_cover']
    (zone,) = compact.zones
    assert zone.area_m2 > 0

    elev_ref, forest_ref = zone.zone
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
    (matrix,) = compact.results.values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]

    assert zone.area_m2 == pytest.approx(float(aoi_raster.array.sum()))


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
    compact = stats.dump_compact()
    (zone,) = [z for z in compact.zones if z.area_m2 > 0]
    (forest_ref,) = zone.zone
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

    compact = stats.dump_compact()
    assert compact.zone_layers == ['terrain.elevation', 'terrain.aspect']
    with_data = [z for z in compact.zones if z.area_m2 > 0]
    assert len(with_data) == 1
    elev_ref, aspect_ref = with_data[0].zone
    assert elev_ref.min == 3000
    # The aspect axis is categorical: a class ref carrying its code + label.
    assert aspect_ref.layer == 'terrain.aspect'
    assert aspect_ref.code == 4
    assert aspect_ref.label == 'flat'


def test_zonal_stats_crosses_northness_band(dataset, pourpoint_geojson, swe_cog):
    # Overwrite the flat-aspect terrain with a uniform northness of 0.6: it lands in
    # the [0.5, 1] bucket, so a northness-crossed query puts all the SWE in exactly
    # that one bucket. 0.6 = cos(~53 deg) is an interior value, well clear of the
    # bucket edges, so the assignment is unambiguous.
    write_terrain(dataset, northness_value=0.6)

    aoi_raster, stats = _crossed_stats(
        dataset,
        pourpoint_geojson,
        [ZoneSelection('terrain.northness')],
    )

    # Four even buckets over [-1, 1]: [-1,-0.5),[-0.5,0),[0,0.5),[0.5,1]; only the
    # last is populated, so the default drops the other three empty buckets.
    assert len(stats.dump_compact(include_empty_zones=True).zones) == 4

    compact = stats.dump_compact()
    assert compact.zone_layers == ['terrain.northness']
    (zone,) = compact.zones
    (north_ref,) = zone.zone
    assert (north_ref.layer, north_ref.min, north_ref.max, north_ref.unit) == (
        'terrain.northness',
        0.5,
        1,
        None,
    )
    (matrix,) = compact.results.values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]
    assert zone.area_m2 == pytest.approx(float(aoi_raster.array.sum()))


def _banded_dataset(directory, band_step_ft):
    """A synthetic dataset whose terrain.elevation configures a non-default band step.

    Mirrors the ``dataset``/``swe_cog`` fixtures but plumbs a ``zones`` config so the
    real ZonalStats path folds the dataset's configured ``band_step_ft`` into the
    elevation scheme, exercising :func:`resolve_zone_axis` on the production path.
    """
    spec = DatasetSpec(
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
        zones={'terrain': {'elevation': BandStepParams(band_step_ft=band_step_ft)}},
    )
    ds = Dataset.create(spec, directory)
    write_terrain(ds)

    date_str = '20180427'
    out_dir = ds._cogs / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cog(
        out_dir / f'{snodas_swe_name(date_str)}.tif',
        numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16),
        transform=ds.grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )
    return ds


def _band_bounds(dataset, pourpoint_geojson, selections):
    """The (layer, min, max) of the single populated elevation band."""
    _, stats = _crossed_stats(dataset, pourpoint_geojson, selections)
    (zone,) = stats.dump_compact().zones
    (elev_ref,) = zone.zone
    return (elev_ref.layer, elev_ref.min, elev_ref.max)


def test_configured_band_step_folds_on_the_real_query_path(tmp_path, pourpoint_geojson):
    # The uniform DEM is 1000 m (~3280 ft). With the dataset configuring a 2000-ft
    # band step, that lands in the (2000, 4000) band; with a per-query override of
    # 500 ft it lands in (3000, 3500) instead -- proving resolve_zone_axis folds the
    # dataset default then lets the explicit override win, on the real pipeline.
    # (Two datasets because rasterize_aoi is converge-by-default: a second query
    # against the same dataset+pourpoint finds the AOI raster already current.)
    default_ds = _banded_dataset(tmp_path / 'default', band_step_ft=2000)
    assert _band_bounds(
        default_ds,
        pourpoint_geojson,
        [ZoneSelection('terrain.elevation')],
    ) == ('terrain.elevation', 2000, 4000)

    override_ds = _banded_dataset(tmp_path / 'override', band_step_ft=2000)
    assert _band_bounds(
        override_ds,
        pourpoint_geojson,
        [ZoneSelection('terrain.elevation', override=500)],
    ) == ('terrain.elevation', 3000, 3500)


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

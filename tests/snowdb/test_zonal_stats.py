"""Unit tests for ZonalStats._calc cell selection, area, and reduction.

These stub the AOI/SNODAS raster I/O so the test pins down the pure numeric
behaviour of _calc directly, with non-uniform pixel areas and a nodata cell --
the cases the uniform end-to-end pipeline test cannot distinguish:

  * area is the cell's geographic area, independent of which pixels are nodata,
  * MEAN is area-weighted over only the pixels that have data, and
  * TOTAL is the area-weighted sum (a basin total) over those pixels.

Banding runs through the elevation layer's :class:`BandedZoning` scheme; each test
tunes the scheme's domain so it yields exactly the bands it wants to exercise.
The single-axis :class:`_ZoneIndex` is the K=1 case of the crossed engine.
"""

import asyncio
import csv
import io
import math

from datetime import date

import numpy

from snowtool.snowdb.constants import M_TO_FT
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.terrain import ELEVATION_NODATA
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit
from snowtool.snowdb.zonal_stats import (
    Result,
    ZonalStats,
    ZoneSelection,
    _ZoneIndex,
    parse_zone_selection,
)
from snowtool.snowdb.zoning import BandedZoning, BandZone, ClassZone, ThresholdZone

NODATA = -9999  # the variable's int16 nodata sentinel for these stubs


def _scheme(domain_max_ft: float, *, domain_min_ft: float = 0.0, step: int = 1000):
    """A banded elevation scheme over ``[domain_min_ft, domain_max_ft]`` feet."""
    return BandedZoning(
        domain_min=domain_min_ft,
        domain_max=domain_max_ft,
        default_step=step,
        unit='ft',
        value_scale=M_TO_FT,
        layer_nodata=ELEVATION_NODATA,
    )


def _variable(reducer: Reducer = Reducer.MEAN) -> DatasetVariable:
    return DatasetVariable(
        key='swe',
        unit=Unit(name='mm', scale_factor=1),
        reducer=reducer,
        dtype='int16',
        nodata=float(NODATA),
        glob='*',
    )


class _FakeRaster:
    """Just enough of DataRaster for _calc: a date."""

    def __init__(self, d: date) -> None:
        self.date = d


class _FakeAOI:
    """Stands in for AOIRaster; load_* just stamps fixed values.

    ``array`` is the AOI raster: per-pixel cell area inside the basin, 0 outside
    -- both the in/out membership signal and the area weights. Elevation is held
    separately (decoupled from the DEM) and fed to the zone index, mirroring the
    real read path where elevation is loaded live from the terrain set. Every
    pixel here carries a positive area, so cell selection covers the whole window.
    """

    def __init__(self, elevation, area, values) -> None:
        self.elevation = elevation
        self.array = area
        self._values = values

    async def load_raster_tiles_into_array(self, raster, values_array, cache):
        values_array[:] = self._values


def _run_calc(aoi, variable, raster, scheme, *, step=None):
    # A single elevation axis: the K=1 case of the crossed index.
    ordinals = scheme.assign(aoi.elevation, step=step)
    zone_index = _ZoneIndex.build(
        [scheme.zones(step=step)], [ordinals], aoi.array,
    )
    return asyncio.run(ZonalStats._calc(aoi, variable, raster, zone_index, cache=None))


def _bounds(result: Result) -> tuple[int, int]:
    (band,) = result.zone
    assert isinstance(band, BandZone)
    return band.min, band.max


def test_calc_area_is_variable_independent_and_mean_is_area_weighted():
    # All four elevations (m) fall inside the single 0..10000 ft band, so the cell
    # selection is the whole window.
    elevations = numpy.array([[500.0, 1000.0], [1500.0, 2000.0]], dtype=numpy.float32)
    # Deliberately non-uniform per-pixel ground areas.
    areas = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float32)
    # The bottom-left pixel (area 30) is nodata for this variable.
    values = numpy.array([[100, 200], [NODATA, 400]], dtype=numpy.int16)

    variable = _variable(Reducer.MEAN)
    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    # domain 0..9999 ft @ 10000 step -> exactly one band (0, 10000).
    (result,) = _run_calc(aoi, variable, raster, _scheme(9999, step=10000))

    assert result.date == raster.date
    assert result.variable is variable
    assert _bounds(result) == (0, 10000)

    # area counts every in-cell pixel, including the one that is nodata for this
    # variable (10 + 20 + 30 + 40).
    assert result.area == 100.0

    # Mean is weighted by each valid pixel's area and excludes the nodata cell,
    # giving an area-weighted average of 300 (vs ~233.3 unweighted).
    assert result.value == 300.0
    assert result.value != (100 + 200 + 400) / 3


def test_calc_total_is_area_weighted_sum():
    elevations = numpy.array([[500.0, 1000.0], [1500.0, 2000.0]], dtype=numpy.float32)
    areas = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200], [NODATA, 400]], dtype=numpy.int16)

    variable = _variable(Reducer.TOTAL)
    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    (result,) = _run_calc(aoi, variable, raster, _scheme(9999, step=10000))

    # Area-weighted sum over the valid pixels gives a basin total of 21000.
    assert result.value == 21000.0


def test_calc_cell_with_terrain_but_no_data_has_area_and_nan_value():
    elevations = numpy.array([[1000.0, 1000.0]], dtype=numpy.float32)
    areas = numpy.array([[5.0, 7.0]], dtype=numpy.float32)
    # Every pixel is nodata for this variable.
    values = numpy.array([[NODATA, NODATA]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    (result,) = _run_calc(aoi, _variable(), raster, _scheme(9999, step=10000))

    # The cell still covers ground (5 + 7) even though no data exists.
    assert result.area == 12.0
    assert math.isnan(result.value)


def test_calc_assigns_pixels_to_their_bands_in_one_pass():
    # 100 m (~328 ft) falls in band (0, 1000) ft; 400 m (~1312 ft) in (1000, 2000).
    elevations = numpy.array([[100.0, 400.0]], dtype=numpy.float32)
    areas = numpy.array([[10.0, 20.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    # domain 0..1000 ft @ 1000 step -> bands (0, 1000) and (1000, 2000).
    low, high = _run_calc(aoi, _variable(Reducer.MEAN), raster, _scheme(1000))

    assert (_bounds(low), low.value, low.area) == ((0, 1000), 100.0, 10.0)
    assert (_bounds(high), high.value, high.area) == ((1000, 2000), 200.0, 20.0)


def test_calc_empty_middle_band_is_nan_with_zero_area():
    # 100 m -> band 0, 700 m (~2297 ft) -> band 2; the (1000, 2000) ft middle is empty.
    elevations = numpy.array([[100.0, 700.0]], dtype=numpy.float32)
    areas = numpy.array([[10.0, 20.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    # domain 0..2000 ft @ 1000 step -> bands (0,1000),(1000,2000),(2000,3000).
    low, mid, high = _run_calc(aoi, _variable(Reducer.TOTAL), raster, _scheme(2000))

    assert low.value == 100.0 * 10.0
    assert high.value == 200.0 * 20.0
    # Empty cell reads nan (not a spurious 0) and carries no area.
    assert math.isnan(mid.value)
    assert mid.area == 0.0


def test_calc_cell_with_no_terrain_has_zero_area_and_nan_value():
    elevations = numpy.array([[1000.0, 1000.0]], dtype=numpy.float32)
    areas = numpy.array([[5.0, 7.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    # Only band (0, 1000) ft exists; 1000 m == ~3281 ft is outside it -> nothing
    # selected.
    (result,) = _run_calc(aoi, _variable(), raster, _scheme(999))

    assert result.area == 0.0
    assert math.isnan(result.value)


# --- the crossed (K=2) index -------------------------------------------------


def test_zone_index_crosses_two_axes_into_product_cells():
    # A 2x2 window. Elevation splits it top (low band) vs bottom (high band);
    # a second binary axis splits it left (class 0) vs right (class 1). Crossed,
    # that is four product cells, each one pixel.
    area = numpy.array([[1.0, 2.0], [4.0, 8.0]], dtype=numpy.float32)

    elev_ordinals = numpy.array([[0, 0], [1, 1]], dtype=numpy.int64)  # rows -> band
    side_ordinals = numpy.array([[0, 1], [0, 1]], dtype=numpy.int64)  # cols -> side

    elev_axis = (
        BandZone(key='0_1000', label='0-1000 ft', min=0, max=1000, unit='ft'),
        BandZone(key='1000_2000', label='1000-2000 ft', min=1000, max=2000, unit='ft'),
    )
    from snowtool.snowdb.zoning import ClassZone

    side_axis = (
        ClassZone(key='L', label='L', code=0),
        ClassZone(key='R', label='R', code=1),
    )

    index = _ZoneIndex.build(
        [elev_axis, side_axis],
        [elev_ordinals, side_ordinals],
        area,
    )

    # 2 x 2 = 4 product cells, in mixed-radix order (elevation outer, side inner).
    assert index.dims == [2, 2]
    assert index.cell_zones == (
        (elev_axis[0], side_axis[0]),
        (elev_axis[0], side_axis[1]),
        (elev_axis[1], side_axis[0]),
        (elev_axis[1], side_axis[1]),
    )
    # Each pixel is its own cell, so the cell areas are just the pixel areas.
    numpy.testing.assert_array_equal(index.areas, [1.0, 2.0, 4.0, 8.0])


def test_calc_reduces_each_crossed_cell_independently():
    # Cross a 2-band elevation axis (by row) with a 2-class side axis (by column);
    # each of the four product cells is one pixel, so each cell's mean is just its
    # own value -- a genuinely non-uniform, multi-populated-cell crossing.
    area = numpy.ones((2, 2), dtype=numpy.float32)
    values = numpy.array([[10, 20], [30, 40]], dtype=numpy.int16)
    aoi = _FakeAOI(numpy.zeros((2, 2), dtype=numpy.float32), area, values)

    elev_ord = numpy.array([[0, 0], [1, 1]], dtype=numpy.int64)
    side_ord = numpy.array([[0, 1], [0, 1]], dtype=numpy.int64)
    elev_axis = (_band(0, 1000), _band(1000, 2000))
    side_axis = (
        ClassZone(key='L', label='L', code=0),
        ClassZone(key='R', label='R', code=1),
    )
    index = _ZoneIndex.build(
        [elev_axis, side_axis], [elev_ord, side_ord], aoi.array,
    )

    results = asyncio.run(
        ZonalStats._calc(aoi, _variable(Reducer.MEAN), _FakeRaster(date(2018, 4, 27)),
                         index, cache=None),
    )
    by_zone = {r.zone: r.value for r in results}
    assert by_zone[(elev_axis[0], side_axis[0])] == 10.0
    assert by_zone[(elev_axis[0], side_axis[1])] == 20.0
    assert by_zone[(elev_axis[1], side_axis[0])] == 30.0
    assert by_zone[(elev_axis[1], side_axis[1])] == 40.0


def test_zone_index_with_no_axes_is_one_whole_basin_cell():
    # The K=0 case: no zone axes -> a single product cell with an empty zone tuple,
    # whose area is the whole in-AOI area. This is the whole-basin default.
    area = numpy.array([[1.0, 2.0], [0.0, 4.0]], dtype=numpy.float32)

    index = _ZoneIndex.build([], [], area)

    assert index.dims == []
    assert index.cell_zones == ((),)
    # Only the area > 0 pixels count (the 0-area pixel is outside the basin).
    assert float(index.areas[0]) == 7.0
    assert index.in_zone.tolist() == [[True, True], [False, True]]


def test_zone_index_excludes_pixels_out_of_any_axis():
    # A pixel out of zone on *either* axis is excluded from every crossed cell.
    area = numpy.array([[1.0, 1.0, 1.0]], dtype=numpy.float32)
    axis_a = numpy.array([[0, -1, 0]], dtype=numpy.int64)  # middle out on axis A
    axis_b = numpy.array([[0, 0, -1]], dtype=numpy.int64)  # last out on axis B

    zones = (BandZone(key='z', label='z', min=0, max=1, unit='x'),)
    index = _ZoneIndex.build([zones, zones], [axis_a, axis_b], area)

    # Only the first pixel is in-zone on both axes.
    assert index.in_zone.tolist() == [[True, False, False]]
    assert float(index.areas[0]) == 1.0


# --- dump_to_csv -------------------------------------------------------------


def _spec_with(variable: DatasetVariable) -> DatasetSpec:
    return DatasetSpec(
        name='test',
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=8,
            rows=8,
            tile_size=8,
        ),
        variables=[variable],
    )


def test_selection_overrides_resolve_per_layer_dataset_defaults():
    # Each axis inherits the dataset's configured default for *its* layer
    # (band_step_ft for elevation, threshold_pct for forest cover), translated to
    # the scheme's kwarg; an explicit selection value always wins.
    from snowtool.snowdb.zone_layer import available_zones
    from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

    spec = DatasetSpec(
        name='t',
        grid_params=GridParams(
            origin_x=-120.0, origin_y=45.0, px_size=0.01, cols=8, rows=8, tile_size=8,
        ),
        zones={
            'terrain': {'elevation': {'band_step_ft': 2000}},
            'landcover': {'forest_cover': {'threshold_pct': 50}},
        },
    )
    registry = available_zones(DEFAULT_ZONE_LAYER_PROVIDERS)
    elevation = registry['terrain.elevation']
    forest = registry['landcover.forest_cover']

    # Elevation inherits the dataset's band_step_ft...
    assert ZonalStats._selection_overrides(
        ZoneSelection('terrain.elevation'), elevation, spec,
    ) == {'step': 2000}
    # ...but an explicit step always wins.
    assert ZonalStats._selection_overrides(
        ZoneSelection('terrain.elevation', step=500), elevation, spec,
    ) == {'step': 500}
    # Forest cover inherits the dataset's threshold_pct.
    assert ZonalStats._selection_overrides(
        ZoneSelection('landcover.forest_cover'), forest, spec,
    ) == {'threshold': 50}
    # A threshold override passes straight through.
    assert ZonalStats._selection_overrides(
        ZoneSelection('landcover.forest_cover', threshold=30), forest, spec,
    ) == {'threshold': 30}


# --- parse_zone_selection (the --zone token parser) --------------------------


def _registry():
    from snowtool.snowdb.zone_layer import available_zones
    from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

    return available_zones(DEFAULT_ZONE_LAYER_PROVIDERS)


def test_parse_zone_selection_bare_layer():
    assert parse_zone_selection('terrain.elevation', _registry()) == ZoneSelection(
        'terrain.elevation',
    )


def test_parse_zone_selection_band_step_override():
    assert parse_zone_selection('terrain.elevation:500', _registry()) == ZoneSelection(
        'terrain.elevation', step=500,
    )


def test_parse_zone_selection_threshold_override():
    assert parse_zone_selection(
        'landcover.forest_cover:40', _registry(),
    ) == ZoneSelection('landcover.forest_cover', threshold=40.0)


def test_parse_zone_selection_unknown_layer():
    import pytest

    with pytest.raises(ValueError, match='Unknown zone layer'):
        parse_zone_selection('terrain.nope', _registry())


def test_parse_zone_selection_categorical_rejects_override():
    import pytest

    with pytest.raises(ValueError, match='takes no override'):
        parse_zone_selection('terrain.aspect:5', _registry())


def test_parse_zone_selection_non_integer_step():
    import pytest

    with pytest.raises(ValueError, match='band step must be an integer'):
        parse_zone_selection('terrain.elevation:x', _registry())


def _band(min_ft: int, max_ft: int) -> BandZone:
    return BandZone(
        key=f'{min_ft}_{max_ft}',
        label=f'{min_ft}-{max_ft} ft',
        min=min_ft,
        max=max_ft,
        unit='ft',
    )


def test_dump_to_csv_renders_a_no_data_cell_as_an_empty_cell():
    # One cell with data and one in-range-but-no-data cell (nan value), so the CSV
    # path's missing-value rendering is exercised. CSV is long form: one row/cell.
    variable = _variable(Reducer.MEAN)
    data_band = _band(0, 10000)
    nodata_band = _band(10000, 20000)
    day = date(2018, 4, 27)

    stats = ZonalStats(
        _spec_with(variable),
        {variable},
        ('terrain.elevation',),
        ((data_band,), (nodata_band,)),
        (day,),
        Result(date=day, zone=(data_band,), variable=variable, value=12.5, area=100.0),
        Result(date=day, zone=(nodata_band,), variable=variable,
               value=float('nan'), area=0.0),
    )

    out = io.StringIO()
    stats.dump_to_csv(out)
    header, data_row, nodata_row = list(csv.reader(io.StringIO(out.getvalue())))

    # A banded axis expands to two unit-bearing columns (min/max); then area + stat.
    assert header == [
        'date',
        'terrain.elevation_min_ft',
        'terrain.elevation_max_ft',
        'area_m2',
        'mean_swe_mm',
    ]
    assert data_row == [day.isoformat(), '0', '10000', '100.0', '12.5']
    # The no-data cell's mean is empty, never the literal 'nan'.
    assert nodata_row == [day.isoformat(), '10000', '20000', '0.0', '']
    assert 'nan' not in out.getvalue()


def test_dump_to_csv_band_axis_splits_categorical_axis_stays_one_column():
    # A crossed (band x categorical) cell: the banded axis is two unit-bearing
    # columns, the categorical (aspect) axis a single label column.
    variable = _variable(Reducer.MEAN)
    band = _band(0, 1000)
    flat = ClassZone(key='flat', label='flat', code=4)
    day = date(2018, 4, 27)
    cell = (band, flat)

    stats = ZonalStats(
        _spec_with(variable),
        {variable},
        ('terrain.elevation', 'terrain.aspect'),
        (cell,),
        (day,),
        Result(date=day, zone=cell, variable=variable, value=5.0, area=10.0),
    )

    out = io.StringIO()
    stats.dump_to_csv(out)
    header, row = list(csv.reader(io.StringIO(out.getvalue())))

    assert header == [
        'date',
        'terrain.elevation_min_ft',
        'terrain.elevation_max_ft',
        'terrain.aspect',
        'area_m2',
        'mean_swe_mm',
    ]
    assert row == [day.isoformat(), '0', '1000', 'flat', '10.0', '5.0']


def test_dump_to_csv_threshold_axis_splits_into_side_and_threshold():
    # A threshold axis expands to a label "side" column + a unit-bearing threshold
    # column, so the split point is structured (not buried in a string).
    variable = _variable(Reducer.MEAN)
    band = _band(0, 1000)
    forested = ThresholdZone(
        key='above', label='forested', threshold=50, unit='%', side='above',
    )
    day = date(2018, 4, 27)
    cell = (band, forested)

    stats = ZonalStats(
        _spec_with(variable),
        {variable},
        ('terrain.elevation', 'landcover.forest_cover'),
        (cell,),
        (day,),
        Result(date=day, zone=cell, variable=variable, value=5.0, area=10.0),
    )

    out = io.StringIO()
    stats.dump_to_csv(out)
    header, row = list(csv.reader(io.StringIO(out.getvalue())))

    assert header == [
        'date',
        'terrain.elevation_min_ft',
        'terrain.elevation_max_ft',
        'landcover.forest_cover_side',
        'landcover.forest_cover_threshold_%',
        'area_m2',
        'mean_swe_mm',
    ]
    assert row == [day.isoformat(), '0', '1000', 'forested', '50', '10.0', '5.0']

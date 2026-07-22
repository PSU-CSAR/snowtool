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

from dataclasses import dataclass
from datetime import date

import numpy
import pytest

from pydantic import ValidationError

from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.config import BandStepParams, ThresholdParams
from snowtool.snowdb.constants import M_TO_FT
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit
from snowtool.snowdb.zonal_stat_models import CompactZone
from snowtool.snowdb.zonal_stats import (
    ZonalStats,
    ZoneSelection,
    _ZoneIndex,
    parse_zone_selection,
    resolve_zone_axis,
)
from snowtool.snowdb.zones.terrain import ELEVATION_NODATA
from snowtool.snowdb.zones.zone_layer import available_zones
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS
from snowtool.snowdb.zones.zoning import (
    BandedZoning,
    BandZone,
    ClassZone,
    ThresholdZone,
    Zone,
)

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

    async def read_window(self, raster, *, dtype, fill, cache):
        return numpy.array(self._values, dtype=dtype)


@dataclass
class _CalcResult:
    """One (cell) reduction reassembled from the ``_calc`` value-vector.

    ``_calc`` now returns just the per-cell value vector (aligned with
    ``zone_index.cell_zones``); these tests still assert per-cell, so a local
    helper pairs each value back with its zone tuple + the index's cell area --
    the same (zone, value, area) the deleted ``Result`` carried, without resurrecting
    it.
    """

    date: date
    variable: DatasetVariable
    zone: tuple[Zone, ...]
    value: float
    area: float


def _calc_results(aoi, variable, raster, zone_index):
    """Run ``_calc`` and pair each cell value with its zone tuple and area.

    Untyped like ``_run_calc`` so mypy does not flag the deliberate ``cache=None``
    (the ``_FakeAOI`` overrides ``read_window``, so the cache is
    never touched) -- matching the pre-existing test convention here.
    """
    values = asyncio.run(
        ZonalStats._calc(aoi, variable, raster, zone_index, cache=None),
    )
    return [
        _CalcResult(
            date=raster.date,
            variable=variable,
            zone=cell,
            value=float(values[idx]),
            area=float(zone_index.areas[idx]),
        )
        for idx, cell in enumerate(zone_index.cell_zones)
    ]


def _run_calc(aoi, variable, raster, scheme):
    # A single elevation axis: the K=1 case of the crossed index.
    ordinals = scheme.assign(aoi.elevation)
    zone_index = _ZoneIndex.build(
        [scheme.zones()],
        [ordinals],
        aoi.array,
    )
    return _calc_results(aoi, variable, raster, zone_index)


def _bounds(result: _CalcResult) -> tuple[int | float, int | float]:
    (band,) = result.zone
    assert isinstance(band, BandZone)
    return band.min, band.max


@pytest.mark.parametrize('nan', [float('nan'), numpy.nan])
def test_dataset_variable_rejects_nan_nodata(nan):
    # The stats reader masks fill pixels with `values != variable.nodata`, and
    # `x != NaN` is always True -- a NaN sentinel would never be excluded and would
    # poison the reduction. Construction must reject it up front.
    with pytest.raises(ValueError, match='finite sentinel'):
        DatasetVariable(
            key='swe',
            unit=Unit(name='mm', scale_factor=1),
            reducer=Reducer.MEAN,
            dtype='float32',
            nodata=nan,
            glob='*',
        )


def test_dataset_variable_accepts_finite_nodata():
    # A finite out-of-range sentinel is the required form; it must construct fine.
    variable = _variable()
    assert variable.nodata == float(NODATA)


@pytest.mark.parametrize(
    ('reducer', 'expected_value'),
    [
        # Mean is weighted by each valid pixel's area and excludes the nodata cell:
        # an area-weighted 300 (vs ~233.3 unweighted -> the exact 300 proves weighting).
        (Reducer.MEAN, 300.0),
        # Total is the area-weighted sum over the valid pixels: a basin total of 21000.
        (Reducer.TOTAL, 21000.0),
    ],
)
def test_calc_area_reducer_independent_and_weighted(reducer, expected_value):
    # All four elevations (m) fall inside the single 0..10000 ft band, so the cell
    # selection is the whole window. The bottom-left pixel (area 30) is nodata.
    elevations = numpy.array([[500.0, 1000.0], [1500.0, 2000.0]], dtype=numpy.float32)
    # Deliberately non-uniform per-pixel ground areas.
    areas = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200], [NODATA, 400]], dtype=numpy.int16)

    variable = _variable(reducer)
    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    # domain 0..9999 ft @ 10000 step -> exactly one band (0, 10000).
    (result,) = _run_calc(aoi, variable, raster, _scheme(9999, step=10000))

    assert result.date == raster.date
    assert result.variable is variable
    assert _bounds(result) == (0, 10000)
    # area counts every in-cell pixel, including the nodata one (10 + 20 + 30 + 40),
    # independent of the reducer.
    assert result.area == 100.0
    assert result.value == expected_value


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
        [elev_axis, side_axis],
        [elev_ord, side_ord],
        aoi.array,
    )

    results = _calc_results(
        aoi,
        _variable(Reducer.MEAN),
        _FakeRaster(date(2018, 4, 27)),
        index,
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


def test_resolve_zone_axis_folds_dataset_default_then_override():
    # resolve_zone_axis folds the dataset's configured default for the layer into a
    # configured scheme (band_step_ft for elevation, threshold_pct for forest cover)
    # and lets an explicit per-query override win on top.
    spec = DatasetSpec(
        name='t',
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=8,
            rows=8,
            tile_size=8,
        ),
        zones={
            'terrain': {'elevation': BandStepParams(band_step_ft=2000)},
            'landcover': {'forest_cover': ThresholdParams(threshold_pct=50)},
        },
    )
    registry = available_zones(DEFAULT_ZONE_LAYER_PROVIDERS)
    elevation = registry['terrain.elevation']
    forest = registry['landcover.forest_cover']

    # Elevation inherits the dataset's band_step_ft; the returned available is the
    # registry entry unchanged.
    available, scheme = resolve_zone_axis(
        ZoneSelection('terrain.elevation'),
        registry,
        spec,
    )
    assert available is elevation
    assert scheme.default_step == 2000
    # ...but an explicit step always wins.
    _, scheme = resolve_zone_axis(
        ZoneSelection('terrain.elevation', override=500),
        registry,
        spec,
    )
    assert scheme.default_step == 500
    # Forest cover inherits the dataset's threshold_pct.
    _, scheme = resolve_zone_axis(
        ZoneSelection('landcover.forest_cover'),
        registry,
        spec,
    )
    assert scheme.default_threshold == 50
    # A threshold override passes straight through.
    available, scheme = resolve_zone_axis(
        ZoneSelection('landcover.forest_cover', override=30),
        registry,
        spec,
    )
    assert available is forest
    assert scheme.default_threshold == 30


def test_resolve_zone_axis_unknown_layer_raises():
    spec = _spec_with(_variable())
    registry = available_zones(DEFAULT_ZONE_LAYER_PROVIDERS)
    with pytest.raises(QueryParameterError, match='Unknown zone layer'):
        resolve_zone_axis(ZoneSelection('terrain.nope'), registry, spec)


# --- parse_zone_selection (the --zone token parser) --------------------------


def _registry():

    return available_zones(DEFAULT_ZONE_LAYER_PROVIDERS)


@pytest.mark.parametrize(
    ('token', 'expected'),
    [
        ('terrain.elevation', ZoneSelection('terrain.elevation')),
        (
            'terrain.elevation:band_step_ft=500',
            ZoneSelection('terrain.elevation', override=500),
        ),
        (
            'landcover.forest_cover:threshold_pct=40',
            ZoneSelection('landcover.forest_cover', override=40.0),
        ),
    ],
)
def test_parse_zone_selection_valid(token, expected):
    assert parse_zone_selection(token, _registry()) == expected


@pytest.mark.parametrize(
    ('token', 'match'),
    [
        ('terrain.nope', 'Unknown zone layer'),
        ('terrain.aspect:band_step_ft=5', 'takes no override'),
        ('terrain.elevation:500', 'expected .*=.*'),
        ('terrain.elevation:nope=500', "its override is 'band_step_ft'"),
        ('terrain.elevation:band_step_ft=x', 'band step must be an integer'),
    ],
)
def test_parse_zone_selection_invalid(token, match):
    with pytest.raises(ValueError, match=match):
        parse_zone_selection(token, _registry())


def _band(min_ft: int, max_ft: int) -> BandZone:
    return BandZone(
        key=f'{min_ft}_{max_ft}',
        label=f'{min_ft}-{max_ft} ft',
        min=min_ft,
        max=max_ft,
        unit='ft',
    )


_DUMP_DAY = date(2018, 4, 27)


def _stats_from_cells(
    variable,
    zone_layers,
    cells,
    day,
    cell_specs,
):
    """Build a ``ZonalStats`` and fill its array through the public ``fill`` seam.

    ``cell_specs`` is ``(zone, value, area)`` per crossed cell, in ``cells`` order.
    Since ``Result`` is gone, the serializer tests fill the array the same way
    :meth:`ZonalStats.calculate` does: one vectorized ``fill(date, variable, values,
    areas)`` per (date, variable). The specs are turned into cell-aligned value and
    area vectors (matching ``cells`` order) and written in one call.
    """
    stats = ZonalStats({variable}, zone_layers, cells, (day,))
    cell_index = {cell: idx for idx, cell in enumerate(cells)}
    values = numpy.empty(len(cells), dtype=numpy.float64)
    areas = numpy.empty(len(cells), dtype=numpy.float64)
    for zone, value, area in cell_specs:
        idx = cell_index[zone]
        values[idx] = value
        areas[idx] = area
    stats.fill(day, variable, values, areas)
    return stats


_FLAT = ClassZone(key='flat', label='flat', code=4)
_FORESTED = ThresholdZone(
    key='above',
    label='forested',
    threshold=50,
    unit='%',
    side='above',
)


@pytest.mark.parametrize(
    ('zone_layers', 'cells', 'results_spec', 'expected_header', 'expected_rows'),
    [
        pytest.param(
            ('terrain.elevation',),
            ((_band(0, 10000),), (_band(10000, 20000),)),
            # One cell with data and one AOI-present-but-no-data cell (positive area,
            # nan value) so the CSV missing-value rendering is exercised. The second
            # cell keeps a nonzero area on purpose: an empty (0-area) cell would be
            # dropped by the default filter (that path is its own test below).
            [
                ((_band(0, 10000),), 12.5, 100.0),
                ((_band(10000, 20000),), float('nan'), 50.0),
            ],
            [
                'date',
                'terrain.elevation_min_ft',
                'terrain.elevation_max_ft',
                'area_m2',
                'mean_swe_mm',
            ],
            # A banded axis -> two unit-bearing columns (min/max); the no-data cell's
            # mean renders empty, never the literal 'nan'.
            [
                [_DUMP_DAY.isoformat(), '0', '10000', '100.0', '12.5'],
                [_DUMP_DAY.isoformat(), '10000', '20000', '50.0', ''],
            ],
            id='nodata-cell-empty',
        ),
        pytest.param(
            ('terrain.elevation', 'terrain.aspect'),
            ((_band(0, 1000), _FLAT),),
            [((_band(0, 1000), _FLAT), 5.0, 10.0)],
            # The categorical (aspect) axis is a single label column.
            [
                'date',
                'terrain.elevation_min_ft',
                'terrain.elevation_max_ft',
                'terrain.aspect',
                'area_m2',
                'mean_swe_mm',
            ],
            [[_DUMP_DAY.isoformat(), '0', '1000', 'flat', '10.0', '5.0']],
            id='band-x-categorical',
        ),
        pytest.param(
            ('terrain.elevation', 'landcover.forest_cover'),
            ((_band(0, 1000), _FORESTED),),
            [((_band(0, 1000), _FORESTED), 5.0, 10.0)],
            # A threshold axis -> a label "side" column + a unit-bearing threshold
            # column, so the split point is structured (not buried in a string).
            [
                'date',
                'terrain.elevation_min_ft',
                'terrain.elevation_max_ft',
                'landcover.forest_cover_side',
                'landcover.forest_cover_threshold_%',
                'area_m2',
                'mean_swe_mm',
            ],
            [[_DUMP_DAY.isoformat(), '0', '1000', 'forested', '50', '10.0', '5.0']],
            id='band-x-threshold',
        ),
    ],
)
def test_dump_to_csv_expands_each_axis_kind(
    zone_layers,
    cells,
    results_spec,
    expected_header,
    expected_rows,
):
    variable = _variable(Reducer.MEAN)
    stats = _stats_from_cells(
        variable,
        zone_layers,
        cells,
        _DUMP_DAY,
        results_spec,
    )

    out = io.StringIO()
    stats.dump_to_csv(out)
    header, *rows = list(csv.reader(io.StringIO(out.getvalue())))

    assert header == expected_header
    assert rows == expected_rows
    assert 'nan' not in out.getvalue()


# --- empty (0-area) zone filtering -------------------------------------------


def _two_band_stats() -> ZonalStats:
    """A single elevation axis with a populated band and an empty (0-area) band.

    The high band is a crossed combination no AOI pixel falls in: 0 area, nan value
    -- exactly the empty cell the default output drops.
    """
    variable = _variable(Reducer.MEAN)
    cells = ((_band(0, 1000),), (_band(1000, 2000),))
    return _stats_from_cells(
        variable,
        ('terrain.elevation',),
        cells,
        _DUMP_DAY,
        [
            (cells[0], 5.0, 10.0),
            (cells[1], float('nan'), 0.0),
        ],
    )


def test_compact_zone_area_must_be_non_negative():
    # Ported from the deleted generated-model tests: CompactZone keeps the same
    # area_m2 >= 0 constraint the per-dataset model used to carry.
    with pytest.raises(ValidationError):
        CompactZone(zone=[], area_m2=-1.0)


def test_dump_compact_drops_empty_zones_by_default():
    # The empty (0-area) high band is dropped; only the populated band survives.
    compact = _two_band_stats().dump_compact()
    (zone,) = compact.zones
    assert zone.area_m2 == 10.0


def test_dump_compact_keeps_empty_zones_when_requested():
    # Opting in restores the full crossed product, empty cells included.
    compact = _two_band_stats().dump_compact(include_empty_zones=True)
    assert [zone.area_m2 for zone in compact.zones] == [10.0, 0.0]


def test_dump_to_csv_drops_empty_zones_by_default():
    out = io.StringIO()
    _two_band_stats().dump_to_csv(out)
    _, *rows = list(csv.reader(io.StringIO(out.getvalue())))
    # Only the populated (0, 1000) band row remains.
    assert rows == [[_DUMP_DAY.isoformat(), '0', '1000', '10.0', '5.0']]


def test_dump_to_csv_keeps_empty_zones_when_requested():
    out = io.StringIO()
    _two_band_stats().dump_to_csv(out, include_empty_zones=True)
    _, *rows = list(csv.reader(io.StringIO(out.getvalue())))
    assert rows == [
        [_DUMP_DAY.isoformat(), '0', '1000', '10.0', '5.0'],
        [_DUMP_DAY.isoformat(), '1000', '2000', '0.0', ''],
    ]


@pytest.mark.parametrize('include_empty_zones', [False, True])
def test_iter_csv_matches_dump_to_csv(include_empty_zones):
    # dump_to_csv is now just out.writelines(iter_csv(...)); pin that the
    # generator's chunks concatenate to exactly what the buffered writer
    # produces, for both the default-filtered and full-product cases.
    stats = _two_band_stats()

    out = io.StringIO()
    stats.dump_to_csv(out, include_empty_zones=include_empty_zones)

    assert ''.join(stats.iter_csv(include_empty_zones=include_empty_zones)) == (
        out.getvalue()
    )


def test_dump_compact_whole_basin_cell_is_never_dropped():
    # The K=0 (unstratified) cell always has area and must survive the default filter.
    variable = _variable(Reducer.MEAN)
    stats = _stats_from_cells(
        variable,
        (),
        ((),),
        _DUMP_DAY,
        [((), 5.0, 100.0)],
    )
    compact = stats.dump_compact()
    (zone,) = compact.zones
    assert zone.area_m2 == 100.0

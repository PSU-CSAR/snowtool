"""Unit tests for ZonalStats._calc band selection, area, and reduction.

These stub the AOI/SNODAS raster I/O so the test pins down the pure numeric
behaviour of _calc directly, with non-uniform pixel areas and a nodata cell --
the cases the uniform end-to-end pipeline test cannot distinguish:

  * area is the band's geographic area, independent of which pixels are nodata,
  * MEAN is area-weighted over only the pixels that have data, and
  * TOTAL is the area-weighted sum (a basin total) over those pixels.

Banding now runs through the elevation layer's :class:`BandedZoning` scheme; each
test tunes the scheme's domain so it yields exactly the bands it wants to exercise.
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
from snowtool.snowdb.zonal_stats import Result, ZonalStats, _BandIndex
from snowtool.snowdb.zoning import BandedZoning, BandZone

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
    """Stands in for AOIRasterWithArea; load_* just stamps fixed values.

    ``array`` is now the boolean AOI mask (decoupled from the DEM); elevation is
    held separately and fed to the band index, mirroring the real read path where
    elevation is loaded live from the terrain set. The mask is all-inside here so
    band selection covers the whole window.
    """

    def __init__(self, elevation, area, values) -> None:
        self.elevation = elevation
        self.array = numpy.ones(elevation.shape, dtype=numpy.uint8)
        self.area = area
        self._values = values

    async def load_raster_tiles_into_array(self, raster, values_array, cache):
        values_array[:] = self._values


def _run_calc(aoi, variable, raster, scheme, *, step=None):
    band_index = _BandIndex.build(
        scheme, aoi.elevation, aoi.array, aoi.area, step=step,
    )
    return asyncio.run(ZonalStats._calc(aoi, variable, raster, band_index, cache=None))


def test_calc_area_is_variable_independent_and_mean_is_area_weighted():
    # All four elevations (m) fall inside the single 0..10000 ft band, so the band
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
    assert (result.elevation_band.min, result.elevation_band.max) == (0, 10000)

    # area counts every in-band pixel, including the one that is nodata for this
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


def test_calc_band_with_terrain_but_no_data_has_area_and_nan_value():
    elevations = numpy.array([[1000.0, 1000.0]], dtype=numpy.float32)
    areas = numpy.array([[5.0, 7.0]], dtype=numpy.float32)
    # Every pixel is nodata for this variable.
    values = numpy.array([[NODATA, NODATA]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))

    (result,) = _run_calc(aoi, _variable(), raster, _scheme(9999, step=10000))

    # The band still covers ground (5 + 7) even though no data exists.
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

    assert ((low.elevation_band.min, low.elevation_band.max), low.value, low.area) == (
        (0, 1000),
        100.0,
        10.0,
    )
    assert (
        (high.elevation_band.min, high.elevation_band.max),
        high.value,
        high.area,
    ) == ((1000, 2000), 200.0, 20.0)


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
    # Empty band reads nan (not a spurious 0) and carries no area.
    assert math.isnan(mid.value)
    assert mid.area == 0.0


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


def _band(min_ft: int, max_ft: int) -> BandZone:
    return BandZone(
        key=f'{min_ft}_{max_ft}',
        label=f'{min_ft}-{max_ft} ft',
        min=min_ft,
        max=max_ft,
        unit='ft',
    )


def test_dump_to_csv_renders_a_no_data_band_as_an_empty_cell():
    # One band with data and one in-range-but-no-data band (nan value), so the
    # CSV path's missing-value rendering is exercised.
    variable = _variable(Reducer.MEAN)
    data_band = _band(0, 10000)
    nodata_band = _band(10000, 20000)
    day = date(2018, 4, 27)

    stats = ZonalStats(
        _spec_with(variable),
        {variable},
        (data_band, nodata_band),
        (day,),
        Result(date=day, elevation_band=data_band, variable=variable,
               value=12.5, area=100.0),
        Result(date=day, elevation_band=nodata_band, variable=variable,
               value=float('nan'), area=0.0),
    )

    out = io.StringIO()
    stats.dump_to_csv(out)
    header, row = list(csv.reader(io.StringIO(out.getvalue())))

    # Columns: date, area/mean for data_band, then area/mean for nodata_band.
    assert header[0] == 'date'
    assert header[2].startswith('mean_swe_mm')
    assert row[0] == day.isoformat()
    assert row[1] == '100.0'  # data band area
    assert row[2] == '12.5'  # data band mean
    assert row[3] == '0.0'  # nodata band area
    assert row[4] == ''  # nodata band mean -> empty, never the literal 'nan'
    assert 'nan' not in out.getvalue()


def test_calc_band_with_no_terrain_has_zero_area_and_nan_value():
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

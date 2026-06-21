"""Unit tests for ZonalStats._calc band selection, area, and reduction.

These stub the AOI/SNODAS raster I/O so the test pins down the pure numeric
behaviour of _calc directly, with non-uniform pixel areas and a nodata cell --
the cases the uniform end-to-end pipeline test cannot distinguish:

  * area is the band's geographic area, independent of which pixels are nodata,
  * MEAN is area-weighted over only the pixels that have data, and
  * TOTAL is the area-weighted sum (a basin total) over those pixels.
"""

import asyncio
import csv
import io
import math

from datetime import date

import numpy
import pytest

from snowtool.snowdb.elevation_band import ElevationBand
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit
from snowtool.snowdb.zonal_stats import Result, ZonalStats, _BandIndex

NODATA = -9999  # the variable's int16 nodata sentinel for these stubs


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
    """Stands in for AOIRasterWithArea; load_* just stamps fixed values."""

    def __init__(self, array, area, values) -> None:
        self.array = array
        self.area = area
        self._values = values

    async def load_raster_tiles_into_array(self, raster, values_array, cache):
        values_array[:] = self._values


def _run_calc(aoi, variable, raster, bands):
    band_index = _BandIndex.build(aoi, bands)
    return asyncio.run(ZonalStats._calc(aoi, variable, raster, band_index, cache=None))


def test_calc_area_is_variable_independent_and_mean_is_area_weighted():
    # All four elevations (m) fall inside the single 0..10000 ft band
    # (0 .. ~3048 m), so the band selection is the whole window.
    elevations = numpy.array([[500.0, 1000.0], [1500.0, 2000.0]], dtype=numpy.float32)
    # Deliberately non-uniform per-pixel ground areas.
    areas = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float32)
    # The bottom-left pixel (area 30) is nodata for this variable.
    values = numpy.array([[100, 200], [NODATA, 400]], dtype=numpy.int16)

    variable = _variable(Reducer.MEAN)
    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))
    band = ElevationBand(0, 10000)

    (result,) = _run_calc(aoi, variable, raster, [band])

    assert result.date == raster.date
    assert result.variable is variable
    assert result.elevation_band == band

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
    band = ElevationBand(0, 10000)

    (result,) = _run_calc(aoi, variable, raster, [band])

    # Area-weighted sum over the valid pixels gives a basin total of 21000.
    assert result.value == 21000.0


def test_calc_band_with_terrain_but_no_data_has_area_and_nan_value():
    elevations = numpy.array([[1000.0, 1000.0]], dtype=numpy.float32)
    areas = numpy.array([[5.0, 7.0]], dtype=numpy.float32)
    # Every pixel is nodata for this variable.
    values = numpy.array([[NODATA, NODATA]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))
    band = ElevationBand(0, 10000)

    (result,) = _run_calc(aoi, _variable(), raster, [band])

    # The band still covers ground (5 + 7) even though no data exists.
    assert result.area == 12.0
    assert math.isnan(result.value)


def test_calc_assigns_pixels_to_their_bands_in_one_pass():
    # 100 m falls in band (0, 1000) ft; 400 m in (1000, 2000) ft.
    elevations = numpy.array([[100.0, 400.0]], dtype=numpy.float32)
    areas = numpy.array([[10.0, 20.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))
    bands = [ElevationBand(0, 1000), ElevationBand(1000, 2000)]

    low, high = _run_calc(aoi, _variable(Reducer.MEAN), raster, bands)

    assert (low.elevation_band, low.value, low.area) == (bands[0], 100.0, 10.0)
    assert (high.elevation_band, high.value, high.area) == (bands[1], 200.0, 20.0)


def test_calc_empty_middle_band_is_nan_with_zero_area():
    # 100 m -> band 0, 700 m -> band 2; the (1000, 2000) ft middle band is empty.
    elevations = numpy.array([[100.0, 700.0]], dtype=numpy.float32)
    areas = numpy.array([[10.0, 20.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    raster = _FakeRaster(date(2018, 4, 27))
    bands = [
        ElevationBand(0, 1000),
        ElevationBand(1000, 2000),
        ElevationBand(2000, 3000),
    ]

    low, mid, high = _run_calc(aoi, _variable(Reducer.TOTAL), raster, bands)

    assert low.value == 100.0 * 10.0
    assert high.value == 200.0 * 20.0
    # Empty band reads nan (not a spurious 0) and carries no area.
    assert math.isnan(mid.value)
    assert mid.area == 0.0


def test_band_index_rejects_noncontiguous_bands():
    aoi = _FakeAOI(
        numpy.zeros((1, 1), dtype=numpy.float32),
        numpy.ones((1, 1), dtype=numpy.float32),
        numpy.zeros((1, 1), dtype=numpy.int16),
    )
    with pytest.raises(ValueError, match='contiguous'):
        _BandIndex.build(aoi, [ElevationBand(0, 1000), ElevationBand(2000, 3000)])


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


def test_dump_to_csv_renders_a_no_data_band_as_an_empty_cell():
    # One band with data and one in-range-but-no-data band (nan value), so the
    # CSV path's missing-value rendering is exercised.
    variable = _variable(Reducer.MEAN)
    data_band = ElevationBand(0, 10000)
    nodata_band = ElevationBand(10000, 20000)
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
    # 1000 m == ~3281 ft, well outside this band -> nothing selected.
    band = ElevationBand(0, 1000)

    (result,) = _run_calc(aoi, _variable(), raster, [band])

    assert result.area == 0.0
    assert math.isnan(result.value)

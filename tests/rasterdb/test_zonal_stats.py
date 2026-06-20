"""Unit tests for ZonalStats._calc band selection, area, and weighting.

These stub the AOI/SNODAS raster I/O so the test pins down the pure numeric
behaviour of _calc directly, with non-uniform pixel areas and a nodata cell --
the cases the uniform end-to-end pipeline test cannot distinguish:

  * area_m2 is the band's geographic area, independent of which product pixels
    are nodata, and
  * the mean is area-weighted over only the pixels that have data.
"""

import asyncio
import math

from datetime import datetime

import numpy

from snowtool.rasterdb.constants import NODATA
from snowtool.rasterdb.elevation_band import ElevationBand
from snowtool.rasterdb.fileinfo import Product
from snowtool.rasterdb.zonal_stats import ZonalStats


class _FakeSNODAS:
    """Just enough of SNODASRaster for _calc: a date and a product."""

    class _FileInfo:
        def __init__(self, dt: datetime, product: Product) -> None:
            self.datetime = dt
            self.product = product

    def __init__(self, dt: datetime, product: Product) -> None:
        self.fileinfo = self._FileInfo(dt, product)


class _FakeAOI:
    """Stands in for AOIRasterWithArea; load_* just stamps fixed values."""

    def __init__(self, array, area, values) -> None:
        self.array = array
        self.area = area
        self._values = values

    async def load_raster_tiles_into_array(self, snodas, values_array, cache):
        values_array[:] = self._values


def _run_calc(aoi, snodas, bands):
    return asyncio.run(ZonalStats._calc(aoi, snodas, bands, cache=None))


def test_calc_area_is_product_independent_and_mean_is_area_weighted():
    # All four elevations (m) fall inside the single 0..10000 ft band
    # (0 .. ~3048 m), so the band selection is the whole window.
    elevations = numpy.array([[500.0, 1000.0], [1500.0, 2000.0]], dtype=numpy.float32)
    # Deliberately non-uniform per-pixel ground areas.
    areas = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float32)
    # The bottom-left pixel (area 30) is nodata for this product.
    values = numpy.array([[100, 200], [NODATA, 400]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    snodas = _FakeSNODAS(datetime(2018, 4, 27), Product.SNOW_WATER_EQUIVALENT)
    band = ElevationBand(0, 10000)

    (result,) = _run_calc(aoi, snodas, [band])

    assert result.date == snodas.fileinfo.datetime.date()
    assert result.product is Product.SNOW_WATER_EQUIVALENT
    assert result.elevation_band == band

    # area_m2 counts every in-band pixel, including the one that is nodata for
    # this product (10 + 20 + 30 + 40).
    assert result.area == 100.0

    # Mean is weighted by each *valid* pixel's area and excludes the nodata
    # cell: (100*10 + 200*20 + 400*40) / (10 + 20 + 40) = 21000 / 70 == 300.
    assert result.mean == 300.0
    # A plain unweighted mean would be (100 + 200 + 400) / 3 == 233.33, so this
    # assertion fails if the area weighting is dropped.
    assert result.mean != (100 + 200 + 400) / 3


def test_calc_band_with_terrain_but_no_data_has_area_and_nan_mean():
    elevations = numpy.array([[1000.0, 1000.0]], dtype=numpy.float32)
    areas = numpy.array([[5.0, 7.0]], dtype=numpy.float32)
    # Every pixel is nodata for this product.
    values = numpy.array([[NODATA, NODATA]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    snodas = _FakeSNODAS(datetime(2018, 4, 27), Product.SNOW_WATER_EQUIVALENT)
    band = ElevationBand(0, 10000)

    (result,) = _run_calc(aoi, snodas, [band])

    # The band still covers ground (5 + 7) even though no product data exists.
    assert result.area == 12.0
    assert math.isnan(result.mean)


def test_calc_band_with_no_terrain_has_zero_area_and_nan_mean():
    elevations = numpy.array([[1000.0, 1000.0]], dtype=numpy.float32)
    areas = numpy.array([[5.0, 7.0]], dtype=numpy.float32)
    values = numpy.array([[100, 200]], dtype=numpy.int16)

    aoi = _FakeAOI(elevations, areas, values)
    snodas = _FakeSNODAS(datetime(2018, 4, 27), Product.SNOW_WATER_EQUIVALENT)
    # 1000 m == ~3281 ft, well outside this band -> nothing selected.
    band = ElevationBand(0, 1000)

    (result,) = _run_calc(aoi, snodas, [band])

    assert result.area == 0.0
    assert math.isnan(result.mean)

"""SNODAS spec facts: variable units and the tenths-of-a-Kelvin temperature scale."""

import numpy
import pytest
import rasterio

from snowtool.exceptions import IngestSourceError
from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS
from snowtool.snowdb.datasets.snodas import (
    SNODAS_SPEC,
    Product,
    SNODASInputRaster,
    SNODASInputRasterSet,
    SNODASName,
)


def test_registered_in_default_specs():
    assert 'snodas' in {s.name for s in DEFAULT_DATASET_SPECS}
    assert SNODAS_SPEC.name == 'snodas'


def test_average_temp_reports_kelvin_from_tenths():
    # SNODAS stores snowpack average temperature in tenths of a Kelvin, so the
    # reporting scale is 10: a raw 2731 (the 273.1 K / 0 degrees C melt cap)
    # scales to ~273 K, not ~2730.
    unit = SNODAS_SPEC.variables['average_temp'].unit
    assert unit.name == 'k'
    assert unit.scale_factor == 10
    assert unit.scale(2731) == pytest.approx(273.1)
    assert unit.scale(2636) == pytest.approx(263.6)


@pytest.mark.parametrize(
    ('key', 'name', 'scale_factor'),
    [
        ('swe', 'mm', 1),
        ('depth', 'mm', 1),
        ('precip_solid', 'kg_per_m2', 10),
        ('precip_liquid', 'kg_per_m2', 10),
        ('average_temp', 'k', 10),
        ('sublimation', 'mm', 100),
        ('sublimation_blowing', 'mm', 100),
        ('runoff', 'mm', 100),
    ],
)
def test_variable_units(key, name, scale_factor):
    unit = Product(key).unit()
    assert unit.name == name
    assert unit.scale_factor == scale_factor


def test_from_names_refuses_duplicate_product():
    # Two stems that parse to the same product (SWE, code 1034) differing only in
    # timecode -- the archive must hold exactly one per product, so a repeat must
    # raise rather than let tar ordering silently pick a last-wins winner.
    swe_snapshot = 'us_ssmv11034SlL00T0001TTNATS2019020205HP001'
    swe_integ = 'us_ssmv11034SlL00T0024TTNATS2019020205DP001'
    with pytest.raises(IngestSourceError) as exc:
        SNODASInputRasterSet.from_names([swe_snapshot, swe_integ])
    message = str(exc.value)
    assert 'swe' in message
    assert swe_snapshot in message
    assert swe_integ in message


def test_trim_header_preserves_within_limit_lines(tmp_path):
    # A header whose lines are within the limit round-trips exactly one newline per
    # line -- no doubled/inserted blank lines (the `for line in f` newline bug).
    hdr = tmp_path / 'header.txt'
    hdr.write_bytes(b'first line\nsecond line\nthird line\n')
    SNODASInputRaster.trim_header(hdr)
    assert hdr.read_bytes() == b'first line\nsecond line\nthird line\n'


def test_trim_header_truncates_over_limit_line(tmp_path):
    # An over-limit line is truncated to line_limit chars + a single newline.
    line_limit = 255
    hdr = tmp_path / 'header.txt'
    long_value = b'x' * 400
    hdr.write_bytes(long_value + b'\nshort\n')
    SNODASInputRaster.trim_header(hdr)
    # Truncated to line_limit + single newline, no doubled newlines anywhere.
    assert hdr.read_bytes() == b'x' * line_limit + b'\nshort\n'


# A minimal parseable SNODAS stem (masked region, pinned 05 time-step) -- the same
# literal test_ingest.py's `_snodas_stems('20190202')[0]` yields, duplicated here so
# this file doesn't reach across test modules for it.
_SNODAS_STEM = 'us_ssmv11034SlL00T0001TTNATS2019020205HP001'


def test_snodas_read_array_refuses_off_grid_shape(tmp_path, monkeypatch):
    """A header whose band is not the dataset grid shape raises, not mis-writes."""
    monkeypatch.setattr(
        'snowtool.snowdb.datasets.snodas.SNODASInputRaster.trim_header',
        staticmethod(lambda hdr: None),
    )
    path = tmp_path / f'{_SNODAS_STEM}.txt'
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        dtype='int16',
        count=1,
        height=3,
        width=3,
        crs='EPSG:4326',
        transform=rasterio.transform.from_origin(-124, 52, 0.5, 0.5),
    ) as dst:
        dst.write(numpy.zeros((1, 3, 3), dtype='int16'))
    raster = SNODASInputRaster(
        SNODASName(_SNODAS_STEM),
        path,
        'v1:abc',
        transform=rasterio.transform.from_origin(-124, 52, 0.5, 0.5),
        crs=rasterio.crs.CRS.from_epsg(4326),
        tile_size=256,
        expected_shape=(4, 4),
    )
    with pytest.raises(IngestSourceError, match='expected the dataset grid shape'):
        raster.read_array()

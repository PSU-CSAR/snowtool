"""SNODAS spec facts: variable units and the tenths-of-a-Kelvin temperature scale."""

import pytest

from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS
from snowtool.snowdb.datasets.snodas import SNODAS_SPEC, Product


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

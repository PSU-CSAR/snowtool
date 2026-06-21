"""Ordering invariants for ElevationBand.

The comparison was previously inconsistent: __gt__ was `not self < other`, so two
equal bands reported both `a == b` and `a > b`. These tests lock the total
ordering down.
"""

from snowtool.snowdb.elevation_band import ElevationBand


def test_equal_bands_are_not_greater_or_less():
    a = ElevationBand(1000, 2000)
    b = ElevationBand(1000, 2000)

    assert a == b
    assert not (a < b)
    assert not (a > b)
    assert a <= b
    assert a >= b


def test_strict_ordering_by_min_then_max():
    lower = ElevationBand(1000, 2000)
    higher_min = ElevationBand(2000, 3000)
    same_min_higher_max = ElevationBand(1000, 3000)

    assert lower < higher_min
    assert higher_min > lower
    assert lower < same_min_higher_max
    assert same_min_higher_max > lower


def test_sorted_orders_bands_ascending():
    bands = [
        ElevationBand(2000, 3000),
        ElevationBand(0, 1000),
        ElevationBand(1000, 2000),
        ElevationBand(1000, 3000),
    ]

    assert sorted(bands) == [
        ElevationBand(0, 1000),
        ElevationBand(1000, 2000),
        ElevationBand(1000, 3000),
        ElevationBand(2000, 3000),
    ]

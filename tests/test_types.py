"""``snowtool.types.to_date``: timezone-independent YYYYMMDD/YYYY-MM-DD parsing."""

from datetime import date

import pytest

from snowtool.types import to_date


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('20240101', date(2024, 1, 1)),
        ('2024-01-01', date(2024, 1, 1)),
        ('20240229', date(2024, 2, 29)),  # leap day
        ('2024-02-29', date(2024, 2, 29)),  # leap day, dashed
        ('20231231', date(2023, 12, 31)),
    ],
)
def test_to_date_parses_exact_value(value, expected):
    assert to_date(value) == expected


@pytest.mark.parametrize(
    'value',
    [
        'not-a-date',
        '2024-02-30',  # not a leap day
        '20240230',
        '',
        '2024/01/01',
    ],
)
def test_to_date_rejects_invalid_input(value):
    with pytest.raises(ValueError, match=r'does not match format|must be in range'):
        to_date(value)

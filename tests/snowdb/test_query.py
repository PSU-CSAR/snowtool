"""The temporal query value objects (date selection + validation)."""

from datetime import date

import pytest

from pydantic import ValidationError

from snowtool.snowdb.query import DateRangeQuery, DOYQuery


def test_date_range_select_filters_available():
    query = DateRangeQuery(start_date=date(2020, 1, 2), end_date=date(2020, 1, 4))
    available = [date(2020, 1, d) for d in range(1, 6)]
    assert query.select(available) == [
        date(2020, 1, 2),
        date(2020, 1, 3),
        date(2020, 1, 4),
    ]


def test_doy_select_picks_the_month_day_across_years():
    query = DOYQuery(month=4, day=1, start_year=2019, end_year=2021)
    available = [date(y, 4, 1) for y in range(2018, 2023)] + [date(2020, 4, 2)]
    assert query.select(available) == [
        date(2019, 4, 1),
        date(2020, 4, 1),
        date(2021, 4, 1),
    ]


# Feb 29 must stay valid (leap years exist in any real span); the impossible
# combinations must be rejected at construction rather than silently selecting
# nothing -- `select` is a filter, so a bad day would mask the typo as 'no data'.
@pytest.mark.parametrize(
    ('month', 'day', 'valid'),
    [
        (2, 29, True),
        (1, 31, True),
        (4, 30, True),
        (2, 30, False),
        (4, 31, False),
        (6, 31, False),
        (9, 31, False),
        (11, 31, False),
    ],
)
def test_doy_rejects_impossible_day_of_month(month, day, valid):
    if valid:
        assert DOYQuery(month=month, day=day, start_year=2000, end_year=2001)
    else:
        with pytest.raises(ValidationError):
            DOYQuery(month=month, day=day, start_year=2000, end_year=2001)

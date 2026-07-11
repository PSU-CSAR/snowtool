"""parse_dates_query: the stats surface's --dates/--years -> domain query."""

from datetime import date

import pytest

from snowtool.cli._dates import parse_dates_query
from snowtool.snowdb.query import DateRangeQuery, DOYQuery


@pytest.mark.parametrize(
    ('dates', 'expected_start', 'expected_end'),
    [
        ('2024-01-01/2024-06-30', date(2024, 1, 1), date(2024, 6, 30)),
        ('2024-01-01/..', date(2024, 1, 1), None),
        ('../2024-06-30', None, date(2024, 6, 30)),
        ('2024-01-01', date(2024, 1, 1), date(2024, 1, 1)),  # instant == 1 day
        (None, None, None),  # absent: no temporal filter
    ],
)
def test_interval_forms(dates, expected_start, expected_end):
    query = parse_dates_query(dates, None)
    assert isinstance(query, DateRangeQuery)
    assert query.start_date == expected_start
    assert query.end_date == expected_end


@pytest.mark.parametrize(
    ('years', 'expected_years'),
    [('2018..2024', (2018, 2024)), ('2020', (2020, 2020))],
)
def test_doy_forms(years, expected_years):
    query = parse_dates_query('04-01', years)
    assert isinstance(query, DOYQuery)
    assert (query.month, query.day) == (4, 1)
    assert (query.start_year, query.end_year) == expected_years


@pytest.mark.parametrize(
    ('dates', 'years'),
    [
        (None, '2018..2024'),  # --years without --dates
        ('2024-01-01/2024-06-30', '2018..2024'),  # interval with --years
        ('04-01', None),  # month-day without --years
        ('not-a-date', None),  # unparseable interval
        ('../..', None),  # open both ends
        ('2024-06-30/2024-01-01', None),  # reversed interval
        ('04-01', '2024..2018'),  # reversed years (DOYQuery validates)
        ('02-30', '2020'),  # invalid day-of-month
    ],
)
def test_bad_inputs_raise_value_error(dates, years):
    with pytest.raises(ValueError, match=r'.'):
        parse_dates_query(dates, years)

"""Date input: the DATE param type and parse_dates_query (--dates/--years)."""

from datetime import date

import click
import pytest

from snowtool.cli._dates import DATE, parse_dates_query
from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.query import DateRangeQuery, DOYQuery


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('20180427', date(2018, 4, 27)),
        ('2018-04-27', date(2018, 4, 27)),
        # Guards timezone independence: to_date must take .date() straight off
        # the parsed datetime, not reinterpret it via astimezone, which would
        # shift the result across the local-TZ boundary (e.g. '20240101' ->
        # 2023-12-31 under TZ=Asia/Tokyo).
        ('20240101', date(2024, 1, 1)),
        ('2024-01-01', date(2024, 1, 1)),
        ('20240229', date(2024, 2, 29)),  # leap day
        ('2024-02-29', date(2024, 2, 29)),  # leap day, dashed
        ('20231231', date(2023, 12, 31)),
    ],
)
def test_date_param_parses_exact_value(value, expected):
    assert DATE.convert(value, None, None) == expected


def test_date_param_passes_through_date():
    assert DATE.convert(date(2018, 4, 27), None, None) == date(2018, 4, 27)


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
def test_date_param_rejects_invalid_input(value):
    with pytest.raises(click.BadParameter):
        DATE.convert(value, None, None)


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
        ('04-01', 'not-years'),  # malformed --years
    ],
)
def test_bad_inputs_raise_query_parameter_error(dates, years):
    # QueryParameterError: typed so the root cli group renders it centrally.
    with pytest.raises(QueryParameterError, match=r'.'):
        parse_dates_query(dates, years)

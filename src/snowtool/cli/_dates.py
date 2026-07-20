"""Date input for the CLI: the ``DATE`` param type and the ``--dates`` parser.

:data:`DATE` is the click argument/option type for a single ``YYYYMMDD`` /
``YYYY-MM-DD`` date. :func:`parse_dates_query` is the ``stats`` surface's
``--dates``/``--years``: ``--dates`` speaks the same OGC ``datetime`` interval
the API's date-range endpoint takes (``2024-01-01/2024-06-30``, ``..`` for an
open end, a bare instant for one day) -- parsed by the same
:class:`gazebo.params.DatetimeInterval` so the two surfaces cannot drift. A
month-day ``--dates MM-DD`` plus ``--years`` selects the day-of-year form (OGC
has no year-recurrence syntax). Parse failures raise
:class:`~snowtool.exceptions.QueryParameterError`, which the root ``cli`` group
renders as one clean usage error.
"""

from __future__ import annotations

import re

from datetime import date, datetime
from typing import Any

import click

from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.query import DateRangeQuery, DOYQuery

_MONTH_DAY = re.compile(r'(\d{2})-(\d{2})\Z')
_YEARS = re.compile(r'(\d{4})(?:\.\.(\d{4}))?\Z')


def _to_date(value: str) -> date:
    """Parse ``YYYYMMDD``/``YYYY-MM-DD`` into a :class:`date`, timezone-independent.

    Takes ``.date()`` straight off the parsed naive datetime rather than
    reinterpreting it via ``astimezone``, which would shift the result across the
    local-TZ boundary (e.g. ``'20240101'`` -> 2023-12-31 under ``TZ=Asia/Tokyo``).
    """
    return datetime.strptime(  # noqa: DTZ007
        value.replace('-', ''),
        '%Y%m%d',
    ).date()


class DateParamType(click.ParamType):
    """A click argument/option type accepting ``YYYYMMDD`` or ``YYYY-MM-DD``."""

    name = 'date'

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> date:
        if isinstance(value, date):
            return value
        try:
            return _to_date(value)
        except ValueError:
            self.fail(
                f'{value!r} is not a valid date (expected YYYYMMDD or YYYY-MM-DD)',
                param,
                ctx,
            )


# A reusable instance for command parameter declarations.
DATE = DateParamType()


def parse_dates_query(
    dates: str | None,
    years: str | None,
) -> DateRangeQuery | DOYQuery:
    """The domain date query selected by ``--dates``/``--years``.

    No ``--dates`` means no temporal filter (an unbounded range), matching the
    API's absent-``datetime`` semantics.
    """
    if years is not None:
        return _doy_query(dates, years)
    if dates is None:
        return DateRangeQuery(start_date=None, end_date=None)
    if _MONTH_DAY.fullmatch(dates):
        raise QueryParameterError(
            f'--dates {dates} is a recurring month-day; add --years '
            '(e.g. --years 2018..2024) or give a full date/interval.',
        )
    return _range_query(dates)


def _range_query(dates: str) -> DateRangeQuery:
    from gazebo.params import DatetimeInterval, ParamError
    from pydantic import ValidationError

    try:
        interval = DatetimeInterval.parse(dates)
    except (ParamError, ValidationError) as e:
        raise QueryParameterError(
            f'invalid --dates {dates!r}: expected an OGC interval like '
            f'2024-01-01/2024-06-30 (open ends with ..), a single date, or '
            f'MM-DD with --years ({e})',
        ) from e
    return DateRangeQuery.from_interval(interval)


def _doy_query(dates: str | None, years: str) -> DOYQuery:
    from pydantic import ValidationError

    month_day = _MONTH_DAY.fullmatch(dates) if dates is not None else None
    if month_day is None:
        raise QueryParameterError(
            '--years needs a recurring month-day --dates MM-DD '
            '(e.g. --dates 04-01 --years 2018..2024).',
        )
    year_span = _YEARS.fullmatch(years)
    if year_span is None:
        raise QueryParameterError("--years must be 'YYYY' or 'YYYY..YYYY'.")
    try:
        # Year order and day-of-month validity are enforced on DOYQuery itself,
        # so the CLI and API share one check.
        return DOYQuery(
            month=int(month_day[1]),
            day=int(month_day[2]),
            start_year=int(year_span[1]),
            end_year=int(year_span[2] or year_span[1]),
        )
    except ValidationError as e:
        raise QueryParameterError(f'invalid day-of-year query: {e}') from e

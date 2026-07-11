"""Parse the ``stats`` surface's ``--dates``/``--years`` into a domain query.

``--dates`` speaks the same OGC ``datetime`` interval the API's date-range
endpoint takes (``2024-01-01/2024-06-30``, ``..`` for an open end, a bare
instant for one day) -- parsed by the same :class:`gazebo.params.DatetimeInterval`
so the two surfaces cannot drift. A month-day ``--dates MM-DD`` plus ``--years``
selects the day-of-year form (OGC has no year-recurrence syntax). Everything
raises plain :class:`ValueError` so the CLI maps it to one clean usage error.
"""

from __future__ import annotations

import re

from snowtool.snowdb.query import DateRangeQuery, DOYQuery

_MONTH_DAY = re.compile(r'(\d{2})-(\d{2})\Z')
_YEARS = re.compile(r'(\d{4})(?:\.\.(\d{4}))?\Z')


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
        raise ValueError(
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
        raise ValueError(
            f'invalid --dates {dates!r}: expected an OGC interval like '
            f'2024-01-01/2024-06-30 (open ends with ..), a single date, or '
            f'MM-DD with --years ({e})',
        ) from e
    return DateRangeQuery(
        start_date=interval.start.date() if interval.start else None,
        end_date=interval.end.date() if interval.end else None,
    )


def _doy_query(dates: str | None, years: str) -> DOYQuery:
    from pydantic import ValidationError

    month_day = _MONTH_DAY.fullmatch(dates) if dates is not None else None
    if month_day is None:
        raise ValueError(
            '--years needs a recurring month-day --dates MM-DD '
            '(e.g. --dates 04-01 --years 2018..2024).',
        )
    year_span = _YEARS.fullmatch(years)
    if year_span is None:
        raise ValueError("--years must be 'YYYY' or 'YYYY..YYYY'.")
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
        raise ValueError(f'invalid day-of-year query: {e}') from e

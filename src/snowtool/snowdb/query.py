"""The temporal query objects: how a stats request selects dates and names output.

A query (:class:`DateRangeQuery` or :class:`DOYQuery`, unified by the discriminated
:data:`PourPointQuery`) carries both the date-selection logic (``select`` filters
the dates a dataset has) and output-naming logic (``csv_name`` builds the download
filename). :class:`DateQuery` is the structural protocol both satisfy.
"""

from __future__ import annotations

import calendar

from datetime import date
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, Self

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

Month = Annotated[
    int,
    Field(..., ge=1, le=12, examples=[4]),
]

Day = Annotated[
    int,
    Field(..., ge=1, le=31, examples=[1]),
]

Year = Annotated[
    int,
    Field(..., ge=1, le=9999, examples=[2008]),
]


class DateQuery(Protocol):  # pragma: no cover
    def csv_name(self: Self, pourpoint_name: str, zone_size: int = 0) -> str: ...
    def select(self: Self, available: Iterable[date]) -> list[date]: ...


def _bound(d: date | None) -> str:
    """An ISO date, or ``'open'`` for an unbounded interval end (filename use)."""
    return d.isoformat() if d is not None else 'open'


class DateRangeQuery(BaseModel):
    type: Literal['DateRange'] = 'DateRange'
    # Either bound may be open (``None``): the query is a *filter* over the dates a
    # dataset actually has, so an open end simply drops that side's constraint (the
    # OGC ``datetime`` interval semantics). The CLI always supplies both.
    start_date: date | None = Field(default=None, examples=['2008-12-01'])
    end_date: date | None = Field(default=None, examples=['2008-12-14'])

    def csv_name(self: Self, pourpoint_name: str, zone_size: int = 0) -> str:
        return '{}_{}-{}{}.csv'.format(
            '-'.join(pourpoint_name.split()),
            _bound(self.start_date),
            _bound(self.end_date),
            f'_zonal_{zone_size}' if zone_size else '',
        )

    def select(self: Self, available: Iterable[date]) -> list[date]:
        """The ``available`` dates within ``[start, end]`` (either bound optional)."""
        return sorted(
            d
            for d in available
            if (self.start_date is None or d >= self.start_date)
            and (self.end_date is None or d <= self.end_date)
        )

    def __str__(self: Self) -> str:
        return f'{_bound(self.start_date)}/{_bound(self.end_date)}'


class DOYFields(BaseModel):
    """The day-of-year field set and its cross-field validators, shared by
    :class:`DOYQuery` and the API's DOY request model so both get the same bounds
    and the same impossible-date/inverted-span checks from one definition.
    """

    month: Month
    day: Day
    start_year: Year
    end_year: Year = Field(examples=[2020])

    @model_validator(mode='after')
    def _check_day_of_month(self: Self) -> Self:
        """Reject a month/day that can occur in no year (e.g. Feb 30, Apr 31).

        ``select`` is a filter over available dates, so an impossible day would
        silently match nothing and mask the typo as 'no data'; catch it here.
        2000 is a leap year, so this treats February as 29 days -- the leap day
        stays a valid day-of-year query (some year in any non-trivial span is a
        leap year).
        """
        if self.day > calendar.monthrange(2000, self.month)[1]:
            raise ValueError(
                f'day {self.day} is out of range for month {self.month}',
            )
        return self

    @model_validator(mode='after')
    def _check_year_order(self: Self) -> Self:
        """Reject an inverted year span.

        ``select`` filters ``start_year <= d.year <= end_year``, so a reversed
        span would silently match nothing rather than surface the typo -- both
        the CLI and the API inherit this from the model instead of checking it
        ad hoc.
        """
        if self.end_year < self.start_year:
            raise ValueError(
                f'end_year {self.end_year} is before start_year {self.start_year}',
            )
        return self


class DOYQuery(DOYFields):
    type: Literal['DayOfYear'] = 'DayOfYear'

    def csv_name(self: Self, pourpoint_name: str, zone_size: int = 0) -> str:
        return '{}_{}-{}_{}-{}{}.csv'.format(
            '-'.join(pourpoint_name.split()),
            self.month,
            self.day,
            self.start_year,
            self.end_year,
            f'_zonal_{zone_size}' if zone_size else '',
        )

    def select(self: Self, available: Iterable[date]) -> list[date]:
        """The ``available`` dates on this month/day across the year span."""
        return sorted(
            d
            for d in available
            if d.month == self.month
            and d.day == self.day
            and self.start_year <= d.year <= self.end_year
        )

    def __str__(self: Self) -> str:
        return f'{self.month}{self.day}/{self.start_year}/{self.end_year}'


PourPointQuery = Annotated[
    DateRangeQuery | DOYQuery,
    Field(..., discriminator='type'),
]

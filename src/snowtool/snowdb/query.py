"""The temporal query objects: how a stats request selects dates and names output.

A query (:class:`DateRangeQuery` or :class:`DOYQuery`, unified by the discriminated
:data:`PourPointQuery`) carries both the date-selection logic (``select`` filters
the dates a dataset has) and output-naming logic (``csv_name`` builds the download
filename). :class:`DateQuery` is the structural protocol both satisfy.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, Self

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

# Largest valid day per month, treating February as 29 so the leap day stays a
# valid day-of-year query (some year in any non-trivial span is a leap year).
_MAX_DAY_IN_MONTH = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

Month = Annotated[
    int,
    Field(..., ge=1, le=12),
]

Day = Annotated[
    int,
    Field(..., ge=1, le=31),
]

Year = Annotated[
    int,
    Field(..., ge=1, le=9999),
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
    start_date: date | None = None
    end_date: date | None = None

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


class DOYQuery(BaseModel):
    type: Literal['DayOfYear'] = 'DayOfYear'
    month: Month
    day: Day
    start_year: Year
    end_year: Year

    @model_validator(mode='after')
    def _check_day_of_month(self: Self) -> Self:
        """Reject a month/day that can occur in no year (e.g. Feb 30, Apr 31).

        ``select`` is a filter over available dates, so an impossible day would
        silently match nothing and mask the typo as 'no data'; catch it here.
        """
        if self.day > _MAX_DAY_IN_MONTH[self.month]:
            raise ValueError(
                f'day {self.day} is out of range for month {self.month}',
            )
        return self

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

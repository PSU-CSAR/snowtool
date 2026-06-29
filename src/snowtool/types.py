from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    Field,
    PlainValidator,
    WithJsonSchema,
)

YYYY = r'\d{4}'
MM = r'(0[1-9]|1[0-2])'
DD = r'(0[1-9]|[1-2][0-9]|3[0-1])'
DATE = f'{YYYY}-?{MM}-?{DD}'
STATION_TRIPLET = r'[a-zA-Z0-9\-]+:[a-zA-Z]{2}:[a-zA-Z]+'


def to_date(value: str) -> date:
    return (
        datetime.strptime(
            value.replace('-', ''),
            '%Y%m%d',
        )
        .astimezone(UTC)
        .date()
    )


Date = Annotated[
    date,
    PlainValidator(to_date),
    WithJsonSchema(
        {
            'type': 'string',
            'pattern': f'^{DATE}$',
            'example': '20230414',
            'description': 'Date in YYYYMMDD format',
        },
        mode='validation',
    ),
]

StationTriplet = Annotated[
    str,
    Field(
        pattern=f'^{STATION_TRIPLET}$',
    ),
    WithJsonSchema(
        {
            'example': '12354500:MT:USGS',
        },
        mode='validation',
    ),
]


def triplet_to_stem(triplet: StationTriplet) -> str:
    """The filename stem for a station triplet (``:`` is not path-safe -> ``_``).

    Inverse of :func:`stem_to_triplet`. The single encoding rule shared by the
    AOI record files (``aois/records/<stem>.geojson``) and the per-dataset burned
    AOI rasters (``<stem>.tif``); both must agree on it for the ``aoi sync`` prune
    diff and the raster cascade to line up.
    """
    return triplet.replace(':', '_')


def stem_to_triplet(stem: str) -> StationTriplet:
    """The station triplet encoded in a record/raster filename stem (``_`` -> ``:``).

    Inverse of :func:`triplet_to_stem`. Lossless because a valid triplet never
    contains ``_`` (see :data:`STATION_TRIPLET`).
    """
    return StationTriplet(stem.replace('_', ':'))


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

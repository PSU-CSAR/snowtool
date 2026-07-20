"""The canonical generic zonal-stats endpoint.

One route family across all datasets — ``{dataset}`` is a path param and the
response is a single generic schema (:class:`CompactStatsResponse`), not a
per-dataset compiled one. Discovery lives in the dataset resource
(``GET /datasets/{dataset}``), which advertises the valid zone keys, override
params, and variables; the ``zone`` query tokens mirror that shape
(``LAYER[:PARAM=VALUE]``). Output is content-negotiated (``?f=json|csv`` or
``Accept``): ``json`` is the compact body (the default), ``csv`` streams the flat
rows via :meth:`ZonalStats.dump_to_csv`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Query
from fastapi.responses import StreamingResponse
from gazebo.ext.fastapi import GazeboRouter
from gazebo.negotiation import FormatEnum, alternate_links, f_description, negotiate
from gazebo.params import DatetimeQuery
from gazebo.rels import MediaType
from pydantic import BaseModel, Field, ValidationError

from snowtool import types
from snowtool.api.dependencies import ReaderDep
from snowtool.api.models.stats import CompactStatsResponse, stats_csv_response
from snowtool.api.tags import Tags
from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.query import DateRangeQuery, DOYQuery

if TYPE_CHECKING:
    from gazebo.negotiation import Representation
    from gazebo.params import DatetimeInterval

    from snowtool.snowdb.reader import SnowDbReader


class _StatsFormat(FormatEnum):
    """The ``?f=`` keys the stats route serves, each carrying its media type."""

    json = 'json', 'application/json'
    csv = 'csv', 'text/csv'


REPRESENTATIONS = _StatsFormat.representations()

# Document every negotiated media type on the route's 200, from the same enum that
# drives ``?f=``/``Accept`` -- one source of truth. ``application/json`` is omitted
# so the ``response_model`` keeps owning it (its ``$ref`` is preserved); the streamed
# ``text/csv`` gets a string-body schema.
_STATS_RESPONSES = _StatsFormat.openapi_responses(schemas={MediaType.JSON: None})

router: GazeboRouter = GazeboRouter(prefix='/datasets/{dataset}/stats')

_ZONE_DESC = (
    'Stratify by a zone layer (repeatable): LAYER or LAYER:PARAM=VALUE. See '
    'GET /datasets/{dataset} for the valid layer keys, override params, and '
    'variables. Default: whole basin.'
)
_VARIABLE_DESC = (
    'Variable to report (repeatable; default: all). Use a variable key from '
    'GET /datasets/{dataset}.'
)
_ALLOW_PARTIAL_DESC = (
    'Permit a basin only partially covered by the dataset grid (default false: a '
    'partially-covered basin is a 409). A wholly off-grid basin always 409s.'
)
_INCLUDE_EMPTY_DESC = (
    'Include crossed zones that no AOI pixel falls in (0 area, all values null). '
    'By default these empty combinations are dropped. No effect on a whole-basin '
    'query.'
)
_DATETIME_EXAMPLES = [
    '2018-01-01/2018-06-30',
    '2018-04-27',
    '2018-01-01/..',
    '../2018-06-30',
]


class _StatsQueryBase(BaseModel):
    zone: list[str] = Field(default_factory=list, description=_ZONE_DESC)
    variable: list[str] = Field(default_factory=list, description=_VARIABLE_DESC)
    allow_partial: bool = Field(default=False, description=_ALLOW_PARTIAL_DESC)
    include_empty_zones: bool = Field(default=False, description=_INCLUDE_EMPTY_DESC)
    f: _StatsFormat | None = Field(
        default=None,
        description=f_description(_StatsFormat),
    )


class DateRangeStatsQuery(_StatsQueryBase):
    datetime: DatetimeQuery = Field(
        default=None,
        examples=_DATETIME_EXAMPLES,
        json_schema_extra={'example': _DATETIME_EXAMPLES[0]},
    )


class DOYStatsQuery(_StatsQueryBase):
    month: int = Field(ge=1, le=12, examples=[4])
    day: int = Field(ge=1, le=31, examples=[27])
    start_year: int = Field(ge=1, le=9999, examples=[2018])
    end_year: int = Field(ge=1, le=9999, examples=[2018])


def _date_range(interval: DatetimeInterval | None) -> DateRangeQuery:
    if interval is None:
        return DateRangeQuery()
    return DateRangeQuery(
        start_date=interval.start.date() if interval.start else None,
        end_date=interval.end.date() if interval.end else None,
    )


async def _run(
    reader: SnowDbReader,
    dataset: str,
    triplet: types.StationTriplet,
    query: DateRangeQuery | DOYQuery,
    params: _StatsQueryBase,
    rep: Representation,
) -> CompactStatsResponse | StreamingResponse:
    stats = await reader.zonal_stats(
        triplet,
        dataset,
        query,
        variable_keys=params.variable or None,
        zone_tokens=params.zone,
        allow_partial=params.allow_partial,
    )
    if rep.key == 'csv':
        return stats_csv_response(
            stats,
            query.csv_name(triplet, zone_size=len(params.zone)),
            include_empty_zones=params.include_empty_zones,
        )
    return CompactStatsResponse.build(
        triplet=triplet,
        dataset=dataset,
        query=query,
        stats=stats.dump_compact(include_empty_zones=params.include_empty_zones),
        alternates=alternate_links(rep, REPRESENTATIONS),
    )


@router.get(
    '/{triplet}/date-range',
    name='stats_date_range',
    response_model=CompactStatsResponse,
    responses=_STATS_RESPONSES,
    tags=[Tags.STATS],
)
async def stats_date_range(
    dataset: str,
    triplet: types.StationTriplet,
    reader: ReaderDep,
    params: Annotated[DateRangeStatsQuery, Query()],
) -> CompactStatsResponse | StreamingResponse:
    rep = negotiate(REPRESENTATIONS, f=params.f)
    return await _run(
        reader,
        dataset,
        triplet,
        _date_range(params.datetime),
        params,
        rep,
    )


@router.get(
    '/{triplet}/doy',
    name='stats_doy',
    response_model=CompactStatsResponse,
    responses=_STATS_RESPONSES,
    tags=[Tags.STATS],
)
async def stats_doy(
    dataset: str,
    triplet: types.StationTriplet,
    reader: ReaderDep,
    params: Annotated[DOYStatsQuery, Query()],
) -> CompactStatsResponse | StreamingResponse:
    rep = negotiate(REPRESENTATIONS, f=params.f)
    try:
        query = DOYQuery(
            month=params.month,
            day=params.day,
            start_year=params.start_year,
            end_year=params.end_year,
        )
    except ValidationError as e:
        raise QueryParameterError(f'Invalid day of year: {e}') from e
    return await _run(reader, dataset, triplet, query, params, rep)

"""The generic compact zonal-stats endpoint.

Unlike the verbose per-dataset ``/stats/`` routes (compiled once per dataset for a
precise response schema), the compact body is generic — zones/variables defined
once, values positional — so this is a *single* route family with ``{dataset}`` a
path param and static query models. Discovery lives in the dataset resource
(``GET /datasets/{dataset}``), which advertises the valid zone keys, override
params, and variables; the ``zone`` query tokens mirror that shape
(``LAYER[:PARAM=VALUE]``). One representation only (compact json), so there is no
``?f=`` negotiation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Query
from gazebo.ext.fastapi import GazeboRouter
from gazebo.params import DatetimeQuery
from pydantic import BaseModel, Field, ValidationError

from snowtool import types
from snowtool.api.dependencies import ReaderDep
from snowtool.api.models.stats import CompactStatsResponse
from snowtool.api.problems import DATASET_NOT_FOUND
from snowtool.api.tags import Tags
from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.query import DateRangeQuery, DOYQuery
from snowtool.snowdb.zonal_stats import parse_zone_selection
from snowtool.snowdb.zones.zone_layer import available_zones

if TYPE_CHECKING:
    from gazebo.params import DatetimeInterval

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.reader import SnowDbReader
    from snowtool.snowdb.zonal_stats import ZonalStats

router: GazeboRouter = GazeboRouter(prefix='/datasets/{dataset}/stats-compact')

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


class _CompactStatsQueryBase(BaseModel):
    zone: list[str] = Field(default_factory=list, description=_ZONE_DESC)
    variable: list[str] = Field(default_factory=list, description=_VARIABLE_DESC)
    allow_partial: bool = Field(default=False, description=_ALLOW_PARTIAL_DESC)
    include_empty_zones: bool = Field(default=False, description=_INCLUDE_EMPTY_DESC)


class CompactDateRangeQuery(_CompactStatsQueryBase):
    datetime: DatetimeQuery = Field(
        default=None,
        examples=_DATETIME_EXAMPLES,
        json_schema_extra={'example': _DATETIME_EXAMPLES[0]},
    )


class CompactDOYStatsQuery(_CompactStatsQueryBase):
    month: int = Field(ge=1, le=12, examples=[4])
    day: int = Field(ge=1, le=31, examples=[27])
    start_year: int = Field(ge=1, le=9999, examples=[2018])
    end_year: int = Field(ge=1, le=9999, examples=[2018])


def _resolve(reader: SnowDbReader, dataset: str) -> Dataset:
    """The active dataset by name, or a 404 problem (mirrors GET /datasets)."""
    try:
        return reader.db[dataset]
    except KeyError as e:
        raise DATASET_NOT_FOUND.exception(
            detail=f'No such dataset: {dataset!r}',
        ) from e


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
    params: _CompactStatsQueryBase,
) -> CompactStatsResponse:
    resolved = _resolve(reader, dataset)
    registry = available_zones(resolved.providers.values())
    selections = [parse_zone_selection(token, registry) for token in params.zone]
    stats: ZonalStats = await reader.zonal_stats(
        triplet,
        dataset,
        query,
        variable_keys=params.variable or None,
        zone_selections=selections,
        allow_partial=params.allow_partial,
    )
    return CompactStatsResponse.build(
        triplet=triplet,
        dataset=dataset,
        query=query,
        stats=stats.dump_compact(include_empty_zones=params.include_empty_zones),
    )


@router.get(
    '/{triplet}/date-range',
    name='stats_compact_date_range',
    tags=[Tags.STATS],
)
async def stats_compact_date_range(
    dataset: str,
    triplet: types.StationTriplet,
    reader: ReaderDep,
    params: Annotated[CompactDateRangeQuery, Query()],
) -> CompactStatsResponse:
    return await _run(
        reader,
        dataset,
        triplet,
        _date_range(params.datetime),
        params,
    )


@router.get('/{triplet}/doy', name='stats_compact_doy', tags=[Tags.STATS])
async def stats_compact_doy(
    dataset: str,
    triplet: types.StationTriplet,
    reader: ReaderDep,
    params: Annotated[CompactDOYStatsQuery, Query()],
) -> CompactStatsResponse:
    try:
        query = DOYQuery(
            month=params.month,
            day=params.day,
            start_year=params.start_year,
            end_year=params.end_year,
        )
    except ValidationError as e:
        raise QueryParameterError(f'Invalid day of year: {e}') from e
    return await _run(reader, dataset, triplet, query, params)

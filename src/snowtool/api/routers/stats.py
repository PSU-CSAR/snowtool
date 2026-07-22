"""The canonical generic zonal-stats endpoint.

One route family across all datasets — ``{dataset}`` is a path param and the
response is a single generic schema (:class:`CompactStatsResponse`), not a
per-dataset compiled one. Discovery lives in the dataset resource
(``GET /datasets/{dataset}``), which advertises the valid zone keys, override
params, and variables; the ``zone`` query tokens mirror that shape
(``LAYER[:PARAM=VALUE]``). Output is content-negotiated (``?f=json|csv`` or
``Accept``): ``json`` is the compact body (the default), ``csv`` streams the flat
rows via :meth:`ZonalStats.iter_csv`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Query
from fastapi.responses import StreamingResponse
from gazebo.ext.fastapi import GazeboRouter
from gazebo.negotiation import alternate_links, negotiate
from gazebo.rels import MediaType

from snowtool import types
from snowtool.api.dependencies import ReaderDep
from snowtool.api.models.stats import (
    CompactStatsResponse,
    DateRangeStatsQuery,
    DOYStatsQuery,
    StatsFormat,
    StatsQueryBase,
    stats_csv_response,
)
from snowtool.api.tags import Tags
from snowtool.snowdb.query import DateRangeQuery, DOYQuery

if TYPE_CHECKING:
    from snowtool.snowdb.reader import SnowDbReader


REPRESENTATIONS = StatsFormat.representations()

# Document every negotiated media type on the route's 200, from the same enum that
# drives ``?f=``/``Accept`` -- one source of truth. ``application/json`` is omitted
# so the ``response_model`` keeps owning it (its ``$ref`` is preserved); the streamed
# ``text/csv`` gets a string-body schema.
_STATS_RESPONSES = StatsFormat.openapi_responses(schemas={MediaType.JSON: None})

router: GazeboRouter = GazeboRouter(prefix='/datasets/{dataset}/stats')


async def _run(
    reader: SnowDbReader,
    dataset: str,
    triplet: types.StationTriplet,
    query: DateRangeQuery | DOYQuery,
    params: StatsQueryBase,
) -> CompactStatsResponse | StreamingResponse:
    rep = negotiate(REPRESENTATIONS, f=params.f)
    stats = await reader.zonal_stats(
        triplet,
        dataset,
        query,
        variable_keys=params.variable or None,
        zones=params.zone,
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
    return await _run(
        reader,
        dataset,
        triplet,
        DateRangeQuery.from_interval(params.datetime),
        params,
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
    return await _run(reader, dataset, triplet, params.to_query(), params)

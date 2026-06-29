"""Per-dataset zonal-stats routes (date-range + day-of-year).

:func:`build_stats_router` is called once per dataset in ``get_app`` so each
dataset's *generated* response model surfaces a precise OpenAPI schema (the point
of the ``model_prefix`` uniqueness check in ``db._index_specs``). Both routes map
onto :meth:`SnowDbReader.zonal_stats`: ``zone`` (repeatable ``LAYER[:override]``),
``variable`` (repeatable), and ``allow_partial``. The date-range endpoint takes the
OGC ``datetime`` interval; the day-of-year endpoint takes ``month``/``day`` over a
year span. No ``zone`` ⇒ the legacy whole-basin "basic stats".

Output is content-negotiated (``?f=json|csv`` or ``Accept``): JSON is the
per-dataset envelope, CSV streams :meth:`ZonalStats.dump_to_csv`. Coverage/lookup/
parse failures propagate to the registered problem handlers (AOICoverageError->409,
AOINotFound/AOIRasterNotFound->404, QueryParameterError->422, ParamError->400).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Query
from fastapi.responses import StreamingResponse
from gazebo.ext.fastapi import DatetimeParam, GazeboRouter, Negotiate
from gazebo.negotiation import JSON, Representation, alternate_links

# DatetimeInterval is imported at runtime (not under TYPE_CHECKING) so
# get_type_hints can resolve the route annotations -- if it fails, gazebo silently
# skips reader injection.
from gazebo.params import DatetimeInterval

from snowtool import types
from snowtool.api.models.stats import StatsResponse, stats_csv_response
from snowtool.api.tags import Tags
from snowtool.snowdb.reader import SnowDbReader
from snowtool.snowdb.zonal_stats import parse_zone_selection
from snowtool.snowdb.zone_layer import available_zones

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset

CSV = Representation('csv', 'text/csv')
_REPRESENTATIONS = [JSON, CSV]

_ZONE = Query(description='Stratify by a zone layer (repeatable).')
_VARIABLE = Query(description='Variable to report (repeatable; default all).')


def _date_range(interval: DatetimeInterval | None) -> types.DateRangeQuery:
    """Map the OGC ``datetime`` interval to a date-range query.

    Either (or both) bounds may be open: selection is a filter over the dates the
    dataset has, so an open end is just an absent constraint. An absent ``datetime``
    parameter (``interval is None``) means no temporal filter -- every ingested date.
    """
    if interval is None:
        return types.DateRangeQuery()
    return types.DateRangeQuery(
        start_date=interval.start.date() if interval.start else None,
        end_date=interval.end.date() if interval.end else None,
    )


def build_stats_router(dataset: Dataset) -> GazeboRouter:
    """A :class:`GazeboRouter` carrying ``dataset``'s two stats routes."""
    name = dataset.spec.name
    response_model = StatsResponse[dataset.spec.zonal_stats_model]  # type: ignore[name-defined]
    registry = available_zones(dataset.providers.values())

    router = GazeboRouter(prefix=f'/datasets/{name}/stats', tags=[Tags.STATS])

    async def run(
        reader: SnowDbReader,
        triplet: types.StationTriplet,
        query: types.DateRangeQuery | types.DOYQuery,
        zone: list[str],
        variable: list[str],
        allow_partial: bool,
        rep: Representation,
    ) -> StreamingResponse | StatsResponse:
        selections = [parse_zone_selection(token, registry) for token in zone]
        stats = await reader.zonal_stats(
            triplet,
            name,
            query,
            variable_keys=variable or None,
            zone_selections=selections,
            allow_partial=allow_partial,
        )
        if rep.key == 'csv':
            return stats_csv_response(
                stats,
                query.csv_name(triplet, zone_size=len(selections)),
            )
        return StatsResponse.build(
            triplet=triplet,
            dataset=name,
            query=query,
            results=stats.dump(),
            alternates=alternate_links(rep, _REPRESENTATIONS),
        )

    @router.get(
        '/{triplet}/date-range',
        name=f'{name}_stats_date_range',
        response_model=response_model,
    )
    async def date_range_stats(
        triplet: types.StationTriplet,
        reader: SnowDbReader,
        rep: Annotated[Representation, Negotiate(_REPRESENTATIONS)],
        interval: Annotated[DatetimeInterval | None, DatetimeParam] = None,
        zone: Annotated[list[str], _ZONE] = [],  # noqa: B006 (FastAPI Query default)
        variable: Annotated[list[str], _VARIABLE] = [],  # noqa: B006
        allow_partial: bool = False,
    ):
        query = _date_range(interval)
        return await run(reader, triplet, query, zone, variable, allow_partial, rep)

    @router.get(
        '/{triplet}/doy',
        name=f'{name}_stats_doy',
        response_model=response_model,
    )
    async def doy_stats(
        triplet: types.StationTriplet,
        reader: SnowDbReader,
        rep: Annotated[Representation, Negotiate(_REPRESENTATIONS)],
        month: Annotated[int, Query(ge=1, le=12)],
        day: Annotated[int, Query(ge=1, le=31)],
        start_year: Annotated[int, Query(ge=1, le=9999)],
        end_year: Annotated[int, Query(ge=1, le=9999)],
        zone: Annotated[list[str], _ZONE] = [],  # noqa: B006
        variable: Annotated[list[str], _VARIABLE] = [],  # noqa: B006
        allow_partial: bool = False,
    ):
        query = types.DOYQuery(
            month=month,
            day=day,
            start_year=start_year,
            end_year=end_year,
        )
        return await run(reader, triplet, query, zone, variable, allow_partial, rep)

    return router

"""Per-dataset zonal-stats routes (date-range + day-of-year).

:func:`build_stats_router` is called once per dataset in ``get_app`` so each
dataset's *generated* response model surfaces a precise OpenAPI schema (the point
of the ``model_prefix`` uniqueness check in ``db._index_specs``). Both routes map
onto :meth:`SnowDbReader.zonal_stats`.

The stratification/negotiation query surface (``zone``/``variable``/``allow_partial``
enums, per-layer overrides, ``f``, ``datetime``) is compiled per dataset in
:mod:`._stats_params`; that module's docstring covers *why* it is one exploded
Pydantic model per endpoint. No ``zone`` => the legacy whole-basin "basic stats".

Output is content-negotiated (``?f=json|csv`` or ``Accept``): JSON is the per-dataset
envelope, CSV streams :meth:`ZonalStats.dump_to_csv`. ``?f=`` is validated by the
folded enum; a one-line :func:`negotiate` call layers the ``Accept`` header back on.
Coverage/lookup failures propagate to the registered problem handlers
(PourpointCoverageError->409, PourpointNotFound/AOIRasterNotFound->404); a
handler-raised QueryParameterError (a well-formed but unprocessable query, e.g. an
impossible day-of-year) is a 422, while a malformed query parameter (bad
``zone``/``variable``/``f``/``datetime``/override, or an override for an unselected
zone) is a schema-layer 400.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import StreamingResponse
from gazebo.ext.fastapi import GazeboRouter
from gazebo.negotiation import alternate_links, negotiate
from pydantic import ValidationError

from snowtool import types
from snowtool.api.dependencies import ReaderDep
from snowtool.api.models.stats import StatsResponse, stats_csv_response
from snowtool.api.routers._stats_params import (
    REPRESENTATIONS,
    DateRangeParams,
    DOYParams,
    StatsParams,
    build_query_models,
    selections,
)
from snowtool.api.tags import Tags
from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.query import DateRangeQuery, DOYQuery
from snowtool.snowdb.reader import SnowDbReader

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from gazebo.negotiation import Representation
    from gazebo.params import DatetimeInterval

    from snowtool.snowdb.dataset import Dataset


def _date_range(interval: DatetimeInterval | None) -> DateRangeQuery:
    """Map the OGC ``datetime`` interval to a date-range query.

    Either (or both) bounds may be open: selection is a filter over the dates the
    dataset has, so an open end is just an absent constraint. An absent ``datetime``
    parameter (``interval is None``) means no temporal filter -- every ingested date.
    """
    if interval is None:
        return DateRangeQuery()
    return DateRangeQuery(
        start_date=interval.start.date() if interval.start else None,
        end_date=interval.end.date() if interval.end else None,
    )


def build_stats_router(dataset: Dataset) -> GazeboRouter:
    """A :class:`GazeboRouter` carrying ``dataset``'s two stats routes."""
    name = dataset.spec.name
    response_model = StatsResponse[dataset.spec.zonal_stats_model]  # type: ignore[name-defined]
    models = build_query_models(dataset)

    router = GazeboRouter(prefix=f'/datasets/{name}/stats', tags=[Tags.STATS])

    async def run(
        reader: SnowDbReader,
        triplet: types.StationTriplet,
        query: DateRangeQuery | DOYQuery,
        params: StatsParams,
        rep: Representation,
    ) -> StreamingResponse | StatsResponse:
        selected = selections(params, models.overrides)
        stats = await reader.zonal_stats(
            triplet,
            name,
            query,
            variable_keys=[str(v) for v in params.variable] or None,
            zone_selections=selected,
            allow_partial=params.allow_partial,
        )
        if rep.key == 'csv':
            return stats_csv_response(
                stats,
                query.csv_name(triplet, zone_size=len(selected)),
                include_empty_zones=params.include_empty_zones,
            )
        return StatsResponse.build(
            triplet=triplet,
            dataset=name,
            query=query,
            results=stats.dump(include_empty_zones=params.include_empty_zones),
            alternates=alternate_links(rep, REPRESENTATIONS),
        )

    async def date_range_stats(
        triplet: types.StationTriplet,
        reader: ReaderDep,
        params: DateRangeParams,
    ):
        rep = negotiate(REPRESENTATIONS, f=params.f)
        return await run(reader, triplet, _date_range(params.datetime), params, rep)

    async def doy_stats(
        triplet: types.StationTriplet,
        reader: ReaderDep,
        params: DOYParams,
    ):
        rep = negotiate(REPRESENTATIONS, f=params.f)
        try:
            query = DOYQuery(
                month=params.month,
                day=params.day,
                start_year=params.start_year,
                end_year=params.end_year,
            )
        except ValidationError as e:
            # An impossible month/day (Feb 30) or inverted year span is a client
            # error, not a 500 -- mirrors the CLI's parse_dates_query handling.
            raise QueryParameterError(f'Invalid day of year: {e}') from e
        return await run(reader, triplet, query, params, rep)

    def register(
        path: str,
        suffix: str,
        handler: Callable[..., Awaitable],
        annotation: object,
    ):
        # The per-dataset query models are locals, so ``params`` cannot be named in
        # a module-level annotation (``from __future__ import annotations`` would
        # stringify the hint and neither FastAPI nor gazebo could resolve the local
        # name). Patch the real Annotated object onto the handler before registering
        # it, which is when both introspect the signature.
        handler.__annotations__['params'] = annotation
        router.get(path, name=f'{name}_{suffix}', response_model=response_model)(
            handler,
        )

    register(
        '/{triplet}/date-range',
        'stats_date_range',
        date_range_stats,
        models.date_range,
    )
    register('/{triplet}/doy', 'stats_doy', doy_stats, models.doy)

    return router

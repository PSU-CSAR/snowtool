"""Per-dataset zonal-stats routes (date-range + day-of-year).

:func:`build_stats_router` is called once per dataset in ``get_app`` so each
dataset's *generated* response model surfaces a precise OpenAPI schema (the point
of the ``model_prefix`` uniqueness check in ``db._index_specs``). Both routes map
onto :meth:`SnowDbReader.zonal_stats`.

Zone stratification is advertised *and* validated in the schema. ``zone`` is a
repeatable enum constrained to the dataset's ``available_zones`` keys (a
per-dataset :class:`enum.StrEnum` built at router-build time), so an unknown zone
is a schema-level 422 rather than a hand-rolled error. Each *overridable* zone
layer contributes one optional, typed query param named ``'<layer_key>.<param_key>'``
(e.g. ``terrain.elevation.band_step_ft``), typed from the matching
:class:`ZoneLayerParams` field; a categorical layer (no override param) contributes
none. These vary per dataset and carry dotted names, so they are gathered into a
per-dataset Pydantic query-params model (:func:`pydantic.create_model` with aliased
fields) consumed via FastAPI's ``Annotated[Model, Query()]`` support, and shared by
both endpoints. A supplied override whose layer is not in the selected ``zone`` list
is a client mistake (:class:`QueryParameterError` -> 422), not a silent no-op.
``variable`` (repeatable) and ``allow_partial`` ride on the same model. No ``zone``
=> the legacy whole-basin "basic stats".

Output is content-negotiated (``?f=json|csv`` or ``Accept``): JSON is the
per-dataset envelope, CSV streams :meth:`ZonalStats.dump_to_csv`. Coverage/lookup/
parse failures propagate to the registered problem handlers
(PourpointCoverageError->409, PourpointNotFound/AOIRasterNotFound->404,
QueryParameterError->422, ParamError->400).
"""

from __future__ import annotations

import enum

from typing import TYPE_CHECKING, Annotated

from fastapi import Query
from fastapi.responses import StreamingResponse
from gazebo.ext.fastapi import DatetimeParam, GazeboRouter, Negotiate
from gazebo.negotiation import JSON, Representation, alternate_links

# DatetimeInterval is imported at runtime (not under TYPE_CHECKING) because it is
# the resolved type of the interval param's annotation.
from gazebo.params import DatetimeInterval
from pydantic import BaseModel, Field, ValidationError, create_model

from snowtool import types
from snowtool.api.dependencies import ReaderDep
from snowtool.api.models.stats import StatsResponse, stats_csv_response
from snowtool.api.tags import Tags
from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.config import ZoneLayerParams
from snowtool.snowdb.query import DateRangeQuery, DOYQuery
from snowtool.snowdb.reader import SnowDbReader
from snowtool.snowdb.zonal_stats import ZoneSelection
from snowtool.snowdb.zones.zone_layer import AvailableZone, available_zones

if TYPE_CHECKING:
    from collections.abc import Mapping

    from snowtool.snowdb.dataset import Dataset

CSV = Representation('csv', 'text/csv')
_REPRESENTATIONS = [JSON, CSV]


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


def _sanitize(key: str) -> str:
    """A valid Python identifier from a dotted/dashed registry key (for a type name
    or a model field name); the query name is restored via an alias."""
    return ''.join(ch if ch.isalnum() else '_' for ch in key)


def _zone_enum(name: str, registry: Mapping[str, AvailableZone]) -> type[enum.StrEnum]:
    """A per-dataset :class:`enum.StrEnum` whose members *are* the registry keys.

    The ``zone`` query param's type: FastAPI validates against the enum values (the
    keys) and renders them as an OpenAPI ``enum``, so an unknown zone is rejected at
    the schema layer. Members are keyed by a sanitized name (dots/dashes aren't
    valid identifiers) but their value is the exact registry key.
    """
    return enum.StrEnum(  # type: ignore[return-value]
        f'{_sanitize(name)}_ZoneKey',
        {_sanitize(key): key for key in sorted(registry)},
    )


def _override_fields(
    registry: Mapping[str, AvailableZone],
) -> dict[str, tuple[str, str, str]]:
    """Overridable layer key -> (model field name, override alias, param key).

    An overridable layer is one whose scheme ``describe()`` names a ``param_key``
    (categorical layers name none). The alias is ``'<layer_key>.<param_key>'`` (the
    dotted query name); the field name is its sanitized, valid-identifier form.
    """
    fields: dict[str, tuple[str, str, str]] = {}
    for key in sorted(registry):
        param_key = registry[key].scheme.describe().param_key
        if param_key is not None:
            alias = f'{key}.{param_key}'
            fields[key] = (_sanitize(alias), alias, param_key)
    return fields


def _base_fields(
    zone_enum: type[enum.StrEnum],
    override_fields: dict[str, tuple[str, str, str]],
) -> dict[str, object]:
    """The query fields both endpoints share: ``zone`` + ``variable`` +
    ``allow_partial`` + one typed, aliased override field per overridable layer.

    ``zone`` is the repeatable per-dataset enum; each override field is typed from
    its :class:`ZoneLayerParams` param and carries the dotted ``'<layer>.<param>'``
    query alias. Returned as a :func:`pydantic.create_model` field spec map so the
    two endpoints can each fold in their own extra fields.
    """
    fields: dict[str, object] = {
        'zone': (
            list[zone_enum],  # type: ignore[valid-type]
            Field(
                default_factory=list,
                description='Stratify by a zone layer (repeatable).',
            ),
        ),
        'variable': (
            list[str],
            Field(
                default_factory=list,
                description='Variable to report (repeatable; default all).',
            ),
        ),
        'allow_partial': (bool, Field(default=False)),
    }
    for key, (field_name, alias, param_key) in override_fields.items():
        annotation = ZoneLayerParams.model_fields[param_key].annotation
        fields[field_name] = (
            annotation,
            Field(
                default=None,
                alias=alias,
                description=f'Override the scheme param for zone {key!r}.',
            ),
        )
    return fields


# FastAPI expands a Pydantic query model into per-field query params only when it is
# the endpoint's *sole* field parameter -- a sibling scalar ``Query`` param collapses
# it back to a single opaque param. So every query param (including the day-of-year
# ``month``/``day``/year span) rides *inside* the model rather than beside it, and the
# doy endpoint gets a model that extends the shared base with those fields.
_DOY_FIELDS: dict[str, object] = {
    'month': (int, Field(ge=1, le=12)),
    'day': (int, Field(ge=1, le=31)),
    'start_year': (int, Field(ge=1, le=9999)),
    'end_year': (int, Field(ge=1, le=9999)),
}


def _query_model(name: str, suffix: str, fields: dict[str, object]) -> type[BaseModel]:
    """A per-dataset query-params model from a ``create_model`` field-spec map.

    ``suffix`` keeps the generated model names unique per dataset *and* per endpoint
    so the OpenAPI component schemas never collide.
    """
    return create_model(f'{_sanitize(name)}_{suffix}', **fields)  # type: ignore[call-overload]


def _selections(
    params: BaseModel,
    override_fields: dict[str, tuple[str, str, str]],
) -> list[ZoneSelection]:
    """The crossed-zone selections from a parsed query model.

    One :class:`ZoneSelection` per selected ``zone`` (in request order), each
    carrying its override value if the client supplied one (else ``None`` -> scheme
    default). An override supplied for a layer *not* in the selected ``zone`` list is
    a client mistake, surfaced as a 422 rather than silently ignored.
    """
    selected = [str(z) for z in params.zone]  # type: ignore[attr-defined]
    selected_set = set(selected)
    overrides: dict[str, int | float] = {}
    for key, (field_name, alias, _) in override_fields.items():
        value = getattr(params, field_name)
        if value is None:
            continue
        if key not in selected_set:
            raise QueryParameterError(
                f'override {alias!r} was supplied but zone {key!r} is not selected; '
                f'add it to the "zone" parameter or drop the override.',
            )
        overrides[key] = value
    return [ZoneSelection(key, overrides.get(key)) for key in selected]


def build_stats_router(dataset: Dataset) -> GazeboRouter:
    """A :class:`GazeboRouter` carrying ``dataset``'s two stats routes."""
    name = dataset.spec.name
    response_model = StatsResponse[dataset.spec.zonal_stats_model]  # type: ignore[name-defined]
    registry = available_zones(dataset.providers.values())

    zone_enum = _zone_enum(name, registry)
    override_fields = _override_fields(registry)
    base_fields = _base_fields(zone_enum, override_fields)
    date_range_model = _query_model(name, 'StatsQuery', base_fields)
    doy_model = _query_model(name, 'DOYStatsQuery', {**base_fields, **_DOY_FIELDS})
    # The query-params models are per-dataset (built here), so they cannot be named in
    # a module-level annotation; ``from __future__ import annotations`` would stringify
    # ``params``' hint and neither FastAPI nor gazebo could resolve the local name.
    # Patch the *real* Annotated object onto each handler's ``__annotations__`` before
    # the route is registered (which is when both introspect the signature).
    date_range_params = Annotated[date_range_model, Query()]  # type: ignore[valid-type]
    doy_params = Annotated[doy_model, Query()]  # type: ignore[valid-type]

    router = GazeboRouter(prefix=f'/datasets/{name}/stats', tags=[Tags.STATS])

    async def run(
        reader: SnowDbReader,
        triplet: types.StationTriplet,
        query: DateRangeQuery | DOYQuery,
        params: BaseModel,
        rep: Representation,
    ) -> StreamingResponse | StatsResponse:
        selections = _selections(params, override_fields)
        stats = await reader.zonal_stats(
            triplet,
            name,
            query,
            variable_keys=params.variable or None,  # type: ignore[attr-defined]
            zone_selections=selections,
            allow_partial=params.allow_partial,  # type: ignore[attr-defined]
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

    async def date_range_stats(
        triplet: types.StationTriplet,
        reader: ReaderDep,
        rep: Annotated[Representation, Negotiate(_REPRESENTATIONS)],
        params,
        interval: Annotated[DatetimeInterval | None, DatetimeParam] = None,
    ):
        query = _date_range(interval)
        return await run(reader, triplet, query, params, rep)

    async def doy_stats(
        triplet: types.StationTriplet,
        reader: ReaderDep,
        rep: Annotated[Representation, Negotiate(_REPRESENTATIONS)],
        params,
    ):
        try:
            query = DOYQuery(
                month=params.month,
                day=params.day,
                start_year=params.start_year,
                end_year=params.end_year,
            )
        except ValidationError as e:
            # An impossible month/day (Feb 30) or inverted year span is a client
            # error, not a 500 -- mirrors the CLI's _build_query handling.
            raise QueryParameterError(f'Invalid day of year: {e}') from e
        return await run(reader, triplet, query, params, rep)

    date_range_stats.__annotations__['params'] = date_range_params
    doy_stats.__annotations__['params'] = doy_params

    router.get(
        '/{triplet}/date-range',
        name=f'{name}_stats_date_range',
        response_model=response_model,
    )(date_range_stats)
    router.get(
        '/{triplet}/doy',
        name=f'{name}_stats_doy',
        response_model=response_model,
    )(doy_stats)

    return router

"""Per-dataset zonal-stats routes (date-range + day-of-year).

:func:`build_stats_router` is called once per dataset in ``get_app`` so each
dataset's *generated* response model surfaces a precise OpenAPI schema (the point
of the ``model_prefix`` uniqueness check in ``db._index_specs``). Both routes map
onto :meth:`SnowDbReader.zonal_stats`.

Zone stratification is advertised *and* validated in the schema. ``zone`` is a
repeatable enum constrained to the dataset's ``available_zones`` keys (a
per-dataset :class:`enum.StrEnum` built at router-build time), so an unknown zone
is a schema-level 400 rather than a hand-rolled error. Each *overridable* zone
layer contributes one typed query param named ``'<layer_key>.<param_key>'``
(e.g. ``terrain.elevation.band_step_ft``), typed from the matching
:class:`ZoneLayerParams` field but made non-nullable and defaulted to the scheme's own
default (shown as the example), so the doc advertises a real value; a categorical layer
(no override param) contributes none. These vary per dataset and carry dotted names, so
they are gathered into a per-dataset Pydantic query-params model
(:func:`pydantic.create_model` with aliased fields) consumed via FastAPI's
``Annotated[Model, Query()]`` support, and shared by both endpoints. An override only
applies to a *selected* zone; because the defaulted field can't be told from an explicit
value, an override for an unselected zone is a harmless no-op.
``variable`` (repeatable), ``allow_partial``, and the content-negotiation ``f`` and
(date-range only) OGC ``datetime`` ride on the same model -- ``f``/``datetime`` are
gazebo field types folded in rather than ``Negotiate``/``DatetimeParam`` dependencies,
whose own query params would collapse the model back into one opaque object. No
``zone`` => the legacy whole-basin "basic stats".

Output is content-negotiated (``?f=json|csv`` or ``Accept``): JSON is the per-dataset
envelope, CSV streams :meth:`ZonalStats.dump_to_csv`. ``?f=`` alone is validated by the
folded enum; a one-line :func:`negotiate` call in the handler layers the ``Accept``
header (read from the ambient request context) back on.
Coverage/lookup failures propagate to the registered problem handlers
(PourpointCoverageError->409, PourpointNotFound/AOIRasterNotFound->404); a
handler-raised QueryParameterError (e.g. an impossible day-of-year) is a 422, while a
malformed query parameter (bad ``zone``/``variable``/``f``/``datetime``/override) is a
schema-layer 400.
"""

from __future__ import annotations

import enum

from typing import TYPE_CHECKING, Annotated, NamedTuple, get_args

from fastapi import Query
from fastapi.responses import StreamingResponse
from gazebo.ext.fastapi import GazeboRouter
from gazebo.negotiation import (
    FormatEnum,
    alternate_links,
    f_description,
    negotiate,
)

# DatetimeQuery is imported at runtime: it is a folded query-model *field type*
# (an annotated ``DatetimeInterval | None`` with the OGC parser + self-documentation).
from gazebo.params import DatetimeQuery
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

    from gazebo.negotiation import Representation
    from gazebo.params import DatetimeInterval

    from snowtool.snowdb.dataset import Dataset


class _StatsFormat(FormatEnum):
    """The ``?f=`` keys the stats routes serve, each carrying its media type.

    Folded into the query model as a field type so ``?f=`` explodes into a
    documented enum query param (rather than riding a separate ``Negotiate``
    dependency, which -- being another query param -- would collapse the whole
    model back into one opaque object). An unknown ``?f=`` is a native pydantic
    ValidationError -> 400. Because members carry their media type, they *are* the
    representation set (:meth:`representations`), and ``Accept``-header negotiation is
    layered back on with a one-line :func:`negotiate` call reading the ambient request.
    """

    json = 'json', 'application/json'
    csv = 'csv', 'text/csv'


_REPRESENTATIONS = _StatsFormat.representations()


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


def _non_optional(annotation: object) -> object:
    """Strip ``None`` from an ``X | None`` annotation, keeping ``X`` (else as-is)."""
    args = [a for a in get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def _param_annotation(param_key: str) -> object:
    """The non-nullable type of a ``ZoneLayerParams`` override param."""
    return _non_optional(ZoneLayerParams.model_fields[param_key].annotation)


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


def _variable_enum(name: str, keys: list[str]) -> type[enum.StrEnum]:
    """A per-dataset :class:`enum.StrEnum` whose members *are* the variable keys.

    The ``variable`` query param's element type: FastAPI validates against the keys
    and renders them as an OpenAPI ``enum`` (a dropdown), so an unknown variable is
    rejected at the schema layer -- like ``zone`` -- rather than reaching the reader.
    """
    return enum.StrEnum(  # type: ignore[return-value]
        f'{_sanitize(name)}_VariableKey',
        {_sanitize(key): key for key in keys},
    )


class _Override(NamedTuple):
    """An overridable zone layer's query-field wiring + its scheme default/unit."""

    field_name: str
    alias: str
    param_key: str
    default: int | float | None
    unit: str | None


def _override_fields(
    registry: Mapping[str, AvailableZone],
) -> dict[str, _Override]:
    """Overridable layer key -> its :class:`_Override` (field name, alias, param key,
    scheme default, unit).

    An overridable layer is one whose scheme ``describe()`` names a ``param_key``
    (categorical layers name none). The alias is ``'<layer_key>.<param_key>'`` (the
    dotted query name); the field name is its sanitized, valid-identifier form. The
    scheme's own ``default`` becomes the (non-nullable) field default and example.
    """
    fields: dict[str, _Override] = {}
    for key in sorted(registry):
        desc = registry[key].scheme.describe()
        if desc.param_key is not None:
            alias = f'{key}.{desc.param_key}'
            fields[key] = _Override(
                _sanitize(alias),
                alias,
                desc.param_key,
                desc.default,
                desc.unit,
            )
    return fields


def _base_fields(
    zone_enum: type[enum.StrEnum],
    variable_enum: type[enum.StrEnum],
    override_fields: dict[str, _Override],
) -> dict[str, object]:
    """The query fields both endpoints share: ``zone`` + ``variable`` +
    ``allow_partial`` + one typed, aliased override field per overridable layer.

    ``zone`` and ``variable`` are repeatable per-dataset enums; each override field is
    typed from its :class:`ZoneLayerParams` param and carries the dotted
    ``'<layer>.<param>'`` query alias. Returned as a :func:`pydantic.create_model`
    field spec map so the two endpoints can each fold in their own extra fields.
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
            list[variable_enum],  # type: ignore[valid-type]
            Field(
                default_factory=list,
                description='Variable to report (repeatable; default all).',
            ),
        ),
        'allow_partial': (bool, Field(default=False)),
        # Content negotiation rides *inside* the model (not a Negotiate dependency) so
        # the model stays the sole query source and explodes. The ``| None`` union hides
        # the enum's own description, so ``f_description`` sets it (naming the actual
        # ``?f=`` keys, json/csv). ``None`` defers to Accept.
        'f': (
            _StatsFormat | None,
            Field(default=None, description=f_description(_StatsFormat)),
        ),
    }
    for key, ov in override_fields.items():
        # Non-nullable, defaulted to the scheme's own default (shown as the example),
        # so the doc advertises a real value rather than an empty ``... | null`` box.
        # The ``ZoneLayerParams`` annotation is optional (``int | None``); strip the
        # ``None`` to keep int-vs-float but drop nullability.
        annotation = _param_annotation(ov.param_key)
        unit = f' {ov.unit}' if ov.unit else ''
        fields[ov.field_name] = (
            annotation,
            Field(
                default=ov.default,
                alias=ov.alias,
                examples=[ov.default],
                description=f'Override the scheme param for zone {key!r} '
                f'(default: {ov.default}{unit}).',
            ),
        )
    return fields


# FastAPI expands a Pydantic query model into per-field query params only when it is
# the endpoint's *sole* query source -- ANY other query param collapses it back to a
# single opaque object, including one declared inside a ``Depends`` sub-dependency (as
# the ``Negotiate``/``DatetimeParam`` adapters do). So every query input rides *inside*
# the model: ``f``/``datetime`` (folded as gazebo field types), and the day-of-year
# ``month``/``day``/year span here, which the doy endpoint folds onto the shared base.
_DOY_FIELDS: dict[str, object] = {
    'month': (int, Field(ge=1, le=12)),
    'day': (int, Field(ge=1, le=31)),
    'start_year': (int, Field(ge=1, le=9999)),
    'end_year': (int, Field(ge=1, le=9999)),
}

# Selection is by *date* here, so these override DatetimeQuery's generic timestamp
# examples with date-style ones covering each form: a single day, a closed interval,
# and open-ended on either side.
_DATETIME_EXAMPLES = [
    '2018-04-27',
    '2018-01-01/2018-06-30',
    '2018-01-01/..',
    '../2018-06-30',
]


def _query_model(name: str, suffix: str, fields: dict[str, object]) -> type[BaseModel]:
    """A per-dataset query-params model from a ``create_model`` field-spec map.

    ``suffix`` keeps the generated model names unique per dataset *and* per endpoint
    so the OpenAPI component schemas never collide.
    """
    return create_model(f'{_sanitize(name)}_{suffix}', **fields)  # type: ignore[call-overload]


def _selections(
    params: BaseModel,
    override_fields: dict[str, _Override],
) -> list[ZoneSelection]:
    """The crossed-zone selections from a parsed query model.

    One :class:`ZoneSelection` per selected ``zone`` (in request order), each carrying
    the override value of its layer (the field is non-nullable and defaults to the
    scheme's own default, so passing it is always equivalent to the scheme default
    unless the client changed it). Override fields for *unselected* zones are ignored:
    with defaulted (never-``None``) fields there is no way to tell an explicit override
    from the default, so an orphan override is a harmless no-op, not an error.
    """
    selected = [str(z) for z in params.zone]  # type: ignore[attr-defined]
    selected_set = set(selected)
    overrides = {
        key: getattr(params, ov.field_name)
        for key, ov in override_fields.items()
        if key in selected_set
    }
    return [ZoneSelection(key, overrides.get(key)) for key in selected]


def build_stats_router(dataset: Dataset) -> GazeboRouter:
    """A :class:`GazeboRouter` carrying ``dataset``'s two stats routes."""
    name = dataset.spec.name
    response_model = StatsResponse[dataset.spec.zonal_stats_model]  # type: ignore[name-defined]
    registry = available_zones(dataset.providers.values())

    zone_enum = _zone_enum(name, registry)
    variable_enum = _variable_enum(name, list(dataset.spec.variables))
    override_fields = _override_fields(registry)
    base_fields = _base_fields(zone_enum, variable_enum, override_fields)
    # ``datetime`` (the OGC interval) rides inside the date-range model too, so the
    # model stays the sole query source and explodes; doy selects by month/day/year
    # instead and has no interval.
    date_range_model = _query_model(
        name,
        'StatsQuery',
        {
            **base_fields,
            'datetime': (
                DatetimeQuery,
                # ``examples`` (array) documents every form; Swagger UI won't pre-fill a
                # query box from it, but it *does* from a singular ``schema.example`` --
                # injected via ``json_schema_extra`` -- so the box shows a usable value.
                Field(
                    default=None,
                    examples=_DATETIME_EXAMPLES,
                    json_schema_extra={'example': _DATETIME_EXAMPLES[0]},
                ),
            ),
        },
    )
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
            variable_keys=[str(v) for v in params.variable] or None,  # type: ignore[attr-defined]
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

    # ``f``/``datetime`` are folded model fields, so the model is the endpoint's sole
    # query source and FastAPI explodes it into individual documented params (no
    # ``Negotiate``/``DatetimeParam`` dependency, whose own query params would collapse
    # it back to one opaque object). ``negotiate`` reads the validated ``?f=`` and, with
    # no explicit ``accept``, falls back to the ambient request's ``Accept`` header.
    async def date_range_stats(
        triplet: types.StationTriplet,
        reader: ReaderDep,
        params,
    ):
        rep = negotiate(_REPRESENTATIONS, f=params.f)
        query = _date_range(params.datetime)
        return await run(reader, triplet, query, params, rep)

    async def doy_stats(
        triplet: types.StationTriplet,
        reader: ReaderDep,
        params,
    ):
        rep = negotiate(_REPRESENTATIONS, f=params.f)
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

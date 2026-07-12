"""The per-dataset query-params compiler behind the stats routes.

Both stats endpoints take the *same* stratification surface -- ``zone`` +
``variable`` + ``allow_partial``, one override param per overridable zone layer,
plus content negotiation (``f``) -- and the date-range endpoint adds the OGC
``datetime`` interval while the day-of-year endpoint adds ``month``/``day``/year
span. :func:`build_query_models` compiles that surface into the two per-dataset
Pydantic models FastAPI explodes into individual, documented query params;
:func:`selections` reads the crossed-zone selection back out of a parsed model.

**Why one model per endpoint, and why everything rides inside it.** FastAPI
expands a Pydantic query model into per-field query params *only* when it is the
endpoint's sole query source -- ANY other query param collapses it back to one
opaque object, including a param declared inside a ``Depends`` sub-dependency (as
gazebo's ``Negotiate``/``DatetimeParam`` adapters do). So every query input rides
*inside* the model: ``f`` and ``datetime`` are folded in as gazebo field types
(:class:`_StatsFormat`, :class:`~gazebo.params.DatetimeQuery`) rather than
dependencies, and the day-of-year fields are added onto the shared base.

**Why the surface is per-dataset (and built here, not annotated).** ``zone`` and
``variable`` are enums constrained to *this* dataset's registry keys, and each
overridable layer contributes a param named ``'<layer_key>.<param_key>'`` (e.g.
``terrain.elevation.band_step_ft``) -- so an unknown zone/variable/format is a
schema-level rejection rather than a hand-rolled error, and the docs advertise the
real, per-dataset set. Because the models are locals built at request-router-build
time they cannot be named in a module-level annotation, so the router patches the
compiled ``Annotated`` object onto each handler's ``__annotations__`` before
registering it. The static shape the handlers see is the :class:`StatsParams`
family of protocols below.

The ``# type: ignore``s here are all one kind: mypy cannot see a type built at
runtime (a functional ``StrEnum``, ``list[<that enum>]``, or ``create_model``).
They are quarantined to this module.
"""

from __future__ import annotations

import enum

from typing import TYPE_CHECKING, Annotated, NamedTuple, Protocol

from fastapi import Query
from gazebo.negotiation import FormatEnum, f_description
from gazebo.params import DatetimeQuery
from pydantic import BaseModel, Field, create_model, model_validator

from snowtool.snowdb.config import ZONE_PARAM_MODELS
from snowtool.snowdb.zonal_stats import ZoneSelection
from snowtool.snowdb.zones.zone_layer import available_zones

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gazebo.params import DatetimeInterval

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.zones.zone_layer import AvailableZone


class _StatsFormat(FormatEnum):
    """The ``?f=`` keys the stats routes serve, each carrying its media type.

    Folded into the query model as a field type so ``?f=`` explodes into a
    documented enum query param (see the module docstring). Because members carry
    their media type they *are* the representation set (:meth:`representations`),
    and ``Accept``-header negotiation is layered back on in the handler with a
    one-line :func:`~gazebo.negotiation.negotiate` call.
    """

    json = 'json', 'application/json'
    csv = 'csv', 'text/csv'


REPRESENTATIONS = _StatsFormat.representations()


class StatsParams(Protocol):
    """The query fields both stats endpoints share (the static view of a compiled
    model; the runtime object is a :func:`pydantic.create_model` instance)."""

    zone: list[str]
    variable: list[str]
    allow_partial: bool
    include_empty_zones: bool
    f: _StatsFormat | None


class DateRangeParams(StatsParams, Protocol):
    """:class:`StatsParams` plus the date-range endpoint's OGC ``datetime`` interval."""

    datetime: DatetimeInterval | None


class DOYParams(StatsParams, Protocol):
    """:class:`StatsParams` plus the day-of-year endpoint's month/day/year span."""

    month: int
    day: int
    start_year: int
    end_year: int


# What the override param controls, per scheme kind -- names the knob in the query
# param's description (a categorical axis has no override param, so is absent).
_OVERRIDE_NOUN = {
    'banded': 'band width',
    'bucketed': 'bucket count',
    'threshold': 'split threshold',
}


class _Override(NamedTuple):
    """An overridable zone layer's query-field wiring + its scheme default/unit."""

    field_name: str
    alias: str
    param_key: str
    default: int | float | None
    unit: str | None
    kind: str


class StatsQueryModels(NamedTuple):
    """The compiled per-dataset query surface: the two ``Annotated[model, Query()]``
    param annotations (patched onto the handlers) and the override map
    :func:`selections` reads back out."""

    date_range: object
    doy: object
    overrides: dict[str, _Override]


def _sanitize(key: str) -> str:
    """A valid Python identifier from a dotted/dashed registry key (for a type name
    or a model field name); the query name is restored via an alias."""
    return ''.join(ch if ch.isalnum() else '_' for ch in key)


def _param_annotation(param_key: str) -> object:
    """The declared type of a zone override param (``int`` or ``float``).

    Read from the param's member model, where the field is required -- so the
    annotation is already non-nullable.
    """
    return ZONE_PARAM_MODELS[param_key].model_fields[param_key].annotation


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


def _override_fields(registry: Mapping[str, AvailableZone]) -> dict[str, _Override]:
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
                desc.kind,
            )
    return fields


def _base_fields(
    zone_enum: type[enum.StrEnum],
    variable_enum: type[enum.StrEnum],
    override_fields: dict[str, _Override],
) -> dict[str, object]:
    """The query fields both endpoints share, as a :func:`pydantic.create_model`
    field-spec map: ``zone`` + ``variable`` (repeatable per-dataset enums),
    ``allow_partial``, the content-negotiation ``f``, and one typed, aliased override
    field per overridable layer.
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
        'allow_partial': (
            bool,
            Field(
                default=False,
                description=(
                    'Permit a basin only partially covered by the dataset grid. By '
                    'default a partially-covered basin is a 409 (guarding against '
                    'stats silently computed over just the covered slice but reported '
                    'as the whole basin); set true to accept a knowingly-clipped '
                    'query over the covered portion. A wholly off-grid basin always '
                    '409s regardless.'
                ),
            ),
        ),
        'include_empty_zones': (
            bool,
            Field(
                default=False,
                description=(
                    'Include crossed zones that no AOI pixel falls in (0 area, all '
                    'stats null). By default these empty combinations are dropped, '
                    'which trims a response that would otherwise grow combinatorically '
                    'with the number of crossed zone axes; set true to emit the full '
                    'zone product. No effect on a whole-basin (unstratified) query.'
                ),
            ),
        ),
        # The ``| None`` union hides the enum's own description, so ``f_description``
        # sets it (naming the actual ``?f=`` keys, json/csv). ``None`` defers to Accept.
        'f': (
            _StatsFormat | None,
            Field(default=None, description=f_description(_StatsFormat)),
        ),
    }
    for key, ov in override_fields.items():
        # Non-nullable, defaulted to the scheme's own default (shown as the example),
        # so the doc advertises a real value rather than an empty ``... | null`` box.
        # The member model's field is required, so the annotation is already
        # non-nullable (int vs float, per member).
        annotation = _param_annotation(ov.param_key)
        unit = f' {ov.unit}' if ov.unit else ''
        noun = _OVERRIDE_NOUN.get(ov.kind, 'scheme param')
        fields[ov.field_name] = (
            annotation,
            Field(
                default=ov.default,
                alias=ov.alias,
                examples=[ov.default],
                description=f'Override the {noun} for zone {key!r} '
                f'(default: {ov.default}{unit}). Applied only when that zone is '
                f'also selected.',
            ),
        )
    return fields


_DOY_FIELDS: dict[str, object] = {
    'month': (int, Field(ge=1, le=12)),
    'day': (int, Field(ge=1, le=31)),
    'start_year': (int, Field(ge=1, le=9999)),
    'end_year': (int, Field(ge=1, le=9999)),
}

# Selection is by *date* here, so these override DatetimeQuery's generic timestamp
# examples with date-style ones covering each form. The first is a closed interval:
# it is the Swagger pre-fill (see ``json_schema_extra``), so the box shows a range.
_DATETIME_EXAMPLES = [
    '2018-01-01/2018-06-30',
    '2018-04-27',
    '2018-01-01/..',
    '../2018-06-30',
]

_DATETIME_FIELD = (
    DatetimeQuery,
    # ``examples`` (array) documents every form; Swagger UI won't pre-fill a query
    # box from it, but it *does* from a singular ``schema.example`` -- injected via
    # ``json_schema_extra`` -- so the box shows a usable value.
    Field(
        default=None,
        examples=_DATETIME_EXAMPLES,
        json_schema_extra={'example': _DATETIME_EXAMPLES[0]},
    ),
)


def _reject_orphan_overrides(overrides: dict[str, _Override]):
    """A model validator rejecting an override changed off its default for a zone that
    isn't selected.

    Such an override can't take effect, so it is a malformed query-parameter
    combination -- a client error. Raised as a pydantic ``ValueError`` from a
    ``@model_validator`` so it fails during schema validation and flows through
    gazebo's ``RequestValidationError`` catchall to a **400**, exactly like an unknown
    ``zone``/``variable`` or a wrong-typed override (not the 422 reserved for
    well-formed-but-unprocessable queries). (FastAPI populates every query-model field,
    defaults included, so a value *equal* to the default is a genuine no-op either way;
    comparing to ``ov.default`` catches exactly the meaningful case.)
    """

    def check(self: StatsParams) -> StatsParams:
        selected = {str(z) for z in self.zone}
        for key, ov in overrides.items():
            if key not in selected and getattr(self, ov.field_name) != ov.default:
                raise ValueError(
                    f'override {ov.alias!r} was set but its zone {key!r} is not '
                    f'selected; add zone={key} or drop the override.',
                )
        return self

    return model_validator(mode='after')(check)


def _query_model(
    name: str,
    suffix: str,
    fields: dict[str, object],
    overrides: dict[str, _Override],
) -> type[BaseModel]:
    """A per-dataset query-params model from a ``create_model`` field-spec map.

    ``suffix`` keeps the generated model names unique per dataset *and* per endpoint
    so the OpenAPI component schemas never collide. The orphan-override rule rides on
    the model as a validator so it fails as schema validation (see
    :func:`_reject_orphan_overrides`).
    """
    validators = {'_reject_orphan_overrides': _reject_orphan_overrides(overrides)}
    return create_model(  # type: ignore[call-overload]
        f'{_sanitize(name)}_{suffix}',
        __validators__=validators,
        **fields,
    )


def build_query_models(dataset: Dataset) -> StatsQueryModels:
    """Compile ``dataset``'s two stats query-params models (see the module docstring).

    Returns the two ``Annotated[model, Query()]`` param annotations (the router
    patches them onto its handlers) and the override map :func:`selections` consumes.
    """
    name = dataset.spec.name
    registry = available_zones(dataset.providers.values())
    zone_enum = _zone_enum(name, registry)
    variable_enum = _variable_enum(name, list(dataset.spec.variables))
    overrides = _override_fields(registry)
    base = _base_fields(zone_enum, variable_enum, overrides)
    date_range_model = _query_model(
        name,
        'StatsQuery',
        {**base, 'datetime': _DATETIME_FIELD},
        overrides,
    )
    doy_model = _query_model(name, 'DOYStatsQuery', {**base, **_DOY_FIELDS}, overrides)
    return StatsQueryModels(
        Annotated[date_range_model, Query()],
        Annotated[doy_model, Query()],
        overrides,
    )


def selections(
    params: StatsParams,
    overrides: dict[str, _Override],
) -> list[ZoneSelection]:
    """The crossed-zone selections from a parsed query model.

    One :class:`ZoneSelection` per selected ``zone`` (in request order), each carrying
    its layer's override value (defaulted to the scheme's own default, so an untouched
    field is equivalent to no override). This is the API counterpart of the CLI's
    ``parse_zone_selection``: both converge on ``list[ZoneSelection]``, differing only
    in input shape -- pre-typed fields here vs. a ``LAYER[:override]`` string there.

    Pure projection: an override for an *unselected* zone has already been rejected at
    schema validation (see :func:`_reject_orphan_overrides`), so a leftover here is a
    default-valued no-op and is simply not read.
    """
    selected = [str(z) for z in params.zone]
    selected_set = set(selected)
    values = {
        key: getattr(params, ov.field_name)
        for key, ov in overrides.items()
        if key in selected_set
    }
    return [ZoneSelection(key, values.get(key)) for key in selected]

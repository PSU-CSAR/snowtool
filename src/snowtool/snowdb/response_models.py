"""Per-dataset zonal-stats response models, generated from a :class:`DatasetSpec`.

A dataset's variables are not known until its spec is defined, so the API
response model for its zonal statistics is built dynamically with
:func:`pydantic.create_model` and cached on the spec
(``spec.zonal_stat_model`` / ``spec.zonal_stats_model``). Each variable
contributes one ``<reducer>_<key>_<unit>`` field (see
:attr:`DatasetVariable.stat_name`); cells with no valid pixels carry a ``nan``
value in memory, which the base model serializes to ``null`` so the payload is
valid JSON.

The response shape is a **flat list of self-describing crossed-zone cells** (not a
nested tree): each cell carries ``zone`` -- one :class:`ZoneRef` per crossed axis
(a discriminated union of a band ref ``{layer, min, max, unit}`` or a class ref
``{layer, code, label}``) -- plus ``area_m2`` and the spec-derived variable stat
fields. A one-axis (elevation-only) query is just a cell whose ``zone`` has a
single ref; crossing more layers lengthens each cell's ``zone`` array without
changing the schema, and the list flattens 1:1 to CSV.
"""

from __future__ import annotations

import math

from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    create_model,
    model_serializer,
)

if TYPE_CHECKING:
    from snowtool.snowdb.spec import DatasetSpec


class BandZoneRef(BaseModel):
    """One crossed-zone axis that is a numeric band ``[min, max)`` in ``unit``."""

    kind: Literal['band'] = 'band'
    layer: str  # the registry key, e.g. 'terrain.elevation'
    min: int
    max: int
    unit: str


class ClassZoneRef(BaseModel):
    """One crossed-zone axis that is a discrete class (its ``code`` + ``label``)."""

    kind: Literal['class'] = 'class'
    layer: str  # the registry key, e.g. 'terrain.aspect'
    code: int
    label: str


class ThresholdZoneRef(BaseModel):
    """One crossed-zone axis that is a threshold split (forested vs unforested).

    ``threshold`` (in ``unit``) is the split point as a real value; ``side`` is
    ``'below'`` or ``'above'`` (at-or-above); ``label`` is the human name of the
    side (e.g. ``'forested'``).
    """

    kind: Literal['threshold'] = 'threshold'
    layer: str  # the registry key, e.g. 'landcover.forest_cover'
    threshold: float
    unit: str
    side: Literal['below', 'above']
    label: str


# A per-axis zone ref: a band, class, or threshold ref, discriminated on ``kind``.
ZoneRef = Annotated[
    BandZoneRef | ClassZoneRef | ThresholdZoneRef,
    Field(discriminator='kind'),
]


class ZonalStatBase(BaseModel):
    """Base for the generated per-cell models: turns any ``nan`` float to null."""

    @model_serializer(mode='wrap')
    def _nan_to_null(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        return {
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in handler(self).items()
        }


def build_zonal_stat_model(spec: DatasetSpec) -> type[BaseModel]:
    """Build the per-crossed-cell zonal-stat model for ``spec``.

    Every cell carries its self-describing ``zone`` refs and ``area_m2``; each
    variable adds one ``<reducer>_<key>_<unit>`` stat field.
    """
    fields: dict[str, Any] = {
        'zone': (list[ZoneRef], ...),
        'area_m2': (Annotated[float, Field(ge=0)], ...),
    }
    for variable in spec.variables.values():
        fields[variable.stat_name] = (float | None, None)

    return create_model(
        f'{spec.model_prefix}ZonalStat',
        __base__=ZonalStatBase,
        **fields,
    )


def build_zonal_stats_model(
    spec: DatasetSpec,
    stat_model: type[BaseModel],
) -> type[BaseModel]:
    """Build the per-date wrapper model (``date`` + crossed axes + the cells)."""
    return create_model(
        f'{spec.model_prefix}ZonalStats',
        date=(date, ...),
        zone_layers=(list[str], ...),
        zones=(list[stat_model], ...),  # type: ignore[valid-type]
    )

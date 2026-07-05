"""Per-dataset zonal-stats models, generated from a :class:`DatasetSpec`.

The shared contract behind a dataset's zonal statistics: a pydantic model family
consumed both by :class:`~snowtool.snowdb.zonal_stats.ZonalStats` (the domain
reducer that builds them) and by the HTTP API (which serializes them).

A dataset's variables are not known until its spec is defined, so the per-cell model
for its zonal statistics is built dynamically with :func:`pydantic.create_model` and
cached on the spec (``spec.zonal_stat_model`` / ``spec.zonal_stats_model``). Each
variable contributes one ``<reducer>_<key>_<unit>`` field (see
:attr:`DatasetVariable.stat_name`) typed :data:`StatValue`, which normalizes a
no-valid-pixels ``nan`` to ``None`` at construction so the JSON is always valid.

The response shape is a flat list of self-describing crossed-zone cells (not a
nested tree): each cell carries ``zone`` -- one :class:`ZoneRef` per crossed axis --
plus ``area_m2`` and the spec-derived variable stat fields. A one-axis query is a
cell whose ``zone`` has a single ref; crossing more layers lengthens each cell's
``zone`` array without changing the schema, and the list flattens 1:1 to CSV.
"""

from __future__ import annotations

import math

from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    create_model,
)

if TYPE_CHECKING:
    from snowtool.snowdb.spec import DatasetSpec


class BandZoneRef(BaseModel):
    """One crossed-zone axis that is a numeric band ``[min, max)`` in ``unit``."""

    kind: Literal['band'] = 'band'
    layer: str = Field(examples=['terrain.elevation'])  # the registry key
    min: int = Field(examples=[6000])
    max: int = Field(examples=[7000])
    unit: str = Field(examples=['ft'])


class ClassZoneRef(BaseModel):
    """One crossed-zone axis that is a discrete class (its ``code`` + ``label``)."""

    kind: Literal['class'] = 'class'
    layer: str = Field(examples=['terrain.aspect'])  # the registry key
    code: int = Field(examples=[0])
    label: str = Field(examples=['N'])


class ThresholdZoneRef(BaseModel):
    """One crossed-zone axis that is a threshold split (forested vs unforested).

    ``threshold`` (in ``unit``) is the split point as a real value; ``side`` is
    ``'below'`` or ``'above'`` (at-or-above); ``label`` is the human name of the
    side (e.g. ``'forested'``).
    """

    kind: Literal['threshold'] = 'threshold'
    layer: str = Field(examples=['landcover.forest_cover'])  # the registry key
    threshold: float = Field(examples=[50.0])
    unit: str = Field(examples=['%'])
    side: Literal['below', 'above'] = Field(examples=['above'])
    label: str = Field(examples=['forested'])


# A per-axis zone ref: a band, class, or threshold ref, discriminated on ``kind``.
ZoneRef = Annotated[
    BandZoneRef | ClassZoneRef | ThresholdZoneRef,
    Field(discriminator='kind'),
]


def _nan_to_none(value: Any) -> Any:
    """A float ``nan`` becomes ``None``; every other value passes through."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# A variable's stat value in a generated cell model. A no-valid-pixels reduction
# arrives as ``nan`` and is normalized to ``None`` at construction, so the model
# always serializes to valid JSON (``null``, never a ``NaN`` literal). Deliberately
# a ``BeforeValidator``, not a ``model_serializer``: a custom model serializer has
# no declared output schema, so FastAPI (which renders response models in
# serialization mode) would collapse every generated cell model to an opaque
# ``{"type": "object"}`` -- hiding the response shape from the OpenAPI docs. No
# consumer reads the raw ``nan`` back (the CSV path formats from the raw array).
StatValue = Annotated[float | None, BeforeValidator(_nan_to_none)]


def build_zonal_stat_model(spec: DatasetSpec) -> type[BaseModel]:
    """Build the per-crossed-cell zonal-stat model for ``spec``.

    Every cell carries its self-describing ``zone`` refs and ``area_m2``; each
    variable adds one ``<reducer>_<key>_<unit>`` stat field.
    """
    fields: dict[str, Any] = {
        'zone': (
            list[ZoneRef],
            Field(
                ...,
                examples=[
                    [
                        {
                            'kind': 'band',
                            'layer': 'terrain.elevation',
                            'min': 6000,
                            'max': 7000,
                            'unit': 'ft',
                        },
                    ],
                ],
            ),
        ),
        'area_m2': (Annotated[float, Field(ge=0)], Field(..., examples=[592891.69])),
    }
    for variable in spec.variables.values():
        fields[variable.stat_name] = (
            StatValue,
            Field(default=None, examples=[42.7]),
        )

    return create_model(
        f'{spec.model_prefix}ZonalStat',
        **fields,
    )


def build_zonal_stats_model(
    spec: DatasetSpec,
    stat_model: type[BaseModel],
) -> type[BaseModel]:
    """Build the per-date wrapper model (``date`` + crossed axes + the cells)."""
    return create_model(
        f'{spec.model_prefix}ZonalStats',
        date=(date, Field(..., examples=['2008-12-14'])),
        zone_layers=(list[str], Field(..., examples=[['terrain.elevation']])),
        zones=(list[stat_model], ...),  # type: ignore[valid-type]
    )

"""Zonal-stats models: the crossed-zone refs and the compact response body.

The shared contract behind a dataset's zonal statistics: a pydantic model family
consumed both by :class:`~snowtool.snowdb.zonal_stats.ZonalStats` (the domain
reducer that builds them) and by the HTTP API (which serializes them).

Each crossed-zone axis contributes one self-describing :class:`ZoneRef` (a band,
class, or threshold ref, discriminated on ``kind``); :data:`StatValue` normalizes a
no-valid-pixels ``nan`` to ``None`` at construction so the JSON is always valid.

The response shape is the compact body (:class:`CompactStats`): zones and
variables are defined once, and each date maps to a bare ``zones x variables``
matrix -- generic across every dataset, with no per-dataset generated model.
"""

from __future__ import annotations

import math

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
)


class BandZoneRef(BaseModel):
    """One crossed-zone axis that is a numeric band ``[min, max)``.

    ``min``/``max`` are ``int`` for an integer-stepped axis (elevation feet) and
    fractional for a bucketed one (the dimensionless ``[-1, 1]`` aspect components);
    ``unit`` is that axis' zone unit, or ``null`` when the measure is dimensionless.
    """

    kind: Literal['band'] = 'band'
    layer: str = Field(examples=['terrain.elevation'])  # the registry key
    min: int | float = Field(examples=[6000])
    max: int | float = Field(examples=[7000])
    unit: str | None = Field(default=None, examples=['ft'])


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


# A variable's stat value in the compact response body. A no-valid-pixels
# reduction arrives as ``nan`` and is normalized to ``None`` at construction, so
# the model always serializes to valid JSON (``null``, never a ``NaN`` literal).
# Deliberately a ``BeforeValidator``, not a ``model_serializer``: a custom model
# serializer has no declared output schema, so FastAPI (which renders response
# models in serialization mode) would collapse the body to an opaque
# ``{"type": "object"}`` -- hiding the response shape from the OpenAPI docs. No
# consumer reads the raw ``nan`` back (the CSV path formats from the raw array).
StatValue = Annotated[float | None, BeforeValidator(_nan_to_none)]


class CompactZone(BaseModel):
    """One crossed-zone cell in the compact form: its refs plus hoisted area.

    ``area_m2`` is date-invariant (a property of the crossed-zone geometry), so it
    is stated once here rather than repeated per date as the verbose per-date model
    does.
    """

    zone: list[ZoneRef] = Field(
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
    )
    area_m2: Annotated[float, Field(ge=0)] = Field(..., examples=[592891.69])


class CompactStats(BaseModel):
    """The normalized compact zonal-stats body, generic across every dataset.

    Zones and variables are defined once; ``results`` maps each date to a bare
    ``zones x variables`` matrix (outer index aligns to ``zones``, inner to
    ``variables``). A ``null`` is a variable with no valid pixels that date (or an
    empty zone). Variables are string ``stat_name``\\ s, so this body carries no
    per-dataset field names and needs no generated model.
    """

    zone_layers: list[str] = Field(..., examples=[['terrain.elevation']])
    variables: list[str] = Field(..., examples=[['mean_swe_mm']])
    zones: list[CompactZone]
    results: dict[date, list[list[StatValue]]] = Field(
        ...,
        examples=[{'2008-12-14': [[42.7], [51.3]]}],
    )

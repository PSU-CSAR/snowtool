"""Per-dataset zonal-stats response models, generated from a :class:`DatasetSpec`.

A dataset's variables are not known until its spec is defined, so the API
response model for its zonal statistics is built dynamically with
:func:`pydantic.create_model` and cached on the spec
(``spec.zonal_stat_model`` / ``spec.zonal_stats_model``). Each variable
contributes one ``<reducer>_<key>_<unit>`` field (see
:attr:`DatasetVariable.stat_name`); bands with no valid pixels carry a ``nan``
value in memory, which the base model serializes to ``null`` so the payload is
valid JSON.
"""

from __future__ import annotations

import math

from datetime import date
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    create_model,
    model_serializer,
)

if TYPE_CHECKING:
    from snowtool.snowdb.spec import DatasetSpec


class ZonalStatBase(BaseModel):
    """Base for the generated per-zone models: turns any ``nan`` float to null."""

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
    """Build the per-elevation-band zonal-stat model for ``spec``."""
    fields: dict[str, Any] = {
        'min_elevation_ft': (float, ...),
        'max_elevation_ft': (float, ...),
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
    """Build the per-date wrapper model (``date`` + a list of ``stat_model``)."""
    return create_model(
        f'{spec.model_prefix}ZonalStats',
        date=(date, ...),
        zones=(list[stat_model], ...),  # type: ignore[valid-type]
    )

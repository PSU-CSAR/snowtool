"""Minimal local GeoJSON geometry models for the pourpoint record + index.

The pourpoint record (``pourpoint.py``) and its derived index
(``pourpoint_index.py``) both need a *typed* stand-in for a raw GeoJSON geometry
mapping -- a coordinate-validated point (the outflow) and a polygon/multipolygon
(the basin). ``geojson-pydantic`` (via ``gazebo[geojson]``) already provides
exactly this, but it is only a *transitive* dependency here (snowdb itself does
not depend on gazebo/the API layer), so this module hand-rolls the tiny subset
actually used rather than taking on a new direct dependency for it. The API layer
(``api/models/pourpoint.py``) converts these to plain mappings at its boundary,
where gazebo's geojson-pydantic-backed ``Feature`` model re-validates them.

Each model exposes a ``.shape`` property converting to a real
:class:`shapely.Geometry` (via ``shapely.geometry.shape``) for the geometry ops
(area, hashing, reprojection) that need one -- the same "typed record, shapely on
demand" split used by ``config.FootprintField``, just as a model instead of a bare
``Annotated[Geometry, ...]`` because these need coordinate-level attribute access
(``.coordinates``) and JSON round-tripping (index ``Feature`` (de)serialization),
not just pass-through storage.
"""

from __future__ import annotations

from typing import Annotated, Literal

import shapely

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from shapely import Geometry

# A GeoJSON position: [lon, lat] or [lon, lat, elevation].
Position = tuple[float, float] | tuple[float, float, float]


class PointGeometry(BaseModel):
    """A GeoJSON ``Point`` geometry (the pourpoint's outflow coordinate)."""

    model_config = ConfigDict(frozen=True)

    type: Literal['Point'] = 'Point'
    coordinates: Position

    @property
    def shape(self) -> Geometry:
        return shapely.geometry.shape(self.model_dump())


class PolygonGeometry(BaseModel):
    """A GeoJSON ``Polygon`` geometry (a basin with no islands/donut holes)."""

    model_config = ConfigDict(frozen=True)

    type: Literal['Polygon'] = 'Polygon'
    coordinates: list[list[Position]]

    @property
    def shape(self) -> Geometry:
        return shapely.geometry.shape(self.model_dump())


class MultiPolygonGeometry(BaseModel):
    """A GeoJSON ``MultiPolygon`` geometry (a multi-part basin)."""

    model_config = ConfigDict(frozen=True)

    type: Literal['MultiPolygon'] = 'MultiPolygon'
    coordinates: list[list[list[Position]]]

    @property
    def shape(self) -> Geometry:
        return shapely.geometry.shape(self.model_dump())


# A pourpoint's basin geometry: a single polygon or a multi-part basin,
# discriminated on the GeoJSON ``type`` string.
BasinGeometry = Annotated[
    PolygonGeometry | MultiPolygonGeometry,
    Field(discriminator='type'),
]

PointGeometryAdapter = TypeAdapter(PointGeometry)
BasinGeometryAdapter: TypeAdapter[PolygonGeometry | MultiPolygonGeometry] = TypeAdapter(
    BasinGeometry,
)

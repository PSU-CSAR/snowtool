from __future__ import annotations

import hashlib

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import shapely

from geojson_pydantic import MultiPolygon, Point, Polygon
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pyproj import CRS, Geod, Transformer
from shapely import Geometry
from shapely.ops import transform as shapely_transform

from snowtool import types
from snowtool.exceptions import GeoJSONValidationError

# A pourpoint's basin geometry: a single polygon or a multi-part basin,
# discriminated on the GeoJSON ``type`` string (geojson-pydantic models).
BasinGeometry = Annotated[Polygon | MultiPolygon, Field(discriminator='type')]

_WGS84 = CRS.from_epsg(4326)
# Geodesic area on the WGS84 ellipsoid -- computes basin area straight from the
# stored lon/lat polygon, so we never depend on the messy `basinarea` property
# (mixed/unknown units) and never have to pick a projected equal-area CRS.
_GEOD = Geod(ellps='WGS84')


# A geometry a pourpoint source's GeometryCollection may hold, discriminated on
# the GeoJSON ``type`` string so a stray member (a LineString, say) fails with
# a precise error.
_SourceGeometry = Annotated[
    Point | Polygon | MultiPolygon,
    Field(discriminator='type'),
]


class _SourceFeature(BaseModel):
    """The point-only pourpoint source form: a GeoJSON ``Feature`` whose
    geometry *must* be the outflow ``Point``.

    ``extra='allow'`` tolerates foreign members (valid GeoJSON); ``id`` is the
    station triplet, pattern-checked here at the parse boundary rather than
    surfacing later as an index-build crash.
    """

    model_config = ConfigDict(extra='allow')

    type: Literal['Feature']
    id: types.StationTriplet
    geometry: Point
    properties: dict[str, Any]

    @property
    def point(self: Self) -> Point:
        return self.geometry

    @property
    def polygon(self: Self) -> None:
        return None


class _SourceGeometryCollection(BaseModel):
    """The basin-bearing source form: a ``GeometryCollection`` of exactly the
    outflow ``Point`` plus the basin ``(Multi)Polygon`` (either order), with
    the triplet ``id`` and ``properties`` as foreign members (the upstream
    convention this importer accepts)."""

    model_config = ConfigDict(extra='allow')

    type: Literal['GeometryCollection']
    id: types.StationTriplet
    geometries: list[_SourceGeometry]
    properties: dict[str, Any]

    @model_validator(mode='after')
    def _exactly_point_plus_basin(self: Self) -> Self:
        if len(self.geometries) != 2:
            raise ValueError(
                'a pourpoint GeometryCollection must hold exactly two '
                f'geometries (point + basin); got {len(self.geometries)}',
            )
        points = [g for g in self.geometries if isinstance(g, Point)]
        if len(points) != 1:
            raise ValueError(
                'a pourpoint GeometryCollection must hold exactly one Point '
                'and one (Multi)Polygon',
            )
        return self

    @property
    def point(self: Self) -> Point:
        return next(g for g in self.geometries if isinstance(g, Point))

    @property
    def polygon(self: Self) -> Polygon | MultiPolygon:
        return next(
            g for g in self.geometries if isinstance(g, (Polygon, MultiPolygon))
        )


# A pourpoint source file, routed by its GeoJSON ``type``: the two accepted
# envelope forms. Anything else (a FeatureCollection, a bare geometry, a JSON
# array) is a discriminator error -> GeoJSONValidationError in from_geojson.
_PourpointSource = Annotated[
    _SourceFeature | _SourceGeometryCollection,
    Field(discriminator='type'),
]
_SOURCE_ADAPTER: TypeAdapter[_SourceFeature | _SourceGeometryCollection] = TypeAdapter(
    _PourpointSource,
)


@dataclass
class Pourpoint:
    """A monitoring/forecast point (station triplet + lon/lat) with an optional
    delineated upstream basin polygon.

    The point is the pourpoint proper -- the outflow through which the basin
    drains -- and is always present; the basin polygon is what gets burned into a
    per-dataset *AOI raster* for zonal queries and may be absent (a point-only
    pourpoint). ``properties`` keeps the full source geojson properties, but only a
    curated few (``awdb_id``/``usgs_id``) are surfaced as attributes; the rest of
    the upstream record is intentionally not part of this model.
    """

    path: Path
    # Verbatim upstream source properties -- a documented exception to the
    # project's typed-modeling default: this is external, open-shaped data (an
    # arbitrary AWDB/USGS property bag), not a snowtool-defined schema, so only
    # the curated fields below (``awdb_id``/``usgs_id``) are pulled out as typed
    # attributes. The source *envelope* (type/id/geometry shape) is typed via
    # :data:`_PourpointSource`; this bag itself is carried through as-is, never
    # round-tripped through validation.
    properties: dict[str, Any]
    station_triplet: types.StationTriplet
    name: str
    point: Point
    polygon: Polygon | MultiPolygon | None = None
    awdb_id: str | None = None
    usgs_id: str | None = None

    @classmethod
    def from_geojson(cls: type[Self], path: Path | str) -> Self:
        """Parse a pourpoint record, classifying *any* unreadable source as invalid.

        The whole read parses through the typed :data:`_PourpointSource` union,
        so garbage bytes, malformed JSON, a wrong envelope ``type``, a
        malformed station-triplet ``id``, a non-Point feature geometry, or a
        mis-shaped GeometryCollection all surface as one
        :class:`GeoJSONValidationError` -- keeping a single bad file in a
        ``pourpoint import``/``sync`` batch in the ``invalid`` list instead of
        aborting the run (``_classify_sources`` catches only
        :class:`GeoJSONValidationError`).
        """
        path = Path(path)
        try:
            source = _SOURCE_ADAPTER.validate_json(path.read_bytes())
        except ValidationError as e:
            raise GeoJSONValidationError(
                f'Pourpoint source is not a valid pourpoint geojson: {e}',
            ) from e

        properties = source.properties
        name = properties.get('nwccname') or properties.get('name')
        if not name:
            raise GeoJSONValidationError(
                "Pourpoint missing required property: 'nwccname' or 'name'",
            )
        return cls(
            path=path,
            properties=properties,
            station_triplet=source.id,
            name=name,
            point=source.point,
            polygon=source.polygon,
            awdb_id=properties.get('awdb_id'),
            usgs_id=properties.get('usgs_id'),
        )

    # Pourpoints are treated as immutable after construction, so the derived
    # geometry/area/hash below are cached_property (computed once per instance;
    # the no-polygon ValueError is deliberately not cached, so it raises anew).
    @cached_property
    def geometry(self: Self) -> Geometry:
        if self.polygon is None:
            raise ValueError('pourpoint does not have a basin polygon')

        # geojson-pydantic geometries expose __geo_interface__, so shapely reads
        # them directly (no intermediate mapping).
        return shapely.geometry.shape(self.polygon)

    @cached_property
    def area_meters(self: Self) -> float:
        """Geodesic area (m^2) of the basin polygon on the WGS84 ellipsoid.

        Computed straight from the stored lon/lat polygon via :data:`_GEOD`, so it
        is authoritative and unit-correct regardless of the source's own
        ``basinarea`` property. Raises if the pourpoint has no basin polygon.
        """
        area, _ = _GEOD.geometry_area_perimeter(self.geometry)
        return abs(area)

    @cached_property
    def geometry_hash(self: Self) -> str:
        """A stable hex sha256 of the basin polygon, identifying its raster.

        Hashes the polygon's canonical WKB (fixed little-endian byte order so the
        digest is machine-independent). Only the basin polygon is hashed -- the
        pourpoint and properties do not affect the burned AOI raster -- so this is
        exactly the signal that should trigger a re-rasterize (see
        ``constants.AOI_HASH_TAG``). Raises if the pourpoint has no basin polygon.
        """
        wkb = shapely.to_wkb(self.geometry, byte_order=1)
        return hashlib.sha256(wkb).hexdigest()

    def geometry_in_crs(self: Self, crs: CRS | str | int) -> Geometry:
        """This pourpoint's basin polygon reprojected from WGS84 (lon/lat) to ``crs``.

        Pourpoints are global and stored as geojson (EPSG:4326); a dataset whose
        grid uses a projected CRS needs the geometry in that CRS before its tile
        extent and pixel mask are computed. Returns the geometry unchanged when
        ``crs`` is already WGS84 (the common geographic case).
        """
        geometry = self.geometry
        dst = CRS.from_user_input(crs)
        if dst == _WGS84:
            return geometry
        transformer = Transformer.from_crs(_WGS84, dst, always_xy=True)
        return shapely_transform(transformer.transform, geometry)

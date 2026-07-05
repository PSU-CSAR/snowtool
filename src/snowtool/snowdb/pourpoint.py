from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Self

import shapely

from geojson_pydantic import MultiPolygon, Point, Polygon
from pydantic import Field, TypeAdapter, ValidationError
from pyproj import CRS, Geod, Transformer
from shapely import Geometry
from shapely.ops import transform as shapely_transform

from snowtool import types
from snowtool.exceptions import GeoJSONValidationError

# A pourpoint's basin geometry: a single polygon or a multi-part basin,
# discriminated on the GeoJSON ``type`` string (geojson-pydantic models).
BasinGeometry = Annotated[Polygon | MultiPolygon, Field(discriminator='type')]
_BASIN_ADAPTER: TypeAdapter[Polygon | MultiPolygon] = TypeAdapter(BasinGeometry)

_WGS84 = CRS.from_epsg(4326)
# Geodesic area on the WGS84 ellipsoid -- computes basin area straight from the
# stored lon/lat polygon, so we never depend on the messy `basinarea` property
# (mixed/unknown units) and never have to pick a projected equal-area CRS.
_GEOD = Geod(ellps='WGS84')

# geojson type strings
GEOM_COLLECTION = 'GeometryCollection'
FEATURE = 'Feature'
POINT = 'Point'
POLYGON = 'Polygon'
MULTIPOLYGON = 'MultiPolygon'

POLYGON_TYPES = [POLYGON, MULTIPOLYGON]


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
    # attributes. It is never round-tripped through validation, only carried.
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

        The read + JSON parse live inside the conversion ``try`` so that garbage
        bytes or malformed JSON surface as :class:`GeoJSONValidationError` -- the
        same error a schema mismatch raises -- rather than a raw
        ``JSONDecodeError``/``UnicodeDecodeError``. That keeps a single bad file in
        a ``pourpoint import``/``sync`` batch landing in the ``invalid`` list
        instead of aborting the whole run (``_classify_sources`` catches only
        :class:`GeoJSONValidationError`).
        """
        path = Path(path)

        kwargs: dict[str, Any] = {}
        try:
            geojson = json.loads(path.read_text())
            if not isinstance(geojson, dict):
                raise GeoJSONValidationError(
                    'Pourpoint source is not a GeoJSON object.',
                )
            if geojson['type'] == FEATURE:
                if geojson['geometry']['type'] != POINT:
                    raise GeoJSONValidationError(
                        'All pourpoints must have a point geometry.',
                    )
                kwargs['point'] = Point.model_validate(geojson['geometry'])
            elif geojson['type'] == GEOM_COLLECTION:
                kwargs.update(cls._parse_geometry_collection(geojson['geometries']))
            else:
                raise GeoJSONValidationError(
                    f"Incompatible type '{geojson['type']}'",
                )

            properties = geojson['properties']
            kwargs['properties'] = properties
            kwargs['station_triplet'] = geojson['id']
            kwargs['name'] = properties.get('nwccname') or properties['name']
            kwargs['awdb_id'] = properties.get('awdb_id')
            kwargs['usgs_id'] = properties.get('usgs_id')
        except KeyError as e:
            raise GeoJSONValidationError(
                'Pourpoint missing required property',
            ) from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise GeoJSONValidationError(
                f'Pourpoint source is not readable geojson: {e}',
            ) from e
        except ValidationError as e:
            raise GeoJSONValidationError(
                f'Pourpoint geometry is invalid: {e}',
            ) from e

        return cls(path=path, **kwargs)

    @staticmethod
    def _parse_geometry_collection(geoms: Any) -> dict[str, Any]:
        """Pull the point (required) + basin polygon (required) out of the two-geom
        ``GeometryCollection`` pourpoint form. Raises :class:`GeoJSONValidationError`
        on the wrong count or a missing point/polygon geometry."""
        if len(geoms) != 2:
            raise GeoJSONValidationError(
                'Multi-geometry pourpoints cannot have more than two geometries',
            )
        kwargs: dict[str, Any] = {}
        if geoms[0]['type'] == POINT:
            kwargs['point'] = Point.model_validate(geoms[0])
        elif geoms[1]['type'] == POINT:
            kwargs['point'] = Point.model_validate(geoms[1])
        else:
            raise GeoJSONValidationError(
                'All pourpoints must have a point geometry.',
            )
        if geoms[0]['type'] in POLYGON_TYPES:
            kwargs['polygon'] = _BASIN_ADAPTER.validate_python(geoms[0])
        elif geoms[1]['type'] in POLYGON_TYPES:
            kwargs['polygon'] = _BASIN_ADAPTER.validate_python(geoms[1])
        else:
            raise GeoJSONValidationError(
                'Multi-geometry pourpoints must have one (Mutli)Polygon geometry',
            )
        return kwargs

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

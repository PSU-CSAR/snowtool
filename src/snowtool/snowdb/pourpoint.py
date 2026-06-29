from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import shapely

from pyproj import CRS, Geod, Transformer
from shapely import Geometry
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from snowtool import types
from snowtool.exceptions import GeoJSONValidationError

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
    properties: dict[str, Any]
    station_triplet: types.StationTriplet
    name: str
    point: dict[str, Any]
    polygon: dict[str, Any] | None = None
    awdb_id: str | None = None
    usgs_id: str | None = None

    @classmethod
    def from_geojson(cls: type[Self], path: Path | str) -> Self:
        path = Path(path)
        geojson = json.loads(path.read_text())

        kwargs: dict[str, Any] = {}
        try:
            if geojson['type'] == FEATURE:
                if geojson['geometry']['type'] != POINT:
                    raise GeoJSONValidationError(
                        'All pourpoints must have a point geometry.',
                    )
                kwargs['point'] = geojson['geometry']
            elif geojson['type'] == GEOM_COLLECTION:
                geoms: Any = geojson['geometries']

                if len(geoms) != 2:
                    raise GeoJSONValidationError(
                        'Multi-geometry pourpoints cannot have '
                        'more than two geometries',
                    )

                if geoms[0]['type'] == POINT:
                    kwargs['point'] = geoms[0]
                elif geoms[1]['type'] == POINT:
                    kwargs['point'] = geoms[1]
                else:
                    raise GeoJSONValidationError(
                        'All pourpoints must have a point geometry.',
                    )

                if geoms[0]['type'] in POLYGON_TYPES:
                    kwargs['polygon'] = geoms[0]
                elif geoms[1]['type'] in POLYGON_TYPES:
                    kwargs['polygon'] = geoms[1]
                else:
                    raise GeoJSONValidationError(
                        'Multi-geometry pourpoints must have one '
                        '(Mutli)Polygon geometry',
                    )
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

        return cls(path=path, **kwargs)

    @property
    def geometry(self: Self) -> Geometry:
        if not self.polygon:
            raise ValueError('pourpoint does not have a basin polygon')

        return shape(self.polygon)

    @property
    def area_meters(self: Self) -> float:
        """Geodesic area (m^2) of the basin polygon on the WGS84 ellipsoid.

        Computed straight from the stored lon/lat polygon via :data:`_GEOD`, so it
        is authoritative and unit-correct regardless of the source's own
        ``basinarea`` property. Raises if the pourpoint has no basin polygon.
        """
        area, _ = _GEOD.geometry_area_perimeter(self.geometry)
        return abs(area)

    @property
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

    def geometry_in_crs(self: Self, crs: Any) -> Geometry:
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

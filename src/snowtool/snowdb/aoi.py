from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pyproj import CRS, Transformer
from shapely import Geometry
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from snowtool import types
from snowtool.exceptions import GeoJSONValidationError

_WGS84 = CRS.from_epsg(4326)

# geojson type strings
GEOM_COLLECTION = 'GeometryCollection'
FEATURE = 'Feature'
POINT = 'Point'
POLYGON = 'Polygon'
MULTIPOLYGON = 'MultiPolygon'

POLYGON_TYPES = [POLYGON, MULTIPOLYGON]


@dataclass
class AOI:
    path: Path
    properties: dict[str, Any]
    station_triplet: types.StationTriplet
    name: str
    source: str
    point: dict[str, Any]
    polygon: dict[str, Any] | None = None

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

            kwargs['properties'] = geojson['properties']
            kwargs['station_triplet'] = geojson['id']
            kwargs['name'] = geojson['properties'].get(
                'nwccname',
                geojson['properties']['name'],
            )
            kwargs['source'] = geojson['properties']['source']
        except KeyError as e:
            raise GeoJSONValidationError(
                'Pourpoint missing required property',
            ) from e

        return cls(path=path, **kwargs)

    @property
    def geometry(self: Self) -> Geometry:
        if not self.polygon:
            raise ValueError('AOI does not have a polygon')

        return shape(self.polygon)

    def geometry_in_crs(self: Self, crs: Any) -> Geometry:
        """This AOI's polygon reprojected from WGS84 (geojson lon/lat) to ``crs``.

        AOIs are global and stored as geojson (EPSG:4326); a dataset whose grid
        uses a projected CRS needs the geometry in that CRS before its tile
        extent and pixel mask are computed. Returns the geometry unchanged when
        ``crs`` is already WGS84 (the common geographic case).
        """
        geometry = self.geometry
        dst = CRS.from_user_input(crs)
        if dst == _WGS84:
            return geometry
        transformer = Transformer.from_crs(_WGS84, dst, always_xy=True)
        return shapely_transform(transformer.transform, geometry)

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from griffine import Point
from shapely import Geometry
from shapely.geometry import shape

from snowtool import types
from snowtool.exceptions import GeoJSONValidationError

if TYPE_CHECKING:
    from griffine.grid import AffineGridTile, TiledAffineGrid

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

    def to_tile_extent(
        self: Self,
        grid: TiledAffineGrid,
    ) -> tuple[AffineGridTile, AffineGridTile]:
        minx, miny, maxx, maxy = self.geometry.bounds
        # TODO: check intersection with SNODAS grid
        upperleft = grid.point_to_tile(Point(minx, maxy))
        bottomright = grid.point_to_tile(Point(maxx, miny))
        return upperleft, bottomright

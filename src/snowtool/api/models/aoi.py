"""AOI response models, on gazebo's coordinate-validated GeoJSON types.

``GET /aois`` returns a :class:`gazebo.geojson.FeatureCollection` (a
``LinkedCollection`` of GeoJSON ``Feature``\\ s -- the OGC API Features shape with
``links``/``numberReturned``/``numberMatched`` for free), paged with
:func:`gazebo.pagination.paginate_offset` and optionally filtered by ``bbox``. Each
feature is one indexed entry (triplet id, pourpoint point, name/source/coverage).
``GET /aois/{triplet}`` returns a single ``Feature`` carrying the full stored record
(point + basin polygon as a ``GeometryCollection``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gazebo.geojson import Feature, FeatureCollection
from gazebo.link import Link
from gazebo.pagination import paginate_offset
from gazebo.rels import MediaType, Rel
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from gazebo.params import BBox

    from snowtool.snowdb.aoi import AOI
    from snowtool.snowdb.aoi_index import AOIIndex, AOIIndexEntry


class AOIProperties(BaseModel):
    """The denormalized list-fields of one indexed AOI (per-dataset coverage incl.)."""

    name: str
    source: str
    active: bool | None = None
    basinarea: float | None = None
    geometry_hash: str
    coverage: dict[str, str] = Field(default_factory=dict)


class AOIRecordProperties(BaseModel):
    """The full stored-record properties (permissive -- whatever the geojson holds)."""

    model_config = ConfigDict(extra='allow')


# The concrete response models (used as ``response_model`` and for OpenAPI).
AOIFeature = Feature[AOIProperties]
AOIFeatureCollection = FeatureCollection[AOIProperties]
AOIDetail = Feature[AOIRecordProperties]


def _aoi_feature(entry: AOIIndexEntry) -> AOIFeature:
    feature = entry.to_feature()
    return AOIFeature(
        id=entry.triplet,
        geometry=feature['geometry'],
        properties=AOIProperties.model_validate(feature['properties']),
        links=[
            Link.to_route(
                'get_aoi',
                rel=Rel.SELF,
                type=MediaType.GEOJSON,
                path={'triplet': entry.triplet},
            ),
        ],
    )


def build_aoi_collection(
    index: AOIIndex,
    *,
    offset: int,
    limit: int,
    bbox: BBox | None = None,
) -> AOIFeatureCollection:
    """One page of the index (triplet-sorted, optionally ``bbox``-filtered) + links.

    Pagination links (``paginate_offset``) rewrite only ``offset``/``limit`` and so
    preserve a ``bbox`` filter across pages.
    """
    entries = [
        entry
        for entry in index  # AOIIndex iterates entries sorted by triplet
        if bbox is None or bbox.contains(*entry.point['coordinates'][:2])
    ]
    total = len(entries)
    page = entries[offset : offset + limit]
    return AOIFeatureCollection(
        items=[_aoi_feature(entry) for entry in page],
        number_matched=total,
        links=[
            Link.root_link(),
            *paginate_offset(
                offset=offset,
                limit=limit,
                total=total,
                type=MediaType.GEOJSON,
            ),
        ],
    )


def build_aoi_detail(aoi: AOI) -> AOIDetail:
    """The single-record feature: point + basin polygon as a ``GeometryCollection``."""
    geometries: list[dict] = [aoi.point]
    if aoi.polygon is not None:
        geometries.append(aoi.polygon)
    # A dict geometry is validated by geojson-pydantic at construction; typed Any
    # so mypy accepts it against the Geometry union (GeometryCollection included).
    geometry: Any = {'type': 'GeometryCollection', 'geometries': geometries}
    return AOIDetail(
        id=aoi.station_triplet,
        geometry=geometry,
        properties=AOIRecordProperties.model_validate(aoi.properties),
        links=[
            Link.self_link(type=MediaType.GEOJSON),
            Link.root_link(),
            Link.to_route('list_aois', rel=Rel.COLLECTION, type=MediaType.GEOJSON),
        ],
    )

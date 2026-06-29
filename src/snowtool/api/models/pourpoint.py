"""Pourpoint response models, on gazebo's coordinate-validated GeoJSON types.

``GET /pourpoints`` returns a :class:`gazebo.geojson.FeatureCollection` (a
``LinkedCollection`` of GeoJSON ``Feature``\\ s -- the OGC API Features shape with
``links``/``numberReturned``/``numberMatched`` for free), paged with
:func:`gazebo.pagination.paginate_offset` and optionally filtered by ``bbox``. The
``geometry`` query param selects each feature's geometry slot -- the pourpoint
``point`` (default, served straight from the index) or the basin ``polygon``
(loaded per record, hence a smaller default page). The pourpoint coordinate is
*always* a property too, so the basin view never loses the outflow point.

``GET /pourpoints/{triplet}`` returns a single ``Feature`` -- it is one record, so
there is no payload pressure and it always carries the basin polygon (falling back
to the point for a point-only pourpoint) plus the full curated property set, with
per-dataset coverage pulled from the index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gazebo.geojson import Feature, FeatureCollection
from gazebo.link import Link
from gazebo.pagination import paginate_offset
from gazebo.rels import MediaType, Rel
from pydantic import BaseModel, Field

from snowtool.snowdb.coverage import Coverage

if TYPE_CHECKING:
    from gazebo.params import BBox

    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.pourpoint import Pourpoint
    from snowtool.snowdb.pourpoint_index import PourpointIndex, PourpointIndexEntry


class PourpointProperties(BaseModel):
    """A pourpoint's summary (list) properties -- the denormalized index fields.

    Fixed regardless of the ``geometry`` view: only the feature ``geometry`` slot
    changes between point and basin mode, never these properties. ``pourpoint`` is
    the outflow ``[lon, lat]`` and is always present (free from the index), so the
    basin view still carries the point.
    """

    name: str
    area_meters: float | None = None
    pourpoint: tuple[float, float]
    coverage: dict[str, Coverage] = Field(default_factory=dict)


class PourpointDetailProperties(PourpointProperties):
    """The single-record (detail) properties: the summary set + curated ids."""

    awdb_id: str | None = None
    usgs_id: str | None = None


# The concrete response models (used as ``response_model`` and for OpenAPI).
PourpointFeature = Feature[PourpointProperties]
PourpointFeatureCollection = FeatureCollection[PourpointProperties]
PourpointDetail = Feature[PourpointDetailProperties]


def _point_coords(point: dict[str, Any]) -> tuple[float, float]:
    lon, lat = point['coordinates'][:2]
    return (lon, lat)


def _pourpoint_feature(
    entry: PourpointIndexEntry,
    geometry: dict[str, Any],
) -> PourpointFeature:
    """One list feature: the chosen ``geometry`` slot + the fixed index properties."""
    return PourpointFeature(
        id=entry.triplet,
        geometry=geometry,  # type: ignore[arg-type]  # dict validated by geojson-pydantic
        properties=PourpointProperties(
            name=entry.name,
            area_meters=entry.area_meters,
            pourpoint=_point_coords(entry.point),
            coverage=entry.coverage,
        ),
        links=[
            Link.to_route(
                'get_pourpoint',
                rel=Rel.SELF,
                type=MediaType.GEOJSON,
                path={'triplet': entry.triplet},
            ),
        ],
    )


def build_pourpoint_collection(
    snowdb: SnowDb,
    *,
    offset: int,
    limit: int,
    basin_geometry: bool = False,
    bbox: BBox | None = None,
) -> PourpointFeatureCollection:
    """One page of the index (triplet-sorted, optionally ``bbox``-filtered) + links.

    ``bbox`` always filters on the pourpoint point (cheap, straight from the
    index). With ``basin_geometry`` the page's basin polygons are loaded per record
    (the expensive view -- the index stores points only), so the route uses a
    smaller default page. Pagination links (``paginate_offset``) rewrite only
    ``offset``/``limit``, preserving the ``bbox``/``geometry`` filters across pages.
    """
    index: PourpointIndex = snowdb.pourpoint_index()
    entries = [
        entry
        for entry in index  # PourpointIndex iterates entries sorted by triplet
        if bbox is None or bbox.contains(*entry.point['coordinates'][:2])
    ]
    total = len(entries)
    page = entries[offset : offset + limit]

    items: list[PourpointFeature] = []
    for entry in page:
        if basin_geometry:
            # The index has no polygon by design; load the record for the basin.
            # The entry is indexed (so basin-bearing), but `.polygon` stays
            # Optional on the model -- fall back to the point to keep this total.
            geometry = snowdb.load_pourpoint(entry.triplet, index=index).polygon or (
                entry.point
            )
        else:
            geometry = entry.point
        items.append(_pourpoint_feature(entry, geometry))

    return PourpointFeatureCollection(
        items=items,
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


def build_pourpoint_detail(
    pourpoint: Pourpoint,
    entry: PourpointIndexEntry,
) -> PourpointDetail:
    """The single-record feature: basin polygon geometry + full properties.

    The basin polygon and the curated record fields come from the loaded
    ``pourpoint``; ``area_meters`` and ``coverage`` are the cached, index-derived
    values from ``entry`` (computed at reindex), not recomputed here.
    """
    geometry: Any = pourpoint.polygon
    return PourpointDetail(
        id=pourpoint.station_triplet,
        geometry=geometry,
        properties=PourpointDetailProperties(
            name=pourpoint.name,
            area_meters=entry.area_meters,
            pourpoint=_point_coords(pourpoint.point),
            coverage=entry.coverage,
            awdb_id=pourpoint.awdb_id,
            usgs_id=pourpoint.usgs_id,
        ),
        links=[
            Link.self_link(type=MediaType.GEOJSON),
            Link.root_link(),
            Link.to_route(
                'list_pourpoints',
                rel=Rel.COLLECTION,
                type=MediaType.GEOJSON,
            ),
        ],
    )

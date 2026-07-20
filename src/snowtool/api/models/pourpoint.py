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
there is no payload pressure and it serves the basin polygon as ``geometry`` (a
``load_pourpoint`` lookup gates on the index, so the served record is always
basin-bearing in practice) plus the full curated property set, with per-dataset
coverage pulled from the index.

Coverage in both responses is filtered to *active* datasets: the index itself
carries coverage for every registered dataset (the admin surfaces -- CLI
``pourpoint list``, diagnostics -- legitimately show inactive coverage), but the
API must not advertise a dataset key its own ``/datasets`` and stats routes
refuse to serve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gazebo.geojson import Feature, FeatureCollection
from gazebo.link import Link
from gazebo.pagination import paginate_offset
from gazebo.rels import MediaType, Rel
from pydantic import BaseModel, Field

from snowtool.api.models.dataset import pourpoint_stats_links
from snowtool.snowdb.coverage import Coverage

if TYPE_CHECKING:
    from collections.abc import Container

    from gazebo.params import BBox
    from geojson_pydantic import MultiPolygon, Point, Polygon

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

    name: str = Field(examples=['Clark Fork R at St. Regis'])
    area_meters: float = Field(examples=[27740389176.98])
    pourpoint: tuple[float, float] = Field(examples=[(-115.087346, 47.301864)])
    coverage: dict[str, Coverage] = Field(
        default_factory=dict,
        examples=[{'snodas': 'full', 'swann-800m': 'full', 'instarr': 'partial'}],
    )


class PourpointDetailProperties(PourpointProperties):
    """The single-record (detail) properties: the summary set + curated ids."""

    awdb_id: str | None = Field(default=None, examples=['12354500'])
    usgs_id: str | None = Field(default=None, examples=['12354500'])


# The concrete response models (used as ``response_model`` and for OpenAPI).
PourpointFeature = Feature[PourpointProperties]
PourpointFeatureCollection = FeatureCollection[PourpointProperties]
PourpointDetail = Feature[PourpointDetailProperties]


def _point_coords(point: Point) -> tuple[float, float]:
    lon, lat = point.coordinates[:2]
    return (lon, lat)


def _active_coverage(
    entry: PourpointIndexEntry,
    active: Container[str],
) -> dict[str, Coverage]:
    """The entry's coverage filtered to active dataset names.

    The index carries coverage for every *registered* dataset; the API serves
    only the active subset, so a response must not expose a coverage key a
    client cannot follow to ``/datasets/{name}`` or the stats routes.
    """
    return {name: cov for name, cov in entry.coverage.items() if name in active}


def _pourpoint_stats_links(
    pourpoint: Pourpoint,
    coverage: dict[str, Coverage],
) -> list[Link]:
    """Stats links for every dataset that can actually serve this basin.

    One (date-range, doy) pair per dataset in ``coverage`` (already filtered to
    active datasets) whose coverage is FULL or PARTIAL -- a NONE dataset always
    409s, so its link would advertise a dead end. Each pair binds the triplet
    and carries the dataset name (see
    :func:`snowtool.api.models.dataset.pourpoint_stats_links`). A point-only
    pourpoint has no basin to query, so it gets none (defensive: such records
    are not indexed, so the detail route cannot reach this today).
    """
    if pourpoint.polygon is None:
        return []
    return [
        link
        for name in sorted(coverage)
        if coverage[name] is not Coverage.NONE
        for link in pourpoint_stats_links(name, pourpoint.station_triplet)
    ]


def _pourpoint_feature(
    entry: PourpointIndexEntry,
    geometry: Point | Polygon | MultiPolygon,
    coverage: dict[str, Coverage],
) -> PourpointFeature:
    """One list feature: the chosen ``geometry`` slot + the fixed index properties."""
    return PourpointFeature(
        id=entry.triplet,
        # gazebo's Feature geometry slot is geojson-pydantic's Geometry union, so
        # our geojson-pydantic geometries pass straight through (no conversion).
        geometry=geometry,
        properties=PourpointProperties(
            name=entry.name,
            area_meters=entry.area_meters,
            pourpoint=_point_coords(entry.point),
            coverage=coverage,
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
        if bbox is None or bbox.contains(*entry.point.coordinates[:2])
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
        items.append(
            _pourpoint_feature(
                entry,
                geometry,
                _active_coverage(entry, snowdb.datasets),
            ),
        )

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
    snowdb: SnowDb,
    pourpoint: Pourpoint,
    entry: PourpointIndexEntry,
) -> PourpointDetail:
    """The single-record feature: basin polygon geometry + full properties.

    The basin polygon and the curated record fields come from the loaded
    ``pourpoint``; ``area_meters`` and ``coverage`` are the cached, index-derived
    values from ``entry`` (computed at reindex), not recomputed here -- coverage
    filtered to ``snowdb``'s active datasets (see :func:`_active_coverage`). The
    response also carries per-dataset stats links, one templated (date-range,
    doy) pair per active dataset covering the basin (see
    :func:`_pourpoint_stats_links`).
    """
    coverage = _active_coverage(entry, snowdb.datasets)
    # The basin polygon passes straight through (see `_pourpoint_feature`); `None`
    # for a point-only pourpoint (GeoJSON allows a null geometry).
    return PourpointDetail(
        id=pourpoint.station_triplet,
        geometry=pourpoint.polygon,
        properties=PourpointDetailProperties(
            name=pourpoint.name,
            area_meters=entry.area_meters,
            pourpoint=_point_coords(pourpoint.point),
            coverage=coverage,
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
            *_pourpoint_stats_links(pourpoint, coverage),
        ],
    )

"""Pourpoint listing + detail routes (catalog-only; inject the read :class:`SnowDb`).

Both reads come off the catalog. ``GET /pourpoints`` pages over the persisted index
(optionally ``bbox``-filtered on the point); its ``geometry`` param picks the
feature geometry -- ``point`` (default) straight from the index, or ``basin``, which
loads each page record's polygon and so defaults to a smaller page. ``GET
/pourpoints/{triplet}`` reads the stored record and always returns the basin (point
fallback), with coverage pulled from the index. A missing record raises
:class:`PourpointNotFoundError` -> 404 via the registered domain handler.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Query

# BBox is imported at runtime (not under TYPE_CHECKING) because it is the resolved
# type of the bbox param's annotation.
from gazebo.ext.fastapi import BBoxParam, GazeboRouter, Inject
from gazebo.params import BBox
from starlette.concurrency import run_in_threadpool

from snowtool import types
from snowtool.api.models.pourpoint import (
    PourpointDetail,
    PourpointFeatureCollection,
    build_pourpoint_collection,
    build_pourpoint_detail,
)
from snowtool.api.tags import Tags
from snowtool.snowdb.db import SnowDb

# SnowDb is registered as an app-scoped constant provider (no __provide__ recipe),
# so injection is opt-in via the Inject marker rather than auto-detected.
CatalogDb = Annotated[SnowDb, Inject]

GeometryView = Literal['point', 'basin']
# Point mode serves geometry from the index (cheap); basin mode parses each record's
# polygon (thousands of coords each), so it pages far smaller. The hard cap is the
# same -- one big request is cheaper than many parallel ones.
POINT_DEFAULT_LIMIT = 100
BASIN_DEFAULT_LIMIT = 25
MAX_LIMIT = 1000

router: GazeboRouter = GazeboRouter()


@router.get('/pourpoints', name='list_pourpoints', tags=[Tags.POURPOINTS])
async def list_pourpoints(
    snowdb: CatalogDb,
    geometry: Annotated[GeometryView, Query()] = 'point',
    bbox: Annotated[BBox | None, BBoxParam] = None,
    limit: Annotated[int | None, Query(ge=1, le=MAX_LIMIT)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PourpointFeatureCollection:
    basin = geometry == 'basin'
    if limit is None:
        limit = BASIN_DEFAULT_LIMIT if basin else POINT_DEFAULT_LIMIT
    # Basin mode parses up to `limit` (<= 1000) basin records, each thousands of
    # coordinate pairs -- synchronous disk + shapely work that would block the
    # event loop. Offload it to a worker thread so other requests keep flowing.
    # (Point mode is a cheap in-memory index scan; left inline.)
    if basin:
        return await run_in_threadpool(
            build_pourpoint_collection,
            snowdb,
            offset=offset,
            limit=limit,
            basin_geometry=basin,
            bbox=bbox,
        )
    return build_pourpoint_collection(
        snowdb,
        offset=offset,
        limit=limit,
        basin_geometry=basin,
        bbox=bbox,
    )


@router.get('/pourpoints/{triplet}', name='get_pourpoint', tags=[Tags.POURPOINTS])
async def get_pourpoint(
    triplet: types.StationTriplet,
    snowdb: CatalogDb,
) -> PourpointDetail:
    # load_pourpoint gates on the index (404 for an unindexed/out-of-band
    # triplet), so the entry is guaranteed present for its derived coverage.
    index = snowdb.pourpoint_index()
    pourpoint = snowdb.load_pourpoint(triplet, index=index)
    return build_pourpoint_detail(pourpoint, index[triplet])

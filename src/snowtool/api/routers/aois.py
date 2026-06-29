"""AOI listing + detail routes (catalog-only; inject the read :class:`SnowDb`).

Both reads come straight off the catalog: ``GET /aois`` pages over the persisted
index (no basin parse), optionally filtered by an OGC ``bbox``; ``GET
/aois/{triplet}`` from the stored record. A missing record raises
:class:`AOINotFoundError` -> 404 via the registered domain handler.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Query

# BBox is imported at runtime (not under TYPE_CHECKING) so get_type_hints can
# resolve the route annotations -- if it fails, gazebo silently skips injection.
from gazebo.ext.fastapi import BBoxParam, GazeboRouter, Inject
from gazebo.params import BBox

from snowtool import types
from snowtool.api.models.aoi import (
    AOIDetail,
    AOIFeatureCollection,
    build_aoi_collection,
    build_aoi_detail,
)
from snowtool.api.tags import Tags
from snowtool.snowdb.db import SnowDb

# SnowDb is registered as an app-scoped constant provider (no __provide__ recipe),
# so injection is opt-in via the Inject marker rather than auto-detected.
CatalogDb = Annotated[SnowDb, Inject]

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

router: GazeboRouter = GazeboRouter()


@router.get('/aois', name='list_aois', tags=[Tags.AOIS])
async def list_aois(
    snowdb: CatalogDb,
    bbox: Annotated[BBox | None, BBoxParam] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AOIFeatureCollection:
    return build_aoi_collection(
        snowdb.aoi_index(),
        offset=offset,
        limit=limit,
        bbox=bbox,
    )


@router.get('/aois/{triplet}', name='get_aoi', tags=[Tags.AOIS])
async def get_aoi(triplet: types.StationTriplet, snowdb: CatalogDb) -> AOIDetail:
    return build_aoi_detail(snowdb.load_aoi(triplet))

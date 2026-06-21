"""Migrate legacy SNODAS AOI-raster tile metadata to ``SNOWTOOL_TILE_BBOX``.

Legacy (snodas) AOI COGs identify the tiles an AOI window spans with Bing-style
**quadkeys**: an origin-tile tag plus a per-tile intersected set. The current
runtime reads only the dataset-agnostic ``SNOWTOOL_TILE_BBOX`` tag
(``"ul_row ul_col br_row br_col"``). This module rewrites the legacy tags into
that bbox tag in place — a tag-only metadata update, no pixel re-encode — and is
the sole remaining home of the quadkey decoder (the runtime no longer reads
quadkeys at all). Driven by the ``snowtool migration aoi-tags`` CLI command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import rasterio

from snowtool.snowdb.constants import TILE_BBOX_TAG

if TYPE_CHECKING:
    from pathlib import Path

# Legacy snodas AOI metadata tags (read-only, migration-time only).
LEGACY_ORIGIN_TILE_TAG = 'SNODAS_ORIGIN_TILE'
LEGACY_TILE_TAG_PREFIX = 'SNODAS_TILE'

# SNODAS native tile zoom: legacy quadkeys are always this many characters.
QUADKEY_ZOOM = 4


def quadkey_to_tile_coords(quadkey: str, zoom: int = QUADKEY_ZOOM) -> tuple[int, int]:
    """Decode a Bing-style quadkey to tile ``(row, col)``."""
    if len(quadkey) != zoom:
        raise ValueError(
            f'Tiles only support native zoom level {zoom}, '
            f'but quadkey is for zoom level {len(quadkey)}.',
        )

    row = 0
    col = 0
    for idx, char in enumerate(reversed(quadkey)):
        mask = 1 << idx
        match char:
            case '0':
                continue
            case '1':
                col |= mask
            case '2':
                row |= mask
            case '3':
                row |= mask
                col |= mask
            case _:
                raise ValueError(f'Invalid quadkey: {quadkey}')

    return row, col


def _legacy_quadkeys(tags: dict[str, str]) -> list[str]:
    """The quadkey values from an origin tag + per-tile intersected set."""
    return [
        value
        for key, value in tags.items()
        if key == LEGACY_ORIGIN_TILE_TAG or key.startswith(LEGACY_TILE_TAG_PREFIX)
    ]


def migrate_aoi_raster_tags(path: Path) -> bool:
    """Rewrite one AOI raster's legacy quadkey tags to ``SNOWTOOL_TILE_BBOX``.

    The bbox is the tile bounding box of the legacy intersected tile set: its
    upper-left tile is ``(min row, min col)`` and its lower-right is
    ``(max row, max col)``. Returns ``True`` if the file was migrated, ``False``
    if it already carried the bbox tag (idempotent). Raises ``ValueError`` if the
    file has neither the bbox tag nor any legacy quadkey tags.
    """
    with rasterio.open(path) as ds:
        tags = ds.tags()

    if TILE_BBOX_TAG in tags:
        return False

    quadkeys = _legacy_quadkeys(tags)
    if not quadkeys:
        raise ValueError(
            f'{path}: no {TILE_BBOX_TAG} and no legacy SNODAS quadkey tags; '
            'nothing to migrate.',
        )

    coords = [quadkey_to_tile_coords(quadkey) for quadkey in quadkeys]
    rows = [row for row, _ in coords]
    cols = [col for _, col in coords]
    bbox = f'{min(rows)} {min(cols)} {max(rows)} {max(cols)}'

    # AOI rasters are written as COGs; GDAL refuses an in-place metadata update
    # unless we accept that it may relocate the metadata (pixels are untouched
    # and the file stays a valid GeoTIFF). These rasters are small and read in
    # full locally, so the COG byte-layout is not relied upon.
    with rasterio.open(path, 'r+', IGNORE_COG_LAYOUT_BREAK='YES') as ds:
        ds.update_tags(**{TILE_BBOX_TAG: bbox})

    return True

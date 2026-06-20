"""The SNODAS analysis grid, built on :mod:`griffine`.

``griffine`` provides the affine/tiled grid math (cell/tile lookup, origins,
tiling, and — when built with a geographic CRS — geodesic cell ``.area``). This
module wraps it with the one SNODAS-specific concern it does not cover: the
Bing-style **quadkey** tile identifiers used as COG metadata keys.
"""

from __future__ import annotations

from typing import NamedTuple

from griffine import Affine, Grid
from griffine.grid import AffineGridCell, AffineGridTile, TiledAffineGrid

from snowtool.rasterdb.constants import (
    COLS,
    ORIGIN_X,
    ORIGIN_Y,
    PX_SIZE,
    ROWS,
    TILE_NATIVE_ZOOM,
    TILE_SIZE,
)


def make_snodas_grid(
    *,
    origin_x: float,
    origin_y: float,
    px_size: float,
    cols: int,
    rows: int,
    tile_size: int,
    crs: int | str = 4326,
) -> TiledAffineGrid:
    """Build a north-up tiled grid.

    The transform maps ``(col, row) -> (x, y)`` with a positive x pixel size and
    a negative y pixel size (north-up). ``crs`` defaults to WGS84 lon/lat, so
    cell/tile ``.area`` is geodesic in square meters. Tests use this with a small
    grid.
    """
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        px_size,
        -px_size,
    )
    return (
        Grid(rows, cols)
        .add_transform(transform, crs=crs)
        .tile_via(Grid(tile_size, tile_size))
    )


SNODAS_GRID: TiledAffineGrid = make_snodas_grid(
    origin_x=ORIGIN_X,
    origin_y=ORIGIN_Y,
    px_size=PX_SIZE,
    cols=COLS,
    rows=ROWS,
    tile_size=TILE_SIZE,
)


class PixelCoord(NamedTuple):
    """A (row, col) index into a grid's base (full-resolution) pixel space."""

    row: int
    col: int


def tile_base_origin(tile: AffineGridTile) -> PixelCoord:
    """Base-grid pixel ``(row, col)`` of a tile's upper-left corner."""
    return PixelCoord(*tile.tile_coords_to_base_coords(0, 0))


def bounds(
    transformable: AffineGridCell | AffineGridTile,
) -> tuple[float, float, float, float]:
    """``(minx, miny, maxx, maxy)`` world bounds of a griffine cell or tile."""
    origin = transformable.origin
    antiorigin = transformable.antiorigin
    return (
        min(origin.x, antiorigin.x),
        min(origin.y, antiorigin.y),
        max(origin.x, antiorigin.x),
        max(origin.y, antiorigin.y),
    )


def tiles_in_bbox(
    grid: TiledAffineGrid,
    ul_row: int,
    ul_col: int,
    br_row: int,
    br_col: int,
) -> list[AffineGridTile]:
    """All tiles in the inclusive tile bounding box ``[ul, br]``."""
    return [
        grid[row, col]
        for row in range(ul_row, br_row + 1)
        for col in range(ul_col, br_col + 1)
    ]


# --- legacy quadkey decoding -------------------------------------------------
# Read-only: existing (snodas) COGs identify tiles by Bing-style quadkey. We
# only ever decode them; new files use the linear index above.


def quadkey_to_tile_coords(
    quadkey: str,
    zoom: int = TILE_NATIVE_ZOOM,
) -> tuple[int, int]:
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


def tile_from_quadkey(
    grid: TiledAffineGrid,
    quadkey: str,
    zoom: int = TILE_NATIVE_ZOOM,
) -> AffineGridTile:
    """Look up the tile identified by a legacy ``quadkey`` within ``grid``."""
    row, col = quadkey_to_tile_coords(quadkey, zoom)
    return grid[row, col]

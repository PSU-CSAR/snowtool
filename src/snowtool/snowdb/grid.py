"""Tiled-grid construction and tile/pixel-coordinate geometry, on :mod:`griffine`.

``griffine`` provides the affine/tiled grid math (cell/tile lookup, origins,
tiling, and — when built with a geographic CRS — geodesic cell ``.area``). This
module adds the small shared helpers on top: building a grid (used by
:class:`~snowtool.snowdb.spec.DatasetSpec`), a tile's base-pixel origin, and the
tiles in a tile bounding box. (Reading an AOI raster's ``SNOWTOOL_TILE_BBOX``
metadata lives with ``AOIRaster`` in :mod:`snowtool.snowdb.raster`.)
"""

from __future__ import annotations

from typing import NamedTuple

from griffine import Affine, Grid, Point
from griffine.grid import AffineGridTile, TiledAffineGrid
from rasterio.warp import transform_bounds


def make_grid(
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
    cell/tile ``.area`` is geodesic in square meters; pass a projected CRS for a
    planar grid with constant cell area.
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


class PixelCoord(NamedTuple):
    """A (row, col) index into a grid's base (full-resolution) pixel space."""

    row: int
    col: int


def tile_base_origin(tile: AffineGridTile) -> PixelCoord:
    """Base-grid pixel ``(row, col)`` of a tile's upper-left corner."""
    return PixelCoord(*tile.tile_coords_to_base_coords(0, 0))


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


def grid_extent_4326(
    grid: TiledAffineGrid,
) -> tuple[float, float, float, float]:
    """The grid's full extent as ``(west, south, east, north)`` in EPSG:4326.

    Used to tell a DEM source which geographic area to fetch. The extent is
    computed in the grid's own CRS then transformed to lon/lat.
    """
    base = grid.base_grid
    t = base.transform
    xmin = t.c
    ymax = t.f
    xmax = t.c + base.cols * t.a
    ymin = t.f + base.rows * t.e
    crs = grid.crs
    if crs is None:  # pragma: no cover - make_grid always sets a CRS
        raise ValueError('grid has no CRS')
    west, south, east, north = transform_bounds(crs, 4326, xmin, ymin, xmax, ymax)
    return (west, south, east, north)


def bounding_tiles(
    grid: TiledAffineGrid,
    bounds: tuple[float, float, float, float],
) -> tuple[AffineGridTile, AffineGridTile]:
    """The upper-left and lower-right tiles covering a world-coord bbox.

    ``bounds`` is ``(minx, miny, maxx, maxy)`` in the grid's own CRS (reproject
    geographic geometry first).
    """
    minx, miny, maxx, maxy = bounds
    return (
        grid.point_to_tile(Point(minx, maxy)),
        grid.point_to_tile(Point(maxx, miny)),
    )

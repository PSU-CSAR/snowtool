"""Tiled-grid construction and tile/pixel-coordinate geometry, on :mod:`griffine`.

``griffine`` provides the affine/tiled grid math (cell/tile lookup, origins,
tiling, and — when built with a geographic CRS — geodesic cell ``.area``). This
module adds the small shared helpers on top: building a grid (used by
:class:`~snowtool.snowdb.spec.DatasetSpec`), a tile's base-pixel origin, and the
tiles in a tile bounding box. (Reading an AOI raster's ``SNOWTOOL_TILE_BBOX``
metadata lives with ``AOIRaster`` in :mod:`snowtool.snowdb.raster`.)
"""

from __future__ import annotations

import math

from typing import NamedTuple

from griffine import Affine, Grid
from griffine.grid import AffineGridTile, TiledAffineGrid
from pydantic import BaseModel, ConfigDict, field_validator
from pyproj import CRS
from pyproj.exceptions import CRSError
from rasterio.warp import transform_bounds

from snowtool.exceptions import GeometryOutsideGridError

# A geographic bounding box: (west, south, east, north) in EPSG:4326.
Bounds = tuple[float, float, float, float]

# A bounding box in some other (typically projected, or the grid's own) CRS:
# (minx, miny, maxx, maxy), not necessarily 4326. Structurally identical to
# ``Bounds``; the two aliases exist to document which convention a given tuple
# follows, not to change behaviour.
Extent = tuple[float, float, float, float]


class GridParams(BaseModel):
    """The parameters defining a dataset's north-up tiled grid.

    A frozen model: it is both the dataset's grid *definition* (carried on a
    :class:`~snowtool.snowdb.spec.DatasetSpec`) and its persisted form (the
    ``grid`` block of a dataset config), so there is one type, not a hand-mirrored
    pair. ``crs`` is an EPSG int or a WKT string.
    """

    model_config = ConfigDict(frozen=True)

    origin_x: float
    origin_y: float
    px_size: float
    cols: int
    rows: int
    tile_size: int
    crs: int | str = 4326

    @field_validator('crs')
    @classmethod
    def _crs_parses(cls: type[GridParams], value: int | str) -> int | str:
        """Reject a CRS pyproj cannot parse at config load, not first grid build.

        The value itself stays as authored (an EPSG int or WKT/authority
        string -- it is the persisted form); only its parseability is checked.
        """
        try:
            CRS.from_user_input(value)
        except CRSError as e:
            raise ValueError(f'not a parseable CRS: {value!r} ({e})') from e
        return value


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


def grid_extent(grid: TiledAffineGrid) -> Extent:
    """The grid's full extent as ``(minx, miny, maxx, maxy)`` in its *own* CRS."""
    base = grid.base_grid
    t = base.transform
    return (t.c, t.f + base.rows * t.e, t.c + base.cols * t.a, t.f)


def grid_extent_4326(
    grid: TiledAffineGrid,
) -> Bounds:
    """The grid's full extent as ``(west, south, east, north)`` in EPSG:4326.

    Used to tell a DEM source which geographic area to fetch. The extent is
    computed in the grid's own CRS then transformed to lon/lat.
    """
    xmin, ymin, xmax, ymax = grid_extent(grid)
    crs = grid.crs
    if crs is None:  # pragma: no cover - make_grid always sets a CRS
        raise ValueError('grid has no CRS')
    west, south, east, north = transform_bounds(crs, 4326, xmin, ymin, xmax, ymax)
    return (west, south, east, north)


def bounding_tiles(
    grid: TiledAffineGrid,
    bounds: Extent,
) -> tuple[AffineGridTile, AffineGridTile]:
    """The upper-left and lower-right tiles covering a world-coord bbox, clamped
    to the grid.

    ``bounds`` is ``(minx, miny, maxx, maxy)`` in the grid's own CRS (reproject
    geographic geometry first). A bbox spilling past a grid edge is clamped to
    the edge tiles, so a basin straddling the boundary gets exactly its in-grid
    window -- griffine would otherwise resolve a negative tile index
    Python-style to the far edge, producing an inverted (negative-sized)
    window. A bbox that does not intersect the grid at all has no window to
    burn and raises :class:`~snowtool.exceptions.GeometryOutsideGridError`.
    """
    minx, miny, maxx, maxy = bounds
    gminx, gminy, gmaxx, gmaxy = grid_extent(grid)
    if minx > gmaxx or maxx < gminx or miny > gmaxy or maxy < gminy:
        raise GeometryOutsideGridError(
            f'geometry bounds {bounds} do not intersect the grid extent '
            f'({gminx}, {gminy}, {gmaxx}, {gmaxy})',
        )

    def clamped_tile(x: float, y: float) -> AffineGridTile:
        # The tiled grid's transform maps world coords -> (col, row) tile space
        # (griffine's point_to_tile does the same inversion, but would raise or
        # wrap out-of-grid indices); floor then clamp into the tile grid so an
        # out-of-grid corner resolves to its nearest edge tile.
        col, row = (math.floor(v) for v in ~grid.transform * (x, y))
        return grid[
            min(max(row, 0), grid.rows - 1),
            min(max(col, 0), grid.cols - 1),
        ]

    return clamped_tile(minx, maxy), clamped_tile(maxx, miny)

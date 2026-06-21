"""Grid math tests.

These pin the griffine-backed SNODAS grid against the legacy hand-rolled
formulas it replaced, and exercise the snowtool-specific tile/area helpers on
both the real grid and a tiny synthetic grid.
"""

import pytest

from griffine import Point
from pyproj import Geod

from snowtool.snowdb.datasets import SNODAS_SPEC
from snowtool.snowdb.grid import make_grid, tiles_in_bbox

SNODAS_GRID = SNODAS_SPEC.grid
_GP = SNODAS_SPEC.grid_params  # SNODAS_SPEC is the source of truth for these


def bounds(transformable) -> tuple[float, float, float, float]:
    """``(minx, miny, maxx, maxy)`` world bounds of a griffine cell or tile."""
    origin = transformable.origin
    antiorigin = transformable.antiorigin
    return (
        min(origin.x, antiorigin.x),
        min(origin.y, antiorigin.y),
        max(origin.x, antiorigin.x),
        max(origin.y, antiorigin.y),
    )

# --- legacy reference implementations (verbatim math from the old code) -------


def legacy_latlon_to_pixel(lat: float, lon: float) -> tuple[int, int]:
    row = int((_GP.origin_y - lat) / _GP.px_size)
    col = int((lon - _GP.origin_x) / _GP.px_size)
    return row, col


def legacy_pixel_to_latlon(row: int, col: int) -> tuple[float, float]:
    lat = _GP.origin_y - (row * _GP.px_size)
    lon = _GP.origin_x + (col * _GP.px_size)
    return lat, lon


# --- grid geometry vs legacy --------------------------------------------------


@pytest.mark.parametrize(
    ('lat', 'lon'),
    [
        (47.301864, -115.087346),  # Clark Fork, MT
        (40.0, -120.0),
        (33.5, -100.25),
    ],
)
def test_point_to_cell_matches_legacy(lat: float, lon: float) -> None:
    cell = SNODAS_GRID.base_grid.point_to_cell(Point(lon, lat))
    assert (cell.row, cell.col) == legacy_latlon_to_pixel(lat, lon)


@pytest.mark.parametrize(('row', 'col'), [(0, 0), (668, 1157), (3350, 6934)])
def test_cell_origin_matches_legacy(row: int, col: int) -> None:
    cell = SNODAS_GRID.base_grid[row, col]
    exp_lat, exp_lon = legacy_pixel_to_latlon(row, col)
    assert cell.origin.x == pytest.approx(exp_lon)
    assert cell.origin.y == pytest.approx(exp_lat)


def test_tiling_shape() -> None:
    # ceil(3351/256)=14 tile rows, ceil(6935/256)=28 tile cols
    assert SNODAS_GRID.size == (14, 28)
    assert SNODAS_GRID.tile_size == (256, 256)


def test_point_to_tile_matches_pixel_floor_div() -> None:
    pt = Point(-115.087346, 47.301864)
    cell = SNODAS_GRID.base_grid.point_to_cell(pt)
    tile = SNODAS_GRID.point_to_tile(pt)
    assert (tile.row, tile.col) == (
        cell.row // _GP.tile_size,
        cell.col // _GP.tile_size,
    )
    # tile's base-pixel origin lines up with the block grid
    base_row, base_col = SNODAS_GRID.tile_coords_to_base_coords(
        0,
        0,
        tile.row,
        tile.col,
    )
    assert (base_row, base_col) == (
        tile.row * _GP.tile_size,
        tile.col * _GP.tile_size,
    )


# --- tile bounding box (the current AOI metadata scheme) ---------------------


def test_tiles_in_bbox() -> None:
    tiles = tiles_in_bbox(SNODAS_GRID, 2, 4, 3, 6)
    assert [(t.row, t.col) for t in tiles] == [
        (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
    ]


def test_tiles_in_bbox_single_tile() -> None:
    tiles = tiles_in_bbox(SNODAS_GRID, 5, 7, 5, 7)
    assert [(t.row, t.col) for t in tiles] == [(5, 7)]


# --- geodesic area (griffine native, via the grid's geographic CRS) -----------


@pytest.mark.parametrize('row', [0, 668, 1500, 3000, 3350])
def test_cell_area_matches_geodesic_reference(row: int) -> None:
    # Independently compute the WGS84 geodesic area of the cell's footprint with
    # pyproj and require griffine's .area to match it across the full latitude
    # range. This confirms the area is the correct geodesic m^2 for the cell
    # (and would fail loudly if the grid fell back to planar deg^2).
    cell = SNODAS_GRID.base_grid[row, 100]
    minx, miny, maxx, maxy = bounds(cell)
    reference, _ = Geod(ellps='WGS84').polygon_area_perimeter(
        [minx, maxx, maxx, minx],
        [maxy, maxy, miny, miny],
    )
    assert cell.area == pytest.approx(abs(reference))


# --- synthetic small grid (the testing workhorse) -----------------------------


def test_small_synthetic_grid() -> None:
    grid = make_grid(
        origin_x=-120.0,
        origin_y=45.0,
        px_size=0.01,
        cols=512,
        rows=512,
        tile_size=256,
    )
    assert grid.size == (2, 2)
    assert grid.tile_size == (256, 256)
    # a point just inside the origin lands in tile (0, 0), cell (0, 0)
    pt = Point(-119.995, 44.995)
    tile = grid.point_to_tile(pt)
    assert (tile.row, tile.col) == (0, 0)
    cell = grid.base_grid.point_to_cell(pt)
    assert (cell.row, cell.col) == (0, 0)

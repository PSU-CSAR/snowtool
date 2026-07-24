"""The TileWindow geometry: byte-identical tag round-trip and offset math.

Isolates the one geometry the AOI write and read paths share. The bar for a
dedicated unit file is met because a live snowdb exists: the ``SNOWTOOL_TILE_BBOX``
tag string format must stay byte-identical so existing burned rasters keep parsing,
which the uniform pipeline test does not pin as an exact string.
"""

from snowtool.snowdb.aoi_raster import TileWindow
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.grid import PixelCoord, bounding_tiles

from ..conftest import TILE


def test_from_corner_tiles_spans_the_whole_window(grid):
    # Corners (0, 0) and (1, 1) -> the full 2x2-tile grid.
    window = TileWindow.from_corner_tiles(grid, grid[0, 0], grid[1, 1])

    assert window.origin == PixelCoord(0, 0)
    assert window.height == 2 * TILE
    assert window.width == 2 * TILE
    assert [(t.row, t.col) for t in window.tiles] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_tag_is_byte_identical_to_the_historical_format(grid):
    # The exact 'ul_row ul_col br_row br_col' string live rasters are stamped with.
    window = TileWindow.from_corner_tiles(grid, grid[0, 0], grid[1, 1])
    assert window.tag == '0 0 1 1'

    single = TileWindow.from_corner_tiles(grid, grid[1, 0], grid[1, 0])
    assert single.tag == '1 0 1 0'


def test_tag_round_trips_through_from_tag(grid):
    # write tag -> parse -> same window
    for start, end in (((0, 0), (1, 1)), ((0, 1), (0, 1)), ((1, 0), (1, 1))):
        written = TileWindow.from_corner_tiles(grid, grid[start], grid[end])
        parsed = TileWindow.from_tag(grid, written.tag)

        assert parsed.tag == written.tag
        assert parsed.origin == written.origin
        assert parsed.height == written.height
        assert parsed.width == written.width
        assert [(t.row, t.col) for t in parsed.tiles] == [
            (t.row, t.col) for t in written.tiles
        ]


def test_from_tags_reads_the_bbox_tag(grid):
    from snowtool.snowdb.aoi_raster import window_from_tags

    window = window_from_tags(grid, {TILE_BBOX_TAG: '0 0 1 1'})
    assert window.tag == '0 0 1 1'


def test_place_offset_is_relative_to_the_window_origin(grid):
    # A window rooted at tile (1, 1): tile (1, 1) sits at offset (0, 0), and the
    # per-tile offset is (tile base origin - window origin).
    window = TileWindow.from_corner_tiles(grid, grid[1, 1], grid[1, 1])
    assert window.place_offset(grid[1, 1]) == PixelCoord(0, 0)

    # A full-grid window: tile (1, 1) sits one tile down and right of the origin.
    full = TileWindow.from_corner_tiles(grid, grid[0, 0], grid[1, 1])
    assert full.place_offset(grid[1, 1]) == PixelCoord(TILE, TILE)


def test_from_corner_tiles_matches_bounding_tiles_on_a_basin_bbox(grid):
    # The write path builds corners via bounding_tiles; a bbox inside tile (0, 0).
    start_tile, end_tile = bounding_tiles(grid, (-119.99, 44.99, -119.98, 45.0))
    window = TileWindow.from_corner_tiles(grid, start_tile, end_tile)

    assert window.tag == '0 0 0 0'
    assert window.origin == PixelCoord(0, 0)
    assert window.height == TILE
    assert window.width == TILE

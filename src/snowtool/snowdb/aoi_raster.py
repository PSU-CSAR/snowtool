"""The burned AOI raster: its model, reader, and writer in one place.

An *AOI raster* is a basin polygon burned onto a dataset grid as per-pixel cell
area (m^2) inside the basin and ``0`` outside -- so the one raster is both the
in/out-of-basin membership mask and the area weights the zonal reduction needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy
import numpy.typing
import rasterio

from rasterio.features import rasterize

from snowtool import types
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.constants import AOI_HASH_TAG, AOI_MASK_NODATA, TILE_BBOX_TAG
from snowtool.snowdb.grid import PixelCoord, tile_base_origin, tiles_in_bbox
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.raster import TiledRaster
from snowtool.snowdb.raster.cog import write_cog

if TYPE_CHECKING:
    from affine import Affine
    from griffine.grid import AffineGrid, AffineGridTile, TiledAffineGrid
    from shapely import Geometry

    from snowtool.snowdb.raster.tiff_cache import TiffCache

# On-disk format version of the burned AOI raster (per-pixel cell area, 0 outside).
# The AOI raster has no ingester/provider -- the Dataset burns it generically -- so
# its version is owned here, by its writer, and stamped onto AOI_HASH_TAG via
# aoi_provenance. Bump on a material format change (e.g. the boolean-mask ->
# cell-area switch) so existing rasters read as stale and re-rasterize.
AOI_RASTER_FORMAT_VERSION = 1


def tiles_from_tags(
    grid: TiledAffineGrid,
    tags: dict[str, str],
) -> tuple[PixelCoord, list[AffineGridTile]]:
    """Resolve an AOI window's origin and tiles from a COG's metadata.

    AOI rasters store a ``ul_row ul_col br_row br_col`` tile bounding box in
    ``SNOWTOOL_TILE_BBOX``. The upper-left tile is the window origin and every
    tile in the box is read (the AOI mask nulls non-AOI pixels).
    """
    try:
        bbox = tags[TILE_BBOX_TAG]
    except KeyError as e:
        raise ValueError('AOI raster is missing tile metadata') from e

    ul_row, ul_col, br_row, br_col = (int(v) for v in bbox.split())
    origin = tile_base_origin(grid[ul_row, ul_col])
    tiles = tiles_in_bbox(grid, ul_row, ul_col, br_row, br_col)
    return origin, tiles


@dataclass
class AOIRaster:
    """A burned AOI: per-pixel cell area inside the basin, 0 outside, over its
    tile-bbox window.

    ``array`` is a ``float32`` of geographic cell area in m^2 for every pixel whose
    centre falls inside the basin polygon, ``0`` elsewhere -- so it is both the
    membership signal (``array > 0``) and the area weights, with no separate area
    raster.
    """

    path: Path
    array: numpy.typing.NDArray[numpy.float32]
    tiles: list[AffineGridTile]
    origin: PixelCoord

    @property
    def station_triplet(self: Self) -> types.StationTriplet:
        return triplet_naming.stem_to_triplet(self.path.stem)

    @classmethod
    def open(
        cls: type[Self],
        path: Path,
        grid: TiledAffineGrid,
    ) -> Self:
        with rasterio.open(path) as ds:
            tags = ds.tags()
            origin, tiles = tiles_from_tags(grid, tags)
            array: numpy.typing.NDArray[numpy.float32] = ds.read(1)

        return cls(
            path=path,
            array=array,
            tiles=tiles,
            origin=origin,
        )

    async def load_raster_tiles_into_array(
        self: Self,
        raster: TiledRaster,
        array: numpy.typing.NDArray[Any],
        cache: TiffCache,
    ) -> None:
        # One coalesced fetch per source COG, then place each block.
        blocks = await raster.load_tiles(self.tiles, cache)
        for tile, block in zip(self.tiles, blocks, strict=True):
            tile_origin = tile_base_origin(tile)
            offset_row = tile_origin.row - self.origin.row
            offset_col = tile_origin.col - self.origin.col
            array[
                offset_row : offset_row + tile.rows,
                offset_col : offset_col + tile.cols,
            ] = block


def make_geometry_mask(
    geometry,
    *,
    out_shape: tuple[int, int],
    transform: Affine,
) -> numpy.typing.NDArray[numpy.bool_]:
    """Rasterize ``geometry`` to a boolean mask, True inside.

    ``geometry`` must already be in the grid/``transform`` CRS.
    """
    burned = rasterize(
        [geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8',
    )
    return burned.astype(bool)


def _window_cell_areas(
    base_grid: AffineGrid,
    start_row: int,
    height: int,
    width: int,
    cell_area: float | None,
) -> numpy.typing.NDArray[numpy.float32]:
    """Per-pixel cell area (m^2) for an AOI window, broadcast to ``(height, width)``.

    A projected grid passes its constant ``cell_area`` (every cell is identical).
    A geographic grid passes ``None``: geodesic cell area depends only on latitude
    (row), so one value per window row is computed from ``base_grid`` and
    broadcast across the columns.
    """
    if cell_area is not None:
        return numpy.broadcast_to(numpy.float32(cell_area), (height, width))
    row_areas = numpy.fromiter(
        (base_grid[start_row + i, 0].area for i in range(height)),
        dtype=numpy.float32,
        count=height,
    )
    return numpy.broadcast_to(row_areas[:, numpy.newaxis], (height, width))


def aoi_provenance(geometry_hash: str) -> str:
    """The versioned tag an AOI raster is stamped with and checked against.

    Combines the AOI's pure geometry digest with the burned-raster format version
    (see :func:`~snowtool.snowdb.provenance.versioned_hash`), so a format change
    invalidates every existing raster through the same equality check that catches
    a geometry change.
    """
    return versioned_hash(AOI_RASTER_FORMAT_VERSION, geometry_hash)


def write_aoi_raster(
    path: Path,
    geometry: Geometry,
    crs: rasterio.crs.CRS,
    start_tile: AffineGridTile,
    end_tile: AffineGridTile,
    tile_size: int,
    provenance: str,
    *,
    base_grid: AffineGrid,
    cell_area: float | None,
) -> None:
    """Burn ``geometry`` to a per-pixel cell-area AOI COG over its tile-bbox window.

    Each pixel whose centre falls inside the basin gets the area (m^2) it rasterizes
    to on this grid; every other pixel is ``0`` (so the one raster is both membership
    signal and area weights). ``cell_area`` is the grid's constant cell area on a
    projected grid, or ``None`` on a geographic grid (per-row geodesic area is
    computed from ``base_grid``).

    ``provenance`` is the versioned AOI tag (see :func:`aoi_provenance`): the AOI
    geometry digest plus the burned-raster format version. Its only AOI-side
    provenance axis is the geometry (``SNOWTOOL_AOI_HASH``); the cell areas are a pure
    function of the fixed grid, and elevation/terrain are read live at query time, so
    a terrain rebuild never invalidates an AOI raster.
    """
    start = tile_base_origin(start_tile)
    end_origin = tile_base_origin(end_tile)
    end_row = end_origin.row + end_tile.rows
    end_col = end_origin.col + end_tile.cols
    height = end_row - start.row
    width = end_col - start.col

    # The tile's own affine is the upper-left transform of the AOI window, at
    # base (full) resolution.
    transform = start_tile.transform

    # ``geometry`` is already in the grid CRS (see Dataset.rasterize_aoi).
    aoi_mask = make_geometry_mask(
        geometry,
        out_shape=(height, width),
        transform=transform,
    )
    areas = _window_cell_areas(base_grid, start.row, height, width, cell_area)
    aoi_area = numpy.where(aoi_mask, areas, numpy.float32(0)).astype(numpy.float32)

    tags = {
        TILE_BBOX_TAG: (
            f'{start_tile.row} {start_tile.col} {end_tile.row} {end_tile.col}'
        ),
        # Records the geometry + format version this raster was burned from, so a
        # changed basin OR a format bump is detected (and re-rasterized) by a cheap
        # tag read.
        AOI_HASH_TAG: provenance,
    }

    write_cog(
        path,
        aoi_area,
        transform=transform,
        crs=crs,
        # 0 = outside the AOI (no real cell has 0 area), so it doubles as the
        # nodata sentinel.
        nodata=AOI_MASK_NODATA,
        tile_size=tile_size,
        tags=tags,
        compute_stats=False,
    )

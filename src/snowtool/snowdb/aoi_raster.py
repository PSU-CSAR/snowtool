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
from rasterio.windows import Window

from snowtool.exceptions import IncompleteDatasetDataError, NodataMaskError
from snowtool.snowdb.constants import AOI_HASH_TAG, AOI_MASK_NODATA, TILE_BBOX_TAG
from snowtool.snowdb.grid import (
    PixelCoord,
    bounding_tiles,
    tile_base_origin,
    tiles_in_bbox,
)
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
        # A burned AOI raster with no tile-bbox tag is corrupt or predates the
        # tagging: a server-side integrity failure the caller fixes by
        # re-rasterizing, not a client error. Typed (not a bare ValueError) so the
        # API surfaces it as an informative 500 problem, not a generic one.
        raise IncompleteDatasetDataError(
            'AOI raster is missing its tile-bbox metadata '
            f'({TILE_BBOX_TAG}); re-rasterize the pourpoint for this dataset.',
        ) from e

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


def _read_nodata_mask_window(
    path: Path,
    base_grid: AffineGrid,
    start: PixelCoord,
    height: int,
    width: int,
) -> numpy.typing.NDArray[numpy.bool_]:
    """The dataset nodata mask's AOI window as a boolean (True = in-domain).

    The mask is a single-band raster on the dataset's *full* grid whose 0
    (= nodata) pixels can never report data; anything nonzero is in-domain.
    Its shape must match the grid exactly -- the window is read by pixel
    offsets, so a mismatched raster would silently misalign; refuse it instead.
    """
    with rasterio.open(path) as ds:
        if ds.shape != (base_grid.rows, base_grid.cols):
            raise NodataMaskError(
                f'nodata mask {path} shape {ds.shape} does not match the '
                f'dataset grid ({base_grid.rows}, {base_grid.cols})',
            )
        band = ds.read(1, window=Window(start.col, start.row, width, height))
    return band != 0


def aoi_provenance(geometry_hash: str, nodata_mask_hash: str | None) -> str:
    """The versioned tag an AOI raster is stamped with and checked against.

    Combines the AOI's pure geometry digest -- plus the dataset's nodata-mask
    file digest, when one is configured -- with the burned-raster format version
    (see :func:`~snowtool.snowdb.provenance.versioned_hash`). A geometry change,
    a mask add/change/remove, or a format bump all invalidate existing rasters
    through the same equality check. An explicit ``None`` (a maskless dataset)
    keeps the digest identical to the pre-mask form, so those datasets never
    see a spurious rebuild. Required rather than defaulted: a call site that
    forgot the mask hash would compute maskless provenance *silently* --
    exactly the staleness bug this tag exists to catch.
    """
    digest = (
        geometry_hash
        if nodata_mask_hash is None
        else f'{geometry_hash}+{nodata_mask_hash}'
    )
    return versioned_hash(AOI_RASTER_FORMAT_VERSION, digest)


def write_aoi_raster(
    path: Path,
    geometry: Geometry,
    grid: TiledAffineGrid,
    geometry_hash: str,
    *,
    cell_area: float | None,
    nodata_mask: Path | None = None,
    nodata_mask_hash: str | None = None,
) -> None:
    """Burn ``geometry`` to a per-pixel cell-area AOI COG over its tile-bbox window.

    Each pixel whose centre falls inside the basin gets the area (m^2) it rasterizes
    to on this grid; every other pixel is ``0`` (so the one raster is both membership
    signal and area weights). ``cell_area`` is the grid's constant cell area on a
    projected grid, or ``None`` on a geographic grid (per-row geodesic area is
    computed from the grid's ``base_grid``).

    ``crs``, the tile-bbox window, ``tile_size``, and ``base_grid`` all derive
    from ``grid`` -- ``geometry`` is already reprojected into the grid's CRS
    (see ``Dataset.rasterize_aoi``), and its bounds pick the tile-bbox window
    via :func:`~snowtool.snowdb.grid.bounding_tiles`.

    The stamped ``SNOWTOOL_AOI_HASH`` tag is :func:`aoi_provenance` of
    ``geometry_hash`` and ``nodata_mask_hash`` -- computed here, from the same
    ``nodata_mask`` this call actually burns, so there is no caller-kept
    sync invariant to maintain. Geometry and mask are the raster's only
    provenance axes: the cell areas are a pure function of the fixed grid, and
    elevation/terrain are read live at query time, so a terrain rebuild never
    invalidates an AOI raster.

    ``nodata_mask``/``nodata_mask_hash`` are the dataset's optional valid-domain
    raster and its file digest (see ``DatasetConfig.nodata_mask``,
    ``Dataset.nodata_mask_hash``): the mask's 0/nodata pixels are burned out of
    the AOI (zero area weight). ``nodata_mask_hash`` is taken as given (not
    hashed from ``nodata_mask`` here) so a convergence loop over many pourpoints
    still hashes the mask file once, not once per AOI -- the two must agree on
    whether a mask is configured (one ``None`` and the other not is a caller
    bug, not a valid maskless call, and is rejected rather than silently
    computing maskless provenance for a masked raster).
    """
    if (nodata_mask is None) != (nodata_mask_hash is None):
        raise ValueError(
            'nodata_mask and nodata_mask_hash must both be given or both be '
            f'None (got nodata_mask={nodata_mask!r}, '
            f'nodata_mask_hash={nodata_mask_hash!r})',
        )
    start_tile, end_tile = bounding_tiles(grid, geometry.bounds)
    # Re-parsing grid.crs (rather than threading Dataset.grid_crs through) is
    # safe here: DatasetSpec.crs is the single source both grid.crs and
    # Dataset.grid_crs are derived from, so the two parses can never disagree.
    crs = rasterio.crs.CRS.from_user_input(grid.crs)
    base_grid = grid.base_grid
    tile_size = grid.tile_rows

    start = tile_base_origin(start_tile)
    end_origin = tile_base_origin(end_tile)
    end_row = end_origin.row + end_tile.rows
    end_col = end_origin.col + end_tile.cols
    height = end_row - start.row
    width = end_col - start.col

    # The tile's own affine is the upper-left transform of the AOI window, at
    # base (full) resolution.
    transform = start_tile.transform

    aoi_mask = make_geometry_mask(
        geometry,
        out_shape=(height, width),
        transform=transform,
    )
    if nodata_mask is not None:
        # Pixels outside the dataset's valid domain (e.g. SNODAS open water)
        # get zero area weight: they are excluded from stats areas exactly as
        # they are excluded from the means, so band stats recombine to
        # whole-basin stats.
        aoi_mask &= _read_nodata_mask_window(
            nodata_mask,
            base_grid,
            start,
            height,
            width,
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
        AOI_HASH_TAG: aoi_provenance(geometry_hash, nodata_mask_hash),
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

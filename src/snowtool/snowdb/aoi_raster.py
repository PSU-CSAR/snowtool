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
from snowtool.snowdb import issues as issues_mod
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


@dataclass(frozen=True)
class TileWindow:
    """The tile-bbox window an AOI raster spans: its tiles, origin, and shape.

    The one geometry the write and read paths share. An AOI raster covers a
    rectangular block of whole tiles; this bundles the tiles that block contains,
    the base-grid pixel :attr:`origin` of its upper-left corner, and the window's
    pixel :attr:`height`/:attr:`width`. The window round-trips through the
    ``SNOWTOOL_TILE_BBOX`` tag (``ul_row ul_col br_row br_col``) via :attr:`tag`
    and :meth:`from_tag`; :meth:`from_corner_tiles` builds it on the write side.
    ``tiles`` is row-major from the upper-left tile to the lower-right, so
    ``tiles[0]``/``tiles[-1]`` are those two corners.
    """

    tiles: list[AffineGridTile]
    origin: PixelCoord
    height: int
    width: int

    @classmethod
    def from_corner_tiles(
        cls: type[Self],
        grid: TiledAffineGrid,
        start_tile: AffineGridTile,
        end_tile: AffineGridTile,
    ) -> Self:
        """The window spanning the inclusive tile box ``[start_tile, end_tile]``."""
        origin = tile_base_origin(start_tile)
        end_origin = tile_base_origin(end_tile)
        return cls(
            tiles=tiles_in_bbox(
                grid,
                start_tile.row,
                start_tile.col,
                end_tile.row,
                end_tile.col,
            ),
            origin=origin,
            height=end_origin.row + end_tile.rows - origin.row,
            width=end_origin.col + end_tile.cols - origin.col,
        )

    @classmethod
    def from_tag(cls: type[Self], grid: TiledAffineGrid, bbox: str) -> Self:
        """The window a ``ul_row ul_col br_row br_col`` tag string encodes."""
        ul_row, ul_col, br_row, br_col = (int(v) for v in bbox.split())
        return cls.from_corner_tiles(grid, grid[ul_row, ul_col], grid[br_row, br_col])

    @property
    def tag(self: Self) -> str:
        """The ``SNOWTOOL_TILE_BBOX`` tag string for this window (write side).

        Byte-identical to the historical hand-formatted string, so existing AOI
        rasters keep round-tripping through :meth:`from_tag`.
        """
        ul, br = self.tiles[0], self.tiles[-1]
        return f'{ul.row} {ul.col} {br.row} {br.col}'

    def place_offset(self: Self, tile: AffineGridTile) -> PixelCoord:
        """A ``tile``'s upper-left pixel offset within this window's array."""
        tile_origin = tile_base_origin(tile)
        return PixelCoord(
            row=tile_origin.row - self.origin.row,
            col=tile_origin.col - self.origin.col,
        )


def window_from_tags(
    grid: TiledAffineGrid,
    tags: dict[str, str],
) -> TileWindow:
    """Resolve an AOI raster's :class:`TileWindow` from a COG's metadata.

    AOI rasters store a ``ul_row ul_col br_row br_col`` tile bounding box in
    ``SNOWTOOL_TILE_BBOX``. Every tile in the box is read (the AOI mask nulls
    non-AOI pixels).
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

    return TileWindow.from_tag(grid, bbox)


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
    window: TileWindow

    @classmethod
    def open(
        cls: type[Self],
        path: Path,
        grid: TiledAffineGrid,
    ) -> Self:
        with rasterio.open(path) as ds:
            tags = ds.tags()
            window = window_from_tags(grid, tags)
            array: numpy.typing.NDArray[numpy.float32] = ds.read(1)

        return cls(
            path=path,
            array=array,
            window=window,
        )

    async def read_window(
        self: Self,
        raster: TiledRaster,
        *,
        dtype: numpy.typing.DTypeLike,
        fill: float | int,
        cache: TiffCache,
    ) -> numpy.typing.NDArray[Any]:
        """Read ``raster`` into a fresh AOI-window array (shape of :attr:`array`).

        Allocates a window-shaped array prefilled with ``fill`` (the caller's
        nodata sentinel), coalesces one fetch per source COG, and places each
        tile block. The tile bbox spans the whole window, so the placed blocks
        are expected to cover every pixel and the prefill never survives -- it is
        a belt-and-suspenders guard, kept here in the one loader rather than
        duplicated at each call site.
        """
        array: numpy.typing.NDArray[Any] = numpy.full(
            self.array.shape,
            fill,
            dtype=dtype,
        )
        # One coalesced fetch per source COG, then place each block.
        tiles = self.window.tiles
        blocks = await raster.load_tiles(tiles, cache)
        for tile, block in zip(tiles, blocks, strict=True):
            offset = self.window.place_offset(tile)
            array[
                offset.row : offset.row + tile.rows,
                offset.col : offset.col + tile.cols,
            ] = block
        return array


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
    see a spurious rebuild.
    """
    digest = (
        geometry_hash
        if nodata_mask_hash is None
        else f'{geometry_hash}+{nodata_mask_hash}'
    )
    return versioned_hash(AOI_RASTER_FORMAT_VERSION, digest)


def aoi_raster_issues(
    path: Path,
    *,
    grid: TiledAffineGrid,
    expected_hash: str | None,
    check_empty: bool = True,
) -> list[issues_mod.Issue]:
    """The health of the AOI raster at ``path``: freshness + structure + grid.

    The single source of truth for "is this AOI raster OK?", shared by the
    write path (skip-unless-issues) and ``doctor`` (report). Reads the raster's
    *own* stored transform, shape, and tags from the file (a missing
    ``SNOWTOOL_TILE_BBOX`` tag or any other read failure short-circuits with the
    corresponding issue), then checks each of the following independently and
    accumulates the results:

    - an all-zero array (:class:`~snowtool.snowdb.issues.EmptyArtifact`) -- the
      *only* check that decodes the band; gate it off with ``check_empty=False``
      when the caller does not need it. The writer does exactly that: emptiness
      is non-actionable (rebuilding an empty-but-current raster reproduces it),
      so ``aoi_raster_is_current`` skips the decode and reads the header alone;
    - grid + structure: the raster's own on-disk transform and shape against
      what ``grid`` expects for the raster's stored tile-bbox window -- the
      upper-left tile transform and the window's height/width (via
      :func:`~snowtool.snowdb.issues.grid_issues`). Reading the transform/shape
      from the *file* (not re-deriving them from ``grid``) is what makes this
      catch a raster burned on an old grid that has since moved: a shifted grid
      yields a different expected UL-tile transform than the one on disk. A
      truncated/corrupt file yields a shape that disagrees with its window.
    - freshness: when ``expected_hash`` is given, the stored ``SNOWTOOL_AOI_HASH``
      tag against it (:class:`~snowtool.snowdb.issues.ContentStale`).
      ``expected_hash=None`` means the caller has no record to compare against
      (an orphan raster), so freshness is skipped entirely -- that case is
      reported elsewhere.

    Empty result means healthy and current.
    """
    try:
        with rasterio.open(path) as ds:
            actual_transform = ds.transform
            actual_shape = (ds.height, ds.width)
            tags = ds.tags()
            # The band decode is only needed for the emptiness check; everything
            # else is header-only, so skip it when the caller opted out.
            array = ds.read(1) if check_empty else None
        # Resolves the stored tile-bbox against the *current* grid: the UL-tile
        # transform + window shape it yields are what an on-grid raster must
        # match. Raises IncompleteDatasetDataError when the tag is absent.
        window = window_from_tags(grid, tags)
    except IncompleteDatasetDataError:
        return [issues_mod.MissingProvenanceTag(TILE_BBOX_TAG)]
    except Exception as e:  # noqa: BLE001 - a health check reports any read failure
        return [issues_mod.Unreadable(str(e))]

    result: list[issues_mod.Issue] = []

    if array is not None and not array.any():
        result.append(issues_mod.EmptyArtifact())

    result.extend(
        issues_mod.grid_issues(
            declared_transform=window.tiles[0].transform,
            actual_transform=actual_transform,
            declared_shape=(window.height, window.width),
            actual_shape=actual_shape,
        ),
    )

    if expected_hash is not None:
        stored = tags.get(AOI_HASH_TAG)
        if stored != expected_hash:
            result.append(
                issues_mod.ContentStale(stored=stored, expected=expected_hash),
            )

    return result


def write_aoi_raster(
    path: Path,
    geometry: Geometry,
    grid: TiledAffineGrid,
    geometry_hash: str,
    *,
    cell_area: float | None,
    nodata_mask: tuple[Path, str] | None = None,
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
    ``nodata_mask`` this call actually burns, so there is no caller-kept sync
    invariant to maintain (see :func:`aoi_provenance` for what invalidates it).

    ``nodata_mask`` pairs the dataset's optional valid-domain raster with its
    file digest (see ``DatasetConfig.nodata_mask``, ``Dataset.nodata_mask_hash``):
    the mask's 0/nodata pixels are burned out of the AOI (zero area weight), and
    the digest is taken as given (not re-hashed here) so a convergence loop over
    many pourpoints hashes the mask file once, not once per AOI. Passing the path
    and hash as one tuple makes the half-specified state unrepresentable.
    """
    mask_path, mask_hash = nodata_mask or (None, None)
    start_tile, end_tile = bounding_tiles(grid, geometry.bounds)
    window = TileWindow.from_corner_tiles(grid, start_tile, end_tile)
    # Re-parsing grid.crs (rather than threading Dataset.grid_crs through) is
    # safe here: DatasetSpec.crs is the single source both grid.crs and
    # Dataset.grid_crs are derived from, so the two parses can never disagree.
    crs = rasterio.crs.CRS.from_user_input(grid.crs)
    base_grid = grid.base_grid
    tile_size = grid.tile_rows

    start = window.origin
    height = window.height
    width = window.width

    # The tile's own affine is the upper-left transform of the AOI window, at
    # base (full) resolution.
    transform = start_tile.transform

    aoi_mask = make_geometry_mask(
        geometry,
        out_shape=(height, width),
        transform=transform,
    )
    if mask_path is not None:
        # Pixels outside the dataset's valid domain (e.g. SNODAS open water)
        # get zero area weight: they are excluded from stats areas exactly as
        # they are excluded from the means, so band stats recombine to
        # whole-basin stats.
        aoi_mask &= _read_nodata_mask_window(
            mask_path,
            base_grid,
            start,
            height,
            width,
        )
    areas = _window_cell_areas(base_grid, start.row, height, width, cell_area)
    aoi_area = numpy.where(aoi_mask, areas, numpy.float32(0)).astype(numpy.float32)

    tags = {
        TILE_BBOX_TAG: window.tag,
        # Records the geometry + format version this raster was burned from, so a
        # changed basin OR a format bump is detected (and re-rasterized) by a cheap
        # tag read.
        AOI_HASH_TAG: aoi_provenance(geometry_hash, mask_hash),
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

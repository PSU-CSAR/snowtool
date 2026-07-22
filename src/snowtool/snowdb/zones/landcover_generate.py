"""Generate a dataset's land-cover set from a fine-resolution NLCD raster.

One streaming pass over a single NLCD land-cover source produces a co-registered
percent-forest-cover layer for *every* target grid at once. Passing one target
degenerates to per-dataset generation; passing several shares the source read.

Unlike :mod:`snowtool.snowdb.zones.terrain_generate`, land cover is categorical, so
there is no slope/aspect, no Horn neighbourhood (hence no block halo), and no
projected work grid -- aspect needs an undistorted metric grid, but a *fraction of
pixels* does not. What the two engines do share -- the pre-flight existence guard,
the generation-digest-then-stamp pass, and the point-in-cell binning arithmetic
(cell assignment, pixel-centre coordinates) -- lives in
:mod:`snowtool.snowdb.zones.generate_common`. Each fine source pixel's centre is
transformed (pyproj) into the target grid's CRS and binned into the cell it lands
in (point-in-cell), so a source in any CRS/resolution bins correctly onto any grid.

Per target cell the engine accumulates a count of valid NLCD pixels and of forest
pixels (classes in :data:`~snowtool.snowdb.constants.FOREST_CLASSES`); the layer
value is ``round(100 * forest / valid)`` as ``uint8``, nodata ``255`` where a cell
caught no valid pixels. The source is read only over the combined extent of the
target grids, not the whole national raster.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Self

import numpy
import numpy.typing
import rasterio

from rasterio.warp import transform_bounds
from rasterio.windows import Window

from snowtool.snowdb.constants import (
    FOREST_CLASSES,
    FOREST_PCT_NODATA,
    NLCD_HASH_TAG,
)
from snowtool.snowdb.grid import grid_extent_4326
from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.zones.generate_common import (
    BinAccumulator,
    Block,
    Loaded,
    StreamingBinner,
    cells_for_points,
    finalize_and_stamp,
    iter_blocks,
    pixel_centre_coords,
    require_absent_layers,
)
from snowtool.snowdb.zones.landcover_layers import (
    FOREST_COVER,
    LANDCOVER_FORMAT_VERSION,
    LANDCOVER_LAYERS,
)
from snowtool.snowdb.zones.parallel import (
    CancelToken,
    effective_workers,
)

if TYPE_CHECKING:
    from affine import Affine

    from snowtool.snowdb.zones.zone_layer import ZoneLayer, ZoneLayerTarget

BLOCK = 2048


class _ForestAccumulator(BinAccumulator):
    """Per-cell forest/valid pixel counts for one target grid.

    Each valid fine NLCD pixel is binned into its grid cell; the per-cell forest
    and valid counts give the percent-forest value. Exact across source block
    boundaries because every fine pixel is placed independently. The
    target/grid/CRS prologue lives on
    :class:`~snowtool.snowdb.zones.generate_common.BinAccumulator`.
    """

    def __init__(self: Self, target: ZoneLayerTarget) -> None:
        super().__init__(target)
        n = self._ncell
        self.forest = numpy.zeros(n, dtype=numpy.int64)
        self.valid = numpy.zeros(n, dtype=numpy.int64)

    def bin_into(
        self: Self,
        xt: numpy.typing.NDArray[numpy.float64],
        yt: numpy.typing.NDArray[numpy.float64],
        *payload: numpy.typing.NDArray,
    ) -> None:
        """Bin already-reprojected valid source pixels into cells.

        ``payload`` is ``(is_forest,)`` (the tuple the streamer splats). See
        :meth:`BinAccumulator.bin_into` for the serial-order contract.
        """
        (is_forest,) = payload
        cell_all, inb = cells_for_points(self._inv, xt, yt, self.width, self.height)
        if not inb.any():
            return
        cell = cell_all[inb]
        n = self._ncell
        self.valid += numpy.bincount(cell, minlength=n)
        forest_cells = cell[is_forest[inb]]
        if forest_cells.size:
            self.forest += numpy.bincount(forest_cells, minlength=n)

    def finalize(self: Self) -> list[tuple[ZoneLayer, numpy.typing.NDArray]]:
        """Reduce the counts to the percent-forest layer (nodata where no pixels)."""
        h, w = self.height, self.width
        valid = self.valid.reshape(h, w)
        forest = self.forest.reshape(h, w)
        out = numpy.full((h, w), FOREST_PCT_NODATA, dtype=numpy.uint8)
        has = valid > 0
        # forest <= valid, so the rounded percentage is always in 0..100.
        out[has] = numpy.rint(100.0 * forest[has] / valid[has]).astype(numpy.uint8)
        return [(FOREST_COVER, out)]


def generate_landcover(
    source: rasterio.io.DatasetReader,
    targets: list[ZoneLayerTarget],
    *,
    workers: int | None = None,
    block_size: int | None = None,
    force: bool = False,
    progress: ProgressReporter = NULL_PROGRESS,
) -> dict[str, str]:
    """Stream ``source`` once, binning percent forest cover into every target grid.

    ``source`` is an opened NLCD land-cover raster (any CRS/resolution; natively
    EPSG:5070 30 m). Only the window over the targets' combined extent is read.
    ``workers`` of ``None`` uses one thread per CPU (capped at
    :data:`~snowtool.snowdb.zones.parallel.MAX_AUTO_WORKERS`), ``1`` forces the
    serial path; ``block_size`` (``None`` -> :data:`BLOCK`) is the per-worker
    memory lever. The result -- including the generation hash -- is independent of
    ``workers``. ``progress`` reports the per-block binning. Returns the single
    generation hash keyed by each target name (every value is equal -- one
    identifier for the whole pass). Refuses to overwrite an existing land-cover
    set unless ``force``.
    """
    if not targets:
        return {}

    n_workers = effective_workers(workers)
    bs = block_size if block_size is not None else BLOCK

    if not force:
        # Check every target before the (potentially large) source read.
        require_absent_layers(targets, LANDCOVER_LAYERS, 'land cover')

    source_crs = source.crs.to_wkt()
    accumulators = [_ForestAccumulator(target) for target in targets]

    # NLCD uses 0 for unclassified/background; treat it (and the file's declared
    # nodata) as invalid so empty cells read as nodata rather than 0% forest.
    forest = numpy.asarray(FOREST_CLASSES)
    src_nodata = source.nodata

    window = _source_window(source, targets)
    if window is not None:
        _LandCoverStreamer(
            source,
            source.transform,
            window,
            forest,
            src_nodata,
            accumulators,
            source_crs,
            bs,
        ).run(n_workers, progress)

    # One generation id for the whole pass: a digest over every target's finalized
    # forest array (its only finalized layer, sorted by name for determinism),
    # stamped identically on every output -- so all layers produced together
    # reconcile as one set.
    return finalize_and_stamp(
        accumulators,
        format_version=LANDCOVER_FORMAT_VERSION,
        hash_tag=NLCD_HASH_TAG,
    )


def _source_window(
    source: rasterio.io.DatasetReader,
    targets: list[ZoneLayerTarget],
) -> Window | None:
    """The source-pixel window covering every target grid's extent, or ``None``.

    The targets' combined extent (in EPSG:4326) is projected into the source CRS
    and clipped to the source, so a national NLCD raster is read only where the
    grids actually need it. ``None`` means no target overlaps the source.
    """
    west = south = math.inf
    east = north = -math.inf
    for target in targets:
        w, s, e, n = grid_extent_4326(target.grid)
        west, south = min(west, w), min(south, s)
        east, north = max(east, e), max(north, n)

    left, bottom, right, top = transform_bounds(
        'EPSG:4326',
        source.crs,
        west,
        south,
        east,
        north,
    )
    full = source.window(left, bottom, right, top)
    clipped = (
        full.intersection(
            Window(0, 0, source.width, source.height),
        )
        .round_offsets()
        .round_lengths()
    )
    if clipped.width <= 0 or clipped.height <= 0:
        return None
    return clipped


class _LandCoverStreamer(StreamingBinner[_ForestAccumulator]):
    """Streams one NLCD window in blocks, binning into every target accumulator.

    The concurrency scaffold (read lock, per-thread Transformers, cancel dance,
    coordinate epilogue, serial reduce, ``ordered_parallel_map`` wiring) lives on
    :class:`~snowtool.snowdb.zones.generate_common.StreamingBinner`. This engine
    supplies only the land-cover-specific ``_load``: the windowed source read (the
    source window offsets its block enumeration) and the forest/valid mask. Its
    payload is ``(is_forest,)`` per :meth:`_ForestAccumulator.bin_into`.
    """

    _label = 'binning land cover'

    def __init__(
        self: Self,
        source: rasterio.io.DatasetReader,
        transform: Affine,
        window: Window,
        forest_classes: numpy.typing.NDArray,
        src_nodata: float | None,
        accumulators: list[_ForestAccumulator],
        source_crs: str,
        block_size: int,
    ) -> None:
        super().__init__(source_crs, accumulators)
        self._source = source
        self._transform = transform
        self._window = window
        self._forest = forest_classes
        self._src_nodata = src_nodata
        self._block_size = block_size

    def _blocks(self: Self) -> list[Block]:
        col_off, row_off = int(self._window.col_off), int(self._window.row_off)
        win_w, win_h = int(self._window.width), int(self._window.height)
        return iter_blocks(
            win_w,
            win_h,
            self._block_size,
            col_off=col_off,
            row_off=row_off,
        )

    def _load(self: Self, block: Block, cancel: CancelToken) -> Loaded:
        """Read one NLCD block, then mask forest/valid pixels.

        ``None`` for an all-invalid block or a read the cancel short-circuited.
        """
        values = self._locked_read(
            cancel,
            lambda: self._source.read(
                1,
                window=Window(block.c0, block.r0, block.bw, block.bh),
            ),
        )
        if values is None:
            return None
        valid = values != 0
        if self._src_nodata is not None:
            valid &= values != self._src_nodata
        if not valid.any():
            return None
        is_forest = numpy.isin(values, self._forest) & valid

        # Absolute source-pixel centres -> source-CRS coordinates. NLCD is
        # north-up (b == d == 0), but the full affine form costs nothing and
        # stays correct for any source.
        x, y = pixel_centre_coords(
            self._transform,
            block.r0,
            block.c0,
            block.bh,
            block.bw,
        )

        keep = valid.ravel()
        return (is_forest.ravel()[keep],), x, y, valid

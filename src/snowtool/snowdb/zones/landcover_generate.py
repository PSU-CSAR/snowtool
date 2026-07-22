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
import threading

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy
import numpy.typing
import rasterio

from pyproj import Transformer
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
    Block,
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
    ordered_parallel_map,
)

if TYPE_CHECKING:
    from affine import Affine

    from snowtool.snowdb.zones.zone_layer import ZoneLayer, ZoneLayerTarget

BLOCK = 2048


class _ForestAccumulator:
    """Per-cell forest/valid pixel counts for one target grid.

    Each valid fine NLCD pixel is binned into its grid cell; the per-cell forest
    and valid counts give the percent-forest value. Exact across source block
    boundaries because every fine pixel is placed independently.
    """

    def __init__(self: Self, target: ZoneLayerTarget) -> None:
        self.target = target
        self.height = target.rows
        self.width = target.cols
        self.transform = target.transform
        # The streamer builds (thread-local) source-CRS -> this-CRS Transformers.
        self.crs = target.crs
        n = self.height * self.width
        self.forest = numpy.zeros(n, dtype=numpy.int64)
        self.valid = numpy.zeros(n, dtype=numpy.int64)
        self._inv = ~self.transform

    @property
    def _ncell(self: Self) -> int:
        return self.height * self.width

    def bin_into(
        self: Self,
        xt: numpy.typing.NDArray[numpy.float64],
        yt: numpy.typing.NDArray[numpy.float64],
        is_forest: numpy.typing.NDArray[numpy.bool_],
    ) -> None:
        """Bin already-reprojected (this grid's CRS) valid source pixels into cells.

        Coordinate transform (source CRS -> this grid's CRS) happens in the
        worker, so this runs serially on the main thread in fixed block order --
        the count accumulation order is identical to the serial pass, keeping the
        generation hash reproducible regardless of worker count.
        """
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


@dataclass(frozen=True)
class _BlockResult:
    """One block's binnable contribution: forest mask + per-target coords.

    ``coords`` holds, per target (same order as the streamer's accumulators), the
    valid fine-pixel centres already reprojected into that target's CRS.
    Everything is flattened and pre-masked to the valid pixels, so the serial
    reducer only has to bin. ``is_forest`` is aligned with those kept pixels.
    """

    is_forest: numpy.typing.NDArray[numpy.bool_]
    coords: list[
        tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
    ]


class _LandCoverStreamer:
    """Streams one NLCD window in blocks, binning into every target accumulator.

    The expensive per-block work -- the per-target pyproj reprojection -- is pure
    and runs on a worker pool. The only shared mutable state is the accumulators,
    so binning is done serially on the main thread in streaming block order; that
    keeps count accumulation order independent of worker count, so the generation
    hash is reproducible (parallel == serial bit for bit). Each worker thread gets
    its own Transformers (a Transformer is not safe to share concurrently).

    The block reads run under a lock: a GDAL dataset is not safe for concurrent
    reads. The read is a small fraction of the per-block cost (the reprojection
    dominates and still runs fully in parallel), so serialising it costs little.

    The parallel-map / serial-reduce machinery -- the sliding window, the
    warm-gate, and the ctrl+c-proof teardown that guarantees no worker is left
    inside the source dataset the caller closes on return -- lives in
    :mod:`snowtool.snowdb.zones.parallel`; ``run`` just wires
    ``_compute``/``_reduce`` into it.
    """

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
        self._source = source
        self._transform = transform
        self._window = window
        self._forest = forest_classes
        self._src_nodata = src_nodata
        self._accumulators = accumulators
        self._source_crs = source_crs
        self._block_size = block_size
        self._local = threading.local()
        # GDAL reads are not concurrency-safe; serialise just the read.
        self._read_lock = threading.Lock()

    def _transformers(self: Self) -> list[Transformer]:
        """Per-thread source-CRS -> target-CRS Transformers (built once per thread)."""
        tfs: list[Transformer] | None = getattr(self._local, 'tfs', None)
        if tfs is None:
            tfs = [
                Transformer.from_crs(self._source_crs, acc.crs, always_xy=True)
                for acc in self._accumulators
            ]
            self._local.tfs = tfs
        return tfs

    def _compute(self: Self, block: Block, cancel: CancelToken) -> _BlockResult | None:
        """Worker step: read, mask, reproject. Pure, no shared writes.

        ``None`` means "no contribution" to the reducer -- true for an all-invalid
        block and for a block abandoned because the run is being cancelled.
        """
        # Bail before queueing on the read lock: once the run is aborting, blocks
        # waiting their turn must not each still pay a read. The re-check under
        # the lock closes the race for a worker that passed the first check just
        # before cancellation and acquired the lock just after it.
        if cancel.cancelled:
            return None
        with self._read_lock:
            if cancel.cancelled:
                return None
            values = self._source.read(
                1,
                window=Window(block.c0, block.r0, block.bw, block.bh),
            )

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
        xf = numpy.broadcast_to(x, values.shape).ravel()[keep]
        yf = numpy.broadcast_to(y, values.shape).ravel()[keep]
        ff = is_forest.ravel()[keep]
        coords = [tf.transform(xf, yf) for tf in self._transformers()]
        return _BlockResult(is_forest=ff, coords=coords)

    def _reduce(self: Self, result: _BlockResult) -> None:
        """Main-thread step: bin one block into every accumulator (serial, ordered)."""
        for acc, (xt, yt) in zip(self._accumulators, result.coords, strict=True):
            acc.bin_into(xt, yt, result.is_forest)

    def run(
        self: Self,
        workers: int,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> None:
        """Stream every block through the ordered parallel-map engine.

        The engine maps ``_compute`` (read + reproject; pure) over the blocks on
        ``workers`` threads and folds each result through ``_reduce`` serially, in
        block order, on this thread -- so the binning (and the generation hash) is
        reproducible regardless of worker count, and no worker survives inside the
        source dataset after this returns. See
        :mod:`snowtool.snowdb.zones.parallel`.
        """
        col_off, row_off = int(self._window.col_off), int(self._window.row_off)
        win_w, win_h = int(self._window.width), int(self._window.height)
        ordered_parallel_map(
            iter_blocks(
                win_w,
                win_h,
                self._block_size,
                col_off=col_off,
                row_off=row_off,
            ),
            self._compute,
            self._reduce,
            workers=workers,
            progress=progress,
            label='binning land cover',
        )

"""Generate a dataset's terrain set from a fine-resolution DEM.

One streaming pass over a single DEM source produces co-registered terrain for
*every* target grid at once: elevation, majority aspect class, the mean orientation
components (northness/eastness = mean cos/sin of aspect), and the normalised
aspect-direction entropy -- the layers of a
:class:`~snowtool.snowdb.zones.terrain.TerrainSet`. Passing one target degenerates to
per-dataset generation; passing several shares the expensive source read.

The three stages
----------------
1. **Resample to a fine, projected work grid (once).** The source is lazily
   reprojected (bilinear, via :class:`rasterio.vrt.WarpedVRT`) into one common
   lattice: a projected CRS at a fine resolution (the source's ``work_crs`` /
   ``work_resolution``; defaults
   :data:`~snowtool.snowdb.zones.terrain_layers.DEFAULT_WORK_CRS` = CONUS Albers,
   :data:`~snowtool.snowdb.zones.terrain_layers.DEFAULT_WORK_RESOLUTION` = 10 m,
   matching 3DEP). This shared intermediate is the only true resample, and it is of
   elevation.
2. **Derive everything at the work resolution.** Streaming in blocks (with a
   one-pixel halo so the 3x3 Horn window is exact across block edges), each fine
   pixel gets a slope, an aspect, an aspect class (N/E/S/W or flat), and
   cos/sin(aspect).
3. **Aggregate fine pixels into target cells.** Each fine pixel's centre is
   transformed (pyproj) into the target grid's CRS and assigned to the cell it lands
   in (point-in-cell, not fractional-area). Per cell the engine accumulates class
   counts (-> majority), an elevation sum (-> mean elevation), and cos/sin sums over
   non-flat pixels (-> mean northness/eastness).

Why a projected CRS
-------------------
Slope and aspect are gradients (dz/dx, dz/dy). A geographic source (e.g. 3DEP 1/3",
EPSG:4326) has pixels measured in degrees that are non-square on the ground and
drift with latitude, so gradients -- hence aspect -- come out distorted. A
projected, metric, near-square CRS (CONUS Albers) makes dx == dy in real metres so
aspect is geometrically valid. This reprojection is required for aspect, not for
elevation.

Elevation rides the same shared work surface (source -> bilinear work grid -> mean
of the fine pixels per cell) rather than a direct ``average`` warp. That costs a
small, sub-metre error (bilinear is a mild low-pass; point-in-cell binning isn't
area-weighted) accepted in exchange for co-registration: every layer comes from the
one surface and the same binning, so elevation and aspect line up by construction.
Because each fine pixel has equal area in the projected CRS, the per-cell mean is an
area-mean, so ``average`` semantics survive for elevation.

``work_resolution`` should match the source's native ground resolution (too fine
invents detail, too coarse discards it), so it is a property of the ``DemSource``
(3DEP pins 10 m); the ``DEFAULT_WORK_*`` constants in ``terrain_layers`` are only
fallbacks.

Parallelism and memory
----------------------
This is an input-driven scatter: the work grid is streamed in blocks and each fine
pixel is dropped (``+=``) into whatever target cell it lands in, sharing the
expensive target-independent work (read, warp, Horn, reproject) across all targets.
Blocks run on a thread pool (the pyproj transform dominates and releases the GIL);
binning stays serial and in block order, so the output -- including the generation
hash -- is identical regardless of worker count. Memory is the fixed target
accumulators (~72 bytes/cell, sized by the target grids) plus transient per-worker
block buffers (~``workers * block_size**2``); ``workers`` and ``block_size`` bound
the latter. The design assumes the targets fit in RAM.
"""

from __future__ import annotations

import math
import warnings

from typing import TYPE_CHECKING, Self

import numpy
import numpy.typing
import rasterio

from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform, transform_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from snowtool.exceptions import SnowtoolWarning
from snowtool.snowdb.constants import DEM_HASH_TAG
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
from snowtool.snowdb.zones.parallel import (
    CancelToken,
    effective_workers,
)
from snowtool.snowdb.zones.terrain_layers import (
    ASPECT_COMPONENT_NODATA,
    ASPECT_E,
    ASPECT_ENTROPY,
    ASPECT_ENTROPY_NODATA,
    ASPECT_FLAT,
    ASPECT_MAJORITY,
    ASPECT_MAJORITY_NODATA,
    ASPECT_N,
    ASPECT_S,
    ASPECT_W,
    EASTNESS,
    ELEVATION,
    ELEVATION_NODATA,
    NORTHNESS,
    TERRAIN_FORMAT_VERSION,
    TERRAIN_LAYERS,
)

if TYPE_CHECKING:
    from affine import Affine

    from snowtool.snowdb.grid import Bounds, Extent
    from snowtool.snowdb.zones.zone_layer import (
        ZoneLayer,
        ZoneLayerSource,
        ZoneLayerTarget,
    )

# Default work-grid block edge (pixels). Block size is a non-lever for throughput
# (the pyproj transform is flat per-pixel; only the ~1-pixel halo re-read and the
# Python loop count scale with it, both negligible), but it *is* the per-worker
# memory lever: a worker holds ~a dozen block-sized float64 arrays while processing
# one block, so transient RAM scales with ``workers * block_size**2``. 1024 keeps
# that to ~2 GB at the default worker cap while staying coarse enough to avoid
# excessive block overhead; shrink it (e.g. 512) to run more workers on less RAM.
# Exposed per call via ``generate_terrain(block_size=...)``.
DEFAULT_BLOCK_SIZE = 1024
# One-pixel halo so the 3x3 Horn window is exact across block boundaries: the
# Horn pass trims exactly one pixel off each edge, so the trimmed inner block
# lines up with the nominal block with no overlap (a larger halo would make
# adjacent blocks' inner regions overlap and double-count pixels).
HALO = 1
# Below this slope a pixel's aspect is unreliable -> the FLAT class.
FLAT_SLOPE_DEG = 2.0

# Internal accumulator layout: four cardinal classes plus flat.
_N_CLASSES = 5


def _slope_aspect(
    z: numpy.typing.NDArray[numpy.float64],
    px: float,
    py: float,
) -> tuple[
    numpy.typing.NDArray[numpy.float64],
    numpy.typing.NDArray[numpy.float64],
    numpy.typing.NDArray[numpy.bool_],
    numpy.typing.NDArray[numpy.float64],
]:
    """Horn 3x3 slope (deg) and aspect (deg, 0=N clockwise) for the inner pixels.

    Returns slope, aspect, a validity mask, and the inner elevation block (all
    trimmed by the one-pixel Horn border the halo provides).
    """
    valid = numpy.isfinite(z)
    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]
    d = z[1:-1, :-2]
    f = z[1:-1, 2:]
    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * px)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * py)
    gx, gy = dz_dx, -dz_dy
    slope_deg = numpy.degrees(numpy.arctan(numpy.hypot(gx, gy)))
    aspect_deg = numpy.degrees(numpy.arctan2(-gx, -gy)) % 360.0
    vint = valid[1:-1, 1:-1] & numpy.isfinite(slope_deg) & numpy.isfinite(aspect_deg)
    return slope_deg, aspect_deg, vint, z[1:-1, 1:-1]


def _classify(
    slope_deg: numpy.typing.NDArray[numpy.float64],
    aspect_deg: numpy.typing.NDArray[numpy.float64],
    valid: numpy.typing.NDArray[numpy.bool_],
) -> numpy.typing.NDArray[numpy.int8]:
    """Per-pixel aspect class (cardinal quadrant or flat); ``-1`` where invalid."""
    cls = numpy.full(aspect_deg.shape, -1, dtype=numpy.int8)
    a = aspect_deg
    cls[(a >= 315) | (a < 45)] = ASPECT_N
    cls[(a >= 45) & (a < 135)] = ASPECT_E
    cls[(a >= 135) & (a < 225)] = ASPECT_S
    cls[(a >= 225) & (a < 315)] = ASPECT_W
    cls[slope_deg < FLAT_SLOPE_DEG] = ASPECT_FLAT
    cls[~valid] = -1
    return cls


class _GridAccumulator(BinAccumulator):
    """Per-cell terrain accumulators for one target grid.

    Each fine source pixel is binned into its grid cell; the cardinal counts give
    the majority, the cos/sin sums give the orientation mean, and the elevation
    sum gives mean elevation. Exact across source block boundaries because every
    fine pixel is placed independently. The target/grid/CRS prologue lives on
    :class:`~snowtool.snowdb.zones.generate_common.BinAccumulator`.
    """

    def __init__(self: Self, target: ZoneLayerTarget) -> None:
        super().__init__(target)
        n = self._ncell
        self.counts = numpy.zeros(_N_CLASSES * n, dtype=numpy.int64)
        self.sum_cos = numpy.zeros(n, dtype=numpy.float64)
        self.sum_sin = numpy.zeros(n, dtype=numpy.float64)
        self.n_nonflat = numpy.zeros(n, dtype=numpy.int64)
        self.sum_z = numpy.zeros(n, dtype=numpy.float64)

    def bin_into(
        self: Self,
        xt: numpy.typing.NDArray[numpy.float64],
        yt: numpy.typing.NDArray[numpy.float64],
        *payload: numpy.typing.NDArray,
    ) -> None:
        """Bin already-reprojected fine-pixel centres into cells.

        ``payload`` is ``(cls, cos, sin, z)`` (the tuple the streamer splats).
        See :meth:`BinAccumulator.bin_into` for the serial-order contract.
        """
        cls, cos, sin, z = payload
        cell_all, inb = cells_for_points(self._inv, xt, yt, self.width, self.height)
        if not inb.any():
            return
        cell = cell_all[inb]
        c = cls[inb]
        nc = self._ncell
        self.counts += numpy.bincount(c * nc + cell, minlength=_N_CLASSES * nc)
        self.sum_z += numpy.bincount(cell, weights=z[inb], minlength=nc)
        nf = c != ASPECT_FLAT  # cos/sin only where aspect is defined
        if nf.any():
            cf = cell[nf]
            self.sum_cos += numpy.bincount(cf, weights=cos[inb][nf], minlength=nc)
            self.sum_sin += numpy.bincount(cf, weights=sin[inb][nf], minlength=nc)
            self.n_nonflat += numpy.bincount(cf, minlength=nc)

    def finalize(
        self: Self,
    ) -> list[tuple[ZoneLayer, numpy.typing.NDArray]]:
        """Reduce the accumulators to each terrain layer, paired with its array.

        Order matches :data:`~snowtool.snowdb.zones.terrain.TERRAIN_LAYERS`
        (elevation, aspect majority, northness, eastness, aspect entropy).
        """
        h, w = self.height, self.width
        counts = self.counts.reshape(_N_CLASSES, h, w)
        total = counts.sum(axis=0)

        majority = counts.argmax(axis=0).astype(numpy.uint8)
        majority[total == 0] = ASPECT_MAJORITY_NODATA

        nf = self.n_nonflat.reshape(h, w)
        with numpy.errstate(invalid='ignore', divide='ignore'):
            # A cell with no non-flat pixels has no defined orientation: mark it
            # with the finite ASPECT_COMPONENT_NODATA sentinel so it digitises
            # cleanly out of the banded northness/eastness schemes.
            northness = numpy.where(
                nf > 0,
                self.sum_cos.reshape(h, w) / nf,
                ASPECT_COMPONENT_NODATA,
            )
            eastness = numpy.where(
                nf > 0,
                self.sum_sin.reshape(h, w) / nf,
                ASPECT_COMPONENT_NODATA,
            )
            elevation = numpy.where(
                total > 0,
                self.sum_z.reshape(h, w) / total,
                ELEVATION_NODATA,
            )
            # Shannon entropy of the 5-class aspect distribution (incl. flat),
            # normalised by ln(5) to [0, 1]; 0*ln(0)=0. Read crossed with the
            # majority, so a flat-dominated cell scores low entropy *and* majority
            # flat -- the flat case stays owned by the majority class.
            p = counts / total
            plogp = numpy.where(p > 0, p * numpy.log(p), 0.0)
            entropy = numpy.where(
                total > 0,
                -plogp.sum(axis=0) / numpy.log(_N_CLASSES),
                ASPECT_ENTROPY_NODATA,
            )

        return [
            (ELEVATION, elevation.astype(numpy.float32)),
            (ASPECT_MAJORITY, majority),
            (NORTHNESS, northness.astype(numpy.float32)),
            (EASTNESS, eastness.astype(numpy.float32)),
            (ASPECT_ENTROPY, entropy.astype(numpy.float32)),
        ]


def _target_bounds_in_work_crs(
    targets: list[ZoneLayerTarget],
    work_crs: str,
) -> Extent:
    """Union of the target grids' extents, expressed in the work CRS.

    The streaming pass only needs to cover where the targets are; this bbox is what
    the work grid is clipped to. It is an axis-aligned bbox of the reprojected target
    edges, densified by :func:`transform_bounds`.

    Known bound: for a target whose CRS curves strongly against the work CRS (e.g.
    MODIS Sinusoidal far from its central meridian), an edge can bulge between densify
    points, under-covering the grid edge by up to the densification error (tens to
    low-hundreds of metres) so the outermost ring of target cells is fed by slightly
    fewer source pixels. Harmless when the clip is a no-op (target >= source, as for
    the continental instarr grid); it only bites a target both
    smaller-than-or-comparable-to the source and strongly curved. Harden via
    ``densify_pts`` and a wider clip margin if needed.
    """
    wests: list[float] = []
    souths: list[float] = []
    easts: list[float] = []
    norths: list[float] = []
    for target in targets:
        t = target.transform
        xmin, ymax = t.c, t.f
        xmax = t.c + target.cols * t.a
        ymin = t.f + target.rows * t.e
        rio_crs = rasterio.crs.CRS.from_wkt(target.crs.to_wkt())
        w, s, e, n = transform_bounds(rio_crs, work_crs, xmin, ymin, xmax, ymax)
        wests.append(w)
        souths.append(s)
        easts.append(e)
        norths.append(n)
    # generate_terrain guards against empty targets, so these are non-empty.
    return min(wests), min(souths), max(easts), max(norths)


def _clip_grid_to_bounds(
    full_transform: Affine,
    full_w: int,
    full_h: int,
    bounds: Extent,
) -> tuple[Affine, int, int] | None:
    """Sub-window of the full work grid covering ``bounds`` (+ halo), same lattice.

    Returns ``(transform, width, height)`` clipped to the work grid, or ``None`` if
    ``bounds`` don't overlap it at all. Because the window snaps outward to whole
    pixels of the *existing* lattice (never rephased), every cell that falls in a
    target is sampled exactly as it would be over the full source -- only empty
    margin is dropped -- so the result and generation hash are unchanged.
    """
    inv = ~full_transform
    west, south, east, north = bounds
    cols, rows = [], []
    for x, y in ((west, north), (east, south), (west, south), (east, north)):
        c, r = inv * (x, y)
        cols.append(c)
        rows.append(r)
    # One extra pixel beyond the Horn halo, as slack against edge rounding. This
    # does NOT fully cover the curved-CRS bbox under-coverage documented on
    # _target_bounds_in_work_crs -- raise it (and densify_pts there) if a small,
    # strongly-curved target ever needs it.
    margin = HALO + 1
    col0 = max(0, math.floor(min(cols)) - margin)
    row0 = max(0, math.floor(min(rows)) - margin)
    col1 = min(full_w, math.ceil(max(cols)) + margin)
    row1 = min(full_h, math.ceil(max(rows)) + margin)
    if col1 <= col0 or row1 <= row0:
        return None
    width, height = col1 - col0, row1 - row0
    transform = window_transform(Window(col0, row0, width, height), full_transform)
    return transform, width, height


def generate_terrain(
    source: ZoneLayerSource,
    targets: list[ZoneLayerTarget],
    bounds: Bounds,
    *,
    workers: int | None = None,
    block_size: int | None = None,
    force: bool = False,
    progress: ProgressReporter = NULL_PROGRESS,
) -> dict[str, str]:
    """Stream ``source`` once, binning terrain into every target grid.

    ``source`` is a :class:`~snowtool.snowdb.zones.terrain_source.DemSource`; the
    engine opens it over ``bounds`` (``(west, south, east, north)`` in EPSG:4326)
    and lazily reprojects the DEM mosaic (any CRS/resolution) to the work grid
    (``source.work_crs`` at ``source.work_resolution`` metres). A
    ``work_resolution`` of ``None`` lets GDAL pick it from the source (right for an
    unknown local DEM); 3DEP pins 10 m. ``workers`` of ``None`` uses one thread per
    CPU (capped at :data:`~snowtool.snowdb.zones.parallel.MAX_AUTO_WORKERS`), ``1``
    forces the serial path; ``block_size`` (``None`` -> :data:`DEFAULT_BLOCK_SIZE`)
    is the per-worker memory lever (see the module docstring). The result --
    including the generation hash -- is independent of ``workers``. ``progress``
    reports the per-block reprojection. Returns the one generation hash keyed by
    each target name (all values equal). Refuses to overwrite an existing terrain
    set unless ``force``.
    """
    from snowtool.snowdb.zones.terrain_source import DemSource

    if not isinstance(source, DemSource):
        raise TypeError(f'terrain generation needs a DemSource, got {source!r}')
    if not targets:
        return {}

    work_crs = source.work_crs
    work_resolution = source.work_resolution
    n_workers = effective_workers(workers)
    bs = block_size if block_size is not None else DEFAULT_BLOCK_SIZE

    if not force:
        # Check every target before the (expensive) source read.
        require_absent_layers(targets, TERRAIN_LAYERS, 'terrain')

    accumulators = [_GridAccumulator(target) for target in targets]

    with source.open(bounds, progress=progress) as src:
        # Mask source fill using the source's own declared nodata. If it declares
        # none, trust it (mask nothing) -- but warn, since an undeclared fill value
        # would otherwise be aggregated as real elevation.
        src_nodata = src.nodata
        if src_nodata is None:
            warnings.warn(
                f'Source DEM {src.name!r} declares no nodata value; treating all '
                'pixels as valid data. Declare a nodata value on the source if it '
                'has fill pixels, or they will be aggregated as real elevations.',
                SnowtoolWarning,
                stacklevel=2,
            )
        full_transform, full_w, full_h = calculate_default_transform(
            src.crs,
            work_crs,
            src.width,
            src.height,
            *src.bounds,
            # None -> GDAL derives the native resolution mapped into the work CRS.
            resolution=work_resolution,
        )
        clipped = _clip_grid_to_bounds(
            full_transform,
            full_w,
            full_h,
            _target_bounds_in_work_crs(targets, work_crs),
        )
        if clipped is not None:
            dst_transform, dst_w, dst_h = clipped
            px, py = abs(dst_transform.a), abs(dst_transform.e)
            with WarpedVRT(
                src,
                crs=work_crs,
                transform=dst_transform,
                width=dst_w,
                height=dst_h,
                resampling=Resampling.bilinear,
                src_nodata=src_nodata,
                # The streaming pass marks no-data with NaN (numpy.isfinite), so the
                # working band must be float. rasterio already promotes the band to
                # float to hold nodata=NaN (even for an integer source); we pin it
                # explicitly so that contract is independent of rasterio's inference.
                # float64 matches the downstream pipeline (the block read casts to
                # float64 anyway).
                dtype='float64',
                nodata=numpy.nan,
            ) as wvrt:
                _TerrainStreamer(
                    wvrt,
                    dst_transform,
                    dst_w,
                    dst_h,
                    px,
                    py,
                    accumulators,
                    work_crs,
                    bs,
                ).run(n_workers, progress)
        # else: no target overlaps the source -> accumulators stay empty (nodata).

    # See finalize_and_stamp for the generation-hash contract.
    return finalize_and_stamp(
        accumulators,
        format_version=TERRAIN_FORMAT_VERSION,
        hash_tag=DEM_HASH_TAG,
    )


def _read_haloed_block(
    wvrt: WarpedVRT,
    c0: int,
    r0: int,
    bw: int,
    bh: int,
    dst_w: int,
    dst_h: int,
) -> numpy.typing.NDArray[numpy.float64]:
    """Read a block plus its 1-px halo (:data:`HALO` per side), NaN-padded at the
    dataset edges.

    WarpedVRT forbids boundless reads, so the requested haloed window is clipped
    to the dataset and the read placed into a NaN-filled array -- giving the Horn
    window its border without reading out of bounds.
    """
    win_c0, win_r0 = c0 - HALO, r0 - HALO
    win_w, win_h = bw + 2 * HALO, bh + 2 * HALO

    read_c0 = max(0, win_c0)
    read_r0 = max(0, win_r0)
    read_c1 = min(dst_w, win_c0 + win_w)
    read_r1 = min(dst_h, win_r0 + win_h)

    z = numpy.full((win_h, win_w), numpy.nan, dtype='float64')
    if read_c1 <= read_c0 or read_r1 <= read_r0:
        return z

    data = wvrt.read(
        1,
        window=Window(read_c0, read_r0, read_c1 - read_c0, read_r1 - read_r0),
    ).astype('float64')
    dst_r, dst_c = read_r0 - win_r0, read_c0 - win_c0
    z[dst_r : dst_r + data.shape[0], dst_c : dst_c + data.shape[1]] = data
    return z


class _TerrainStreamer(StreamingBinner[_GridAccumulator]):
    """Streams one work grid in blocks, binning into every target accumulator.

    The concurrency scaffold (read lock, per-thread Transformers, cancel dance,
    coordinate epilogue, serial reduce, ``ordered_parallel_map`` wiring) lives on
    :class:`~snowtool.snowdb.zones.generate_common.StreamingBinner`. This engine
    supplies only the terrain-specific ``_load``: the haloed WarpedVRT read (so
    the 3x3 Horn window is exact across block edges) and the Horn slope/aspect
    derivation. Its payload is ``(cls, cos, sin, z)`` per
    :meth:`_GridAccumulator.bin_into`.
    """

    _label = 'reprojecting DEM'

    def __init__(
        self: Self,
        wvrt: WarpedVRT,
        dst_transform: Affine,
        dst_w: int,
        dst_h: int,
        px: float,
        py: float,
        accumulators: list[_GridAccumulator],
        work_crs: str,
        block_size: int,
    ) -> None:
        super().__init__(work_crs, accumulators)
        self._wvrt = wvrt
        self._dst_transform = dst_transform
        self._dst_w = dst_w
        self._dst_h = dst_h
        self._px = px
        self._py = py
        self._block_size = block_size

    def _blocks(self: Self) -> list[Block]:
        return iter_blocks(self._dst_w, self._dst_h, self._block_size)

    def _load(self: Self, block: Block, cancel: CancelToken) -> Loaded:
        """Read a haloed block (:data:`HALO` per side), then Horn slope/aspect.

        ``None`` for an all-nodata block or a read the cancel short-circuited.
        """
        z = self._locked_read(
            cancel,
            lambda: _read_haloed_block(
                self._wvrt,
                block.c0,
                block.r0,
                block.bw,
                block.bh,
                self._dst_w,
                self._dst_h,
            ),
        )
        if z is None or not numpy.isfinite(z).any():
            return None

        slope_deg, aspect_deg, vint, zint = _slope_aspect(z, self._px, self._py)
        if not vint.any():
            return None

        cls = _classify(slope_deg, aspect_deg, vint)
        rad = numpy.radians(aspect_deg)
        cos = numpy.cos(rad)
        sin = numpy.sin(rad)

        # The Horn pass trims one pixel off each edge of the haloed read, so the
        # inner block's global origin is (r0 - HALO + 1, c0 - HALO + 1) and its
        # shape is zint's; deriving the cell coords this way keeps them aligned
        # with cls/zint regardless of HALO or edge clipping.
        inner_h, inner_w = zint.shape
        x, y = pixel_centre_coords(
            self._dst_transform,
            block.r0 - HALO + 1,
            block.c0 - HALO + 1,
            inner_h,
            inner_w,
        )

        # Unmasked, block-shaped payload: the base (_compute) applies the single
        # ``keep`` mask to each array. cls carries the mask (``>= 0`` is exactly the
        # keep predicate), so it need not be computed twice.
        keep = cls >= 0
        payload = (cls.astype(numpy.int64), cos, sin, zint)
        return payload, x, y, keep

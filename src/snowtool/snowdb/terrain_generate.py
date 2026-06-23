"""Generate a dataset's terrain set from a fine-resolution DEM.

Overview
--------
One streaming pass over a single DEM source produces co-registered terrain for
*every* target grid at once: elevation, majority aspect class, and the mean
orientation components (northness/eastness = mean cos/sin of aspect) -- the layers
of a :class:`~snowtool.snowdb.terrain.TerrainSet`. Passing one target degenerates
to per-dataset generation; passing several shares the (expensive) source read.

This is a rework of the project's original 3DEP sample onto rasterio only (no
``osgeo``): the lazy reprojection is a :class:`rasterio.vrt.WarpedVRT`, and the
source mosaic is built by :mod:`~snowtool.snowdb.dem_source`, not ``gdal.BuildVRT``.

The three stages
----------------
1. **Resample to a fine, projected work grid (once).** The source is lazily
   reprojected (bilinear) into a single common lattice: a projected CRS at a fine
   resolution (``work_crs`` / ``work_resolution``; defaults
   :data:`DEFAULT_WORK_CRS` = CONUS Albers, :data:`DEFAULT_WORK_RESOLUTION` = 10 m,
   which matches 3DEP). This is *not* any dataset's target grid -- it is a shared
   intermediate. It is the only true resample, and it is of elevation.
2. **Derive everything at the work resolution.** Streaming in blocks (with a
   one-pixel halo so the 3x3 Horn window is exact across block edges), each fine
   pixel gets a slope, an aspect, an aspect class (N/E/S/W or flat), and
   cos/sin(aspect).
3. **Aggregate fine pixels into target cells.** Each fine pixel's centre is
   transformed (pyproj) into the target grid's CRS and assigned to the cell it
   lands in (point-in-cell, not fractional-area). Per cell the engine accumulates
   class counts (-> majority), an elevation sum (-> mean elevation), and cos/sin
   sums over non-flat pixels (-> mean northness/eastness).

Why reproject to a projected CRS at all
---------------------------------------
The source (e.g. 3DEP 1/3") is geographic (EPSG:4326): pixels are measured in
*degrees*, and a degree of longitude shrinks with latitude while a degree of
latitude is ~constant -- so pixels are non-square on the ground and the x-scale
drifts north/south. Slope and aspect are gradients (dz/dx, dz/dy); computed in
degree space they are anisotropically distorted and aspect is rotated/biased.
A projected, metric, near-square CRS (CONUS Albers) makes dx == dy in real metres
so the gradient -- hence aspect -- is geometrically valid. **This reprojection is
required for aspect, not for elevation.**

Elevation: tradeoff vs. a direct resample
-----------------------------------------
Elevation does *not* need the projected intermediate; it could be averaged
directly from the source onto each target grid (what the old elevation-only tool
did). Here it instead rides the shared work surface: source -> (bilinear) work
grid -> (mean of the fine pixels per cell) target cell. That adds a small,
nonzero error vs. a single ``average`` warp -- the bilinear step is a mild
low-pass filter and is not exactly mean-preserving, and point-in-cell binning
isn't fractional-area weighted (negligible at ~10 m pixels in ~1 km cells). The
magnitude is typically sub-metre, far below the elevation-band width it feeds, so
it is immaterial here. We accept it for **co-registration**: every terrain layer
comes from the one surface and the same binning, so elevation and aspect line up
by construction. (If metre-level elevation fidelity ever matters more than that
guarantee, compute elevation by a direct ``average`` warp and use the work surface
only for aspect.)

Note: because each fine pixel has equal area in the projected CRS, the plain
per-cell mean of elevation *is* an area-mean, so the ``average`` semantics survive
for elevation.

Resolution is source-dependent
------------------------------
``work_resolution`` should match the source's native ground resolution: too fine
upsamples and invents detail (aspect on interpolated values), too coarse throws
detail away. It is therefore a property of the ``DemSource`` (3DEP pins 10 m; a
local file defaults to its own native resolution), not a global constant -- the
values here are only the fallbacks.

Parallelism, memory, and the scatter vs. gather tradeoff
--------------------------------------------------------
This is an **input-driven scatter**: the work grid is streamed in blocks, and each
fine pixel is reprojected and dropped (``+=``) into whatever cell it lands in, in
*every* target at once. The expensive, target-independent work -- read, warp, Horn
slope/aspect, reprojection -- is therefore done once and shared across all targets;
that single shared read is the whole point of binning many grids in one pass.

Blocks run on a thread pool (the pyproj transform dominates and releases the GIL);
binning stays serial and in block order, so the output -- including the generation
hash -- is identical regardless of worker count. Two memory terms result:

* the **target accumulators** -- one shared set per target, ``~72 bytes/cell`` --
  a *fixed* cost (independent of worker count), sized by the target grids; and
* the **transient per-worker block buffers** -- each worker holds ~a dozen
  block-sized float64 arrays mid-block, so this scales with
  ``workers * block_size**2``.

The two knobs that bound this are ``workers`` and ``block_size`` (the latter is the
per-worker lever, since block size costs nothing on throughput). Neither bounds the
*accumulator* term: this design assumes the targets fit in RAM.

Scaling past RAM -- the materialize-then-gather refinement (future)
-------------------------------------------------------------------
When the targets stop fitting in RAM, the principled fix is **not** to throttle
workers but to switch the binning stage to **output-driven gather**, in two phases:

1. *Materialize* the shared work surface (elevation, slope, aspect, cos/sin) to a
   temporary tiled raster, once, streaming -- bounded memory, sequential write
   (this cold, write-once/read-windowed intermediate is the one place spilling to
   SSD genuinely helps, unlike the hot random-access accumulators).
2. *Gather* per target by output tile: each tile reads its footprint from the
   materialized surface, reprojects + bins independently and in parallel, and is
   written out -- memory bounded by tile size, no whole-grid accumulators.

This keeps the single shared read (phase 1) while making each output tile an
independent, bounded, parallel unit (phase 2). Naive output-tiling *without* the
materialize step would instead re-read/re-warp/re-Horn the overlapping work-grid
region once per target per tile, losing the shared-read efficiency -- which is why
the current single-pass scatter is the right call while the targets still fit.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import warnings

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy
import numpy.typing
import rasterio

from pyproj import Transformer
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform, transform_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from snowtool.exceptions import SNODASWarning
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import DEM_HASH_TAG
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.terrain import (
    ASPECT_COMPONENTS,
    ASPECT_E,
    ASPECT_FLAT,
    ASPECT_MAJORITY,
    ASPECT_MAJORITY_NODATA,
    ASPECT_N,
    ASPECT_S,
    ASPECT_W,
    ELEVATION,
    ELEVATION_NODATA,
    TERRAIN_FORMAT_VERSION,
    TERRAIN_LAYERS,
)

if TYPE_CHECKING:
    from snowtool.snowdb.zone_layer import ZoneLayerTarget

# Defaults for the projected, fine work grid aspect is computed on. CONUS Albers
# (metres, near-square) keeps slope/aspect undistorted; 10 m matches 3DEP. These
# are only fallbacks -- the DemSource supplies the right values for its data (see
# the module docstring), since the work resolution must track the source's native
# resolution and the work CRS its region.
DEFAULT_WORK_CRS = 'EPSG:5070'
DEFAULT_WORK_RESOLUTION = 10.0
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
# Whether the flat class can win the per-cell majority vote.
MAJORITY_INCLUDES_FLAT = True
# Whether to weight the cos/sin orientation mean by sin(slope) (steep dominates).
COSSIN_SLOPE_WEIGHTED = False

# Internal accumulator layout: four cardinal classes plus flat.
_N_CLASSES = 5

# Default cap on the *auto* worker count (``workers=None``): one thread per CPU but
# never more than this. Beyond ~here the per-block reprojection stops scaling (reads
# are serialised under a lock and the serial bin/reduce starts to bind) while memory
# and lock contention keep climbing, so more threads mostly cost RAM. An explicit
# ``workers`` is always honoured -- the caller owns that tradeoff (see the module
# docstring's memory notes; ``block_size`` is the lever for bounding per-worker RAM).
MAX_AUTO_WORKERS = 16


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


class _GridAccumulator:
    """Per-cell terrain accumulators for one target grid.

    Each fine source pixel is binned into its grid cell; the cardinal counts give
    the majority, the cos/sin sums give the orientation mean, and the elevation
    sum gives mean elevation. Exact across source block boundaries because every
    fine pixel is placed independently.
    """

    def __init__(self: Self, target: ZoneLayerTarget) -> None:
        self.target = target
        base = target.grid.base_grid
        self.height = base.rows
        self.width = base.cols
        self.transform = base.transform
        crs = target.grid.crs
        if crs is None:  # pragma: no cover - make_grid always sets a CRS
            raise ValueError(f'{target.name}: grid has no CRS')
        # The streamer builds (thread-local) work-CRS -> this-CRS Transformers.
        self.crs = crs
        n = self.height * self.width
        self.counts = numpy.zeros(_N_CLASSES * n, dtype=numpy.int64)
        self.sum_cos = numpy.zeros(n, dtype=numpy.float64)
        self.sum_sin = numpy.zeros(n, dtype=numpy.float64)
        self.sum_wt = numpy.zeros(n, dtype=numpy.float64)
        self.sum_z = numpy.zeros(n, dtype=numpy.float64)
        self._inv = ~self.transform

    @property
    def _ncell(self: Self) -> int:
        return self.height * self.width

    def bin_into(
        self: Self,
        xt: numpy.typing.NDArray[numpy.float64],
        yt: numpy.typing.NDArray[numpy.float64],
        cls: numpy.typing.NDArray[numpy.int64],
        cos: numpy.typing.NDArray[numpy.float64],
        sin: numpy.typing.NDArray[numpy.float64],
        wt: numpy.typing.NDArray[numpy.float64],
        z: numpy.typing.NDArray[numpy.float64],
    ) -> None:
        """Bin already-reprojected fine-pixel centres (this grid's CRS) into cells.

        Coordinate transform (work CRS -> this grid's CRS) happens in the worker,
        so this runs serially on the main thread in fixed block order -- the float
        accumulation order is identical to the serial pass, keeping the generation
        hash reproducible regardless of worker count.
        """
        col = numpy.floor(self._inv.a * xt + self._inv.b * yt + self._inv.c).astype(
            numpy.int64,
        )
        row = numpy.floor(self._inv.d * xt + self._inv.e * yt + self._inv.f).astype(
            numpy.int64,
        )
        inb = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        if not inb.any():
            return
        cell = row[inb] * self.width + col[inb]
        c = cls[inb]
        nc = self._ncell
        self.counts += numpy.bincount(c * nc + cell, minlength=_N_CLASSES * nc)
        self.sum_z += numpy.bincount(cell, weights=z[inb], minlength=nc)
        nf = c != ASPECT_FLAT  # cos/sin only where aspect is defined
        if nf.any():
            cf, wf = cell[nf], wt[inb][nf]
            self.sum_cos += numpy.bincount(cf, weights=cos[inb][nf] * wf, minlength=nc)
            self.sum_sin += numpy.bincount(cf, weights=sin[inb][nf] * wf, minlength=nc)
            self.sum_wt += numpy.bincount(cf, weights=wf, minlength=nc)

    def finalize(
        self: Self,
    ) -> tuple[
        numpy.typing.NDArray[numpy.uint8],
        numpy.typing.NDArray[numpy.float32],
        numpy.typing.NDArray[numpy.float32],
    ]:
        """Reduce the accumulators to the (majority, components, elevation) arrays."""
        h, w = self.height, self.width
        counts = self.counts.reshape(_N_CLASSES, h, w)
        total = counts.sum(axis=0)

        pool = counts if MAJORITY_INCLUDES_FLAT else counts[:4]
        majority = pool.argmax(axis=0).astype(numpy.uint8)
        majority[total == 0] = ASPECT_MAJORITY_NODATA

        wt = self.sum_wt.reshape(h, w)
        with numpy.errstate(invalid='ignore', divide='ignore'):
            northness = numpy.where(wt > 0, self.sum_cos.reshape(h, w) / wt, numpy.nan)
            eastness = numpy.where(wt > 0, self.sum_sin.reshape(h, w) / wt, numpy.nan)
            elevation = numpy.where(
                total > 0,
                self.sum_z.reshape(h, w) / total,
                ELEVATION_NODATA,
            )

        components = numpy.stack(
            [northness.astype(numpy.float32), eastness.astype(numpy.float32)],
        )
        return majority, components, elevation.astype(numpy.float32)

    def write_layers(
        self: Self,
        majority: numpy.typing.NDArray[numpy.uint8],
        components: numpy.typing.NDArray[numpy.float32],
        elevation: numpy.typing.NDArray[numpy.float32],
        dem_hash: str,
    ) -> None:
        """Write the three layer COGs, all stamped with the generation ``dem_hash``."""
        tags = {DEM_HASH_TAG: dem_hash}

        self.target.directory.mkdir(parents=True, exist_ok=True)
        rio_crs = rasterio.crs.CRS.from_wkt(self.crs.to_wkt())
        common = {
            'transform': self.transform,
            'crs': rio_crs,
            'tile_size': self.target.tile_size,
            'tags': tags,
        }

        write_cog(
            self.target.directory / ELEVATION.filename,
            elevation,
            nodata=ELEVATION.nodata,
            band_descriptions=ELEVATION.band_descriptions,
            **common,
        )
        write_cog(
            self.target.directory / ASPECT_MAJORITY.filename,
            majority,
            nodata=ASPECT_MAJORITY.nodata,
            band_descriptions=ASPECT_MAJORITY.band_descriptions,
            **common,
        )
        write_cog(
            self.target.directory / ASPECT_COMPONENTS.filename,
            components,
            nodata=ASPECT_COMPONENTS.nodata,
            band_descriptions=ASPECT_COMPONENTS.band_descriptions,
            # NaN nodata: the stats filter can't exclude it, so skip stats.
            compute_stats=False,
            **common,
        )


def _target_bounds_in_work_crs(
    targets: list[ZoneLayerTarget],
    work_crs: str,
) -> tuple[float, float, float, float]:
    """Union of the target grids' extents, expressed in the work CRS.

    The streaming pass only needs to cover where the targets are; this bbox (a
    conservative outer bound, since :func:`transform_bounds` densifies the reprojected
    edge) is what the work grid is clipped to.

    Known bound -- bbox under-coverage for small, heavily-curved target CRSs. The
    footprint is an axis-aligned bbox of the reprojected target edges, densified by
    ``transform_bounds`` (default 21 pts/edge). For a target whose CRS curves
    strongly against the work CRS (e.g. MODIS Sinusoidal far from its central
    meridian), a long edge can bulge outward *between* densify points, so the bbox
    can under-cover at the grid edge by up to that densification error (~tens to
    low-hundreds of metres); the ``+ HALO`` slack in :func:`_clip_grid_to_bounds`
    only partly absorbs it, so the outermost ring of target cells could be fed by
    slightly fewer source pixels. This is harmless whenever the clip is a no-op --
    i.e. when a target is as large as or larger than the source, which is the case
    for the continental instarr/MODIS-sinusoidal grid (the source is always the
    smaller extent, so the full source is processed). It only bites a target that is
    *both* comparable-to-or-smaller-than the source *and* strongly curved. If that
    ever matters, harden by raising ``transform_bounds(..., densify_pts=...)`` and
    widening the clip margin; left as-is since no current dataset hits it.
    """
    wests: list[float] = []
    souths: list[float] = []
    easts: list[float] = []
    norths: list[float] = []
    for target in targets:
        base = target.grid.base_grid
        t = base.transform
        xmin, ymax = t.c, t.f
        xmax = t.c + base.cols * t.a
        ymin = t.f + base.rows * t.e
        crs = target.grid.crs
        if crs is None:  # pragma: no cover - make_grid always sets a CRS
            raise ValueError(f'{target.name}: grid has no CRS')
        rio_crs = rasterio.crs.CRS.from_wkt(crs.to_wkt())
        w, s, e, n = transform_bounds(rio_crs, work_crs, xmin, ymin, xmax, ymax)
        wests.append(w)
        souths.append(s)
        easts.append(e)
        norths.append(n)
    # generate_terrain guards against empty targets, so these are non-empty.
    return min(wests), min(souths), max(easts), max(norths)


def _clip_grid_to_bounds(
    full_transform: Any,
    full_w: int,
    full_h: int,
    bounds: tuple[float, float, float, float],
) -> tuple[Any, int, int] | None:
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
    source: rasterio.io.DatasetReader,
    targets: list[ZoneLayerTarget],
    *,
    work_crs: str = DEFAULT_WORK_CRS,
    work_resolution: float | None = DEFAULT_WORK_RESOLUTION,
    workers: int | None = None,
    block_size: int | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Stream ``source`` once, binning terrain into every target grid.

    ``source`` is an opened DEM mosaic (any CRS/resolution); it is lazily
    reprojected to the fine projected work grid (``work_crs`` at
    ``work_resolution`` metres) by a :class:`WarpedVRT`. ``work_resolution`` should
    track the source's native resolution -- ``None`` lets GDAL pick it from the
    source (the right default for an unknown local DEM); 3DEP pins 10 m. Returns
    the single generation hash keyed by each target name (every value is equal --
    it is one identifier for the whole pass). Refuses to overwrite an existing
    terrain set unless ``force``.

    The per-block reprojection dominates runtime and releases the GIL, so blocks
    are processed on a thread pool: ``workers`` of ``None`` (the default) uses one
    thread per CPU (capped at :data:`MAX_AUTO_WORKERS`), ``1`` forces the serial
    path; an explicit value is honoured as-is. Binning stays serial and in block
    order, so the result -- including the generation hash -- is identical regardless
    of ``workers``. ``block_size`` (``None`` -> :data:`DEFAULT_BLOCK_SIZE`) is the
    per-worker memory lever: transient RAM scales with ``workers * block_size**2``
    at no throughput cost, so shrink it to run more workers on less RAM. Neither
    knob bounds the (fixed) target accumulators -- see the module docstring's memory
    and scatter-vs-gather notes.
    """
    if not targets:
        return {}

    n_workers = _effective_workers(workers)
    bs = block_size if block_size is not None else DEFAULT_BLOCK_SIZE

    if not force:
        # Check every target before the (expensive) source read.
        for target in targets:
            existing = [
                layer.filename
                for layer in TERRAIN_LAYERS
                if (target.directory / layer.filename).is_file()
            ]
            if existing:
                raise FileExistsError(
                    f'Could not generate terrain for {target.name}: '
                    f'{target.directory} already has {", ".join(existing)}. '
                    'Remove and try again or use force=True.',
                )

    accumulators = [_GridAccumulator(target) for target in targets]

    # Mask source fill using the source's own declared nodata. If it declares
    # none, trust it (mask nothing) -- but warn, since an undeclared fill value
    # would otherwise be aggregated as real elevation.
    src_nodata = source.nodata
    if src_nodata is None:
        warnings.warn(
            f'Source DEM {source.name!r} declares no nodata value; treating all '
            'pixels as valid data. Declare a nodata value on the source if it '
            'has fill pixels, or they will be aggregated as real elevations.',
            SNODASWarning,
            stacklevel=2,
        )
    full_transform, full_w, full_h = calculate_default_transform(
        source.crs,
        work_crs,
        source.width,
        source.height,
        *source.bounds,
        # None -> GDAL derives the native resolution mapped into the work CRS.
        resolution=work_resolution,
    )
    # Process only the part of the (full-source) work grid that actually feeds a
    # target. The reprojection is lazy and per-block (a WarpedVRT over COG tiles), so
    # clipping the work grid to the union of target footprints means the range reads
    # only ever touch the *intersecting portions* of the source tiles -- not the
    # whole tile files, even when a grid clips just a corner of them. The clip keeps
    # the source lattice, so any cell that lands in a target is sampled identically
    # to processing the whole source; only empty margin is skipped (the output, and
    # the generation hash, are unchanged).
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
            source,
            crs=work_crs,
            transform=dst_transform,
            width=dst_w,
            height=dst_h,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            # The streaming pass marks no-data with NaN (numpy.isfinite), so the
            # working band must be float. rasterio already promotes the band to float
            # to hold nodata=NaN (even for an integer source); we pin it explicitly so
            # that contract is independent of rasterio's inference. float64 matches
            # the downstream pipeline (the block read casts to float64 anyway).
            dtype='float64',
            nodata=numpy.nan,
        ) as wvrt:
            _stream_blocks(
                wvrt,
                dst_transform,
                dst_w,
                dst_h,
                px,
                py,
                accumulators,
                work_crs,
                n_workers,
                bs,
            )
    # else: no target overlaps the source -> accumulators stay empty (nodata terrain).

    # One generation id for the whole pass: a digest over every target's
    # finalized elevation (sorted by name for determinism), stamped identically on
    # every layer of every terrain set. It identifies the generation, not an
    # individual raster -- so all rasters produced together reconcile as one set.
    finalized = []
    digest = hashlib.sha256()
    for acc in sorted(accumulators, key=lambda a: a.target.name):
        majority, components, elevation = acc.finalize()
        finalized.append((acc, majority, components, elevation))
        digest.update(acc.target.name.encode('utf-8'))
        digest.update(elevation.tobytes())
    dem_hash = versioned_hash(TERRAIN_FORMAT_VERSION, digest.hexdigest())

    for acc, majority, components, elevation in finalized:
        acc.write_layers(majority, components, elevation, dem_hash)
    return dict.fromkeys((acc.target.name for acc in accumulators), dem_hash)


def _read_haloed_block(
    wvrt: WarpedVRT,
    c0: int,
    r0: int,
    bw: int,
    bh: int,
    dst_w: int,
    dst_h: int,
) -> numpy.typing.NDArray[numpy.float64]:
    """Read a block plus its 2-px halo, NaN-padded at the dataset edges.

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


@dataclass(frozen=True)
class _Block:
    """A nominal (un-haloed) work-grid block, in row-major streaming order."""

    c0: int
    r0: int
    bw: int
    bh: int


@dataclass(frozen=True)
class _BlockResult:
    """One block's binnable contribution: shared per-pixel arrays + per-target coords.

    ``coords`` holds, per target (same order as the streamer's accumulators), the
    fine-pixel centres already reprojected into that target's CRS. Everything is
    flattened and pre-masked to the valid (``cls >= 0``) pixels, so the serial
    reducer only has to bin.
    """

    cls: numpy.typing.NDArray[numpy.int64]
    cos: numpy.typing.NDArray[numpy.float64]
    sin: numpy.typing.NDArray[numpy.float64]
    wt: numpy.typing.NDArray[numpy.float64]
    z: numpy.typing.NDArray[numpy.float64]
    coords: list[
        tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
    ]


def _effective_workers(requested: int | None) -> int:
    """Resolve the worker count.

    ``requested`` of ``None`` means auto -- one thread per CPU, but never more than
    :data:`MAX_AUTO_WORKERS`. ``1`` (or anything <= 1) means the serial path. An
    explicit request is honoured as-is: the caller owns the memory tradeoff (bound
    per-worker RAM with ``block_size``; see the module docstring). Always >= 1.
    """
    if requested is None:
        return min(os.cpu_count() or 1, MAX_AUTO_WORKERS)
    return max(1, requested)


class _TerrainStreamer:
    """Streams one work grid in blocks, binning into every target accumulator.

    The expensive per-block work -- the Horn slope/aspect and the per-target pyproj
    reprojection -- is pure and runs on a worker pool. The only shared mutable state
    is the accumulators, so binning is done serially on the main thread, consuming
    block results in streaming order. That keeps the float accumulation order
    independent of the worker count, so the generation hash is reproducible
    (parallel == serial bit for bit). Each worker thread gets its own Transformers;
    a Transformer is not safe to share across threads concurrently.

    The haloed reads run under a lock: a GDAL dataset (and the shared
    :class:`WarpedVRT` over it) is not safe for concurrent reads -- unsynchronised
    reads corrupt blocks nondeterministically. The read is a small fraction of the
    per-block cost (the reprojection dominates and still runs fully in parallel), so
    serialising just the read costs little while guaranteeing correctness.

    Future improvement: if the read lock ever becomes the bottleneck (it is not at
    the ~10-worker scale measured -- the transform dominates), parallelise the reads
    by giving each worker thread its OWN dataset handle instead of sharing one
    WarpedVRT. That means reopening the source per thread (``rasterio.open`` of the
    mosaic) and wrapping each in its own WarpedVRT, since the unsafety is in the
    underlying GDAL dataset -- per-thread VRTs over one shared source would still
    race. Hold these in the existing thread-local (alongside the Transformers) and
    drop the lock. The cost is N source handles + N GDAL block caches resident at
    once, which is why it is deferred until the read actually dominates.
    """

    def __init__(
        self: Self,
        wvrt: WarpedVRT,
        dst_transform: Any,
        dst_w: int,
        dst_h: int,
        px: float,
        py: float,
        accumulators: list[_GridAccumulator],
        work_crs: str,
        block_size: int,
    ) -> None:
        self._wvrt = wvrt
        self._dst_transform = dst_transform
        self._dst_w = dst_w
        self._dst_h = dst_h
        self._px = px
        self._py = py
        self._accumulators = accumulators
        self._work_crs = work_crs
        self._block_size = block_size
        self._local = threading.local()
        # GDAL/WarpedVRT reads are not concurrency-safe; serialise just the read.
        self._read_lock = threading.Lock()

    def _blocks(self: Self) -> list[_Block]:
        bs = self._block_size
        nbx = math.ceil(self._dst_w / bs)
        nby = math.ceil(self._dst_h / bs)
        blocks = []
        for by in range(nby):
            for bx in range(nbx):
                c0, r0 = bx * bs, by * bs
                blocks.append(
                    _Block(
                        c0=c0,
                        r0=r0,
                        bw=min(bs, self._dst_w - c0),
                        bh=min(bs, self._dst_h - r0),
                    ),
                )
        return blocks

    def _transformers(self: Self) -> list[Transformer]:
        """Per-thread work-CRS -> target-CRS Transformers (built once per thread)."""
        tfs: list[Transformer] | None = getattr(self._local, 'tfs', None)
        if tfs is None:
            tfs = [
                Transformer.from_crs(self._work_crs, acc.crs, always_xy=True)
                for acc in self._accumulators
            ]
            self._local.tfs = tfs
        return tfs

    def _compute(self: Self, block: _Block) -> _BlockResult | None:
        """Worker step: read, derive terrain, reproject. Pure, no shared writes."""
        with self._read_lock:
            z = _read_haloed_block(
                self._wvrt,
                block.c0,
                block.r0,
                block.bw,
                block.bh,
                self._dst_w,
                self._dst_h,
            )
        if not numpy.isfinite(z).any():
            return None

        slope_deg, aspect_deg, vint, zint = _slope_aspect(z, self._px, self._py)
        if not vint.any():
            return None

        cls = _classify(slope_deg, aspect_deg, vint)
        rad = numpy.radians(aspect_deg)
        cos = numpy.cos(rad)
        sin = numpy.sin(rad)
        wt = (
            numpy.sin(numpy.radians(slope_deg))
            if COSSIN_SLOPE_WEIGHTED
            else numpy.ones_like(slope_deg)
        )

        # The Horn pass trims one pixel off each edge of the haloed read, so the
        # inner block's global origin is (r0 - HALO + 1, c0 - HALO + 1) and its
        # shape is zint's; deriving the cell coords this way keeps them aligned
        # with cls/zint regardless of HALO or edge clipping.
        inner_h, inner_w = zint.shape
        rows = (numpy.arange(inner_h) + (block.r0 - HALO + 1))[:, None]
        cols = (numpy.arange(inner_w) + (block.c0 - HALO + 1))[None, :]
        x = (
            self._dst_transform.c
            + (cols + 0.5) * self._dst_transform.a
            + (rows + 0.5) * self._dst_transform.b
        )
        y = (
            self._dst_transform.f
            + (cols + 0.5) * self._dst_transform.d
            + (rows + 0.5) * self._dst_transform.e
        )

        keep = cls.ravel() >= 0
        xf = numpy.broadcast_to(x, cls.shape).ravel()[keep]
        yf = numpy.broadcast_to(y, cls.shape).ravel()[keep]
        coords = [tf.transform(xf, yf) for tf in self._transformers()]
        return _BlockResult(
            cls=cls.ravel()[keep].astype(numpy.int64),
            cos=cos.ravel()[keep],
            sin=sin.ravel()[keep],
            wt=wt.ravel()[keep],
            z=zint.ravel()[keep],
            coords=coords,
        )

    def _reduce(self: Self, result: _BlockResult) -> None:
        """Main-thread step: bin one block into every accumulator (serial, ordered)."""
        for acc, (xt, yt) in zip(self._accumulators, result.coords, strict=True):
            acc.bin_into(
                xt, yt, result.cls, result.cos, result.sin, result.wt, result.z,
            )

    def run(self: Self, workers: int) -> None:
        blocks = self._blocks()
        if workers <= 1:
            for block in blocks:
                result = self._compute(block)
                if result is not None:
                    self._reduce(result)
            return

        # Parallel map, serial ordered reduce. A sliding window of at most
        # ``workers`` in-flight futures keeps every worker fed while bounding the
        # transient memory of buffered block results; popping left to right reduces
        # in block order, so the accumulation stays deterministic.
        pending = iter(blocks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            window: deque[Future[_BlockResult | None]] = deque()
            for block in pending:
                window.append(pool.submit(self._compute, block))
                if len(window) >= workers:
                    break
            while window:
                result = window.popleft().result()
                if result is not None:
                    self._reduce(result)
                next_block = next(pending, None)
                if next_block is not None:
                    window.append(pool.submit(self._compute, next_block))


def _stream_blocks(
    wvrt: WarpedVRT,
    dst_transform: Any,
    dst_w: int,
    dst_h: int,
    px: float,
    py: float,
    accumulators: list[_GridAccumulator],
    work_crs: str,
    workers: int,
    block_size: int,
) -> None:
    _TerrainStreamer(
        wvrt, dst_transform, dst_w, dst_h, px, py, accumulators, work_crs, block_size,
    ).run(workers)

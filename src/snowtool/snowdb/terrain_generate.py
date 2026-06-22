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
"""

from __future__ import annotations

import hashlib
import math
import warnings

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy
import numpy.typing
import rasterio

from pyproj import Transformer
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform

from snowtool.exceptions import SNODASWarning
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import DEM_HASH_TAG
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
    TERRAIN_LAYERS,
)

if TYPE_CHECKING:
    from pathlib import Path

    from griffine.grid import TiledAffineGrid

# Defaults for the projected, fine work grid aspect is computed on. CONUS Albers
# (metres, near-square) keeps slope/aspect undistorted; 10 m matches 3DEP. These
# are only fallbacks -- the DemSource supplies the right values for its data (see
# the module docstring), since the work resolution must track the source's native
# resolution and the work CRS its region.
DEFAULT_WORK_CRS = 'EPSG:5070'
DEFAULT_WORK_RESOLUTION = 10.0
BLOCK = 2048
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


@dataclass(frozen=True)
class TerrainTarget:
    """A grid to bin into and where to write its terrain set."""

    name: str
    grid: TiledAffineGrid
    tile_size: int
    directory: Path


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

    def __init__(self: Self, target: TerrainTarget, work_crs: str) -> None:
        self.target = target
        base = target.grid.base_grid
        self.height = base.rows
        self.width = base.cols
        self.transform = base.transform
        crs = target.grid.crs
        if crs is None:  # pragma: no cover - make_grid always sets a CRS
            raise ValueError(f'{target.name}: grid has no CRS')
        self._crs = crs
        n = self.height * self.width
        self.counts = numpy.zeros(_N_CLASSES * n, dtype=numpy.int64)
        self.sum_cos = numpy.zeros(n, dtype=numpy.float64)
        self.sum_sin = numpy.zeros(n, dtype=numpy.float64)
        self.sum_wt = numpy.zeros(n, dtype=numpy.float64)
        self.sum_z = numpy.zeros(n, dtype=numpy.float64)
        self._inv = ~self.transform
        # Fine-pixel centres arrive in the work CRS; map them to this grid's CRS.
        self._tf = Transformer.from_crs(work_crs, self._crs, always_xy=True)

    @property
    def _ncell(self: Self) -> int:
        return self.height * self.width

    def add(
        self: Self,
        x_src: numpy.typing.NDArray[numpy.float64],
        y_src: numpy.typing.NDArray[numpy.float64],
        cls: numpy.typing.NDArray[numpy.int64],
        cos: numpy.typing.NDArray[numpy.float64],
        sin: numpy.typing.NDArray[numpy.float64],
        wt: numpy.typing.NDArray[numpy.float64],
        z: numpy.typing.NDArray[numpy.float64],
    ) -> None:
        xt, yt = self._tf.transform(x_src, y_src)
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
        rio_crs = rasterio.crs.CRS.from_wkt(self._crs.to_wkt())
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


def generate_terrain(
    source: rasterio.io.DatasetReader,
    targets: list[TerrainTarget],
    *,
    work_crs: str = DEFAULT_WORK_CRS,
    work_resolution: float | None = DEFAULT_WORK_RESOLUTION,
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
    """
    if not targets:
        return {}

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

    accumulators = [_GridAccumulator(target, work_crs) for target in targets]

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
    dst_transform, dst_w, dst_h = calculate_default_transform(
        source.crs,
        work_crs,
        source.width,
        source.height,
        *source.bounds,
        # None -> GDAL derives the native resolution mapped into the work CRS.
        resolution=work_resolution,
    )
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
        # that contract is independent of rasterio's inference. float64 matches the
        # downstream pipeline (the block read casts to float64 anyway).
        dtype='float64',
        nodata=numpy.nan,
    ) as wvrt:
        _stream_blocks(wvrt, dst_transform, dst_w, dst_h, px, py, accumulators)

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
    dem_hash = digest.hexdigest()

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
    from rasterio.windows import Window

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


def _stream_blocks(
    wvrt: WarpedVRT,
    dst_transform: Any,
    dst_w: int,
    dst_h: int,
    px: float,
    py: float,
    accumulators: list[_GridAccumulator],
) -> None:
    nbx = math.ceil(dst_w / BLOCK)
    nby = math.ceil(dst_h / BLOCK)

    for by in range(nby):
        for bx in range(nbx):
            c0, r0 = bx * BLOCK, by * BLOCK
            bw = min(BLOCK, dst_w - c0)
            bh = min(BLOCK, dst_h - r0)

            z = _read_haloed_block(wvrt, c0, r0, bw, bh, dst_w, dst_h)
            if not numpy.isfinite(z).any():
                continue

            slope_deg, aspect_deg, vint, zint = _slope_aspect(z, px, py)
            if not vint.any():
                continue

            cls = _classify(slope_deg, aspect_deg, vint)
            rad = numpy.radians(aspect_deg)
            cos = numpy.cos(rad)
            sin = numpy.sin(rad)
            wt = (
                numpy.sin(numpy.radians(slope_deg))
                if COSSIN_SLOPE_WEIGHTED
                else numpy.ones_like(slope_deg)
            )

            # The Horn pass trims one pixel off each edge of the haloed read, so
            # the inner block's global origin is (r0 - HALO + 1, c0 - HALO + 1)
            # and its shape is zint's; deriving the cell coords this way keeps
            # them aligned with cls/zint regardless of HALO or edge clipping.
            inner_h, inner_w = zint.shape
            rows = (numpy.arange(inner_h) + (r0 - HALO + 1))[:, None]
            cols = (numpy.arange(inner_w) + (c0 - HALO + 1))[None, :]
            x = (
                dst_transform.c
                + (cols + 0.5) * dst_transform.a
                + (rows + 0.5) * dst_transform.b
            )
            y = (
                dst_transform.f
                + (cols + 0.5) * dst_transform.d
                + (rows + 0.5) * dst_transform.e
            )

            keep = cls.ravel() >= 0
            xf = numpy.broadcast_to(x, cls.shape).ravel()[keep]
            yf = numpy.broadcast_to(y, cls.shape).ravel()[keep]
            clsf = cls.ravel()[keep].astype(numpy.int64)
            cosf = cos.ravel()[keep]
            sinf = sin.ravel()[keep]
            wtf = wt.ravel()[keep]
            zf = zint.ravel()[keep]

            for acc in accumulators:
                acc.add(xf, yf, clsf, cosf, sinf, wtf, zf)

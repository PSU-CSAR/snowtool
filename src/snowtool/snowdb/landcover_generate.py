"""Generate a dataset's land-cover set from a fine-resolution NLCD raster.

One streaming pass over a single NLCD land-cover source produces a co-registered
percent-forest-cover layer for *every* target grid at once. Passing one target
degenerates to per-dataset generation; passing several shares the source read.

This parallels :mod:`snowtool.snowdb.terrain_generate` but is much simpler:
land cover is categorical, so there is no slope/aspect, no Horn neighbourhood
(hence no block halo), and no projected work grid -- aspect needs an undistorted
metric grid, but a *fraction of pixels* does not. The one shared idea is the
final stage: each fine source pixel's centre is transformed (pyproj) into the
target grid's CRS and binned into the cell it lands in (point-in-cell), so a
source in any CRS/resolution bins correctly onto any grid.

Per target cell the engine accumulates a count of valid NLCD pixels and a count
of forest pixels (classes in :data:`~snowtool.snowdb.constants.FOREST_CLASSES`);
the layer value is ``round(100 * forest / valid)`` as ``uint8``, nodata ``255``
where a cell caught no valid pixels.

The source is read only over the combined extent of the target grids (not the
whole national raster), so a CONUS NLCD file is not streamed in its entirety to
fill a handful of small grids.
"""

from __future__ import annotations

import hashlib
import math

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy
import numpy.typing
import rasterio

from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import FOREST_CLASSES, FOREST_PCT_NODATA, NLCD_HASH_TAG
from snowtool.snowdb.grid import grid_extent_4326
from snowtool.snowdb.landcover import FOREST_COVER, LANDCOVER_LAYERS

if TYPE_CHECKING:
    from pathlib import Path

    from griffine.grid import TiledAffineGrid

BLOCK = 2048


@dataclass(frozen=True)
class LandCoverTarget:
    """A grid to bin into and where to write its land-cover set."""

    name: str
    grid: TiledAffineGrid
    tile_size: int
    directory: Path


class _ForestAccumulator:
    """Per-cell forest/valid pixel counts for one target grid.

    Each valid fine NLCD pixel is binned into its grid cell; the per-cell forest
    and valid counts give the percent-forest value. Exact across source block
    boundaries because every fine pixel is placed independently.
    """

    def __init__(self: Self, target: LandCoverTarget, source_crs: str) -> None:
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
        self.forest = numpy.zeros(n, dtype=numpy.int64)
        self.valid = numpy.zeros(n, dtype=numpy.int64)
        self._inv = ~self.transform
        # Fine-pixel centres arrive in the source CRS; map them to this grid's CRS.
        self._tf = Transformer.from_crs(source_crs, self._crs, always_xy=True)

    @property
    def _ncell(self: Self) -> int:
        return self.height * self.width

    def add(
        self: Self,
        x_src: numpy.typing.NDArray[numpy.float64],
        y_src: numpy.typing.NDArray[numpy.float64],
        is_forest: numpy.typing.NDArray[numpy.bool_],
    ) -> None:
        """Bin a block of (already-valid) source pixels into this grid's cells."""
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
        n = self._ncell
        self.valid += numpy.bincount(cell, minlength=n)
        forest_cells = cell[is_forest[inb]]
        if forest_cells.size:
            self.forest += numpy.bincount(forest_cells, minlength=n)

    def finalize(self: Self) -> numpy.typing.NDArray[numpy.uint8]:
        """Reduce the counts to the percent-forest array (nodata where no pixels)."""
        h, w = self.height, self.width
        valid = self.valid.reshape(h, w)
        forest = self.forest.reshape(h, w)
        out = numpy.full((h, w), FOREST_PCT_NODATA, dtype=numpy.uint8)
        has = valid > 0
        # forest <= valid, so the rounded percentage is always in 0..100.
        out[has] = numpy.rint(100.0 * forest[has] / valid[has]).astype(numpy.uint8)
        return out

    def write_layer(
        self: Self,
        forest_pct: numpy.typing.NDArray[numpy.uint8],
        nlcd_hash: str,
    ) -> None:
        """Write the forest-cover COG, stamped with the generation ``nlcd_hash``."""
        self.target.directory.mkdir(parents=True, exist_ok=True)
        rio_crs = rasterio.crs.CRS.from_wkt(self._crs.to_wkt())
        write_cog(
            self.target.directory / FOREST_COVER.filename,
            forest_pct,
            transform=self.transform,
            crs=rio_crs,
            nodata=FOREST_COVER.nodata,
            tile_size=self.target.tile_size,
            band_descriptions=FOREST_COVER.band_descriptions,
            tags={NLCD_HASH_TAG: nlcd_hash},
        )


def generate_landcover(
    source: rasterio.io.DatasetReader,
    targets: list[LandCoverTarget],
    *,
    force: bool = False,
) -> dict[str, str]:
    """Stream ``source`` once, binning percent forest cover into every target grid.

    ``source`` is an opened NLCD land-cover raster (any CRS/resolution; natively
    EPSG:5070 30 m). Only the window over the targets' combined extent is read.
    Returns the single generation hash keyed by each target name (every value is
    equal -- one identifier for the whole pass). Refuses to overwrite an existing
    land-cover set unless ``force``.
    """
    if not targets:
        return {}

    if not force:
        # Check every target before the (potentially large) source read.
        for target in targets:
            existing = [
                layer.filename
                for layer in LANDCOVER_LAYERS
                if (target.directory / layer.filename).is_file()
            ]
            if existing:
                raise FileExistsError(
                    f'Could not generate land cover for {target.name}: '
                    f'{target.directory} already has {", ".join(existing)}. '
                    'Remove and try again or use force=True.',
                )

    source_crs = source.crs.to_wkt()
    accumulators = [_ForestAccumulator(target, source_crs) for target in targets]

    # NLCD uses 0 for unclassified/background; treat it (and the file's declared
    # nodata) as invalid so empty cells read as nodata rather than 0% forest.
    forest = numpy.asarray(FOREST_CLASSES)
    src_nodata = source.nodata

    window = _source_window(source, targets)
    if window is not None:
        _stream_blocks(source, window, forest, src_nodata, accumulators)

    # One generation id for the whole pass: a digest over every target's finalized
    # forest array (sorted by name for determinism), stamped identically on every
    # output -- so all layers produced together reconcile as one set.
    finalized = []
    digest = hashlib.sha256()
    for acc in sorted(accumulators, key=lambda a: a.target.name):
        forest_pct = acc.finalize()
        finalized.append((acc, forest_pct))
        digest.update(acc.target.name.encode('utf-8'))
        digest.update(forest_pct.tobytes())
    nlcd_hash = digest.hexdigest()

    for acc, forest_pct in finalized:
        acc.write_layer(forest_pct, nlcd_hash)
    return dict.fromkeys((acc.target.name for acc in accumulators), nlcd_hash)


def _source_window(
    source: rasterio.io.DatasetReader,
    targets: list[LandCoverTarget],
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


def _stream_blocks(
    source: rasterio.io.DatasetReader,
    window: Window,
    forest: numpy.typing.NDArray,
    src_nodata: float | None,
    accumulators: list[_ForestAccumulator],
) -> None:
    transform = source.transform
    col_off, row_off = int(window.col_off), int(window.row_off)
    win_w, win_h = int(window.width), int(window.height)
    nbx = math.ceil(win_w / BLOCK)
    nby = math.ceil(win_h / BLOCK)

    for by in range(nby):
        for bx in range(nbx):
            c0 = col_off + bx * BLOCK
            r0 = row_off + by * BLOCK
            bw = min(BLOCK, col_off + win_w - c0)
            bh = min(BLOCK, row_off + win_h - r0)

            values = source.read(1, window=Window(c0, r0, bw, bh))

            valid = values != 0
            if src_nodata is not None:
                valid &= values != src_nodata
            if not valid.any():
                continue
            is_forest = numpy.isin(values, forest) & valid

            # Absolute source-pixel centres -> source-CRS coordinates. NLCD is
            # north-up (b == d == 0), but the full affine form costs nothing and
            # stays correct for any source.
            rows = (numpy.arange(bh) + r0)[:, None]
            cols = (numpy.arange(bw) + c0)[None, :]
            x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
            y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e

            keep = valid.ravel()
            xf = numpy.broadcast_to(x, values.shape).ravel()[keep]
            yf = numpy.broadcast_to(y, values.shape).ravel()[keep]
            ff = is_forest.ravel()[keep]

            for acc in accumulators:
                acc.add(xf, yf, ff)

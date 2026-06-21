"""A dataset's *terrain set*: the elevation + aspect layers derived from a DEM.

Replaces the single ``dem.tif``. Because aspect cannot be resampled after the
fact (it must be computed from elevation at the source resolution), terrain is
generated once from a fine-resolution DEM (see
:mod:`snowtool.snowdb.terrain_generate`) into a small family of co-registered
layers on the dataset grid, stored under ``data/<name>/terrain/``:

* ``elevation.tif`` -- ``float32`` mean elevation (m).
* ``aspect_majority.tif`` -- ``uint8`` majority aspect class per cell
  (``0`` N, ``1`` E, ``2`` S, ``3`` W, ``4`` flat); nodata ``255``.
* ``aspect_components.tif`` -- two ``float32`` bands, ``northness`` =
  mean ``cos(aspect)`` and ``eastness`` = mean ``sin(aspect)`` over the cell's
  non-flat pixels (the first circular moment; ``hypot(northness, eastness)`` is
  the orientation purity in ``[0, 1]``).

Every layer carries a :data:`~snowtool.snowdb.constants.DEM_HASH_TAG` tag -- the
sha256 of the generated elevation array -- so the whole set's provenance can be
read back cheaply.

The single elevation layer is the one the read path needs at query time (it
drives elevation banding), so it is exposed as an :class:`ElevationRaster`, a
tiled COG read like any other (see :class:`~snowtool.snowdb.raster.TiledRaster`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy
import rasterio

from snowtool.snowdb.constants import DEM_HASH_TAG
from snowtool.snowdb.raster import TiledRaster

if TYPE_CHECKING:
    from pathlib import Path

# Aspect majority classes (cell's modal cardinal quadrant, or flat).
ASPECT_N = 0
ASPECT_E = 1
ASPECT_S = 2
ASPECT_W = 3
ASPECT_FLAT = 4

# Per-layer nodata sentinels. Elevation uses a far-below-bracket sentinel rather
# than NaN so it digitizes cleanly out of every elevation band (a NaN would not
# compare out), which is exactly how query-time banding excludes uncovered cells.
ELEVATION_NODATA = -9999.0
ASPECT_MAJORITY_NODATA = 255
ASPECT_COMPONENTS_NODATA = float('nan')


@dataclass(frozen=True)
class TerrainLayer:
    """A single on-disk terrain layer: its filename, dtype, nodata, and bands.

    Shared by the generator (which writes it) and :class:`TerrainSet` (which
    locates/reads it) so the two can never disagree on layout.
    """

    filename: str
    dtype: str
    nodata: float | int
    band_descriptions: tuple[str, ...]

    @property
    def count(self: Self) -> int:
        return len(self.band_descriptions)


ELEVATION = TerrainLayer(
    filename='elevation.tif',
    dtype='float32',
    nodata=ELEVATION_NODATA,
    band_descriptions=('elevation_mean_m',),
)
ASPECT_MAJORITY = TerrainLayer(
    filename='aspect_majority.tif',
    dtype='uint8',
    nodata=ASPECT_MAJORITY_NODATA,
    band_descriptions=('majority_cls_0N1E2S3W4flat',),
)
ASPECT_COMPONENTS = TerrainLayer(
    filename='aspect_components.tif',
    dtype='float32',
    nodata=ASPECT_COMPONENTS_NODATA,
    band_descriptions=('northness_mean_cos_aspect', 'eastness_mean_sin_aspect'),
)

# Every layer of a complete terrain set, in write order.
TERRAIN_LAYERS = (ELEVATION, ASPECT_MAJORITY, ASPECT_COMPONENTS)


class ElevationRaster(TiledRaster[numpy.float32]):
    """The mean-elevation terrain layer, read tile-by-tile at query time.

    Elevation banding reads this windowed to an AOI's tiles (the AOI raster is a
    bare geometry mask and no longer carries elevation), so it is a plain tiled
    COG reader -- no extra behavior beyond :class:`TiledRaster`.
    """


class TerrainSet:
    """A dataset's ``terrain/`` directory and the layers within it.

    Filesystem-only: it locates the layer files, reports which exist, and reads
    the shared provenance hash. Generation lives in
    :mod:`snowtool.snowdb.terrain_generate`.
    """

    def __init__(self: Self, directory: Path) -> None:
        self.directory = directory

    def layer_path(self: Self, layer: TerrainLayer) -> Path:
        return self.directory / layer.filename

    @property
    def elevation_path(self: Self) -> Path:
        return self.layer_path(ELEVATION)

    @property
    def aspect_majority_path(self: Self) -> Path:
        return self.layer_path(ASPECT_MAJORITY)

    @property
    def aspect_components_path(self: Self) -> Path:
        return self.layer_path(ASPECT_COMPONENTS)

    def present(self: Self) -> bool:
        """Whether every layer of a complete terrain set exists on disk."""
        return all(self.layer_path(layer).is_file() for layer in TERRAIN_LAYERS)

    def missing_layers(self: Self) -> list[TerrainLayer]:
        """The terrain layers that are not present on disk (the report selection)."""
        return [
            layer for layer in TERRAIN_LAYERS if not self.layer_path(layer).is_file()
        ]

    def elevation_raster(self: Self) -> ElevationRaster:
        """The elevation layer as a tiled reader (for query-time banding)."""
        return ElevationRaster(self.elevation_path)

    def dem_hash(self: Self) -> str | None:
        """The terrain set's provenance hash, or ``None`` if it isn't built.

        Reads only the elevation layer's tags (no array decode); returns ``None``
        when the layer is absent or predates the :data:`DEM_HASH_TAG` tagging.
        """
        if not self.elevation_path.is_file():
            return None
        with rasterio.open(self.elevation_path) as ds:
            return ds.tags().get(DEM_HASH_TAG)

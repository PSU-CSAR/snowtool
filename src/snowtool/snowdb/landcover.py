"""A dataset's *land-cover set*: the NLCD-derived percent-forest-cover layer.

Parallels :mod:`snowtool.snowdb.terrain` but is kept separate because it comes
from a different source (NLCD land cover, not a DEM) and so carries its own
provenance (:data:`~snowtool.snowdb.constants.NLCD_HASH_TAG`, not the DEM hash).
It is generated once from a fine-resolution NLCD raster (see
:mod:`snowtool.snowdb.landcover_generate`) onto the dataset grid, stored under
``data/<name>/landcover/``:

* ``forest_cover_pct.tif`` -- ``uint8`` percent forest cover (0..100), the share
  of the cell's NLCD pixels classed as forest (see
  :data:`~snowtool.snowdb.constants.FOREST_CLASSES`); nodata ``255``.

The layer is exposed as a :class:`ForestCoverRaster`, a tiled COG read like any
other (see :class:`~snowtool.snowdb.raster.TiledRaster`), so a future query path
can read it windowed to an AOI's tiles exactly as elevation is read for banding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy
import rasterio

from snowtool.snowdb.constants import FOREST_PCT_NODATA, NLCD_HASH_TAG
from snowtool.snowdb.raster import TiledRaster

# The on-disk layer descriptor is the same generic shape as terrain's (filename,
# dtype, nodata, band names); reuse it rather than redefining an identical class.
from snowtool.snowdb.terrain import TerrainLayer

if TYPE_CHECKING:
    from pathlib import Path


FOREST_COVER = TerrainLayer(
    filename='forest_cover_pct.tif',
    dtype='uint8',
    nodata=FOREST_PCT_NODATA,
    band_descriptions=('forest_cover_percent_0_100',),
)

# Every layer of a complete land-cover set, in write order.
LANDCOVER_LAYERS = (FOREST_COVER,)


class ForestCoverRaster(TiledRaster[numpy.uint8]):
    """The percent-forest-cover layer, read tile-by-tile at query time.

    A plain tiled COG reader (no extra behavior beyond :class:`TiledRaster`); it
    mirrors :class:`~snowtool.snowdb.terrain.ElevationRaster` so the read path can
    band/mask on forest cover the same way it bands on elevation.
    """


class LandCoverSet:
    """A dataset's ``landcover/`` directory and the layers within it.

    Filesystem-only: it locates the layer files, reports which exist, and reads
    the shared provenance hash. Generation lives in
    :mod:`snowtool.snowdb.landcover_generate`. Parallels
    :class:`~snowtool.snowdb.terrain.TerrainSet`.
    """

    def __init__(self: Self, directory: Path) -> None:
        self.directory = directory

    def layer_path(self: Self, layer: TerrainLayer) -> Path:
        return self.directory / layer.filename

    @property
    def forest_cover_path(self: Self) -> Path:
        return self.layer_path(FOREST_COVER)

    def present(self: Self) -> bool:
        """Whether every layer of a complete land-cover set exists on disk."""
        return all(self.layer_path(layer).is_file() for layer in LANDCOVER_LAYERS)

    def missing_layers(self: Self) -> list[TerrainLayer]:
        """The land-cover layers that are not present on disk."""
        return [
            layer for layer in LANDCOVER_LAYERS if not self.layer_path(layer).is_file()
        ]

    def forest_cover_raster(self: Self) -> ForestCoverRaster:
        """The forest-cover layer as a tiled reader (for query-time banding)."""
        return ForestCoverRaster(self.forest_cover_path)

    def nlcd_hash(self: Self) -> str | None:
        """The land-cover set's provenance hash, or ``None`` if it isn't built.

        Reads only the forest-cover layer's tags (no array decode); returns
        ``None`` when the layer is absent or predates the :data:`NLCD_HASH_TAG`
        tagging.
        """
        if not self.forest_cover_path.is_file():
            return None
        with rasterio.open(self.forest_cover_path) as ds:
            return ds.tags().get(NLCD_HASH_TAG)

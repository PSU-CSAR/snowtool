"""COG raster-I/O primitives: tiled reads, the handle cache, and COG writing.

The dataset-agnostic raster plumbing the rest of snowdb sits on, grouped away from
the catalog/query logic and free of any snow-domain knowledge:

* :mod:`.tiled` -- :class:`TiledRaster` / :class:`DataRaster`, the async block-read
  surface over a COG (re-exported here as the package's public face).
* :mod:`.collection` -- :class:`RasterCollection`, a query's selected COGs.
* :mod:`.cog` -- :func:`write_cog` and the WGS84 default CRS.
* :mod:`.tiff_cache` -- :class:`TiffCache`, the loop-affine async TIFF-handle cache.

Related: the burned *AOI* raster lives beside this package in
:mod:`snowtool.snowdb.aoi_raster`, and grid geometry in :mod:`snowtool.snowdb.grid`.
"""

from __future__ import annotations

from snowtool.snowdb.raster.tiled import DataRaster, TiledRaster

__all__ = ['DataRaster', 'TiledRaster']

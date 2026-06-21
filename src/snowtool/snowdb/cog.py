"""Shared helpers for writing Cloud-Optimized GeoTIFFs with rasterio.

rasterio's bundled GDAL supports writing the ``COG`` driver directly, so these
wrap that with the project's creation options and embed band statistics (read
back later by :meth:`AOIRaster.open` via the ``STATISTICS_*`` tags).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import rasterio

from rasterio.crs import CRS

if TYPE_CHECKING:
    from affine import Affine

WGS84 = CRS.from_epsg(4326)


def _default_predictor(dtype: numpy.dtype) -> int:
    # 3 = floating-point predictor, 2 = horizontal differencing for integers.
    return 3 if numpy.issubdtype(dtype, numpy.floating) else 2


def write_cog(
    path,
    array: numpy.ndarray,
    *,
    transform: Affine,
    tile_size: int,
    crs: CRS = WGS84,
    nodata: float | int | None = None,
    predictor: int | None = None,
    compute_stats: bool = True,
    tags: dict[str, str] | None = None,
    band_descriptions: tuple[str, ...] | None = None,
) -> None:
    """Write ``array`` to ``path`` as a tiled, DEFLATE-compressed COG.

    ``array`` may be 2D (single band) or 3D ``(bands, rows, cols)``. ``tags`` are
    written as dataset-level metadata (e.g. AOI tile quadkeys).
    ``band_descriptions``, when given, names each band (one per band, in order).
    """
    if array.ndim == 2:
        array = array[numpy.newaxis, ...]

    count, height, width = array.shape

    profile = {
        'driver': 'COG',
        'dtype': array.dtype,
        'count': count,
        'height': height,
        'width': width,
        'crs': crs,
        'transform': transform,
        'nodata': nodata,
        'blocksize': tile_size,
        'compress': 'DEFLATE',
        'predictor': predictor if predictor is not None else _default_predictor(
            array.dtype,
        ),
        # rasterio's bundled GDAL caps DEFLATE at 9 (no libdeflate).
        'level': 9,
    }

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array)
        if tags:
            dst.update_tags(**tags)
        if band_descriptions:
            for idx, description in enumerate(band_descriptions):
                dst.set_band_description(idx + 1, description)
        if compute_stats:
            _embed_stats(dst, array, nodata)


def _embed_stats(dst, array: numpy.ndarray, nodata: float | int | None) -> None:
    for idx in range(array.shape[0]):
        band = array[idx]
        if nodata is not None:
            band = band[band != nodata]
        if band.size == 0:
            continue
        dst.update_tags(
            idx + 1,
            STATISTICS_MINIMUM=float(band.min()),
            STATISTICS_MAXIMUM=float(band.max()),
            STATISTICS_MEAN=float(band.mean()),
            STATISTICS_STDDEV=float(band.std()),
        )

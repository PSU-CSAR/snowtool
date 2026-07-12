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

from snowtool.exceptions import ArtifactExistsError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date

    from affine import Affine

WGS84 = CRS.from_epsg(4326)

# Ingest provenance tag: a ``versioned_hash`` of the source artifact's bytes (see
# Dataset.write_date_cogs / INGEST_FORMAT_VERSION), stamped on every COG a date is
# ingested into. A same-name re-release with different bytes changes this, so the
# per-date skip rebuilds instead of wrongly skipping. Spelled inline like the other
# SOURCE_* keys below (a source-record tag, not one of the SNOWTOOL_* geometry tags
# in constants.py), but named once here since the skip check reads it back too.
SOURCE_HASH_TAG = 'SOURCE_HASH'


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
    written as dataset-level metadata (e.g. the AOI tile bbox / provenance hashes).
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
        'predictor': predictor
        if predictor is not None
        else _default_predictor(
            array.dtype,
        ),
        # rasterio's bundled GDAL caps DEFLATE at 9 (no libdeflate).
        'level': 9,
        # No overviews: the read path only ever reads full resolution (z=0).
        # Skipping them keeps the output smaller and deterministic.
        'overviews': 'NONE',
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


def write_cog_guarded(
    path,
    array: numpy.ndarray,
    *,
    force: bool,
    transform: Affine,
    tile_size: int,
    crs: CRS = WGS84,
    nodata: float | int | None = None,
    tags: dict[str, str] | None = None,
    predictor: int | None = 2,
) -> None:
    """:func:`write_cog` with the shared ingest existence-guard.

    Every ingester writes its COGs the same way: refuse to clobber an existing
    file unless ``force``, then write with the integer ``predictor=2`` ingest
    default. This keeps the guard message and that default in one place.
    """
    if not force and path.exists():
        raise ArtifactExistsError(
            f'Unable to write COG: {path} already exists. '
            'Remove file and try again or use `force=True`.',
        )
    write_cog(
        path,
        array,
        transform=transform,
        tile_size=tile_size,
        crs=crs,
        nodata=nodata,
        predictor=predictor,
        tags=tags,
    )


def source_tags(
    *,
    dataset: str,
    date: date,
    variable: str,
    files: str,
    source_hash: str,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The standard ``SOURCE_*`` provenance tags every ingester embeds in a COG.

    The shared keys (dataset/date/variable/files/hash) are spelled once here;
    ``extra`` carries each dataset's kind-specific tags (e.g. ``SOURCE_STAGE``,
    ``SOURCE_COLLECTION``, the SNODAS time-step fields). ``source_hash`` is the
    :func:`~snowtool.snowdb.provenance.versioned_hash` of the source artifact the
    date was ingested from; the per-date skip in
    :meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs` reads it back to catch
    a same-name different-bytes re-release.
    """
    tags = {
        'SOURCE_DATASET': dataset,
        'SOURCE_DATE': date.isoformat(),
        'SOURCE_VARIABLE': variable,
        'SOURCE_FILES': files,
        SOURCE_HASH_TAG: source_hash,
    }
    if extra:
        tags.update(extra)
    return tags


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

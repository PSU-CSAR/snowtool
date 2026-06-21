"""Pluggable DEM sources for terrain generation.

A :class:`DemSource` knows how to present a single opened DEM mosaic covering a
requested geographic extent; the terrain engine
(:func:`~snowtool.snowdb.terrain_generate.generate_terrain`) reprojects and
streams whatever it is handed. The source is a property of the *snow database*,
not of any one dataset: ``init`` reads one source and bins it into every grid.

* :class:`LocalFile` -- an on-disk raster or VRT the operator already has (this
  is the "import from a file" path).
* :class:`ThreeDEP` -- streams USGS 3DEP 1/3 arc-second tiles from the public
  ``prd-tnm`` S3 bucket. The tiles for the extent are enumerated (anonymous S3
  ``head_object`` existence checks) and stitched into a **VRT mosaic built as
  XML** -- the project is deliberately off the ``osgeo`` Python bindings, so
  ``gdal.BuildVRT`` is not used; rasterio's bundled GDAL reads the ``.vrt`` we
  write.
"""

from __future__ import annotations

import math
import tempfile

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Self
from xml.sax.saxutils import escape

import rasterio

from snowtool.snowdb.terrain_generate import (
    DEFAULT_WORK_CRS,
    DEFAULT_WORK_RESOLUTION,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# (west, south, east, north) in EPSG:4326.
Bounds = tuple[float, float, float, float]

# USGS 3DEP 1/3 arc-second staged COGs on the public registry-of-open-data bucket.
_S3_BUCKET = 'prd-tnm'
_S3_PREFIX = 'StagedProducts/Elevation/13/TIFF/current'
_TIF_NAME = 'USGS_13_{tile}.tif'

# rasterio/GDAL config for anonymous, range-read access to the COG tiles.
_S3_ENV = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'VSI_CACHE': 'TRUE',
    'GDAL_HTTP_MULTIRANGE': 'YES',
    'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
    'AWS_NO_SIGN_REQUEST': 'YES',
}

# rasterio dtype string -> GDAL data-type name (for the VRT band declaration).
_GDAL_DTYPES = {
    'uint8': 'Byte',
    'int16': 'Int16',
    'uint16': 'UInt16',
    'int32': 'Int32',
    'uint32': 'UInt32',
    'float32': 'Float32',
    'float64': 'Float64',
}


class DemSource(ABC):
    """A source of fine-resolution elevation, opened over a geographic extent.

    Carries the projected work grid terrain is derived on: ``work_crs`` (a metric,
    near-square CRS appropriate to the source's region -- needed so slope/aspect
    are undistorted) and ``work_resolution`` (metres; should track the source's
    native resolution, ``None`` lets GDAL derive it). These belong to the source,
    not the engine, because the right values depend on the data -- see
    :mod:`snowtool.snowdb.terrain_generate`.
    """

    def __init__(
        self: Self,
        *,
        work_crs: str = DEFAULT_WORK_CRS,
        work_resolution: float | None = DEFAULT_WORK_RESOLUTION,
    ) -> None:
        self.work_crs = work_crs
        self.work_resolution = work_resolution

    @abstractmethod
    def open(
        self: Self,
        bounds: Bounds,
    ) -> AbstractContextManager[rasterio.io.DatasetReader]:
        """Context manager yielding an opened DEM mosaic covering ``bounds``.

        ``bounds`` is ``(west, south, east, north)`` in EPSG:4326.
        """
        raise NotImplementedError


class LocalFile(DemSource):
    """A DEM the operator already has on disk (a raster or a ``.vrt``).

    ``work_resolution`` defaults to ``None`` -- GDAL derives the source's native
    resolution -- so an arbitrary DTM is processed at its own resolution rather
    than a hard-coded one. Pass it (and ``work_crs`` for non-CONUS data) to
    override.
    """

    def __init__(
        self: Self,
        path: Path,
        *,
        work_crs: str = DEFAULT_WORK_CRS,
        work_resolution: float | None = None,
    ) -> None:
        super().__init__(work_crs=work_crs, work_resolution=work_resolution)
        self.path = Path(path)

    @contextmanager
    def open(self: Self, bounds: Bounds) -> Iterator[rasterio.io.DatasetReader]:
        # The whole file is the source; the engine clips to the target grids.
        with rasterio.open(self.path) as src:
            yield src


class ThreeDEP(DemSource):
    """Stream USGS 3DEP 1/3 arc-second tiles from the public ``prd-tnm`` bucket.

    Defaults to the 10 m CONUS-Albers work grid (3DEP's native resolution); the
    work CRS/resolution can still be overridden via the base constructor.
    """

    @contextmanager
    def open(self: Self, bounds: Bounds) -> Iterator[rasterio.io.DatasetReader]:
        uris = self._tile_uris(bounds)
        if not uris:
            raise RuntimeError(
                f'No 3DEP tiles found for extent {bounds}; check the bounds.',
            )
        with rasterio.Env(**_S3_ENV), tempfile.TemporaryDirectory() as tmp:
            vrt_path = build_mosaic_vrt(uris, Path(tmp) / 'source.vrt')
            with rasterio.open(vrt_path) as src:
                yield src

    def _tile_uris(self: Self, bounds: Bounds) -> list[str]:
        keys = existing_tile_keys(bounds)
        return [f'/vsis3/{_S3_BUCKET}/{key}' for key in keys]


def candidate_tiles(bounds: Bounds) -> list[str]:
    """The 1-degree 3DEP tile names (e.g. ``n40w106``) covering ``bounds``.

    3DEP tiles are named by the *north-west* corner of each 1-degree cell, with
    west longitudes positive in the ``wNNN`` field; this assumes the western
    hemisphere (CONUS), as the snow domain is.
    """
    west, south, east, north = bounds
    tiles = set()
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            tiles.add(f'n{lat + 1:02d}w{-lon:03d}')
    return sorted(tiles)


def existing_tile_keys(bounds: Bounds) -> list[str]:
    """The S3 keys of the candidate tiles that actually exist (anonymous check)."""
    import boto3

    from botocore import UNSIGNED
    from botocore.client import Config

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    keys: list[str] = []
    for tile in candidate_tiles(bounds):
        key = f'{_S3_PREFIX}/{tile}/{_TIF_NAME.format(tile=tile)}'
        try:
            s3.head_object(Bucket=_S3_BUCKET, Key=key)
        except Exception:  # noqa: BLE001, S112 - any error means "tile absent"
            continue
        keys.append(key)
    return keys


def _gdal_dtype(dtype: str) -> str:
    try:
        return _GDAL_DTYPES[dtype]
    except KeyError as e:
        raise ValueError(f'Unsupported source dtype for VRT mosaic: {dtype}') from e


def build_mosaic_vrt(uris: list[str], out_path: Path) -> Path:
    """Write a single-band VRT mosaic over ``uris`` (same CRS + resolution).

    Reads each source's header (not its pixels) to place it in the mosaic, then
    writes GDAL VRT XML. This is the ``osgeo``-free stand-in for
    ``gdal.BuildVRT``; rasterio can open the result like any dataset.
    """
    sources = []
    crs_wkt: str | None = None
    px: float | None = None
    py: float | None = None
    dtype: str | None = None
    nodata: float | None = None
    for uri in uris:
        with rasterio.open(uri) as ds:
            t = ds.transform
            if crs_wkt is None:
                crs_wkt = ds.crs.to_wkt() if ds.crs else ''
                px, py = t.a, t.e
                dtype = ds.dtypes[0]
                nodata = ds.nodata
            sources.append((uri, t, ds.width, ds.height))

    if px is None or py is None or dtype is None:  # pragma: no cover - empty uris
        raise ValueError('Cannot build a VRT mosaic from no sources.')

    xmin = min(t.c for _, t, _, _ in sources)
    ymax = max(t.f for _, t, _, _ in sources)
    xmax = max(t.c + w * t.a for _, t, w, _ in sources)
    ymin = min(t.f + h * t.e for _, t, _, h in sources)
    width = round((xmax - xmin) / px)
    height = round((ymax - ymin) / -py)

    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        f'  <SRS>{escape(crs_wkt or "")}</SRS>',
        f'  <GeoTransform>{xmin}, {px}, 0.0, {ymax}, 0.0, {py}</GeoTransform>',
        f'  <VRTRasterBand dataType="{_gdal_dtype(dtype)}" band="1">',
    ]
    if nodata is not None:
        lines.append(f'    <NoDataValue>{nodata}</NoDataValue>')
    for uri, t, w, h in sources:
        dst_xoff = round((t.c - xmin) / px)
        dst_yoff = round((ymax - t.f) / -py)
        lines.extend(
            [
                '    <SimpleSource>',
                f'      <SourceFilename relativeToVRT="0">{escape(str(uri))}'
                '</SourceFilename>',
                '      <SourceBand>1</SourceBand>',
                f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>',
                f'      <DstRect xOff="{dst_xoff}" yOff="{dst_yoff}" '
                f'xSize="{w}" ySize="{h}"/>',
                '    </SimpleSource>',
            ],
        )
    lines.extend(['  </VRTRasterBand>', '</VRTDataset>'])

    out_path.write_text('\n'.join(lines))
    return out_path

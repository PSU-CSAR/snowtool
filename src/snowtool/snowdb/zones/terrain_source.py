"""Pluggable DEM sources for terrain generation.

A :class:`DemSource` knows how to present a single opened DEM mosaic covering a
requested geographic extent; the terrain engine
(:func:`~snowtool.snowdb.zones.terrain_generate.generate_terrain`) reprojects and
streams whatever it is handed. The source is a property of the *snow database*,
not of any one dataset: ``init`` reads one source and bins it into every grid.

* :class:`LocalFile` -- an on-disk raster or VRT the operator already has (this
  is the "import from a file" path).
* :class:`ThreeDEP` -- streams USGS 3DEP 1/3 arc-second tiles from the public
  ``prd-tnm`` S3 bucket. The tiles for the extent are discovered in a single
  concurrent pass on the ``async-tiff`` store layer the COG read path already uses
  (:func:`discover_tiles`): each candidate is opened anonymously, which both proves
  it exists *and* yields its geo-header, so the existing tiles are stitched into a
  VRT mosaic written as XML (no ``osgeo``/``gdal.BuildVRT``) that rasterio's bundled
  GDAL reads. The actual pixel streaming is GDAL over ``/vsis3/``.
"""

from __future__ import annotations

import asyncio
import math
import tempfile

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self
from xml.sax.saxutils import escape

import rasterio

from async_tiff import TIFF, ImageFileDirectory
from async_tiff.enums import SampleFormat
from async_tiff.store import S3Store
from rasterio.crs import CRS

from snowtool.snowdb.grid import Bounds
from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.zones.terrain_generate import (
    DEFAULT_WORK_CRS,
    DEFAULT_WORK_RESOLUTION,
)
from snowtool.snowdb.zones.zone_layer import ZoneLayerSource

if TYPE_CHECKING:
    from collections.abc import Iterator

# USGS 3DEP 1/3 arc-second staged COGs on the public registry-of-open-data bucket.
_S3_BUCKET = 'prd-tnm'
_S3_REGION = 'us-west-2'
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

# (TIFF SampleFormat, bits-per-sample) -> rasterio dtype string. Covers the
# integer/float widths 3DEP (and any sane DEM) actually ships; an unmapped pair
# raises rather than guessing.
_SAMPLE_DTYPES = {
    (SampleFormat.Uint, 8): 'uint8',
    (SampleFormat.Uint, 16): 'uint16',
    (SampleFormat.Uint, 32): 'uint32',
    (SampleFormat.Int, 16): 'int16',
    (SampleFormat.Int, 32): 'int32',
    (SampleFormat.Float, 32): 'float32',
    (SampleFormat.Float, 64): 'float64',
}


class DemSource(ZoneLayerSource):
    """A source of fine-resolution elevation, opened over a geographic extent.

    Carries the projected work grid terrain is derived on: ``work_crs`` (a metric,
    near-square CRS appropriate to the source's region -- needed so slope/aspect
    are undistorted) and ``work_resolution`` (metres; should track the source's
    native resolution, ``None`` lets GDAL derive it). These belong to the source,
    not the engine, because the right values depend on the data -- see
    :mod:`snowtool.snowdb.zones.terrain_generate`.
    """

    def __init__(
        self: Self,
        *,
        work_crs: str = DEFAULT_WORK_CRS,
        work_resolution: float | None = DEFAULT_WORK_RESOLUTION,
    ) -> None:
        self.work_crs = work_crs
        self.work_resolution = work_resolution


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
    def open(
        self: Self,
        bounds: Bounds,
        *,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> Iterator[rasterio.io.DatasetReader]:
        # The whole file is the source; the engine clips to the target grids.
        # Nothing to download, so ``progress`` is unused.
        with rasterio.open(self.path) as src:
            yield src


class ThreeDEP(DemSource):
    """Stream USGS 3DEP 1/3 arc-second tiles from the public ``prd-tnm`` bucket.

    Defaults to the 10 m CONUS-Albers work grid (3DEP's native resolution); the
    work CRS/resolution can still be overridden via the base constructor.
    """

    @contextmanager
    def open(
        self: Self,
        bounds: Bounds,
        *,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> Iterator[rasterio.io.DatasetReader]:
        # 3DEP tiles stream lazily over range reads, so there is no discrete
        # download step to report; ``progress`` is accepted for the uniform source
        # contract and unused.
        tiles = discover_tiles(bounds)
        if not tiles:
            raise RuntimeError(
                f'No 3DEP tiles found for extent {bounds}; check the bounds.',
            )
        with rasterio.Env(**_S3_ENV), tempfile.TemporaryDirectory() as tmp:
            vrt_path = build_mosaic_vrt(tiles, Path(tmp) / 'source.vrt')
            with rasterio.open(vrt_path) as src:
                yield src


@dataclass(frozen=True)
class MosaicTile:
    """A discovered 3DEP tile: its stream URI plus the geo-header for the VRT.

    Populated from the tile's GeoTIFF header during discovery, so building the
    mosaic needs no further per-tile reads. ``py`` is the (negative) north-up
    pixel height; ``crs_wkt`` is the source CRS as WKT.
    """

    uri: str
    origin_x: float
    origin_y: float
    px: float
    py: float
    width: int
    height: int
    dtype: str
    nodata: float | None
    crs_wkt: str


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


def _tile_key(tile: str) -> str:
    return f'{_S3_PREFIX}/{tile}/{_TIF_NAME.format(tile=tile)}'


def _anonymous_store() -> S3Store:
    """An unsigned ``async-tiff`` S3 store for the public ``prd-tnm`` bucket."""
    return S3Store(
        _S3_BUCKET,
        config={'skip_signature': 'true', 'region': _S3_REGION},
    )


def _parse_geo_header(ifd: ImageFileDirectory, uri: str) -> MosaicTile:
    """Build a :class:`MosaicTile` from an opened tile's first IFD.

    3DEP tiles are north-up GeoTIFFs, so the model pixel-scale + tiepoint give
    the transform directly and the GeoKey EPSG gives the CRS -- no pixel reads.
    The georeferencing tags are all optional on the IFD, so a tile missing the
    pixel scale, tiepoint, GeoKey directory, or a CRS code is rejected with a
    clear error (it is not a usable north-up mosaic input) rather than crashing on
    a ``None``.
    """
    scale = ifd.model_pixel_scale
    tiepoint = ifd.model_tiepoint
    gk = ifd.geo_key_directory
    if scale is None or tiepoint is None or gk is None:
        raise ValueError(
            f'{uri}: not a georeferenced north-up GeoTIFF (missing model pixel '
            'scale, tiepoint, or GeoKey directory).',
        )

    fmt = ifd.sample_format[0]
    bits = ifd.bits_per_sample[0]
    try:
        dtype = _SAMPLE_DTYPES[(fmt, bits)]
    except KeyError as e:
        raise ValueError(
            f'Unsupported sample format/bits for VRT mosaic: {fmt}/{bits}',
        ) from e

    epsg = gk.projected_type or gk.geographic_type
    if epsg is None:
        raise ValueError(
            f'{uri}: GeoKey directory declares no projected or geographic CRS.',
        )

    raw_nodata = ifd.gdal_nodata
    return MosaicTile(
        uri=uri,
        origin_x=tiepoint[3],
        origin_y=tiepoint[4],
        px=scale[0],
        py=-scale[1],
        width=ifd.image_width,
        height=ifd.image_height,
        dtype=dtype,
        nodata=None if raw_nodata is None else float(raw_nodata),
        crs_wkt=CRS.from_epsg(epsg).to_wkt(),
    )


async def _probe_tile(store: S3Store, tile: str) -> MosaicTile | None:
    """Open one candidate tile: its header if it exists, else ``None``.

    Only a genuine not-found (404) means "tile not published" and is swallowed;
    everything else (throttling, auth, a transient 5xx, a network error) must
    surface rather than silently drop a real tile and leave a hole in the mosaic.
    ``async-tiff`` maps a missing key to ``FileNotFoundError`` and all other S3
    failures to its own exception, so a single typed catch preserves that.
    """
    key = _tile_key(tile)
    try:
        tiff = await TIFF.open(key, store=store)
    except FileNotFoundError:
        return None
    return _parse_geo_header(tiff.ifd(0), f'/vsis3/{_S3_BUCKET}/{key}')


async def _discover_async(bounds: Bounds, store: S3Store) -> list[MosaicTile]:
    """Probe every candidate tile concurrently; return the existing ones sorted."""
    tiles = await asyncio.gather(
        *(_probe_tile(store, tile) for tile in candidate_tiles(bounds)),
    )
    return sorted((t for t in tiles if t is not None), key=lambda t: t.uri)


def discover_tiles(bounds: Bounds) -> list[MosaicTile]:
    """The existing 3DEP tiles covering ``bounds``, with their geo-headers.

    One concurrent anonymous pass over the public bucket: each probe both proves
    existence and reads the header the VRT needs, so there is no second per-tile
    read. Sync bridge over a one-shot event loop (no shared loop, so the tiff
    handle cache's loop-binding caveat does not apply here).
    """
    return asyncio.run(_discover_async(bounds, _anonymous_store()))


def _gdal_dtype(dtype: str) -> str:
    try:
        return _GDAL_DTYPES[dtype]
    except KeyError as e:
        raise ValueError(f'Unsupported source dtype for VRT mosaic: {dtype}') from e


def build_mosaic_vrt(tiles: list[MosaicTile], out_path: Path) -> Path:
    """Write a single-band VRT mosaic over ``tiles`` (same CRS + resolution).

    Pure assembly from the geo-headers gathered during discovery -- no source is
    re-opened. This is the ``osgeo``-free stand-in for ``gdal.BuildVRT``;
    rasterio can open the result like any dataset.
    """
    if not tiles:  # pragma: no cover - callers guard against empty discovery
        raise ValueError('Cannot build a VRT mosaic from no sources.')

    px = tiles[0].px
    py = tiles[0].py
    dtype = tiles[0].dtype
    nodata = tiles[0].nodata
    crs_wkt = tiles[0].crs_wkt

    # The VRT header is taken wholesale from tile 0, so a heterogeneous mosaic would
    # silently mis-place or mis-type the odd tile. Enforce the documented precondition
    # (same CRS + resolution + dtype + nodata) instead of assuming it.
    for i, t in enumerate(tiles[1:], start=1):
        if (t.crs_wkt, t.px, t.py, t.dtype, t.nodata) != (
            crs_wkt,
            px,
            py,
            dtype,
            nodata,
        ):
            raise ValueError(
                'Cannot build a VRT mosaic from heterogeneous tiles: tile '
                f'{i} disagrees with tile 0 on CRS/resolution/dtype/nodata.',
            )

    xmin = min(t.origin_x for t in tiles)
    ymax = max(t.origin_y for t in tiles)
    xmax = max(t.origin_x + t.width * t.px for t in tiles)
    ymin = min(t.origin_y + t.height * t.py for t in tiles)
    width = round((xmax - xmin) / px)
    height = round((ymax - ymin) / -py)

    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        f'  <SRS>{escape(crs_wkt)}</SRS>',
        f'  <GeoTransform>{xmin}, {px}, 0.0, {ymax}, 0.0, {py}</GeoTransform>',
        f'  <VRTRasterBand dataType="{_gdal_dtype(dtype)}" band="1">',
    ]
    if nodata is not None:
        lines.append(f'    <NoDataValue>{nodata}</NoDataValue>')
    for t in tiles:
        dst_xoff = round((t.origin_x - xmin) / px)
        dst_yoff = round((ymax - t.origin_y) / -py)
        lines.extend(
            [
                '    <SimpleSource>',
                f'      <SourceFilename relativeToVRT="0">{escape(t.uri)}'
                '</SourceFilename>',
                '      <SourceBand>1</SourceBand>',
                f'      <SrcRect xOff="0" yOff="0" xSize="{t.width}" '
                f'ySize="{t.height}"/>',
                f'      <DstRect xOff="{dst_xoff}" yOff="{dst_yoff}" '
                f'xSize="{t.width}" ySize="{t.height}"/>',
                '    </SimpleSource>',
            ],
        )
    lines.extend(['  </VRTRasterBand>', '</VRTDataset>'])

    out_path.write_text('\n'.join(lines))
    return out_path

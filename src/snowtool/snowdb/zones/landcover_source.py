"""Pluggable NLCD land-cover sources for the land-cover generator.

A :class:`LandCoverSource` presents a single opened NLCD land-cover raster
covering a requested geographic extent; the generator
(:func:`~snowtool.snowdb.zones.landcover_generate.generate_landcover`) bins whatever it
is handed onto each target grid. Like the DEM source, this belongs to the *snow
database*, not any one dataset: one source bins into every grid in a single pass.

* :class:`AnnualNLCD` -- the default. Downloads the MRLC Annual NLCD land-cover
  bundle once (a single national GeoTIFF), caches the extracted raster, and reads it
  locally, so ``snowdb init`` and the ``dataset`` commands build land cover out of
  the box.
* :class:`LocalFile` -- an NLCD raster the operator already has on disk.

Why download-and-cache rather than stream like 3DEP: the Annual NLCD raster
bucket (``s3://usgs-landcover/...``) is *requester-pays* (it cannot be read
anonymously the way the DEM's open ``prd-tnm`` bucket can), but NLCD is a single
static national file, so the open MRLC HTTPS bundle is fetched once and cached.
(The ``.tif`` inside the ``.zip`` is deflate-compressed, so it is extracted to the
cache rather than read in place -- ``/vsizip/`` cannot range-read a compressed
member.)
"""

from __future__ import annotations

import tempfile
import urllib.request
import zipfile

from abc import abstractmethod
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Self

import rasterio

from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.zones.zone_layer import Bounds, ZoneLayerSource

if TYPE_CHECKING:
    from collections.abc import Iterator

# Download read size: stream the ~1.5 GB bundle a chunk at a time so progress can
# advance (and memory stays flat) rather than buffering the whole response.
_DOWNLOAD_CHUNK_BYTES = 1 << 20

# The current Annual NLCD land-cover (LndCov) CONUS bundle: Collection 1 Version 1,
# data year 2024 (the latest published; EPSG:5070, 30 m, categorical uint8). An
# open MRLC HTTPS download -- no credentials, unlike the requester-pays S3 bucket.
DEFAULT_NLCD_URL = (
    'https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/'
    'Annual_NLCD_LndCov_2024_CU_C1V1.zip'
)


class LandCoverSource(ZoneLayerSource):
    """A source of fine-resolution NLCD land cover, opened over an extent.

    Adds an optional ``progress`` to :meth:`open` over the base contract: a source
    that fetches its data (:class:`AnnualNLCD`) reports the download through it;
    one that reads a local file ignores it.
    """

    @abstractmethod
    def open(
        self: Self,
        bounds: Bounds,
        *,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> AbstractContextManager[rasterio.io.DatasetReader]:
        """Context manager yielding an opened NLCD raster covering ``bounds``."""
        raise NotImplementedError


class LocalFile(LandCoverSource):
    """An NLCD land-cover raster the operator already has on disk."""

    def __init__(self: Self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def open(
        self: Self,
        bounds: Bounds,
        *,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> Iterator[rasterio.io.DatasetReader]:
        # The whole file is the source; the generator reads only the window over
        # the target grids' extent. Nothing to download, so ``progress`` is unused.
        with rasterio.open(self.path) as src:
            yield src


class AnnualNLCD(LandCoverSource):
    """Download the MRLC Annual NLCD land-cover bundle once, then read it locally.

    The bundle is a ``.zip`` holding one national land-cover GeoTIFF; it is
    fetched to ``cache_dir`` and the raster extracted there on first use, then
    reused on subsequent opens. The download is lazy (only :meth:`open` triggers
    it), so constructing the source -- e.g. as ``SnowDb``'s default -- is cheap.
    """

    def __init__(
        self: Self,
        *,
        url: str = DEFAULT_NLCD_URL,
        cache_dir: Path | None = None,
    ) -> None:
        self.url = url
        # A persistent default keyed by nothing-in-particular so repeat runs reuse
        # the ~1.5 GB download; SnowDb passes a cache dir under the snowdb root.
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.gettempdir()) / 'snowtool-landcover-cache'
        )
        self._zip_name = url.rsplit('/', 1)[-1]
        self._member = self._zip_name.removesuffix('.zip') + '.tif'

    @property
    def raster_path(self: Self) -> Path:
        """Where the extracted land-cover GeoTIFF lives in the cache."""
        return self.cache_dir / self._member

    def _ensure_local(
        self: Self,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> Path:
        """Fetch + extract the raster if it isn't cached yet; return its path."""
        raster = self.raster_path
        if raster.is_file():
            return raster

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.cache_dir / self._zip_name
        if not zip_path.is_file():
            # Download to a .part sidecar and rename, so an interrupted fetch
            # never leaves a truncated zip that looks complete.
            part = zip_path.with_suffix(zip_path.suffix + '.part')
            with urllib.request.urlopen(self.url) as resp, part.open('wb') as out:  # noqa: S310 - fixed https MRLC URL
                # .headers.get (not .getheader) so this reads identically off an
                # http.client.HTTPResponse and a file:// addinfourl (the test path).
                content_length = resp.headers.get('Content-Length')
                total = int(content_length) if content_length else None
                # Stream in chunks (not shutil.copyfileobj) so progress can advance
                # by bytes written; memory stays flat regardless of the ~1.5 GB size.
                with progress.track('downloading NLCD', total=total) as task:
                    while chunk := resp.read(_DOWNLOAD_CHUNK_BYTES):
                        out.write(chunk)
                        task.advance(len(chunk))
            part.replace(zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(self._member, self.cache_dir)
        # The raster is extracted; drop the ~1.5 GB zip to reclaim the space.
        zip_path.unlink(missing_ok=True)
        return raster

    @contextmanager
    def open(
        self: Self,
        bounds: Bounds,
        *,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> Iterator[rasterio.io.DatasetReader]:
        raster = self._ensure_local(progress)
        with rasterio.open(raster) as src:
            yield src

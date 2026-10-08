"""

download.py

download.py takes a date and attempts an import of the raw data for that given date
The data sources it attempts to import from are
    - SWANN (University of Arizona)
    - SNODAS (NSIDC)
    - INSTARR (NSIDC)

Usage
-----
    # Download data from all sources for a given date
    snowtool download date 2026-03-06

    # Download data for a single source
    snowtool download date 2026-03-06 --source swann

    # Download data for multiple specific sources
    snowtool download date 2026-03-06 --source swann --source instarr

    # Retry all failed/missing downloads across all sources
    snowtool download retry

    # Retry failed/missing downloads for a specific source
    snowtool download retry --source swann

    # Retry failed/missing downloads for multiple sources
    snowtool download retry --source swann --source instarr

    # Retry failed/missing downloads for a date range
    snowtool download retry --source swann --start 2023-12-23 --end 2024-05-06


Output layout
-------------
SWANN:
    {dest}/{year}/{month}/UA_SWE_Depth_800m_v1_{YYYYMMDD}_{qualifier}.nc

INSTARR (grouped by date so completeness of a tile-set is easy to check):
    {dest}/{tile}/{YYYYMMDD}/SPIRES_NRT_{tile}_MOD09GA061_{YYYYMMDD}_V1.0.nc

SNODAS:
    {dest}/{year}/{month}/SNODAS_{year}{month}{day}.tar
"""

from __future__ import annotations

import logging
import posixpath
import shutil

from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import click

from curl_cffi import requests
from curl_cffi.requests.exceptions import HTTPError, RequestException

from snowtool.cli._context import config_option, pass_snowdb
from snowtool.cli._dates import DATE
from snowtool.snowdb import diagnostics
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.downloads import (
    BaseUrl,
    Downloader,
    DownloadResult,
    INSTARRUrls,
    SNODASUrl,
    SWANNUrl,
)

CHUNK_SIZE: int = 1024 * 1024

DEFAULT_TIMEOUT_SECONDS: int = 60


# NOTE: hand-maintained alongside the dataset registry in snowdb/datasets/.
# A dataset registered there but missing here will KeyError in `download`.
SOURCE_MODELS: dict[str, type[BaseUrl]] = {
    'swann': SWANNUrl,
    'instarr': INSTARRUrls,
    'snodas': SNODASUrl,
}


def _fetch_ftp(url: str, tmp_dest: Path) -> None:
    """Stream an ``ftp://`` URL to ``tmp_dest`` using the standard library."""
    with (
        urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response,  # noqa: S310
        tmp_dest.open('wb') as f,
    ):
        shutil.copyfileobj(response, f, CHUNK_SIZE)


def _fetch_http(url: str, tmp_dest: Path, session: Downloader) -> None:
    """Stream an ``http(s)://`` URL to ``tmp_dest`` through ``session``."""
    response = session.get(url, impersonate='chrome', stream=True)
    try:
        response.raise_for_status()
        with tmp_dest.open('wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)
    finally:
        response.close()


def _get_file(
    url: str,
    dest: Path,
    session: Downloader | None = None,
) -> DownloadResult:
    """Fetch one file over HTTP(S) or FTP and write it into ``dest``.

    Args:
        url: Fully-qualified URL of the file to fetch.
        dest: Directory to write into. The filename is taken from ``url``.
        session: Transport for HTTP(S) fetches. Defaults to a new ``curl_cffi``
            session, closed before returning. Unused for ``ftp://``.

    Returns:
        A ``DownloadResult`` whose ``status`` is ``True`` on success, or
        ``False`` with the failure in ``detail``.
    """
    filename = Path(posixpath.basename(urlparse(url).path))
    dest = dest / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + '.part')

    is_ftp = urlparse(url).scheme == 'ftp'
    owns_session = session is None and not is_ftp
    try:
        if is_ftp:
            _fetch_ftp(url, tmp_dest)
        else:
            session = session if session is not None else requests.Session()
            _fetch_http(url, tmp_dest, session)
        tmp_dest.rename(dest)
    except (HTTPError, RequestException, URLError, OSError) as e:
        tmp_dest.unlink(missing_ok=True)
        return DownloadResult(status=False, detail=str(e))
    finally:
        if owns_session and session is not None:
            session.close()

    return DownloadResult(status=True, detail='')


def _download_sources(
    sources: list[str],
    dates: list[date],
    snowdb: SnowDb,
    session: Downloader,
) -> None:
    """Fetch every file each source publishes for each date, logging outcomes."""
    for source_iter in sources:
        download_root = snowdb.download_root(source_iter)
        for d in dates:
            model = SOURCE_MODELS[source_iter]._for_date(d)
            for url, dest in model._iter_downloads():
                result = _get_file(url, download_root / dest, session)
                if result.status:
                    logger.info('[%s] %s downloaded', source_iter, d)
                else:
                    logger.warning(
                        '[%s] %s failed: %s',
                        source_iter,
                        d,
                        result.detail,
                    )


logger = logging.getLogger(__name__)


@click.group()
def download() -> None:
    """Download Import Commands"""


@download.command('date')
@click.argument('date', type=click.DateTime(formats=['%Y-%m-%d']), required=True)
@click.option('--source', '-s', type=str, multiple=True)
@config_option
@pass_snowdb
def download_dates(
    snowdb: SnowDb,
    date: datetime,
    source: tuple[str, ...] | None,
) -> None:
    """
    download_dates will attempt to grab the files from each specified source
    for the requested date, writes to the record ledger if the attempt fails
    or if it's a version that can be upgraded later

    Args:
        date (datetime): Date to grab data from each source for
        source (tuple[str, ...] | None, optional): List of sources to iterate through
                                         Defaults to ['snodas', 'instarr', 'swann']
    """
    sources = list(source) if source else ['snodas', 'instarr', 'swann']
    session = requests.Session()
    try:
        _download_sources(sources, [date.date()], snowdb, session)
    finally:
        session.close()


@download.command('retry')
@click.option('--source', '-s', type=str, multiple=True)
@click.option('--start', type=DATE, default=None)
@click.option('--end', type=DATE, default=None)
@config_option
@pass_snowdb
def retry_download(
    snowdb: SnowDb,
    source: tuple[str, ...] | None,
    start: date | None,
    end: date | None,
) -> None:
    """
    retry_download will find all missing daily uploads in a given date range
    for a data source and attempt to reimport them. If a date range is not
    provided, function will default to three weeks

    Args:
        source (tuple[str, ...] | None): List of sources to iterate through
                                         Defaults to ['snodas', 'instarr', 'swann']
        start                      date: Start of date range to search
                                         Defaults to none
        end                        date: End of date range to search
                                         Defaults to none
    """
    retry_start = start if start else date.today() - timedelta(weeks=3)  # noqa: DTZ011
    retry_end = end if end else date.today()  # noqa: DTZ011
    sources = list(source) if source else ['snodas', 'instarr', 'swann']

    session = requests.Session()
    try:
        for source_iter in sources:
            ds = snowdb.registered_dataset(source_iter)
            missing = diagnostics.missing_dates(ds, start=retry_start, end=retry_end)

            if not missing:
                logger.info('[%s] has no missing dates in range', source_iter)
                continue

            click.echo(f'[{source_iter}] {len(missing)} missing date(s) found.')
            _download_sources([source_iter], list(missing), snowdb, session)
    finally:
        session.close()

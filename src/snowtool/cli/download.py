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

from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import click

from curl_cffi import requests
from curl_cffi.requests.exceptions import HTTPError, RequestException

from snowtool.api.models.downloads import (
    BaseUrl,
    DownloadResult,
    INSTARRUrls,
    SNODASUrl,
    SWANNUrl,
)
from snowtool.cli._context import config_option, pass_snowdb
from snowtool.cli._dates import DATE
from snowtool.snowdb import diagnostics
from snowtool.snowdb.db import SnowDb

DEFAULT_TIMEOUT_SECONDS: int = 60
"""
Some servers (e.g. climate.arizona.edu) reject requests with no/generic
User-Agent headers. Need to identify as a real browser to avoid spurious 403s.
"""
REQUEST_HEADERS: dict[str, str] = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
}

SOURCE_MODELS: dict[str, type[BaseUrl]] = {
    'swann': SWANNUrl,
    'instarr': INSTARRUrls,
    'snodas': SNODASUrl,
}


def _get_file(url: str, dest: Path) -> DownloadResult:
    """
    _get_file requests the file from the specified source
    (http or ftp), and writes it to the desired destination

    Args:
        url (str): url for requested file
        dest (Path): _description_

    Returns:
        DownloadResult: _description_
    """
    filename = Path(posixpath.basename(urlparse(url).path))
    dest = dest / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + '.part')
    try:
        response = requests.get(url, impersonate='chrome')
        response.raise_for_status()
        with tmp_dest.open('wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        tmp_dest.rename(dest)
        status = True
        detail = ''
    except HTTPError as e:
        tmp_dest.unlink(missing_ok=True)
        status = False
        detail = str(e)
    except RequestException as e:
        tmp_dest.unlink(missing_ok=True)
        status = False
        detail = str(e)
    except OSError as e:
        status = False
        detail = str(e)

    return DownloadResult(
        status=status,
        detail=detail,
    )


logger = logging.getLogger(__name__)


@click.group()
def download() -> None:
    """Download Import Commands"""


@download.command('date')
@click.argument('date', type=click.DateTime(formats=['%Y-%m-%d']), required=True)
@click.option('--source', '-s', type=str, multiple=True)
def download_dates(
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
    for source_iter in sources:
        model = SOURCE_MODELS[source_iter]._for_date(date.date())
        for url, dest in model._iter_downloads():
            result = _get_file(url, dest)
            if result.status:
                logger.info('[%s] %s downloaded', source_iter, date.date())
            else:
                logger.warning(
                    '[%s] %s failed: %s',
                    source_iter,
                    date.date(),
                    result.detail,
                )


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

    for source_iter in sources:
        ds = snowdb.registered_dataset(source_iter)
        missing = diagnostics.missing_dates(ds, start=retry_start, end=retry_end)

        if not missing:
            logging.info('[%s] has no missing dates in range', source_iter)
            continue

        click.echo(f'[{source_iter}] {len(missing)} missing date(s) found.')
        for d in missing:
            model = SOURCE_MODELS[source_iter]._for_date(d)
            for url, dest in model._iter_downloads():
                result = _get_file(url, Path(dest))
                if result.status:
                    logger.info('[%s] %s downloaded', source_iter, d)
                else:
                    logger.warning('[%s] %s failed: %s', source_iter, d, result.detail)

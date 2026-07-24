"""

download.py

download.py takes a date and attempts an import of that given days data
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

    # Retry SWANN downloads and attempt to upgrade early/provisional
    # files to a more stable qualifier
    # Note: --upgrade requires --source swann
    snowtool download retry --source swann --upgrade

    # --upgrade without --source swann will raise a UsageError
    # snowtool download retry --source instarr --upgrade  <- invalid



Output layout
-------------
SWANN:
    {dest}/{year}/{month}/UA_SWE_Depth_800m_v1_{YYYYMMDD}_{qualifier}.nc

INSTARR (grouped by date so completeness of a tile-set is easy to check):
    {dest}/{tile}/{YYYYMMDD}/SPIRES_NRT_{tile}_MOD09GA061_{YYYYMMDD}_V1.0.nc

SNODAS:
    {dest}/{year}/{month}/SNODAS_{year}{month}{day}.tar

Writes to sqlite3 ledger file about failed results of download attempts for data

"""

from __future__ import annotations

import posixpath

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import click

from curl_cffi import requests
from curl_cffi.requests.exceptions import HTTPError, RequestException

from snowtool.api.models.downloads import BaseUrl, INSTARRUrls, SNODASUrl, SWANNUrl
from snowtool.api.models.ledger import DownloadResult, Ledger

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
        return DownloadResult(url=url, dest=dest, status='success')
    except HTTPError as e:
        tmp_dest.unlink(missing_ok=True)
        if e.response.status_code == 404:
            return DownloadResult(
                url=url,
                dest=dest,
                status='missing',
                detail='HTTP 404',
            )
        return DownloadResult(url=url, dest=dest, status='error', detail=str(e))
    except RequestException as e:
        tmp_dest.unlink(missing_ok=True)
        return DownloadResult(url=url, dest=dest, status='error', detail=str(e))
    except OSError as e:
        tmp_dest.unlink(missing_ok=True)
        return DownloadResult(url=url, dest=dest, status='error', detail=str(e))


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
            if result.status != 'success':
                Ledger._write_to_record(
                    source=source_iter,
                    result=result,
                )
            if isinstance(model, SWANNUrl) and model.qualifier in (
                'early',
                'provisional',
            ):
                Ledger._write_to_record(
                    'swann',
                    result,
                    qualifier=model.qualifier,
                )


@download.command('retry')
@click.option('--source', '-s', type=str, multiple=True)
@click.option(
    '--upgrade',
    is_flag=True,
    default=False,
    help=(
        'Re-attempt SWANN downloads as a more stable version (requires --source swann)',
    ),
)
def retry_download(
    source: tuple[str, ...] | None,
    upgrade: bool,
) -> None:
    """
    retry_download will parse the record ledger
    and retry previously failed downloads
    or upgrade swann to a more stable version of code

    Args:
        source (tuple[str, ...] | None): List of sources to iterate through
                                         Defaults to ['snodas', 'instarr', 'swann']
        upgrade (bool): Flag to set if downloading the more stable iteration
                        of a SWANN dataset

    Raises:
        click.UsageError: Raised when --upgrade is passed without SWANN
    """
    sources = list(source) if source else ['snodas', 'instarr', 'swann']

    if upgrade and 'swann' not in sources:
        raise click.UsageError(
            '--upgrade can only be used when --source includes swann',
        )

    for source_iter in sources:
        missing_sets = Ledger._get_records(source_iter)
        for url, dest in missing_sets:
            result = _get_file(url, dest)
            if result.status == 'success':
                Ledger._clean_records(url)
            else:
                Ledger._write_to_record(source_iter, result)

        if source_iter == 'swann' and upgrade:
            for url, dest, qualifier in Ledger._get_records('swann', update=True):
                match qualifier:
                    case 'early':
                        next_ver = 'provisional'
                    case 'provisional':
                        next_ver = 'stable'
                    case _:
                        continue
                # Might be fragile if UA ever ends up changing how they structure URLS
                # Worth revisiting later?
                update_url = url.replace(qualifier, next_ver)
                result = _get_file(update_url, Path(dest))

                if result.detail != 'success':
                    Ledger._write_to_record(
                        source='swann',
                        result=result,
                        qualifier=qualifier,
                    )
                else:
                    Ledger._clean_records(url=url)  # Remove old version from record
                    Ledger._write_to_record(  # Write record with new qualifier
                        source='swann',
                        result=result,
                        qualifier=next_ver,
                    )

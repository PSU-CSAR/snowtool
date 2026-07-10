"""

download.py

download.py takes a date and attempts an import of that given days data
The data sources it attempts to import from are
    - SWANN (University of Arizona)
    - SNODAS (NSIDC)
    - INSTARR (NSIDC)

Usage
-----
    # Attempt to download data files for a given date
    snowtool download 2026-03-06

    # Attempt to download a specified date for a singular (or multiple sources)
    snowtool download 2026-03-06 --source instarr
    snowtool download 2026-03-06 --source instarr --source swann

    # Attempt a download with a number of retries
    snowtool download 2026-03-06 --retries 5


Output layout
-------------
SWANN:
    {dest}/{year}/{month}/UA_SWE_Depth_800m_v1_{YYYYMMDD}_{qualifier}.nc

INSTARR (grouped by date so completeness of a tile-set is easy to check):
    {dest}/{tile}/{YYYYMMDD}/SPIRES_NRT_{tile}_MOD09GA061_{YYYYMMDD}_V1.0.nc

Writes to sqlite3 ledger file about failed results of download attempts for data

Dependencies: requests, netCDF4
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

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


def get_file(url: str, dest: Path) -> DownloadResult:
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
    source: tuple[str, ...] | None = None,
) -> None:
    sources = list(source) if source else ['snodas', 'instarr', 'swann']
    for source_iter in sources:
        model = SOURCE_MODELS[source_iter]._for_date(date.date())
        for url, dest in model._iter_downloads():
            result = get_file(url, dest)
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

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

Writes to sqlite3 ledger file about result of download attempts for data

Dependencies: requests, netCDF4
"""

from __future__ import annotations

import shutil
import sqlite3
import urllib.error
import urllib.request

from datetime import date, datetime
from pathlib import Path

import click
import requests

from snowtool.api.models.download import DownloadResult

try:
    import netCDF4  # type: ignore[import]
except ImportError:
    netCDF4 = None  # verification step degrades gracefully if unavailable  # noqa: N816

# Constants
BASE_URLS: dict[str, str] = {
    'swann': 'https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{year}/UA_SWE_Depth_800m_v1_{year}{month}{day}_{qualifier}.nc',
    'instarr': 'ftp://dtn.rc.colorado.edu/shares/snow-today/gridded_data/SPIRES_NRT_V01/{tile}/{year}/SPIRES_NRT_{tile}_MOD09GA061_{year}{month}{day}_V1.0.nc',
    'snodas': 'https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{year}/{month}_{month_abbr}/SNODAS_{year}{month}{day}.tar',
}

INSTARR_TILES: tuple[str, ...] = ('h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v04')

INGEST_RECORD_DB: Path = Path('/d/projects/gisdata/ingest_record.db')

DEFAULT_RETRY_ATTEMPTS: int = 3

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


@click.group()
def download() -> None:
    """Download Import Commands"""


def _water_year(d: date) -> int:
    """
    _water_year returns the water year for a given date.
    Water year N runs from Oct 1 of year (N-1) through Sep 30 of year N.

    Inputs:
        d: date to check
    Outputs:
        Water Year
    """
    return d.year + 1 if d.month >= 10 else d.year


def _download_file_ftp(
    url: str,
    dest: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> DownloadResult:
    """
    _download_file_ftp downloads a single file from an FTP server to a destination,
    skipping if it already exists.


    Inputs:
        url: url to request file from
        dest: destination to download file to
        timeout: Time to wait for download before erroring out
    Output:
        Returns a DownloadResult describing what happened.
    """

    if dest.exists() and dest.stat().st_size > 0:
        return DownloadResult(url, dest, 'skipped_exists')

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + '.part')

    try:
        with (
            urllib.request.urlopen(url, timeout=timeout) as response,  # noqa: S310
            Path.open(tmp_dest, 'wb') as f,
        ):
            shutil.copyfileobj(response, f)

        tmp_dest.rename(dest)
        return DownloadResult(url, dest, 'downloaded')

    except urllib.error.URLError as e:
        tmp_dest.unlink(missing_ok=True)
        # urllib raises URLError for both "not found" and network errors on FTP
        reason = str(e.reason) if hasattr(e, 'reason') else str(e)
        status = 'missing' if '550' in reason else 'error'
        return DownloadResult(url, dest, status, reason)
    except Exception as e:  # noqa: BLE001
        tmp_dest.unlink(missing_ok=True)
        return DownloadResult(url, dest, 'error', str(e))


def _download_file(
    url: str,
    dest: Path,
    session: requests.Session,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> DownloadResult:
    """
    _download_file downloads a single file from an HTTP server to a destination,
    skipping if it already exists.


    Inputs:
        url: url to request file from
        dest: destination to download file to
        session: requests object that initiates the request
    Output:
        Returns a DownloadResult describing what happened.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return DownloadResult(url, dest, 'skipped_exists')

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + '.part')

    try:
        with session.get(
            url,
            stream=True,
            timeout=timeout,
            headers=REQUEST_HEADERS,
        ) as response:
            if response.status_code == 404:
                return DownloadResult(url, dest, 'missing', 'HTTP 404')
            response.raise_for_status()

            expected_size = response.headers.get('Content-Length')

            with Path.open(tmp_dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

        if expected_size is not None:
            actual_size = tmp_dest.stat().st_size
            if int(expected_size) != actual_size:
                tmp_dest.unlink(missing_ok=True)
                return DownloadResult(
                    url,
                    dest,
                    'error',
                    f'size mismatch: expected {expected_size}, got {actual_size}',
                )

        tmp_dest.rename(dest)
        return DownloadResult(url, dest, 'downloaded')

    except requests.RequestException as e:
        tmp_dest.unlink(missing_ok=True)
        return DownloadResult(url, dest, 'error', str(e))


def _verify_netcdf(path: Path) -> bool:
    """
    _verify_netcdf is a Best-effort verification
    that a downloaded file is a readable NetCDF.
    Input:
        path: path to downloaded file
    Output:
        Returns True if netCDF4 isn't installed (verification skipped) or if
        the file opens successfully; False if it's corrupt/truncated.
    """
    if netCDF4 is None:
        return True
    try:
        with netCDF4.Dataset(path, 'r'):
            return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_ledger(conn: sqlite3.Connection):
    """
    _ensure_ledger makes sure the sqlite3 file used to track missing
    or failed imports exists

    Inputs:
        conn (sqlite3.Connection): Connection to the ledger file
    Outputs:
        None
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL,
            file        TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            status      TEXT    NOT NULL,
            qualifier   TEXT,
            detail      TEXT    NOT NULL DEFAULT '',
            attempted_at TEXT   NOT NULL DEFAULT (datetime('now'))
        )
    """,
    )


def _write_to_record(
    source: str,
    filename: str,
    date: date,
    result: DownloadResult,
    ingest_record: Path,
    qualifier: str = '',
) -> None:
    """
    _write_to_record Writes the results of a download attempt for a data file
    to the ledger database

    Args:
        source (str): Data source (swann | instarr)
        filename (str): Name of the file
        date (date): date of the file
        result (DownloadResult): Results from the download attempt of the file
        ingest_record (Path): Path to the ledger file
        qualifier (str, optional): Used for SWANN to identify early/provisional files to
            track for their next iteration. Defaults to "".
    """
    if result.status == 'skipped_exists':
        # No need to write to ledger if record already exists
        return

    with sqlite3.connect(ingest_record) as conn:
        _ensure_ledger(conn)
        cursor = conn.cursor()
        if result.status == 'downloaded':
            cursor.execute(
                'INSERT INTO ledger (source,file, date, status, detail, qualifier) VALUES (?,?,?,?,?,?)',  # noqa: E501
                (source, filename, date, 'success', '', qualifier),
            )
        else:
            cursor.execute(
                'INSERT INTO ledger (source,file, date, status, detail, qualifier) VALUES (?,?,?,?,?,?)',  # noqa: E501
                (source, filename, date, 'failed', result.detail, qualifier),
            )
        conn.commit()


def _write_missing_log(results: list[DownloadResult], dest_root: Path) -> None:
    """
    _write_missing_log writes the entirety of the failed download results
    to a missing_date.log in the directory that the files were downloaded into

    Inputs:
        results (list[DownloadResult]): list of all download attempts and their status
        dest_root (Path): Path to place missing_dates.log
    Output:
        None
    """
    problems = [r for r in results if r.status in ('missing', 'error', 'verify_failed')]
    if not problems:
        print('\nNo missing dates or errors — backlog import complete.')  # noqa: T201
        return

    log_path = dest_root / 'missing_dates.log'
    with Path.open(log_path, 'w') as f:
        for r in problems:
            f.write(f'{r.status}\t{r.url}\t{r.detail}\n')

    print(  # noqa: T201
        f'\n{len(problems)} file(s) missing or failed. See {log_path} for details.',
    )


@click.argument('date', type=click.DateTime(formats=['%Y-%m-%d']), required=True)
@click.option('--source', '-s', type=str, multiple=True)
@click.option('--retries', '-r', type=int)
def download_dates(
    date: datetime,
    sources: list[str] | None = None,
    retries: int = DEFAULT_RETRY_ATTEMPTS,
) -> None:
    if sources is None:
        sources = ['snodas', 'instarr', 'swann']

    for source in sources:
        download_url = BASE_URLS[source]
        match source:
            case 'snodas':
                download_url.format()
            case 'swann':
                download_url.format()
            case 'instarr':
                download_url.format()

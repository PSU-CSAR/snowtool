"""
backlog_import.py

Crawl SWANN (University of Arizona) and INSTARR (CU Boulder SPIReS)
servers to download the full historical backlog of daily SWE data,
up to (but not including) today.

Both sources use predictable, constructible URL patterns, so no
directory listing or HTML crawling is required.

Usage
-----
    # Download everything from each source's earliest available date
    # through yesterday
    python backlog_import.py swann --start 1981-10-01  --dest /d/projects/gisdata/swann/unprocessed
    python backlog_import.py instarr --start 2001-01-01 --dest /d/projects/gisdata/instarr/unprocessed

    # Resume a partially-completed run (already-downloaded files are skipped)
    python backlog_import.py swann --start 1981-10-01 --dest /d/projects/gisdata/swann/unprocessed

    # Limit request rate to prevent the sources from blacklisting us
    python backlog_import.py instarr --start 2001-01-01 --dest /d/projects/gisdata/instarr --delay 1.0

Output layout
-------------
SWANN:
    {dest}/{year}/{month}/UA_SWE_Depth_800m_v1_{YYYYMMDD}_{qualifier}.nc

INSTARR (grouped by date so completeness of a tile-set is easy to check):
    {dest}/{YYYYMMDD}/SPIRES_NRT_{tile}_MOD09GA061_{YYYYMMDD}_V1.0.nc

A summary log of genuinely missing dates (404s, not just skipped/cached
files) is written to {dest}/missing_dates.log at the end of the run.

Dependencies: requests, netCDF4
"""  # noqa: E501

from __future__ import annotations

import shutil
import sqlite3
import time
import urllib.error
import urllib.request

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import requests

try:
    import netCDF4  # type: ignore[import]
except ImportError:
    netCDF4 = None  # verification step degrades gracefully if unavailable  # noqa: N816


# Constants
SWANN_BASE_URL: str = 'https://climate.arizona.edu/data/UA_SWE/DailyData_800m'
INSTARR_BASE_URL: str = (
    'ftp://dtn.rc.colorado.edu/shares/snow-today/gridded_data/SPIRES_{qualifier}_V01'
)

# The 5 MODIS tiles INSTARR publishes, matching snodas-download.bash
INSTARR_TILES: tuple[str, ...] = ('h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v04')

# Sqlite file to track record of ingests
# TODO: Pull ledger path from snowtool settings
INGEST_RECORD_DB: Path = Path('/d/projects/gisdata/ingest_record.db')

"""
Per the UA readme: current month = early, 1-6 months back = provisional,
> 6 months back = stable
"""
SWANN_PROVISIONAL_WINDOW_DAYS: int = 186  # ~6 months

DEFAULT_REQUEST_DELAY_SECONDS: float = 0.5
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
def backlog() -> None:
    """Backlog import commands"""


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


def _swann_qualifier_for_date(d: date, today: date) -> str:
    """
    _swann_qualifier_for_date will determine which SWANN qualifier
    to request for a given date, per the UA readme's documented schedule.
    For backfill purposes, we treat 'early' as unavailable beyond the current month.

    Inputs:
        d: date to check against
        today: Todays date to compare with
    Outputs:
        Qualifier based on time difference between files
    """
    age_days = (today - d).days
    if age_days < 0:
        raise ValueError(f'Cannot backfill a future date: {d}')
    if age_days <= 31:
        return 'early'
    if age_days <= SWANN_PROVISIONAL_WINDOW_DAYS:
        return 'provisional'
    return 'stable'


def _daterange(start: date, end: date):
    """
    _daterange yields each date from start to end, inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------


@dataclass
class DownloadResult:
    url: str
    dest: Path
    status: str  # "downloaded", "skipped_exists", "missing", "error", "verify_failed"
    detail: str = ''


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


# SWANN backlog
@backlog.command('swann')
@click.argument('dest_root', type=click.Path(exists=True, path_type=Path))
@click.option('--start', type=click.DateTime(formats=['%Y-%m-%d']), required=True)
@click.option('--end', type=click.DateTime(formats=['%Y-%m-%d']))
@click.option(
    '--delay',
    type=float,
    default=DEFAULT_REQUEST_DELAY_SECONDS,
    show_default=True,
)
def run_swann_backlog(
    dest_root: Path,
    delay: float,
    start: datetime,
    end: datetime,
) -> None:
    """
    run_swann_backlog imports the backlog of data
    from the SWANN HTTP server into the desired location

    SWANN has three different versions of data: early, provisional
    and stable. Most backlog data will be stable, but early/provisional
    versions will be catalogued for later attempts to import the newer iteration

    Inputs:
        dest_root (Path): Base root path of INSTARR unprocessed data
        delay (float): Grace period between download attempts to avoid throttling
        start (datetime): Start of desired date range to import
        end (datetime): End of desired date range to import  Will default to today)
    Output:
        None
    """
    results: list[DownloadResult] = []
    session = requests.Session()
    today = date.today()  # noqa: DTZ011

    start_date = start.date()
    end_date = end.date() if end else date.today() - timedelta(days=1)  # noqa: DTZ011

    for d in _daterange(start_date, end_date):
        qualifier = _swann_qualifier_for_date(d, today)
        wy = _water_year(d)
        filename = f'UA_SWE_Depth_800m_v1_{d:%Y%m%d}_{qualifier}.nc'
        url = f'{SWANN_BASE_URL}/WY{wy}/{filename}'
        out_path = dest_root / f'{d.year}' / f'{d.month:02d}' / filename

        result = _download_file(url, out_path, session)

        if result.status == 'downloaded' and not _verify_netcdf(result.dest):
            result.dest.unlink(missing_ok=True)
            result = DownloadResult(
                url,
                out_path,
                'verify_failed',
                'failed to open as NetCDF',
            )

        results.append(result)
        _log_result('SWANN', d, result)
        _write_to_record('swann', filename, d, result, INGEST_RECORD_DB, qualifier)

        if result.status == 'downloaded':
            time.sleep(delay)
    _write_missing_log(results, dest_root)


# INSTARR backlog
@backlog.command('instarr')
@click.argument('dest_root', type=click.Path(exists=True, path_type=Path))
@click.option('--start', type=click.DateTime(formats=['%Y-%m-%d']), required=True)
@click.option('--end', type=click.DateTime(formats=['%Y-%m-%d']))
def run_instarr_backlog(
    dest_root: Path,
    delay: float,
    start: datetime,
    end: datetime,
) -> None:
    """
    run_instarr_backlog imports the backlog of data
    from the INSTARR ftp server into the desired location

    INSTARR is separated into two verions: NRT and HIST. function
    iterates through date range to figure out which to import

    Inputs:
        dest_root (Path): Base root path of INSTARR unprocessed data
        start (datetime): Start of desired date range to import
        end (datetime): End of desired date range to import  Will default to today)
    Output:
        None
    """
    results: list[DownloadResult] = []

    start_date = start.date()
    end_date = end.date() if end else date.today() - timedelta(days=1)  # noqa: DTZ011

    for d in _daterange(start_date, end_date):
        qualifier = 'HIST' if d < date(2025, 10, 1) else 'NRT'

        for tile in INSTARR_TILES:
            dest_dir = dest_root / tile / f'{d.year}' / f'{d.month:02d}'
            filename = f'SPIRES_{qualifier}_{tile}_MOD09GA061_{d:%Y%m%d}_V1.0.nc'
            dest_filename = f'SPIRES_NRT_{tile}_MOD09GA061_{d:%Y%m%d}_V1.0.nc'
            url = f'{INSTARR_BASE_URL.format(qualifier=qualifier)}/{tile}/{d.year}/{filename}'  # noqa: E501
            out_path = dest_dir / dest_filename
            result = _download_file_ftp(url, out_path)

            if result.status == 'downloaded' and not _verify_netcdf(result.dest):
                result.dest.unlink(missing_ok=True)
                result = DownloadResult(
                    url,
                    out_path,
                    'verify_failed',
                    'failed to open as NetCDF',
                )
            results.append(result)
            _log_result(f'INSTARR/{tile}', d, result)
            _write_to_record('instarr', filename, d, result, INGEST_RECORD_DB)
            if result.status == 'downloaded':
                time.sleep(delay)
    _write_missing_log(results, dest_root)


def _log_result(source_label: str, d: date, result: DownloadResult) -> None:
    """
    _log_result outputs results of importing to stdout

    Inputs:
        source_label (str): Data Source (swann | instarr)
        d (date): Date of the data that is being imported
        result (DownloadResult): Results from the download attempt of the data
    Outputs:
        None
    """
    if result.status == 'downloaded':
        print(f'[{source_label}] {d}  downloaded')  # noqa: T201
    elif result.status == 'skipped_exists':
        print(f'[{source_label}] {d}  already present, skipped')  # noqa: T201
    elif result.status == 'missing':
        print(f'[{source_label}] {d}  NOT FOUND on server (404)')  # noqa: T201
    else:
        print(f'[{source_label}] {d}  ERROR: {result.detail}')  # noqa: T201


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

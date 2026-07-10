import sqlite3

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from snowtool.exceptions import LedgerError


@dataclass
class DownloadResult:
    url: str
    dest: Path
    status: str  # "downloaded", "missing", "error", "verify_failed"
    detail: str = ''


class Ledger:
    BASE_LEDGER_PATH: ClassVar[Path] = Path('/d/projects/gisdata/reimport_ledger.db')

    def __init__(self) -> None:
        self._ensure_ledger()

    @classmethod
    def _ensure_ledger(cls) -> None:
        """
        _ensure_ledger makes sure the sqlite3 file used to track missing
        or failed imports exists

        Inputs:
            conn (sqlite3.Connection): Connection to the ledger file
        Outputs:
            None
        """
        try:
            with sqlite3.connect(cls.BASE_LEDGER_PATH) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ledger (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        source      TEXT    NOT NULL,
                        url         TEXT    UNIQUE   NOT NULL,
                        dest        TEXT    NOT NULL,
                        qualifier   TEXT    DEFAULT '',
                        status      TEXT    NOT NULL,
                        detail      TEXT    NOT NULL DEFAULT '',
                        attempts    INTEGER NOT NULL DEFAULT 1,
                        last_attempt   TEXT   NOT NULL DEFAULT (datetime('now'))
                    )
                """,
                )
        except sqlite3.Error as e:
            raise LedgerError from e
        finally:
            conn.close()

    @classmethod
    def _write_to_record(
        cls,
        source: str,
        result: DownloadResult,
        qualifier: str = '',
    ) -> None:
        """
        _write_to_record Writes the results of a failed (or early) download attempt
        for a data file to the ledger database

        Args:
            - source        (str):    Data source (swann | instarr)
            - result(DownloadResult): Results from the download attempt of the file
            - qualifier     (str):    Used for SWANN to identify early/provisional files
                                      to track for their next iteration. Defaults to "".

        """
        with sqlite3.connect(cls.BASE_LEDGER_PATH) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO ledger (source, url, dest, qualifier, status, detail)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(url)
                    DO UPDATE SET
                        attempts = attempts + 1,
                        last_attempt = (datetime('now'))
                    """,
                    (
                        source,
                        result.url,
                        result.dest,
                        qualifier,
                        result.status,
                        result.detail,
                    ),
                )
                conn.commit()
            except sqlite3.Error as e:
                raise LedgerError from e
            finally:
                conn.close()

    @classmethod
    def _get_records(
        cls,
        source: str,
        update: bool = False,
    ) -> list:
        """
        _get_records will retrieve records
        from the ledger

        Args:
            source (str): Data Source to isolate
            update (bool): Used for updating SWANN datasets with the
                           more stable version of the record

        Raises:
            APIError: _description_
        """

        if source == 'swann' and update:
            stmt = """
            SELECT url, dest, qualifier from LEDGER
            WHERE source = 'swann'
            AND qualifier IN ('early', 'provisional')
            """
        else:
            stmt = """
            SELECT url, dest from LEDGER
            WHERE status IN ('missing','error')
            AND attempts < 5
            """

        with sqlite3.connect(cls.BASE_LEDGER_PATH) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(stmt)
                return cursor.fetchall()
            except sqlite3.Error as e:
                raise LedgerError from e
            finally:
                conn.close()

    @classmethod
    def _clean_records(cls) -> None:
        with sqlite3.connect(cls.BASE_LEDGER_PATH) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM ledger
                    WHERE attempts >= 5
                    """,
                )
                conn.commit()
            except sqlite3.Error as e:
                raise LedgerError from e
            finally:
                conn.close()

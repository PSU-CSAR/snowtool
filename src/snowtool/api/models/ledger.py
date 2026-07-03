import sqlite3

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Ledger:
    BASE_LEDGER_PATH: Path = Path('/d/projects/gisdata/reimport_ledger.db')

    @staticmethod
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
                date        TEXT    NOT NULL,
                qualifier   TEXT    DEFAULT '',
                tile        TEXT    DEFAULT '',
                status      TEXT    NOT NULL,
                detail      TEXT    NOT NULL DEFAULT '',
                attempts    INTEGER NOT NULL DEFAULT 1,
                last_attempt   TEXT   NOT NULL DEFAULT (datetime('now'))
            )
        """,
        )

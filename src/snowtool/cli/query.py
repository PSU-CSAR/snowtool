"""The ``query`` command group: date listings.

``query dates`` lists a dataset's ingested dates. It is a read, so it takes
:func:`pass_snowdb` and tolerates an un-initialized root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from snowtool.cli._context import config_option, pass_snowdb
from snowtool.cli._datasets import format_option, get_dataset
from snowtool.cli._render import DATE, _emit

if TYPE_CHECKING:
    from datetime import date

    from snowtool.snowdb.db import SnowDb


@click.group()
def query() -> None:
    """Date listings over a snowdb."""


@query.command('dates')
@click.option(
    '--dataset',
    '-d',
    'name',
    required=True,
    help='The dataset whose ingested dates to list (exactly one).',
)
@click.option('--start', type=DATE, default=None, help='Only dates on/after this.')
@click.option('--end', type=DATE, default=None, help='Only dates on/before this.')
@format_option
@config_option
@pass_snowdb
def dates(
    snowdb: SnowDb,
    name: str,
    start: date | None,
    end: date | None,
    fmt: str,
) -> None:
    """List a dataset's ingested dates (optionally within a range)."""
    dataset = get_dataset(snowdb, name)
    rows = [
        {'date': d.isoformat()}
        for d in dataset.available_dates()
        if (start is None or d >= start) and (end is None or d <= end)
    ]
    _emit(rows, fmt)

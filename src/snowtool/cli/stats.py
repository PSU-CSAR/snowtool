"""The top-level ``stats`` command: crossed zonal statistics, the analyst surface.

A thin shell over :meth:`SnowDbReader.zonal_stats` -- the shared read seam that
guards coverage, loads the burned AOI raster, and runs the crossed-zone
reduction. One dataset per invocation (each has its own variables/grid and a
differently-shaped output); whole-basin by default, ``--zone`` adds
stratification axes. ``--dates`` speaks the API's OGC ``datetime`` interval so
the two query surfaces share one syntax (see :mod:`snowtool.cli._dates`).
"""

from __future__ import annotations

import asyncio
import io
import json

from typing import TYPE_CHECKING

import click

from snowtool.cli import _console
from snowtool.cli._context import config_option, pass_snowdb
from snowtool.cli._datasets import nested_format_option
from snowtool.cli._dates import parse_dates_query
from snowtool.snowdb.reader import SnowDbReader

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb


@click.command('stats')
@click.argument('dataset_name', metavar='DATASET')
@click.argument('triplet')
@click.option(
    '--dates',
    default=None,
    help='OGC interval (2024-01-01/2024-06-30; .. for an open end), a single '
    'date, or MM-DD with --years. Default: every ingested date.',
)
@click.option(
    '--years',
    default=None,
    help="Year span for a month-day --dates: 'YYYY' or 'YYYY..YYYY'.",
)
@click.option(
    '--zone',
    'zones',
    multiple=True,
    help=(
        'Stratify by a zone layer (repeatable; default: whole basin). '
        'LAYER[:PARAM=VALUE], e.g. terrain.elevation:band_step_ft=500 or '
        'landcover.forest_cover:threshold_pct=40.'
    ),
)
@click.option(
    '--variable',
    'variables',
    multiple=True,
    help='Variable to report (repeatable; default: all of the dataset).',
)
@click.option(
    '--allow-partial',
    is_flag=True,
    default=False,
    help='Permit a clipped result over an AOI the grid only partially covers.',
)
@click.option(
    '--include-empty-zones',
    is_flag=True,
    default=False,
    help=(
        'Include crossed zones no AOI pixel falls in (0 area, null stats). '
        'By default these are dropped; the output otherwise grows '
        'combinatorically with the number of --zone axes.'
    ),
)
@nested_format_option
@config_option
@pass_snowdb
def stats(
    snowdb: SnowDb,
    dataset_name: str,
    triplet: str,
    dates: str | None,
    years: str | None,
    zones: tuple[str, ...],
    variables: tuple[str, ...],
    allow_partial: bool,
    include_empty_zones: bool,
    fmt: str,
) -> None:
    """Zonal statistics for pourpoint TRIPLET over DATASET (whole-basin by default).

    \b
    Examples:
      snowtool stats snodas 13120:CO:SNTL --dates 2024-01-01/2024-06-30
      snowtool stats snodas 13120:CO:SNTL --dates 04-01 --years 2018..2024 \\
          --zone terrain.elevation --format json
    """
    date_query = parse_dates_query(dates, years)

    reader = SnowDbReader(snowdb)
    with _console.err().status(f'querying {dataset_name} for {triplet}...'):
        # A fresh reader used once in a CLI process that has no prior event loop,
        # so drive the async query directly.
        result = asyncio.run(
            reader.zonal_stats(
                triplet,
                dataset_name,
                date_query,
                variable_keys=variables or None,
                zones=zones,
                allow_partial=allow_partial,
            ),
        )

    if fmt == 'json':
        # Always pretty-printed: this is the human-facing surface, and it is
        # deterministic regardless of whether stdout is a terminal or a pipe. Pipe
        # through ``jq -c`` (or similar) for a compact/minified rendering. (The API
        # serves minified JSON, which is the right default for a machine client.)
        click.echo(
            json.dumps(
                result.dump_compact(
                    include_empty_zones=include_empty_zones,
                ).model_dump(mode='json'),
                indent=2,
            ),
        )
        return
    buffer = io.StringIO()
    result.dump_to_csv(buffer, include_empty_zones=include_empty_zones)
    click.echo(buffer.getvalue(), nl=False)

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
import sys

from typing import TYPE_CHECKING

import click

from snowtool.cli import _console
from snowtool.cli._context import config_option, pass_snowdb
from snowtool.cli._datasets import get_dataset, nested_format_option
from snowtool.cli._dates import parse_dates_query
from snowtool.snowdb.zonal_stats import parse_zone_selection
from snowtool.snowdb.zones.zone_layer import available_zones

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.zonal_stats import ZonalStats


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
    dataset = get_dataset(snowdb, dataset_name, include_inactive=False)
    date_query = parse_dates_query(dates, years)

    registry = available_zones(dataset.providers.values())
    # The CLI's ``LAYER[:PARAM=VALUE]`` string tokens and the HTTP API's ``zone``
    # query params both parse through :func:`parse_zone_selection` into the same
    # ``list[ZoneSelection]`` (see ``api.routers.stats._run``); only the input
    # shape differs.
    selections = [parse_zone_selection(token, registry) for token in zones]

    async def run() -> ZonalStats:
        # Build the reader inside the loop that will use it: the cache it owns is
        # loop-affine (alru_cache binds to the loop that first awaits it).
        from snowtool.snowdb.reader import SnowDbReader

        reader = SnowDbReader(snowdb)
        return await reader.zonal_stats(
            triplet,
            dataset_name,
            date_query,
            variable_keys=variables or None,
            zone_selections=selections,
            allow_partial=allow_partial,
        )

    with _console.err().status(f'querying {dataset_name} for {triplet}...'):
        result = asyncio.run(run())

    if fmt == 'json':
        payload = result.dump_compact(
            include_empty_zones=include_empty_zones,
        ).model_dump(mode='json')
        # Pretty-print for a human at a terminal; minify when piped/redirected
        # (jq/git convention). CliRunner and pipelines are non-TTY -> minified.
        if sys.stdout.isatty():
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(json.dumps(payload, separators=(',', ':')))
        return
    buffer = io.StringIO()
    result.dump_to_csv(buffer, include_empty_zones=include_empty_zones)
    click.echo(buffer.getvalue(), nl=False)

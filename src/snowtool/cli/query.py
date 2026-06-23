"""The ``query`` command group: zonal statistics and date listings.

``query stats`` is the analyst surface over :meth:`SnowDb.zonal_stats` -- the
shared read seam that guards coverage, loads the burned AOI raster, and runs the
crossed-zone reduction. It operates on **one** dataset per invocation (each
dataset has its own variables/grid and a differently-shaped output) and defaults
to a whole-basin reduction; ``--zone`` adds one or more stratification axes.
``query dates`` lists a dataset's ingested dates. Both are reads, so they take
:func:`pass_snowdb` and tolerate an un-initialized root.
"""

from __future__ import annotations

import asyncio
import io
import json

from typing import TYPE_CHECKING

import click

from pydantic import ValidationError

from snowtool import types
from snowtool.cli._context import pass_snowdb
from snowtool.cli._datasets import format_option, get_dataset
from snowtool.cli._render import DATE, _emit
from snowtool.exceptions import SNODASError
from snowtool.snowdb.zonal_stats import parse_zone_selection
from snowtool.snowdb.zone_layer import available_zones

if TYPE_CHECKING:
    from datetime import date

    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.zonal_stats import ZonalStats


@click.group()
def query() -> None:
    """Zonal statistics and date listings over a snowdb."""


def _build_query(
    start: date | None,
    end: date | None,
    doy: tuple[int, int] | None,
    start_year: int | None,
    end_year: int | None,
) -> types.DateRangeQuery | types.DOYQuery:
    """Build a date-range or day-of-year query from the mutually-exclusive flags.

    Exactly one mode must be fully specified: a ``--start``/``--end`` range, or a
    ``--doy MONTH DAY`` with ``--start-year``/``--end-year``. Anything else is a
    clean usage error.
    """
    range_given = start is not None or end is not None
    doy_given = doy is not None or start_year is not None or end_year is not None

    if range_given and doy_given:
        raise click.ClickException(
            'Use either --start/--end or --doy/--start-year/--end-year, not both.',
        )

    if range_given:
        if start is None or end is None:
            raise click.ClickException('A date range needs both --start and --end.')
        if end < start:
            raise click.ClickException('--end must be on or after --start.')
        return types.DateRangeQuery(start_date=start, end_date=end)

    if doy_given:
        if doy is None or start_year is None or end_year is None:
            raise click.ClickException(
                'A day-of-year query needs --doy MONTH DAY, --start-year, and '
                '--end-year.',
            )
        if end_year < start_year:
            raise click.ClickException('--end-year must be on or after --start-year.')
        month, day = doy
        try:
            return types.DOYQuery(
                month=month,
                day=day,
                start_year=start_year,
                end_year=end_year,
            )
        except ValidationError as e:
            raise click.ClickException(f'Invalid day of year: {e}') from e

    raise click.ClickException(
        'Provide a date range (--start/--end) or a day-of-year query '
        '(--doy MONTH DAY --start-year --end-year).',
    )


@query.command('stats')
@click.argument('triplet')
@click.option(
    '--dataset',
    '-d',
    'name',
    required=True,
    help='The dataset to query (exactly one; each has its own variables/grid).',
)
@click.option('--start', type=DATE, default=None, help='Range start (inclusive).')
@click.option('--end', type=DATE, default=None, help='Range end (inclusive).')
@click.option(
    '--doy',
    type=int,
    nargs=2,
    default=None,
    help='Day-of-year query: MONTH DAY (with --start-year/--end-year).',
)
@click.option('--start-year', type=int, default=None, help='Day-of-year start year.')
@click.option('--end-year', type=int, default=None, help='Day-of-year end year.')
@click.option(
    '--zone',
    'zones',
    multiple=True,
    help=(
        'Stratify by a zone layer (repeatable; default: whole basin). '
        'LAYER[:override], e.g. terrain.elevation:500 or landcover.forest_cover:40.'
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
    '--format',
    'fmt',
    type=click.Choice(('csv', 'json')),
    default='csv',
    help='Output format (the zonal output is nested, so no table form).',
)
@pass_snowdb
def stats(
    snowdb: SnowDb,
    triplet: str,
    name: str,
    start: date | None,
    end: date | None,
    doy: tuple[int, int] | None,
    start_year: int | None,
    end_year: int | None,
    zones: tuple[str, ...],
    variables: tuple[str, ...],
    allow_partial: bool,
    fmt: str,
) -> None:
    """Zonal statistics for AOI TRIPLET over a dataset (whole-basin by default)."""
    dataset = get_dataset(snowdb, name)
    date_query = _build_query(start, end, doy, start_year, end_year)

    registry = available_zones(dataset.providers.values())
    try:
        selections = [parse_zone_selection(token, registry) for token in zones]
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    async def run() -> ZonalStats:
        return await snowdb.zonal_stats(
            triplet,
            name,
            date_query,
            variable_keys=variables or None,
            zone_selections=selections,
            allow_partial=allow_partial,
        )

    try:
        result = asyncio.run(run())
    except (FileNotFoundError, ValueError, SNODASError) as e:
        raise click.ClickException(str(e)) from e

    if fmt == 'json':
        click.echo(
            json.dumps(
                [model.model_dump(mode='json') for model in result.dump()],
                indent=2,
            ),
        )
        return
    buffer = io.StringIO()
    result.dump_to_csv(buffer)
    click.echo(buffer.getvalue(), nl=False)


@query.command('dates')
@click.argument('name')
@click.option('--start', type=DATE, default=None, help='Only dates on/after this.')
@click.option('--end', type=DATE, default=None, help='Only dates on/before this.')
@format_option
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

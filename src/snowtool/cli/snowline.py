from __future__ import annotations

import asyncio
import json

from typing import TYPE_CHECKING

import click

from snowtool.cli import _console
from snowtool.cli._context import config_option, pass_snowdb
from snowtool.cli._dates import parse_dates_query
from snowtool.snowdb.reader import SnowDbReader
from snowtool.snowdb.snowline import snow_line_elevation

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb


@click.command('snowline')
@click.argument('dataset_name', metavar='DATASET')
@click.argument('triplet')
@click.option(
    '--dates',
    default=None,
    help='OGC interval, single date, or MM-DD with --years.',
)
@click.option('--years', default=None)
@click.option(
    '--band-step-ft',
    default=500,
    show_default=True,
    help='Elevation band width; smaller gives finer snow line resolution.',
)
@click.option(
    '--threshold',
    default=50.0,
    show_default=True,
    help='Snow fraction percent defining the snow line.',
)
@click.option(
    '--allow-partial',
    is_flag=True,
    default=False,
    help='Permit a clipped result over an AOI the grid only partially covers.',
)
@click.option('--variable', default='snow_fraction', show_default=True)
@config_option
@pass_snowdb
def snowline(
    snowdb: SnowDb,
    dataset_name: str,
    triplet: str,
    dates: str | None,
    years: str | None,
    band_step_ft: int,
    threshold: float,
    allow_partial: bool,
    variable: str,
) -> None:
    date_query = parse_dates_query(dates, years)
    reader = SnowDbReader(snowdb)

    with _console.err().status(f'querying {dataset_name} for {triplet}...'):
        result = asyncio.run(
            reader.zonal_stats(
                triplet,
                dataset_name,
                date_query,
                variable_keys=(variable,),
                allow_partial=allow_partial,
                zones=(f'terrain.elevation:band_step_ft={band_step_ft}',),
            ),
        )

    stats = result.dump_compact(include_empty_zones=False)
    lines = snow_line_elevation(stats, threshold=threshold)
    click.echo(
        json.dumps(
            {d.isoformat(): (v.model_dump() if v else None) for d, v in lines.items()},
            indent=2,
        ),
    )

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import CliContext, config_option, pass_snowdb
from snowtool.cli._render import FORMATS, _emit

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb


@click.group()
def snowdb() -> None:
    """Snow-database management commands."""


@snowdb.command('status')
@click.option(
    '--format',
    'fmt',
    type=click.Choice(FORMATS),
    default='table',
    help='Output format.',
)
@config_option
@pass_snowdb
def snowdb_status(snowdb: SnowDb, fmt: str) -> None:
    """Overview of every registered dataset: active flag, artifacts, date span."""
    from snowtool.snowdb.diagnostics import dataset_status

    rows = []
    for name in sorted(snowdb.registered):
        status = dataset_status(snowdb.registered[name])
        artifacts = status.artifacts
        row = {
            'dataset': status.name,
            'active': name in snowdb.datasets,
            'present': status.present,
        }
        # One column per configured zone-layer provider (terrain, landcover, ...).
        for provider_name, present in sorted(artifacts.zone_layers.items()):
            row[provider_name] = present
        row.update(
            {
                'cogs': artifacts.cogs,
                'aoi_rasters': artifacts.aoi_rasters,
                'dates': status.date_count,
                'first': status.first_date.isoformat() if status.first_date else '',
                'last': status.last_date.isoformat() if status.last_date else '',
            },
        )
        rows.append(row)
    _emit(rows, fmt)


@snowdb.command('init')
@click.argument(
    'path',
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@config_option
@click.pass_obj
def snowdb_init(cli_ctx: CliContext, path: Path | None) -> None:
    """Create an empty snowdb at PATH (or the ``--config`` / env-var root).

    Lays out the root config (``snowdb_conf.json``), ``pourpoints/``, and
    ``data/``. No datasets are registered: ``dataset create`` stages one (area
    raster + zone layers) and registers it inactive; ``dataset activate`` makes
    it live. Idempotent -- an existing root config is left untouched.
    """
    from snowtool.snowdb.manager import SnowDbManager

    if path is not None:
        root = path
    elif cli_ctx.config is not None:
        root = cli_ctx.config
    else:
        raise click.ClickException(
            'No snowdb location. Pass PATH, --config/-C, or set '
            'SNOWTOOL_SNOWDB_CONFIG.',
        )

    SnowDbManager.initialize(
        root,
        zone_layer_providers=cli_ctx.zone_layer_providers,
    )
    click.echo(f'initialized snowdb: {root}')

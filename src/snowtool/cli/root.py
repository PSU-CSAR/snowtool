"""Root-level snow-database commands: ``init`` and ``status``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import CliContext, config_option, pass_snowdb
from snowtool.cli._render import emit, format_option
from snowtool.snowdb.diagnostics import dataset_status
from snowtool.snowdb.manager import SnowDbManager

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb


@click.command('status')
@format_option
@config_option
@pass_snowdb
def status(snowdb: SnowDb, fmt: str) -> None:
    """Overview of every registered dataset: active flag, artifacts, date span."""
    rows = [
        dataset_status(snowdb.registered[name]).to_row(active=name in snowdb.datasets)
        for name in sorted(snowdb.registered)
    ]
    emit(rows, fmt)


@click.command('init')
@click.argument(
    'path',
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@config_option
@click.pass_obj
def init_snowdb(cli_ctx: CliContext, path: Path | None) -> None:
    """Create an empty snowdb at PATH (or the ``--config`` / env-var root).

    Lays out the root config (``snowdb_conf.json``), ``pourpoints/``, and
    ``data/``. No datasets are registered: ``dataset create`` stages one (area
    raster + zone layers) and registers it inactive; ``dataset activate`` makes
    it live. Idempotent -- an existing root config is left untouched.
    """
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

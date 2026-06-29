"""The ``snowtool`` CLI: a thin shell over the snowdb Python API.

The root ``cli`` group exposes ``--config`` and seeds a :class:`CliContext` on
``ctx.obj``; subcommand groups live in sibling modules and are registered here.
Command bodies stay thin -- they resolve a SnowDb (via
:func:`snowtool.cli._context.pass_snowdb`), call a domain method, and render with
:func:`snowtool.cli._render._emit`. New logic belongs on ``SnowDb``/``Dataset``
or in ``snowdb/diagnostics.py``, not in click callbacks.
"""

from pathlib import Path

import click

from snowtool.cli._context import CliContext
from snowtool.cli.dataset import dataset
from snowtool.cli.migration import migration
from snowtool.cli.pourpoint import pourpoint
from snowtool.cli.query import query
from snowtool.cli.report import report
from snowtool.cli.snowdb import snowdb
from snowtool.cli.version import version


@click.group()
@click.option(
    '--config',
    '-C',
    type=click.Path(path_type=Path),
    default=None,
    help='Snowdb config file or its directory '
    '(defaults to the SNOWTOOL_SNOWDB_CONFIG setting).',
)
@click.pass_context
def cli(ctx: click.Context, config: Path | None) -> None:
    # Honor a CliContext already placed on ctx.obj (tests inject one carrying
    # synthetic specs); otherwise build one from --config for the normal entrypoint.
    if not isinstance(ctx.obj, CliContext):
        ctx.obj = CliContext(config=config)


cli.add_command(version)
cli.add_command(snowdb)
cli.add_command(dataset)
cli.add_command(pourpoint)
cli.add_command(report)
cli.add_command(query)
cli.add_command(migration)

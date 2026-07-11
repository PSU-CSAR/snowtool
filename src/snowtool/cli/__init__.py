"""The ``snowtool`` CLI: a thin shell over the snowdb Python API.

The root ``cli`` group seeds a :class:`CliContext` on ``ctx.obj``; subcommand
groups live in sibling modules and are registered here. Commands that open a
snowdb take the shared ``config_option`` (``--config``) rather than a root flag,
so a command that needs no database (``api serve``, ``--version``) carries none.
Command bodies stay thin -- they resolve a SnowDb (via
:func:`snowtool.cli._context.pass_snowdb`), call a domain method, and render with
:func:`snowtool.cli._render._emit`. New logic belongs on ``SnowDb``/``Dataset``
or in ``snowdb/diagnostics.py``, not in click callbacks.
"""

import click

from snowtool import __version__
from snowtool.cli._context import CliContext
from snowtool.cli.api import api
from snowtool.cli.dataset import dataset
from snowtool.cli.doctor import doctor
from snowtool.cli.pourpoint import pourpoint
from snowtool.cli.snowdb import init_snowdb, status
from snowtool.cli.stats import stats
from snowtool.cli.windows import windows


@click.group(context_settings={'auto_envvar_prefix': 'SNOWTOOL'})
@click.version_option(__version__, '--version', prog_name='snowtool')
@click.option(
    '--color',
    type=click.Choice(['auto', 'always', 'never']),
    default='auto',
    help='Colorize output (auto: only on a TTY; NO_COLOR is honored).',
)
@click.option(
    '--quiet',
    '-q',
    is_flag=True,
    default=False,
    help='Suppress progress bars and status messages (stderr); data output '
    'is unaffected.',
)
@click.pass_context
def cli(ctx: click.Context, color: str, quiet: bool) -> None:
    from snowtool.cli import _console

    _console.configure(color=color, quiet=quiet)
    # Seed the per-invocation CliContext (unless a test injected one carrying
    # synthetic specs). --config is a per-command option (see config_option), so a
    # command that opens a snowdb sets its config here via that option's callback.
    if not isinstance(ctx.obj, CliContext):
        ctx.obj = CliContext()


cli.add_command(init_snowdb)
cli.add_command(status)
cli.add_command(dataset)
cli.add_command(doctor)
cli.add_command(pourpoint)
cli.add_command(stats)
cli.add_command(api)
cli.add_command(windows)

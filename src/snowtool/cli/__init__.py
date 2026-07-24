"""The ``snowtool`` CLI: a thin shell over the snowdb Python API.

The root ``cli`` group seeds a :class:`CliContext` on ``ctx.obj`` and carries
``--color``/``--quiet`` (each with an env-var fallback via
``auto_envvar_prefix='SNOWTOOL'``); everything else is a command or group
registered here:

- ``init``, ``status``, ``doctor``, ``stats`` -- top-level, single-purpose
  commands (``init``/``status`` manage the snowdb itself; ``doctor`` runs the
  four health checks -- grid, dates, files, pourpoints -- and exits 1 on any
  finding; ``stats DATASET TRIPLET`` is the analyst-facing crossed zonal-stats
  query, taking an OGC ``--dates``/``--years`` interval).
- ``dataset`` -- list/info/dates/values/create/register/activate/deactivate/
  ingest/generate-zones/remove-date.
- ``pourpoint`` -- import/sync/list/show/dump/reindex/remove/rasterize.
- ``api`` -- ``serve``, the read-only HTTP API.
- ``windows`` -- Windows-only admin commands; hidden (not disabled) unless
  running on ``win32``.

Commands that open a snowdb take the shared ``config_option`` (``--config``)
rather than a root flag, so a command that needs no database (``api serve``,
``--version``) carries none. Output goes through a pair of rich consoles in
``_console.py`` -- stdout for data, stderr for progress/status -- so piping a
command's stdout never mixes in status noise; write commands additionally
accept ``--format json`` to emit machine-readable rows instead of prose, and
destructive commands (``dataset remove-date``, ``pourpoint remove``) gate on
``confirm_destructive`` (``_confirm.py``) unless ``--yes`` is passed. Command
bodies stay thin -- they resolve a SnowDb (via
:func:`snowtool.cli._context.pass_snowdb`), call a domain method, and render
with :func:`snowtool.cli._render.emit`. New logic belongs on
``SnowDb``/``Dataset`` or in ``snowdb/diagnostics.py``, not in click callbacks.

Error handling is centralized: every operator-facing domain error is a
:class:`~snowtool.exceptions.SnowtoolError`, and the root group maps that base
to a clean ``ClickException`` for *every* command -- so command bodies carry no
``try/except`` ceremony. A command adds its own ``except`` only to *tailor* a
message (e.g. ``pourpoint import`` pointing a directory SRC at ``sync``);
anything else that escapes is a bug and gets a traceback on purpose.
"""

import click

from snowtool import __version__
from snowtool.cli import _console
from snowtool.cli._context import CliContext
from snowtool.cli.api import api
from snowtool.cli.dataset import dataset
from snowtool.cli.doctor import doctor
from snowtool.cli.pourpoint import pourpoint
from snowtool.cli.root import init_snowdb, status
from snowtool.cli.stats import stats
from snowtool.cli.windows import windows
from snowtool.exceptions import SnowtoolError


class _SnowtoolCli(click.Group):
    """The root group, mapping domain errors to clean CLI errors centrally.

    Every command runs inside this ``invoke``, so any
    :class:`~snowtool.exceptions.SnowtoolError` -- the base of every
    operator-facing domain error -- renders as one clean ``Error: ...`` line
    (exit 1) instead of a traceback. Deliberately *only* that base: a bare
    ``ValueError``/``OSError`` escaping the domain is a bug, and hiding it
    behind a clean message would just bury the evidence.
    """

    def invoke(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        except SnowtoolError as e:
            raise click.ClickException(str(e)) from e


@click.group(cls=_SnowtoolCli, context_settings={'auto_envvar_prefix': 'SNOWTOOL'})
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
    _console.configure(color=color, quiet=quiet)
    # Unless a test injected one carrying synthetic specs.
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

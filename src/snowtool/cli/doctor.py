"""The ``doctor`` command: run health checks, report findings, exit 1 on any.

Each check maps a dataset to uniform finding rows (``check``/``dataset``/
``target``/``issue``) built from :mod:`snowtool.snowdb.diagnostics`, so the
output is one flat, machine-readable table across every check -- empty means
healthy, which is the cron/CI contract (exit 1 otherwise). The four checks:

- ``grid``: declaration vs reality -- an ingester with no variables, or an
  on-disk COG whose shape/transform disagrees with the declared grid.
- ``dates``: ingested dates missing one or more of the dataset's variables.
- ``files``: expected artifacts absent (zone layers, COGs, AOI rasters) or
  zone layers stamped with a stale on-disk format version.
- ``pourpoints``: basins without burned rasters, orphan rasters, partial/zero
  grid coverage, and AOI rasters that won't read cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from snowtool.cli import _console
from snowtool.cli._context import config_option, pass_snowdb
from snowtool.cli._datasets import dataset_option, format_option, resolve_datasets
from snowtool.cli._progress import RichProgress
from snowtool.cli._render import _emit
from snowtool.snowdb import diagnostics

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb


@click.command('doctor')
@click.argument('checks', nargs=-1, metavar='[CHECK]...')
@dataset_option
@click.option(
    '--include-inactive',
    is_flag=True,
    help='Also check registered-but-inactive datasets (default: active only, '
    'so a half-built staged dataset does not fail the cron/CI gate).',
)
@format_option
@config_option
@pass_snowdb
def doctor(
    snowdb: SnowDb,
    checks: tuple[str, ...],
    dataset_names: tuple[str, ...],
    include_inactive: bool,
    fmt: str,
) -> None:
    """Run health checks; print findings and exit 1 if there are any.

    Checks: grid, dates, files, pourpoints (default: all). Empty output
    (``[]`` for json) means healthy. An explicit ``-d`` NAME always resolves
    from everything registered.

    \b
    Examples:
      snowtool doctor
      snowtool doctor files pourpoints -d snodas --format json
    """
    unknown = sorted(set(checks) - set(diagnostics.HEALTH_CHECK_NAMES))
    if unknown:
        raise click.ClickException(
            f'Unknown check(s): {", ".join(unknown)}. '
            f'Known checks: {", ".join(diagnostics.HEALTH_CHECK_NAMES)}.',
        )
    selected = (
        list(dict.fromkeys(checks)) if checks else list(diagnostics.HEALTH_CHECK_NAMES)
    )
    datasets = resolve_datasets(
        snowdb,
        dataset_names,
        include_inactive=include_inactive,
    )

    findings = diagnostics.run_health_checks(
        snowdb,
        datasets,
        selected,
        progress=RichProgress(),
    )

    _emit(findings, fmt)
    if findings:
        _console.err().print(f'[red]{len(findings)} problem(s) found[/red]')
        raise SystemExit(1)
    _console.err().print('[green]ok: no problems found[/green]')

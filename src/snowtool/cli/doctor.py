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
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb

type _Finding = dict[str, str]


def _finding(check: str, dataset: str, target: str, issue: str) -> _Finding:
    return {'check': check, 'dataset': dataset, 'target': target, 'issue': issue}


def _check_grid(snowdb: SnowDb, ds: Dataset) -> list[_Finding]:
    name = ds.spec.name
    return [
        _finding('grid', name, '', issue)
        for issue in diagnostics.grid_validation_report(ds)
    ]


def _check_dates(snowdb: SnowDb, ds: Dataset) -> list[_Finding]:
    name = ds.spec.name
    return [
        _finding(
            'dates',
            name,
            inc.date.isoformat(),
            f'missing {", ".join(inc.missing)}',
        )
        for inc in diagnostics.completeness_report(ds)
    ]


def _check_files(snowdb: SnowDb, ds: Dataset) -> list[_Finding]:
    name = ds.spec.name
    findings = [
        _finding('files', name, artifact, 'missing')
        for artifact in diagnostics.missing_artifacts(ds)
    ]
    findings.extend(
        _finding(
            'files',
            name,
            stale.provider,
            f'stale zone-layer format (stored {stale.stored} != '
            f'current {stale.expected})',
        )
        for stale in diagnostics.stale_format_zone_layers(ds)
    )
    return findings


def _check_pourpoints(snowdb: SnowDb, ds: Dataset) -> list[_Finding]:
    name = ds.spec.name
    coverage = diagnostics.pourpoint_coverage_report(snowdb, ds)
    findings = [
        _finding('pourpoints', name, triplet, issue)
        for issue, triplets in (
            ('no raster', coverage.unrasterized),
            ('orphan raster', coverage.orphan_rasters),
            ('partial coverage', coverage.partial),
            ('no coverage', coverage.uncovered),
        )
        for triplet in triplets
    ]
    for health in diagnostics.aoi_health_report(ds):
        if health.ok:
            continue
        assert health.issue is not None  # noqa: S101 - ok=False always carries an issue
        findings.append(_finding('pourpoints', name, health.triplet, health.issue))
    return findings


CHECKS = {
    'grid': _check_grid,
    'dates': _check_dates,
    'files': _check_files,
    'pourpoints': _check_pourpoints,
}


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
    unknown = sorted(set(checks) - CHECKS.keys())
    if unknown:
        raise click.ClickException(
            f'Unknown check(s): {", ".join(unknown)}. '
            f'Known checks: {", ".join(CHECKS)}.',
        )
    selected = list(dict.fromkeys(checks)) if checks else list(CHECKS)
    datasets = resolve_datasets(
        snowdb,
        dataset_names,
        include_inactive=include_inactive,
    )

    findings: list[_Finding] = []
    with RichProgress().track(
        'doctor',
        total=len(datasets) * len(selected),
    ) as task:
        for ds in datasets:
            for check in selected:
                findings.extend(CHECKS[check](snowdb, ds))
                task.advance()

    _emit(findings, fmt)
    if findings:
        _console.err().print(f'[red]{len(findings)} problem(s) found[/red]')
        raise SystemExit(1)
    _console.err().print('[green]ok: no problems found[/green]')

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import CliContext, pass_snowdb
from snowtool.cli._datasets import dataset_option, resolve_datasets
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
@pass_snowdb
def snowdb_status(snowdb: SnowDb, fmt: str) -> None:
    """Overview of every dataset: artifacts present, date span, and counts."""
    from snowtool.snowdb.diagnostics import dataset_status

    rows = []
    for name in sorted(snowdb):
        status = dataset_status(snowdb[name])
        artifacts = status.artifacts
        row = {'dataset': status.name, 'present': status.present}
        # One column per configured zone-layer provider (terrain, landcover, ...).
        for provider_name, present in sorted(artifacts.zone_layers.items()):
            row[provider_name] = present
        row.update(
            {
                'area': 'n/a' if artifacts.area is None else artifacts.area,
                'cogs': artifacts.cogs,
                'aoi_rasters': artifacts.aoi_rasters,
                'dates': status.date_count,
                'first': status.first_date.isoformat() if status.first_date else '',
                'last': status.last_date.isoformat() if status.last_date else '',
            },
        )
        rows.append(row)
    _emit(rows, fmt)


@snowdb.command('validate')
@dataset_option
@pass_snowdb
def snowdb_validate(snowdb: SnowDb, dataset_names: tuple[str, ...]) -> None:
    """Roll up the read-only health checks; exit non-zero if any problem is found.

    Aggregates completeness, missing-files, aoi-coverage, and aoi-health across
    the selected datasets (default: all). Prints one line per problem and exits 1
    when there are any, so it can gate cron/CI.
    """
    from snowtool.snowdb import diagnostics

    findings: list[str] = []
    for ds in resolve_datasets(snowdb, dataset_names):
        name = ds.spec.name
        for inc in diagnostics.completeness_report(ds):
            missing_vars = ', '.join(inc.missing)
            findings.append(
                f'incomplete: {name} {inc.date.isoformat()} missing {missing_vars}',
            )
        missing = diagnostics.missing_artifacts(ds)
        if missing:
            findings.append(f'missing-files: {name}: {", ".join(missing)}')
        coverage = diagnostics.aoi_coverage_report(snowdb, ds)
        findings.extend(f'aoi-no-raster: {name} {t}' for t in coverage.unrasterized)
        findings.extend(f'aoi-orphan: {name} {t}' for t in coverage.orphan_rasters)
        findings.extend(f'aoi-partial: {name} {t}' for t in coverage.partial)
        findings.extend(f'aoi-uncovered: {name} {t}' for t in coverage.uncovered)
        for health in diagnostics.aoi_health_report(ds):
            if not health.ok:
                findings.append(f'aoi-health: {name} {health.triplet}: {health.issue}')

    for finding in findings:
        click.echo(finding)
    if findings:
        click.echo(f'{len(findings)} problem(s) found')
        raise SystemExit(1)
    click.echo('ok: no problems found')


@snowdb.command('init')
@click.argument(
    'path',
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.pass_obj
def snowdb_init(cli_ctx: CliContext, path: Path | None) -> None:
    """Create an empty snowdb at PATH (defaults to the snowdb_path setting).

    Lays out the root config (``snowdb_conf.json``), ``aois/``, and ``data/``. No
    datasets are registered: a dataset goes live by staging it (``dataset
    create``) and registering it (``dataset create --activate`` or ``dataset
    add``), which is also where area + zone-layer generation happens. Idempotent --
    an existing root config is left untouched.
    """
    from snowtool.settings import Settings
    from snowtool.snowdb.manager import SnowDbManager

    if path is not None:
        root = path
    elif cli_ctx.root is not None:
        root = cli_ctx.root
    else:
        root = Settings().snowdb_path

    SnowDbManager.initialize(
        root,
        zone_layer_providers=cli_ctx.zone_layer_providers,
        zone_layer_sources=cli_ctx.zone_layer_sources,
    )
    click.echo(f'initialized snowdb: {root}')

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import CliContext, pass_snowdb
from snowtool.cli._datasets import dataset_option, get_dataset, resolve_datasets
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
        rows.append(
            {
                'dataset': status.name,
                'present': status.present,
                'terrain': artifacts.terrain,
                'area': 'n/a' if artifacts.area is None else artifacts.area,
                'cogs': artifacts.cogs,
                'aoi_rasters': artifacts.aoi_rasters,
                'dates': status.date_count,
                'first': status.first_date.isoformat() if status.first_date else '',
                'last': status.last_date.isoformat() if status.last_date else '',
            },
        )
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
@click.option(
    '--quick',
    is_flag=True,
    help='Create the directory skeleton only; skip area + terrain generation.',
)
@click.option(
    '--dataset-dem',
    'dataset_dems',
    nargs=2,
    multiple=True,
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    metavar='DATASET PATH',
    help='Generate DATASET terrain from a local DEM file instead of the default '
    'source (repeatable).',
)
@click.pass_obj
def snowdb_init(
    cli_ctx: CliContext,
    path: Path | None,
    quick: bool,
    dataset_dems: tuple[tuple[str, Path], ...],
) -> None:
    """Create the base snowdb layout at PATH (defaults to the snowdb_path setting).

    Lays out ``aois/``, ``data/``, and a ``data/<dataset>/`` directory for every
    configured dataset, then ensures each dataset's area raster (geographic grids
    only) and terrain set exist. Terrain is generated from the database's default
    DEM source in one shared pass, unless ``--dataset-dem`` overrides a dataset
    with a local file. ``--quick`` skips all generation (skeleton only). Idempotent
    -- existing area rasters and terrain sets are left untouched.

    Generation is heavy: it reprojects the whole DEM source to a 10 m work grid
    and keeps per-grid accumulators for every dataset in the shared pass resident
    at once, so a full multi-dataset run over a continental source is memory- and
    bandwidth-intensive (the default 3DEP source streams from S3). Use ``--quick``
    for a fast skeleton, or ``dataset generate`` per dataset to bound the work.
    """
    from snowtool.settings import Settings
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.dem_source import LocalFile

    if path is not None:
        root = path
    elif cli_ctx.root is not None:
        root = cli_ctx.root
    else:
        root = Settings().snowdb_path

    db = SnowDb.initialize(root, cli_ctx.specs, dem_source=cli_ctx.dem_source)
    click.echo(f'initialized snowdb: {root}')

    if quick:
        return

    for name in sorted(db):
        if db[name].ensure_area_raster():
            click.echo(f'built area raster for {name}')

    overrides = {name: Path(p) for name, p in dataset_dems}
    for name in overrides:
        get_dataset(db, name)  # validate before doing expensive work

    # Generate every default-source dataset that lacks terrain in one shared pass.
    default_names = [
        name
        for name in sorted(db)
        if name not in overrides and not db[name].terrain.present()
    ]
    if default_names:
        joined = ', '.join(default_names)
        click.echo(f'generating terrain (default source) for: {joined}')
        db.generate_terrain(default_names, force=True)

    # Overrides are explicit, so (re)generate them from the given file.
    for name, dem_path in overrides.items():
        click.echo(f'generating terrain for {name} from {dem_path}')
        db[name].generate_terrain(LocalFile(dem_path), force=True)

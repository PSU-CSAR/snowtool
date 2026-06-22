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
                'landcover': artifacts.landcover,
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
@click.option(
    '--dataset-nlcd',
    'dataset_nlcds',
    nargs=2,
    multiple=True,
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    metavar='DATASET PATH',
    help='Generate DATASET land cover from a local NLCD file instead of the '
    'default source (repeatable).',
)
@click.pass_obj
def snowdb_init(
    cli_ctx: CliContext,
    path: Path | None,
    quick: bool,
    dataset_dems: tuple[tuple[str, Path], ...],
    dataset_nlcds: tuple[tuple[str, Path], ...],
) -> None:
    """Create the base snowdb layout at PATH (defaults to the snowdb_path setting).

    Lays out ``aois/``, ``data/``, and a ``data/<dataset>/`` directory for every
    configured dataset, then ensures each dataset's area raster (geographic grids
    only), terrain set, and land-cover set exist. Terrain comes from the database's
    default DEM source and land cover from the default NLCD source, each in one
    shared pass, unless ``--dataset-dem`` / ``--dataset-nlcd`` override a dataset
    with a local file. ``--quick`` skips all generation (skeleton only). Idempotent
    -- existing area rasters, terrain sets, and land-cover sets are left untouched.

    Generation is heavy: terrain reprojects the whole DEM source to a 10 m work
    grid (the default 3DEP source streams from S3), and land cover downloads the
    MRLC Annual NLCD national raster (~1.5 GB) on first use, cached under the
    snowdb root. Use ``--quick`` for a fast skeleton, or the per-dataset
    ``dataset generate`` / ``dataset generate-landcover`` commands to bound the work.
    """
    from snowtool.settings import Settings
    from snowtool.snowdb.db import SnowDb

    if path is not None:
        root = path
    elif cli_ctx.root is not None:
        root = cli_ctx.root
    else:
        root = Settings().snowdb_path

    db = SnowDb.initialize(
        root,
        cli_ctx.specs,
        dem_source=cli_ctx.dem_source,
        landcover_source=cli_ctx.landcover_source,
    )
    click.echo(f'initialized snowdb: {root}')

    if quick:
        return

    for name in sorted(db):
        if db[name].ensure_area_raster():
            click.echo(f'built area raster for {name}')

    overrides = {name: Path(p) for name, p in dataset_dems}
    nlcd_overrides = {name: Path(p) for name, p in dataset_nlcds}
    for name in overrides.keys() | nlcd_overrides.keys():
        get_dataset(db, name)  # validate before doing expensive work

    _generate_terrain(db, overrides)
    _generate_landcover(db, nlcd_overrides)


def _generate_terrain(db: SnowDb, overrides: dict[str, Path]) -> None:
    """init's terrain step: a default-source shared pass + per-dataset overrides.

    Generates every default-source dataset that lacks terrain in one pass, then
    (re)generates each ``--dataset-dem`` override from its given local file.
    """
    from snowtool.snowdb.dem_source import LocalFile

    default_names = [
        name
        for name in sorted(db)
        if name not in overrides and not db[name].terrain.present()
    ]
    if default_names:
        joined = ', '.join(default_names)
        click.echo(f'generating terrain (default source) for: {joined}')
        db.generate_terrain(default_names, force=True)

    for name, dem_path in overrides.items():
        click.echo(f'generating terrain for {name} from {dem_path}')
        db[name].generate_terrain(LocalFile(dem_path), force=True)


def _generate_landcover(db: SnowDb, overrides: dict[str, Path]) -> None:
    """init's land-cover step, the same shape as :func:`_generate_terrain`."""
    from snowtool.snowdb.landcover_source import LocalFile

    default_names = [
        name
        for name in sorted(db)
        if name not in overrides and not db[name].landcover.present()
    ]
    if default_names:
        joined = ', '.join(default_names)
        click.echo(f'generating land cover (default source) for: {joined}')
        db.generate_landcover(default_names, force=True)

    for name, nlcd_path in overrides.items():
        click.echo(f'generating land cover for {name} from {nlcd_path}')
        db[name].generate_landcover(LocalFile(nlcd_path), force=True)

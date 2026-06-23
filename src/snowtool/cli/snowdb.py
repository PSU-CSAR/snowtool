from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import CliContext, pass_snowdb
from snowtool.cli._datasets import dataset_option, get_dataset, resolve_datasets
from snowtool.cli._render import FORMATS, _emit

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.zone_layer import ZoneLayerProvider


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
@click.option(
    '--quick',
    is_flag=True,
    help='Create the directory skeleton only; skip area + zone-layer generation.',
)
@click.option(
    '--dataset-source',
    'dataset_sources',
    nargs=3,
    multiple=True,
    type=(str, str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    metavar='PROVIDER DATASET PATH',
    help='Generate PROVIDER zone layers (e.g. terrain, landcover) for DATASET from '
    'a local PATH instead of the default source (repeatable).',
)
@click.option(
    '--workers',
    default=None,
    type=click.IntRange(min=1),
    help='Terrain-generation worker threads (default: one per CPU; 1 = serial). '
    'Block reprojection is parallelized; the result is identical regardless.',
)
@click.option(
    '--block-size',
    default=None,
    type=click.IntRange(min=64),
    help='Terrain work-grid block edge in pixels (default 1024). Lower it to bound '
    'per-worker memory (~workers x block_size^2); no effect on the result.',
)
@click.pass_obj
def snowdb_init(
    cli_ctx: CliContext,
    path: Path | None,
    quick: bool,
    dataset_sources: tuple[tuple[str, str, Path], ...],
    workers: int | None,
    block_size: int | None,
) -> None:
    """Create the base snowdb layout at PATH (defaults to the snowdb_path setting).

    Lays out ``aois/``, ``data/``, and a ``data/<dataset>/`` directory for every
    configured dataset, then ensures each dataset's area raster (geographic grids
    only) and every configured zone layer (terrain, land cover, ...) exist. Each
    zone layer comes from its provider's default source in one shared pass, unless
    ``--dataset-source PROVIDER DATASET PATH`` overrides a dataset with a local
    file. ``--quick`` skips all generation (skeleton only). Idempotent -- existing
    area rasters and zone-layer sets are left untouched.

    Generation is heavy: terrain reprojects the whole DEM source to a 10 m work
    grid (the default 3DEP source streams from S3), and land cover downloads the
    MRLC Annual NLCD national raster (~1.5 GB) on first use, cached under the
    snowdb root. Use ``--quick`` for a fast skeleton, or the per-dataset
    ``dataset generate-zones`` command to bound the work.
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
        zone_layer_providers=cli_ctx.zone_layer_providers,
        zone_layer_sources=cli_ctx.zone_layer_sources,
    )
    click.echo(f'initialized snowdb: {root}')

    if quick:
        return

    for name in sorted(db):
        if db[name].ensure_area_raster():
            click.echo(f'built area raster for {name}')

    # provider -> {dataset: local path}; validate provider + dataset names before
    # any (expensive) generation work.
    overrides: dict[str, dict[str, Path]] = {}
    for provider_name, dataset_name, p in dataset_sources:
        if provider_name not in db.zone_layer_providers:
            raise click.ClickException(f'No such zone-layer provider: {provider_name}')
        get_dataset(db, dataset_name)
        overrides.setdefault(provider_name, {})[dataset_name] = Path(p)

    for provider_name, provider in db.zone_layer_providers.items():
        _generate_zone_layers(
            db, provider, overrides.get(provider_name, {}), workers, block_size,
        )


def _generate_zone_layers(
    db: SnowDb,
    provider: ZoneLayerProvider,
    overrides: dict[str, Path],
    workers: int | None,
    block_size: int | None,
) -> None:
    """init's per-provider step: a default-source shared pass + per-dataset overrides.

    Generates every default-source dataset that lacks this provider's set in one
    pass, then (re)generates each ``--dataset-source`` override from its local file.
    ``workers``/``block_size`` are engine knobs the terrain provider honours and
    others ignore.
    """
    default_names = [
        name
        for name in sorted(db)
        if name not in overrides and not db[name].zones[provider.name].present()
    ]
    if default_names:
        joined = ', '.join(default_names)
        click.echo(f'generating {provider.name} (default source) for: {joined}')
        db.generate_zone_layers(
            provider.name,
            default_names,
            force=True,
            workers=workers,
            block_size=block_size,
        )

    for name, source_path in overrides.items():
        click.echo(f'generating {provider.name} for {name} from {source_path}')
        db[name].generate_zone_layers(
            provider,
            provider.local_source(source_path),
            force=True,
            workers=workers,
            block_size=block_size,
        )

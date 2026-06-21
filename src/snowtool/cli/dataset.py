"""The ``dataset`` command group: per-dataset management.

Thin wrappers over the ``Dataset`` API: each command resolves the dataset,
calls a domain method, and renders. Write commands (create/ingest/set-dem/
rebuild-area/remove-date/prune) first require an initialized snowdb root.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import pass_snowdb
from snowtool.cli._datasets import format_option, get_dataset, require_initialized
from snowtool.cli._render import DATE, _emit, _emit_record
from snowtool.exceptions import SNODASError

if TYPE_CHECKING:
    from datetime import date

    from snowtool.snowdb.db import SnowDb


@click.group()
def dataset() -> None:
    """Dataset management commands."""


@dataset.command('list')
@format_option
@pass_snowdb
def list_datasets(snowdb: SnowDb, fmt: str) -> None:
    """List configured datasets and whether each has data on disk."""
    from snowtool.snowdb.diagnostics import dataset_status

    rows = []
    for name in sorted(snowdb):
        status = dataset_status(snowdb[name])
        rows.append(
            {
                'dataset': status.name,
                'present': status.present,
                'dates': status.date_count,
            },
        )
    _emit(rows, fmt)


@dataset.command('info')
@click.argument('name')
@format_option
@pass_snowdb
def dataset_info(snowdb: SnowDb, name: str, fmt: str) -> None:
    """Show a dataset's grid, variables, and on-disk artifacts."""
    from snowtool.snowdb.constants import MAX_ELEVATION_M, MIN_ELEVATION_M
    from snowtool.snowdb.diagnostics import dataset_status

    ds = get_dataset(snowdb, name)
    spec = ds.spec
    grid = spec.grid_params
    status = dataset_status(ds)
    artifacts = status.artifacts

    record = {
        'name': spec.name,
        'present': status.present,
        'crs': str(grid.crs),
        'is_geographic': spec.is_geographic,
        'rows': grid.rows,
        'cols': grid.cols,
        'tile_size': grid.tile_size,
        'cell_area_m2': 'varies (geographic)' if spec.is_geographic else spec.cell_area,
        'band_step_ft': spec.band_step_ft,
        'elevation_bracket_m': f'{MIN_ELEVATION_M} .. {MAX_ELEVATION_M}',
        'variables': sorted(spec.variables),
        'dem': artifacts.dem,
        'area': 'n/a' if artifacts.area is None else artifacts.area,
        'cogs': artifacts.cogs,
        'aoi_rasters': artifacts.aoi_rasters,
        'dates': status.date_count,
        'first_date': status.first_date.isoformat() if status.first_date else None,
        'last_date': status.last_date.isoformat() if status.last_date else None,
    }
    _emit_record(record, fmt)


@dataset.command('create')
@click.argument('name')
@click.option(
    '--dem',
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Source DEM to resample onto the dataset grid.',
)
@pass_snowdb
def create_dataset(snowdb: SnowDb, name: str, dem: Path) -> None:
    """Create dataset NAME's directory, area raster, and resampled DEM.

    Idempotent: if the dataset is already created this is a no-op. To replace its
    DEM or area raster, use ``set-dem`` / ``rebuild-area``.
    """
    from snowtool.snowdb.dataset import Dataset

    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    try:
        Dataset.create(ds.spec, ds.path, dem)
    except FileExistsError:
        click.echo(
            f'dataset {name} already created (use set-dem/rebuild-area to rebuild)',
        )
        return
    except SNODASError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f'created dataset {name} at {ds.path}')


@dataset.command('ingest')
@click.argument('name')
@click.argument(
    'archives',
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@pass_snowdb
def ingest_dataset(
    snowdb: SnowDb,
    name: str,
    archives: tuple[Path, ...],
) -> None:
    """Ingest one or more source ARCHIVES into dataset NAME.

    Idempotent: re-ingesting an archive overwrites that date's COGs with the
    same (deterministic) result.
    """
    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    for archive in archives:
        try:
            dates = ds.ingest(archive, force=True)
        except (FileExistsError, SNODASError) as e:
            raise click.ClickException(str(e)) from e
        for ingested in dates:
            click.echo(f'ingested {name} {ingested.isoformat()} from {archive}')


@dataset.command('set-dem')
@click.argument('name')
@click.argument('dem', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@pass_snowdb
def set_dem(snowdb: SnowDb, name: str, dem: Path) -> None:
    """Re-resample dataset NAME's DEM from source DEM (overwrites the existing DEM)."""
    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    try:
        ds.create_resampled_dem(dem, force=True)
    except SNODASError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f'set dem for {name} from {dem}')


@dataset.command('rebuild-area')
@click.argument('name')
@pass_snowdb
def rebuild_area(snowdb: SnowDb, name: str) -> None:
    """Rebuild dataset NAME's per-pixel area raster (no-op on a projected grid)."""
    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    if not ds.spec.is_geographic:
        click.echo(f'{name}: projected grid uses a constant cell area; nothing to do.')
        return
    ds.make_area_raster(force=True)
    click.echo(f'rebuilt area raster for {name}')


@dataset.command('remove-date')
@click.argument('name')
@click.argument('removal_date', metavar='DATE', type=DATE)
@click.option('--dry-run', is_flag=True, help='Show what would be removed only.')
@pass_snowdb
def remove_date(
    snowdb: SnowDb,
    name: str,
    removal_date: date,
    dry_run: bool,
) -> None:
    """Remove a single ingested DATE from dataset NAME."""
    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    iso = removal_date.isoformat()

    if dry_run:
        present = removal_date in ds.available_dates()
        click.echo(f'would remove {name} {iso}' if present else f'{name} {iso}: absent')
        return

    if ds.remove_date(removal_date):
        click.echo(f'removed {name} {iso}')
    else:
        click.echo(f'{name} {iso}: absent (nothing removed)')


@dataset.command('prune')
@click.argument('name')
@click.option(
    '--before',
    required=True,
    type=DATE,
    help='Remove all ingested dates strictly before this date.',
)
@click.option('--dry-run', is_flag=True, help='Show what would be removed only.')
@pass_snowdb
def prune_dates(snowdb: SnowDb, name: str, before: date, dry_run: bool) -> None:
    """Remove every ingested date in dataset NAME older than --before."""
    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    targets = ds.dates_before(before)

    if not targets:
        click.echo(f'{name}: no dates before {before.isoformat()}')
        return

    for target in targets:
        iso = target.isoformat()
        if dry_run:
            click.echo(f'would remove {name} {iso}')
        else:
            ds.remove_date(target)
            click.echo(f'removed {name} {iso}')

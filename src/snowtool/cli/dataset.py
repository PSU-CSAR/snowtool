"""The ``dataset`` command group: per-dataset management.

Thin wrappers over the ``Dataset`` API: each command resolves the dataset,
calls a domain method, and renders. Write commands (create/ingest/generate/
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
        'terrain': artifacts.terrain,
        'dem_hash': ds.terrain.dem_hash(),
        'landcover': artifacts.landcover,
        'nlcd_hash': ds.landcover.nlcd_hash(),
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
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Generate terrain from this local DEM file instead of the default source.',
)
@click.option(
    '--nlcd',
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Generate land cover from this local NLCD file instead of the default source.',
)
@click.option(
    '--quick',
    is_flag=True,
    help='Create the directory + area raster only; skip terrain + land cover.',
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
@pass_snowdb
def create_dataset(
    snowdb: SnowDb,
    name: str,
    dem: Path | None,
    nlcd: Path | None,
    quick: bool,
    workers: int | None,
    block_size: int | None,
) -> None:
    """Create dataset NAME's directory + area raster, then its terrain + land cover.

    Mirrors ``snowdb init`` for one dataset: terrain comes from the database's
    default DEM source (unless ``--dem`` supplies a local file) and land cover
    from the default NLCD source (unless ``--nlcd`` does); ``--quick`` skips both.
    Idempotent -- existing area raster / terrain / land-cover sets are left
    untouched. To rebuild, use ``generate`` / ``generate-landcover`` / ``rebuild-area``.
    """
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.dem_source import LocalFile
    from snowtool.snowdb.landcover_source import LocalFile as LocalNLCD

    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    try:
        Dataset.create(ds.spec, ds.path)
    except FileExistsError:
        click.echo(f'dataset {name} already created')
    else:
        click.echo(f'created dataset {name} at {ds.path}')

    if quick:
        return

    if not ds.terrain.present():
        source = LocalFile(dem) if dem is not None else snowdb.dem_source
        try:
            ds.generate_terrain(
                source, workers=workers, block_size=block_size, force=True,
            )
        except (FileExistsError, SNODASError) as e:
            raise click.ClickException(str(e)) from e
        click.echo(f'generated terrain for {name}')

    if not ds.landcover.present():
        lc_source = LocalNLCD(nlcd) if nlcd is not None else snowdb.landcover_source
        try:
            ds.generate_landcover(lc_source, force=True)
        except (FileExistsError, SNODASError) as e:
            raise click.ClickException(str(e)) from e
        click.echo(f'generated land cover for {name}')


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


@dataset.command('generate')
@click.argument('name')
@click.option(
    '--source',
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Generate from this local DEM file instead of the default source.',
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
@pass_snowdb
def generate_terrain(
    snowdb: SnowDb,
    name: str,
    source: Path | None,
    workers: int | None,
    block_size: int | None,
) -> None:
    """(Re)generate dataset NAME's terrain set, overwriting any existing layers.

    Uses the database's default DEM source unless ``--source`` supplies a local
    file. Terrain is the DEM-derived elevation + aspect layers. Generation is
    heavy -- it reprojects the whole DEM source to a 10 m work grid -- and the
    default 3DEP source streams tiles from S3.
    """
    from snowtool.snowdb.dem_source import LocalFile

    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    dem_source = LocalFile(source) if source is not None else snowdb.dem_source
    try:
        ds.generate_terrain(
            dem_source, workers=workers, block_size=block_size, force=True,
        )
    except (FileExistsError, SNODASError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f'generated terrain for {name}')


@dataset.command('generate-landcover')
@click.argument('name')
@click.option(
    '--source',
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Generate from this local NLCD file instead of the default source.',
)
@pass_snowdb
def generate_landcover(snowdb: SnowDb, name: str, source: Path | None) -> None:
    """(Re)generate dataset NAME's land-cover set, overwriting any existing layer.

    Uses the database's default NLCD source unless ``--source`` supplies a local
    file. Land cover is the NLCD-derived percent-forest-cover layer. The default
    source downloads the MRLC Annual NLCD national raster (~1.5 GB) on first use,
    cached under the snowdb root.
    """
    from snowtool.snowdb.landcover_source import LocalFile

    require_initialized(snowdb)
    ds = get_dataset(snowdb, name)
    lc_source = LocalFile(source) if source is not None else snowdb.landcover_source
    try:
        ds.generate_landcover(lc_source, force=True)
    except (FileExistsError, SNODASError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f'generated land cover for {name}')


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

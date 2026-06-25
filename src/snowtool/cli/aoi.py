"""The ``aoi`` command group: global AOI storage and rasterization.

Thin wrappers over the ``SnowDb`` AOI API. ``import`` is additive; ``sync``
reconciles storage to a directory (with a guarded prune); ``list`` reads the
derived ``index.geojson`` while ``show``/``dump`` read the per-AOI records;
``rasterize`` burns AOIs onto each dataset grid (missing-or-stale by default);
``remove`` cascade-deletes a record and its per-dataset rasters.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import pass_manager, pass_snowdb
from snowtool.cli._datasets import (
    dataset_option,
    format_option,
    resolve_datasets,
)
from snowtool.cli._render import _emit, _emit_record
from snowtool.exceptions import AOIPruneDestinationRequiredError, SNODASError

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.manager import AOIImportResult, SnowDbManager


@click.group()
def aoi() -> None:
    """Global AOI management commands."""


def _echo_import(result: AOIImportResult, *, dry_run: bool) -> None:
    """Print the imported / skipped / invalid summary (invalids to stderr)."""
    verb = 'would import' if dry_run else 'imported'
    click.echo(f'{verb} {len(result.imported)} AOI(s)')
    if result.skipped:
        click.echo(f'skipped {len(result.skipped)} point-only pourpoint(s)')
    for path, message in result.invalid:
        click.echo(f'invalid: {path}: {message}', err=True)


def _fail_if_invalid(result: AOIImportResult) -> None:
    """Exit nonzero when any source file failed to parse (clean runs exit 0)."""
    if result.invalid:
        raise click.ClickException(
            f'{len(result.invalid)} invalid source file(s); see messages above.',
        )


@aoi.command('import')
@click.argument('src', type=click.Path(exists=True, path_type=Path))
@click.option('--dry-run', is_flag=True, help='Classify sources without writing.')
@pass_manager
def import_aois(manager: SnowDbManager, src: Path, dry_run: bool) -> None:
    """Additively import AOI(s) from a file or directory into the snowdb.

    Imports polygon-bearing pourpoints, skips point-only ones, and reports
    unparseable files (nonzero exit). Never removes a stored AOI.
    """
    result = manager.import_aois(src, dry_run=dry_run)
    _echo_import(result, dry_run=dry_run)
    _fail_if_invalid(result)


@aoi.command('sync')
@click.argument('src', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--prune-to',
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help='Archive directory for AOIs removed because they are absent from SRC.',
)
@click.option('--dry-run', is_flag=True, help='Show the import + prune plan only.')
@pass_manager
def sync_aois(
    manager: SnowDbManager,
    src: Path,
    prune_to: Path | None,
    dry_run: bool,
) -> None:
    """Mirror directory SRC into the snowdb: import it, then prune absent AOIs.

    Any stored AOI whose triplet is not in SRC is dumped to ``--prune-to`` and
    removed (cascading to its per-dataset rasters). If a prune would happen and
    ``--prune-to`` is absent, the command errors before changing anything.
    """
    try:
        result = manager.sync_aois(src, prune_to=prune_to, dry_run=dry_run)
    except AOIPruneDestinationRequiredError as e:
        raise click.ClickException(str(e)) from e

    _echo_import(result, dry_run=dry_run)
    if result.pruned:
        verb = 'would prune' if dry_run else 'pruned'
        dest = f' to {prune_to}' if prune_to else ''
        click.echo(
            f'{verb} {len(result.pruned)} AOI(s){dest}: '
            f'{", ".join(result.pruned)}',
        )
    _fail_if_invalid(result)


@aoi.command('list')
@format_option
@pass_snowdb
def list_aois(snowdb: SnowDb, fmt: str) -> None:
    """List stored AOIs from the index (triplet, name, source, active, area)."""
    rows = [
        {
            'triplet': entry.triplet,
            'name': entry.name,
            'source': entry.source,
            'active': entry.active,
            'basinarea': entry.basinarea,
        }
        for entry in snowdb.aoi_index()
    ]
    _emit(rows, fmt)


@aoi.command('show')
@click.argument('triplet')
@format_option
@pass_snowdb
def show_aoi(snowdb: SnowDb, triplet: str, fmt: str) -> None:
    """Show a stored AOI's details (from its record geojson)."""
    try:
        aoi_ = snowdb.load_aoi(triplet)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    lon, lat = aoi_.point['coordinates'][:2]
    record = {
        'triplet': aoi_.station_triplet,
        'name': aoi_.name,
        'source': aoi_.source,
        'active': aoi_.properties.get('active'),
        'basinarea': aoi_.properties.get('basinarea'),
        'huc': aoi_.properties.get('huc'),
        'point_lon': lon,
        'point_lat': lat,
        'geometry_hash': aoi_.geometry_hash,
    }
    _emit_record(record, fmt)


@aoi.command('dump')
@click.argument('triplet')
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(file_okay=False, path_type=Path),
    default='.',
    help='Directory to write the record geojson into (default: cwd).',
)
@pass_snowdb
def dump_aoi(snowdb: SnowDb, triplet: str, output_dir: Path) -> None:
    """Copy a stored AOI's record geojson out to OUTPUT_DIR (round-trip)."""
    try:
        dest = snowdb.dump_aoi(triplet, output_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f'dumped {triplet} to {dest}')


@aoi.command('reindex')
@pass_manager
def reindex_aois(manager: SnowDbManager) -> None:
    """Rebuild the index.geojson manifest from the stored records."""
    index = manager.reindex_aois()
    click.echo(f'reindexed {len(index)} AOI(s) into {manager.db.aoi_index_path}')


@aoi.command('remove')
@click.argument('triplet')
@click.option('--dry-run', is_flag=True, help='Show what would be removed only.')
@pass_manager
def remove_aoi(manager: SnowDbManager, triplet: str, dry_run: bool) -> None:
    """Remove a stored AOI and its per-dataset rasters (cascade)."""
    if dry_run:
        present = manager.db.aoi_record_path(triplet).is_file()
        click.echo(f'would remove {triplet}' if present else f'{triplet}: absent')
        return
    if manager.remove_aoi(triplet):
        click.echo(f'removed {triplet}')
    else:
        click.echo(f'{triplet}: absent (nothing removed)')


@aoi.command('rasterize')
@click.argument('triplet', required=False)
@click.option('--all', 'all_aois', is_flag=True, help='Rasterize every stored AOI.')
@click.option('--rebuild', is_flag=True, help='Rebuild even rasters that are current.')
@dataset_option
@pass_manager
def rasterize_aois(
    manager: SnowDbManager,
    triplet: str | None,
    all_aois: bool,
    rebuild: bool,
    dataset_names: tuple[str, ...],
) -> None:
    """Burn AOI(s) onto each dataset grid, building missing or stale rasters.

    Provide a single TRIPLET or ``--all``. By default only missing/stale rasters
    are (re)built; ``--rebuild`` forces all selected.
    """
    if bool(triplet) == bool(all_aois):
        raise click.ClickException('Provide exactly one of TRIPLET or --all.')

    if all_aois:
        aois = list(manager.db.aois())
    else:
        try:
            aois = [manager.db.load_aoi(triplet)]  # type: ignore[arg-type]
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e

    datasets = resolve_datasets(manager.db, dataset_names)
    try:
        result = manager.rasterize_aois(aois, datasets, rebuild=rebuild)
    except (FileNotFoundError, SNODASError) as e:
        raise click.ClickException(str(e)) from e

    for triplet_, dataset_name in result.built:
        click.echo(f'built {triplet_} [{dataset_name}]')
    click.echo(
        f'built {len(result.built)}, '
        f'skipped {len(result.skipped)} (already current)',
    )

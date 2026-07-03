"""The ``pourpoint`` command group: global pourpoint storage and rasterization.

Thin wrappers over the ``SnowDb`` pourpoint API. ``import`` is additive; ``sync``
reconciles storage to a directory (with a guarded prune); ``list`` reads the
derived ``index.geojson`` while ``show``/``dump`` read the per-pourpoint records;
``rasterize`` burns each pourpoint's basin onto every dataset grid into an AOI
raster (missing-or-stale by default); ``remove`` cascade-deletes a record and its
per-dataset rasters.
"""

from __future__ import annotations

import sys

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import config_option, pass_manager, pass_snowdb
from snowtool.cli._datasets import (
    dataset_option,
    format_option,
    resolve_datasets,
)
from snowtool.cli._progress import ClickProgress
from snowtool.cli._render import _emit, _emit_record
from snowtool.exceptions import PourpointPruneDestinationRequiredError, SnowtoolError

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.manager import PourpointImportResult, SnowDbManager


@click.group()
def pourpoint() -> None:
    """Global pourpoint management commands."""


def _echo_import(result: PourpointImportResult, *, dry_run: bool) -> None:
    """Print the imported / skipped / invalid summary (invalids to stderr)."""
    verb = 'would import' if dry_run else 'imported'
    click.echo(f'{verb} {len(result.imported)} pourpoint(s)')
    if result.skipped:
        click.echo(f'skipped {len(result.skipped)} point-only pourpoint(s)')
    for path, message in result.invalid:
        click.echo(f'invalid: {path}: {message}', err=True)


def _fail_if_invalid(result: PourpointImportResult) -> None:
    """Exit nonzero when any source file failed to parse (clean runs exit 0)."""
    if result.invalid:
        raise click.ClickException(
            f'{len(result.invalid)} invalid source file(s); see messages above.',
        )


@pourpoint.command('import')
@click.argument('src', type=click.Path(exists=True, path_type=Path))
@click.option('--dry-run', is_flag=True, help='Classify sources without writing.')
@config_option
@pass_manager
def import_pourpoints(manager: SnowDbManager, src: Path, dry_run: bool) -> None:
    """Additively import pourpoint(s) from a file or directory into the snowdb.

    Imports basin-bearing pourpoints, skips point-only ones, and reports
    unparseable files (nonzero exit). Never removes a stored pourpoint.
    """
    result = manager.import_pourpoints(src, dry_run=dry_run)
    _echo_import(result, dry_run=dry_run)
    _fail_if_invalid(result)


@pourpoint.command('sync')
@click.argument('src', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--prune-to',
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help='Archive directory for pourpoints removed because they are absent from SRC.',
)
@click.option('--dry-run', is_flag=True, help='Show the import + prune plan only.')
@config_option
@pass_manager
def sync_pourpoints(
    manager: SnowDbManager,
    src: Path,
    prune_to: Path | None,
    dry_run: bool,
) -> None:
    """Mirror directory SRC into the snowdb: import it, then prune absent pourpoints.

    Any stored pourpoint whose triplet is not in SRC is dumped to ``--prune-to`` and
    removed (cascading to its per-dataset rasters). If a prune would happen and
    ``--prune-to`` is absent, the command errors before changing anything.
    """
    try:
        result = manager.sync_pourpoints(src, prune_to=prune_to, dry_run=dry_run)
    except PourpointPruneDestinationRequiredError as e:
        raise click.ClickException(str(e)) from e

    _echo_import(result, dry_run=dry_run)
    if result.pruned:
        verb = 'would prune' if dry_run else 'pruned'
        dest = f' to {prune_to}' if prune_to else ''
        click.echo(
            f'{verb} {len(result.pruned)} pourpoint(s){dest}: '
            f'{", ".join(result.pruned)}',
        )
    _fail_if_invalid(result)


@pourpoint.command('list')
@format_option
@config_option
@pass_snowdb
def list_pourpoints(snowdb: SnowDb, fmt: str) -> None:
    """List stored pourpoints from the index (triplet, name, area, coverage)."""
    rows = [
        {
            'triplet': entry.triplet,
            'name': entry.name,
            'area_meters': entry.area_meters,
            'coverage': {name: cov.value for name, cov in entry.coverage.items()},
        }
        for entry in snowdb.pourpoint_index()
    ]
    _emit(rows, fmt)


@pourpoint.command('show')
@click.argument('triplet')
@format_option
@config_option
@pass_snowdb
def show_pourpoint(snowdb: SnowDb, triplet: str, fmt: str) -> None:
    """Show a stored pourpoint's details (from its record geojson)."""
    index = snowdb.pourpoint_index()
    try:
        pp = snowdb.load_pourpoint(triplet, index=index)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    # area_meters and geometry_hash are cached on the index entry (computed at
    # reindex); read them rather than recomputing from the basin polygon.
    entry = index[triplet]
    lon, lat = pp.point['coordinates'][:2]
    record = {
        'triplet': pp.station_triplet,
        'name': pp.name,
        'awdb_id': pp.awdb_id,
        'usgs_id': pp.usgs_id,
        'area_meters': entry.area_meters,
        'point_lon': lon,
        'point_lat': lat,
        'geometry_hash': entry.geometry_hash,
    }
    _emit_record(record, fmt)


@pourpoint.command('dump')
@click.argument('triplet')
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(file_okay=False, path_type=Path),
    default='.',
    help='Directory to write the record geojson into (default: cwd).',
)
@config_option
@pass_snowdb
def dump_pourpoint(snowdb: SnowDb, triplet: str, output_dir: Path) -> None:
    """Copy a stored pourpoint's record geojson out to OUTPUT_DIR (round-trip)."""
    try:
        dest = snowdb.dump_pourpoint(triplet, output_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f'dumped {triplet} to {dest}')


@pourpoint.command('reindex')
@config_option
@pass_manager
def reindex_pourpoints(manager: SnowDbManager) -> None:
    """Rebuild the index.geojson manifest from the stored records."""
    index = manager.reindex_pourpoints()
    click.echo(
        f'reindexed {len(index)} pourpoint(s) into {manager.db.pourpoint_index_path}',
    )


@pourpoint.command('remove')
@click.argument('triplet')
@click.option('--dry-run', is_flag=True, help='Show what would be removed only.')
@config_option
@pass_manager
def remove_pourpoint(manager: SnowDbManager, triplet: str, dry_run: bool) -> None:
    """Remove a stored pourpoint and its per-dataset rasters (cascade)."""
    if dry_run:
        present = manager.db.pourpoint_record_path(triplet).is_file()
        click.echo(f'would remove {triplet}' if present else f'{triplet}: absent')
        return
    if manager.remove_pourpoint(triplet):
        click.echo(f'removed {triplet}')
    else:
        click.echo(f'{triplet}: absent (nothing removed)')


@pourpoint.command('rasterize')
@click.argument('triplet', required=False)
@click.option(
    '--all',
    'all_pourpoints',
    is_flag=True,
    help='Rasterize every stored pourpoint.',
)
@click.option('--rebuild', is_flag=True, help='Rebuild even rasters that are current.')
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='List each built raster, not just the totals.',
)
@dataset_option
@config_option
@pass_manager
def rasterize_aois(
    manager: SnowDbManager,
    triplet: str | None,
    all_pourpoints: bool,
    rebuild: bool,
    verbose: bool,
    dataset_names: tuple[str, ...],
) -> None:
    """Burn pourpoint basin(s) onto each dataset grid, building missing/stale rasters.

    Provide a single TRIPLET or ``--all``. By default only missing/stale rasters
    are (re)built; ``--rebuild`` forces all selected. A live bar shows progress on a
    TTY; ``--verbose`` (or a non-TTY, where the bar is hidden) lists each built
    raster.
    """
    if bool(triplet) == bool(all_pourpoints):
        raise click.ClickException('Provide exactly one of TRIPLET or --all.')

    # The guard above leaves exactly one branch live; testing ``triplet`` directly
    # (truthy => a non-empty str, excluding None) narrows it for the typed load.
    if triplet:
        try:
            pourpoints = [manager.db.load_pourpoint(triplet)]
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
    else:
        pourpoints = list(manager.db.pourpoints())

    datasets = resolve_datasets(manager.db, dataset_names)
    try:
        result = manager.rasterize_aois(
            pourpoints,
            datasets,
            rebuild=rebuild,
            progress=ClickProgress(),
        )
    except (FileNotFoundError, SnowtoolError) as e:
        raise click.ClickException(str(e)) from e

    # The bar (stderr) only renders on a TTY; when asked (--verbose) or when it was
    # hidden (piped/non-TTY), list what built so the detail isn't lost. Printed after
    # the bar closes, so the two never interleave.
    if verbose or not sys.stderr.isatty():
        for triplet_, dataset_name in result.built:
            click.echo(f'built {triplet_} [{dataset_name}]')
    click.echo(
        f'built {len(result.built)}, skipped {len(result.skipped)} (already current)',
    )

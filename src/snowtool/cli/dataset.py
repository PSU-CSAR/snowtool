"""The ``dataset`` command group: per-dataset management.

Thin wrappers over the ``Dataset`` API: each command resolves the dataset,
calls a domain method, and renders. Write commands (create/ingest/generate/
remove-date/prune) first require an initialized snowdb root.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._context import config_option, pass_manager, pass_snowdb
from snowtool.cli._datasets import format_option, get_dataset
from snowtool.cli._progress import RichProgress
from snowtool.cli._render import DATE, _emit, _emit_record
from snowtool.exceptions import SnowtoolError
from snowtool.snowdb.zones.zone_layer import GenerationOptions

if TYPE_CHECKING:
    from datetime import date

    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.manager import SnowDbManager


@click.group()
def dataset() -> None:
    """Dataset management commands."""


@dataset.command('list')
@format_option
@config_option
@pass_snowdb
def list_datasets(snowdb: SnowDb, fmt: str) -> None:
    """List every registered dataset and whether readers serve it (``active``).

    One row per dataset the root config registers, active or not -- for
    presence, date spans, and artifact counts, see ``snowtool status``.
    """
    _emit(
        [
            {'dataset': name, 'active': name in snowdb.datasets}
            for name in sorted(snowdb.registered)
        ],
        fmt,
    )


@dataset.command('info')
@click.argument('name')
@format_option
@config_option
@pass_snowdb
def dataset_info(snowdb: SnowDb, name: str, fmt: str) -> None:
    """Show a dataset's grid, variables, and on-disk artifacts.

    Resolves any *registered* dataset -- an inactive one is still inspectable
    (its ``active`` field says whether readers serve it).
    """
    from snowtool.snowdb.constants import MAX_ELEVATION_M, MIN_ELEVATION_M
    from snowtool.snowdb.diagnostics import dataset_status, grid_report

    ds = get_dataset(snowdb, name, include_inactive=True)
    spec = ds.spec
    grid = spec.grid_params
    status = dataset_status(ds)
    grid_details = grid_report(ds)
    artifacts = status.artifacts

    record = {
        'name': spec.name,
        'active': name in snowdb.datasets,
        'present': status.present,
        'crs': str(grid.crs),
        'is_geographic': spec.is_geographic,
        'rows': grid.rows,
        'cols': grid.cols,
        'tile_size': grid.tile_size,
        'cell_area_m2': 'varies (geographic)' if spec.is_geographic else spec.cell_area,
        'px_size': grid_details.px_size,
        'n_tiles': grid_details.n_tiles,
        'extent': list(grid_details.extent),
        'zones': {
            provider: {layer: params.model_dump() for layer, params in layers.items()}
            for provider, layers in spec.zones.items()
        },
        'elevation_bracket_m': f'{MIN_ELEVATION_M} .. {MAX_ELEVATION_M}',
        'variables': sorted(spec.variables),
        # One entry per configured zone-layer provider: whether its set is present
        # and the provenance hash it was generated with.
        'zone_layers': {
            name: {
                'present': artifacts.zone_layers[name],
                'hash': ds.zones[name].provenance_hash(),
            }
            for name in ds.zones
        },
        'cogs': artifacts.cogs,
        'aoi_rasters': artifacts.aoi_rasters,
        'dates': status.date_count,
        'first_date': status.first_date.isoformat() if status.first_date else None,
        'last_date': status.last_date.isoformat() if status.last_date else None,
    }
    _emit_record(record, fmt)


@dataset.command('dates')
@click.argument('name')
@click.option('--start', type=DATE, default=None, help='Only dates on/after this.')
@click.option('--end', type=DATE, default=None, help='Only dates on/before this.')
@click.option(
    '--gaps',
    is_flag=True,
    help='Summarize the span and interior gaps instead of listing every date.',
)
@format_option
@config_option
@pass_snowdb
def dataset_dates(
    snowdb: SnowDb,
    name: str,
    start: date | None,
    end: date | None,
    gaps: bool,
    fmt: str,
) -> None:
    """Ingested dates for dataset NAME (or, with --gaps, its span and gaps).

    Resolves any *registered* dataset -- active or not -- like ``dataset info``.
    """
    from snowtool.snowdb import diagnostics

    ds = get_dataset(snowdb, name, include_inactive=True)
    if gaps:
        result = diagnostics.coverage_report(ds)
        _emit_record(
            {
                'dataset': result.name,
                'dates': result.date_count,
                'first': result.first_date.isoformat() if result.first_date else '',
                'last': result.last_date.isoformat() if result.last_date else '',
                'gaps': len(result.gaps),
                'gap_ranges': '; '.join(
                    f'{gap_start.isoformat()}..{gap_end.isoformat()}'
                    for gap_start, gap_end in result.gaps
                ),
            },
            fmt,
        )
        return
    rows = [
        {'date': d.isoformat()}
        for d in ds.available_dates()
        if (start is None or d >= start) and (end is None or d <= end)
    ]
    _emit(rows, fmt)


@dataset.command('values')
@click.argument('name')
@click.option(
    '--date',
    'on_date',
    type=DATE,
    default=None,
    help='Date to report (default: latest ingested).',
)
@format_option
@config_option
@pass_snowdb
def dataset_values(
    snowdb: SnowDb,
    name: str,
    on_date: date | None,
    fmt: str,
) -> None:
    """Per-variable min/max/mean (unit-scaled) and nodata % for one date."""
    from snowtool.snowdb import diagnostics

    ds = get_dataset(snowdb, name, include_inactive=True)
    if on_date is None:
        dates = ds.available_dates()
        if not dates:
            raise click.ClickException(f'{name} has no ingested dates')
        on_date = dates[-1]

    rows = [
        {
            'variable': result.variable,
            'unit': result.unit,
            'min': result.minimum,
            'max': result.maximum,
            'mean': result.mean,
            'nodata_pct': round(result.nodata_pct, 3),
        }
        for result in diagnostics.value_ranges_report(ds, on_date)
    ]
    if not rows:
        raise click.ClickException(
            f'{name} has no variable files for {on_date.isoformat()}',
        )
    _emit(rows, fmt)


@dataset.command('create')
@click.argument('name')
@click.option(
    '--template',
    default=None,
    help='Stamp a built-in dataset template (e.g. snodas, swann-800m, instarr) as '
    'the new dataset NAME, instead of staging an already-registered dataset.',
)
@config_option
@pass_manager
def create_dataset(
    manager: SnowDbManager,
    name: str,
    template: str | None,
) -> None:
    """Create dataset NAME: stage its artifacts and register it (inactive).

    Staging writes the directory skeleton, the area raster, the dataset config
    (``data/NAME/dataset.json``), an AOI raster of every indexed pourpoint's
    basin on the new grid, and each pourpoint's coverage of that grid -- never
    zone layers. Zone-layer generation is a separate explicit operation:
    ``dataset generate-zones NAME ...``, which shares one source read across
    every named dataset (so create several datasets first, then generate their
    zone layers together). The dataset is registered *inactive*: it exists
    (manageable by name -- ingest, generate-zones, diagnostics) but stays
    invisible to readers (query/API) until ``dataset activate NAME`` flips it
    live. The dataset's definition comes from ``--template`` (a built-in) for a
    brand-new dataset, or from an already-registered dataset of this NAME
    otherwise. Converge-by-default, like ingest: a re-create leaves current
    artifacts untouched (an AOI raster rebuilds only when its provenance tag
    reads stale) and never touches an existing registration (its active state
    and link are preserved). To force-rebuild AOI rasters regardless, use
    ``pourpoint rasterize --all --rebuild -d NAME``.
    """
    from snowtool.snowdb.config import DATASET_CONFIG_FILENAME

    ds, config = _resolve_create_target(manager.db, name, template)

    # Stage the dataset config beside its data so `stage_dataset` can build from
    # it and `register_dataset` can link it. Idempotent overwrite.
    ds.path.mkdir(parents=True, exist_ok=True)
    config_path = ds.path / DATASET_CONFIG_FILENAME
    config.save(config_path)

    try:
        staged = manager.stage_dataset(name, config_path, progress=RichProgress())
    except (ValueError, FileExistsError, SnowtoolError) as e:
        raise click.ClickException(str(e)) from e

    if staged.created:
        click.echo(f'created dataset {name} at {staged.dataset.path}')
    else:
        click.echo(f'dataset {name} already created')

    # Ensure the staged dataset is registered (inactive). An existing
    # registration -- whatever its link or active state -- is left untouched so
    # an idempotent re-create never deactivates or relinks a live dataset.
    if name not in manager.db.registered:
        manager.register_dataset(
            name,
            config_path,
            coverage=staged.coverage,
            active=False,
        )
        click.echo(
            f'registered {name} (inactive; generate zones with '
            f"'dataset generate-zones {name}', activate with "
            f"'dataset activate {name}')",
        )


def _resolve_create_target(snowdb: SnowDb, name: str, template: str | None):
    """The (Dataset, DatasetConfig) to stage for ``dataset create``.

    With ``--template`` the dataset is brand-new: its config is the named built-in
    template and a fresh :class:`Dataset` is bound at ``data/<name>/``. Otherwise
    the dataset must already be registered under ``name`` (active or not) -- its
    config is derived from the bound spec.
    """
    from snowtool.snowdb.config import DatasetConfig
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.datasets import DATASET_TEMPLATES, config_from_spec
    from snowtool.snowdb.spec import DatasetSpec

    config: DatasetConfig
    if template is not None:
        if template not in DATASET_TEMPLATES:
            known = ', '.join(sorted(DATASET_TEMPLATES))
            raise click.ClickException(
                f'No such template: {template!r}. Known templates: {known}.',
            )
        config = DATASET_TEMPLATES[template]
        spec = DatasetSpec.from_config(config, name)
        ds = Dataset(
            spec,
            snowdb.dataset_dir(name, config),
            snowdb.zone_layer_providers.values(),
        )
        return ds, config
    ds = get_dataset(snowdb, name, include_inactive=True)
    return ds, config_from_spec(ds.spec)


def _resolve_source_overrides(
    snowdb: SnowDb,
    sources: tuple[tuple[str, Path], ...],
) -> dict[str, Path]:
    """Validate ``--source PROVIDER PATH`` pairs into a ``{provider: path}`` map."""
    overrides: dict[str, Path] = {}
    for provider_name, path in sources:
        if provider_name not in snowdb.zone_layer_providers:
            raise click.ClickException(f'No such zone-layer provider: {provider_name}')
        overrides[provider_name] = Path(path)
    return overrides


@dataset.command('add')
@click.argument('name')
@click.argument(
    'config_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@config_option
@pass_manager
def add_dataset(
    manager: SnowDbManager,
    name: str,
    config_path: Path,
) -> None:
    """Register dataset NAME from its config at CONFIG_PATH (writes the link).

    The escape hatch for a dataset built *out of tree* (or otherwise out of
    band): ``dataset create`` already stages *and* registers, so ``add`` exists
    only to link an externally built config into the root config. Registration
    is inactive: the dataset becomes manageable by name (ingest, generate-zones,
    diagnostics) but invisible to readers until ``dataset activate NAME``.
    The config is validated (it must parse and its ingester must resolve) before
    the link is written. Idempotent -- re-adding a name overwrites its link.
    Since ``add`` skips staging, pourpoint coverage for the new dataset is
    unknown until the next ``pourpoint reindex``.
    """
    from pydantic import ValidationError

    from snowtool.snowdb.config import DatasetConfig
    from snowtool.snowdb.spec import DatasetSpec

    try:
        config = DatasetConfig.load(config_path)
        DatasetSpec.from_config(config, name)  # validate it resolves (ingester, ...)
    except (ValidationError, ValueError) as e:
        raise click.ClickException(
            f'Not a usable dataset config ({config_path}): {e}',
        ) from e

    manager.register_dataset(name, config_path, active=False)
    click.echo(
        f'registered {name} -> {config_path} '
        f"(inactive; run 'pourpoint reindex' to compute coverage, then "
        f"'dataset activate {name}')",
    )


@dataset.command('activate')
@click.argument('name')
@config_option
@pass_manager
def activate_dataset(manager: SnowDbManager, name: str) -> None:
    """Make registered dataset NAME visible to readers (query CLI + API).

    Activation only toggles reader visibility: the dataset was already fully
    manageable by name (ingest, generate-zones, diagnostics) while inactive, and
    stays so either way. Idempotent.
    """
    try:
        manager.set_dataset_active(name, True)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    # Activation never needs a reindex: `dataset create` folds coverage into the
    # index at registration. The one exception (`dataset add`, which skips
    # staging) prints its own reindex guidance at add time.
    click.echo(f'activated {name} (restart the API to pick it up)')


@dataset.command('deactivate')
@click.argument('name')
@config_option
@pass_manager
def deactivate_dataset(manager: SnowDbManager, name: str) -> None:
    """Hide registered dataset NAME from readers (query CLI + API).

    The dataset stays registered and fully manageable by name (ingest,
    generate-zones, diagnostics) -- only reader visibility changes. Idempotent.
    """
    try:
        manager.set_dataset_active(name, False)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f'deactivated {name} (restart the API to pick it up)')


def _resolve_managed_dataset(manager: SnowDbManager, token: str):
    """Resolve a management-surface dataset token, CLI-cleanly.

    Wraps :meth:`SnowDbManager.resolve_dataset` (a bare token is a *registered*
    NAME -- active or not; a token with a path separator or a ``.json`` suffix
    is a dataset config file path), converting its :class:`ValueError` to a
    :class:`click.ClickException`.
    """
    try:
        return manager.resolve_dataset(token)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@dataset.command('ingest')
@click.argument('name')
@click.argument(
    'source',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--force',
    is_flag=True,
    help='Rebuild dates even when the stored source hash already matches.',
)
@config_option
@pass_manager
def ingest_dataset(
    manager: SnowDbManager,
    name: str,
    source: Path,
    force: bool,
) -> None:
    """Ingest a single SOURCE into dataset NAME.

    SOURCE is one source artifact per invocation, and its shape is the
    dataset's: a single *file* for snodas (a daily tar archive) and swann (a
    daily NetCDF) -- one file == one date -- or a *directory* of SPIRES ``.nc``
    tiles for instarr. Always pass instarr the directory: a date's mosaic is
    built from ALL of its tiles in one ingest call (per-tile calls would each
    rebuild the date from a single tile, last write wins). Batch driving
    belongs to the shell -- e.g.
    ``ls /data/snodas/*.tar | xargs -n1 -P4 snowtool dataset ingest snodas``
    -- and parallel runs are safe across distinct dates because each date
    commits via an atomic whole-directory swap.

    NAME is a registered dataset name (active or not) or a dataset config path
    -- ingest is a management op, so reader visibility is irrelevant (the point
    of the register/activate split is populating a dataset *before* serving
    it). Converge-by-default: a date whose COGs already carry the same source
    hash is left untouched (reported ``up to date``); a re-release under the
    same filename with different bytes rebuilds. ``--force`` rebuilds every
    date regardless.
    """
    ds = _resolve_managed_dataset(manager, name)
    try:
        result = ds.ingest(
            source,
            force=force,
            progress=RichProgress(prefix=f'{name} ingest: '),
        )
    except (FileExistsError, SnowtoolError) as e:
        raise click.ClickException(str(e)) from e
    for ingested in result.ingested:
        click.echo(f'ingested {name} {ingested.isoformat()} from {source}')
    for skipped in result.skipped:
        click.echo(f'up to date {name} {skipped.isoformat()} from {source}')


@dataset.command('generate-zones')
@click.argument('names', nargs=-1, required=True, metavar='NAME...')
@click.option(
    '--provider',
    'provider_names',
    multiple=True,
    help='Limit generation to these zone-layer providers (default: all).',
)
@click.option(
    '--source',
    'sources',
    nargs=2,
    multiple=True,
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    metavar='PROVIDER PATH',
    help='Generate PROVIDER zone layers from a local PATH instead of the default '
    'source (repeatable).',
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
@config_option
@pass_manager
def generate_zones(
    manager: SnowDbManager,
    names: tuple[str, ...],
    provider_names: tuple[str, ...],
    sources: tuple[tuple[str, Path], ...],
    workers: int | None,
    block_size: int | None,
) -> None:
    """(Re)generate zone layers for one or more datasets, sharing each source read.

    Each NAME is a registered dataset name (active or not); a token containing
    a path separator or ending in ``.json`` is instead a dataset config path.
    Activation does not matter -- zone layers live under ``data/<name>/``
    independent of the root-config link's visibility flag, so you can generate
    them for freshly created (registered-inactive) datasets before any of
    them is activated. Every
    configured provider (terrain, land cover, ...) is generated unless
    ``--provider`` limits the selection; each provider's source is read *once* over
    the combined extent of all the named datasets and binned into every one -- so
    standing up several datasets that share a provider pays that provider's
    expensive read (terrain reprojects the whole DEM source to a 10 m work grid;
    land cover downloads the ~1.5 GB MRLC Annual NLCD national raster) a single
    time. ``--source PROVIDER PATH`` supplies a local file for a provider instead
    of its default source. Always rebuilds, overwriting existing layers.
    """
    overrides = _resolve_source_overrides(manager.db, sources)

    try:
        datasets = [manager.resolve_dataset(name) for name in names]
        generated = manager.generate_zone_layers_for(
            datasets,
            provider_names=provider_names or None,
            source_overrides=overrides,
            force=True,
            options=GenerationOptions(workers=workers, block_size=block_size),
            progress_factory=lambda provider_name: RichProgress(
                prefix=f'{provider_name}: ',
            ),
        )
    except (ValueError, FileExistsError, SnowtoolError) as e:
        raise click.ClickException(str(e)) from e
    for provider_name, hashes in generated.items():
        for ds_name in hashes:
            click.echo(f'generated {provider_name} for {ds_name}')


@dataset.command('remove-date')
@click.argument('name')
@click.argument('removal_date', metavar='DATE', type=DATE)
@click.option('--dry-run', is_flag=True, help='Show what would be removed only.')
@config_option
@pass_manager
def remove_date(
    manager: SnowDbManager,
    name: str,
    removal_date: date,
    dry_run: bool,
) -> None:
    """Remove a single ingested DATE from dataset NAME.

    NAME is a registered dataset name (active or not) or a dataset config path.
    """
    ds = _resolve_managed_dataset(manager, name)
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
@config_option
@pass_manager
def prune_dates(manager: SnowDbManager, name: str, before: date, dry_run: bool) -> None:
    """Remove every ingested date in dataset NAME older than --before.

    NAME is a registered dataset name (active or not) or a dataset config path.
    """
    ds = _resolve_managed_dataset(manager, name)
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

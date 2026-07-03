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
from snowtool.cli._progress import ClickProgress
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
    """List the names of configured datasets.

    Just the names -- for presence, date spans, and artifact counts, see
    ``snowtool snowdb status``.
    """
    _emit([{'dataset': name} for name in sorted(snowdb)], fmt)


@dataset.command('info')
@click.argument('name')
@format_option
@config_option
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
        'zones': spec.zones,
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


@dataset.command('create')
@click.argument('name')
@click.option(
    '--template',
    default=None,
    help='Stamp a built-in dataset template (e.g. snodas, swann-800m, instarr) as '
    'the new dataset NAME, instead of staging an already-registered dataset.',
)
@click.option(
    '--source',
    'sources',
    nargs=2,
    multiple=True,
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    metavar='PROVIDER PATH',
    help='Generate PROVIDER zone layers (e.g. terrain, landcover) from a local '
    'PATH instead of the default source (repeatable).',
)
@click.option(
    '--activate',
    is_flag=True,
    help='Register the dataset in the root config after staging it (writes its '
    'link). Going live still needs `pourpoint reindex` + a service restart.',
)
@click.option(
    '--quick',
    is_flag=True,
    help='Create the directory + area raster only; skip zone-layer generation.',
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
def create_dataset(
    manager: SnowDbManager,
    name: str,
    template: str | None,
    sources: tuple[tuple[str, Path], ...],
    activate: bool,
    quick: bool,
    workers: int | None,
    block_size: int | None,
) -> None:
    """Create dataset NAME's directory + area raster, then its zone layers.

    Stages the dataset: it writes the directory skeleton, the area raster, the
    dataset config (``data/NAME/dataset.json``), and -- unless ``--quick`` -- every
    configured zone layer (terrain, land cover, ...) from its provider's default
    source (or ``--source PROVIDER PATH``). The dataset's definition comes from
    ``--template`` (a built-in) for a brand-new dataset, or from an
    already-registered dataset of this NAME otherwise. Staging does *not* register
    the dataset unless ``--activate`` is passed (or use ``dataset add`` later);
    going live also needs a ``pourpoint reindex`` + restart. Idempotent -- existing
    artifacts are left untouched.
    """
    from snowtool.snowdb.config import DATASET_CONFIG_FILENAME
    from snowtool.snowdb.dataset import Dataset

    ds, config = _resolve_create_target(manager.db, name, template)
    overrides = _resolve_source_overrides(manager.db, sources)
    try:
        Dataset.create(ds.spec, ds.path)
    except FileExistsError:
        click.echo(f'dataset {name} already created')
    else:
        click.echo(f'created dataset {name} at {ds.path}')

    # Stage the dataset config beside its data so it can be registered (now via
    # --activate, or later via `dataset add`). Idempotent overwrite.
    config_path = ds.path / DATASET_CONFIG_FILENAME
    config.save(config_path)
    if activate:
        manager.register_dataset(name, config_path)
        click.echo(
            f'registered {name} (run `pourpoint reindex` + restart to go live)',
        )

    if quick:
        return

    try:
        generated = manager.generate_dataset_zone_layers(
            ds,
            source_overrides=overrides,
            skip_present=True,
            force=True,
            options=GenerationOptions(workers=workers, block_size=block_size),
            progress_factory=lambda provider_name: ClickProgress(
                prefix=f'{name} {provider_name}: ',
            ),
        )
    except (ValueError, FileExistsError, SnowtoolError) as e:
        raise click.ClickException(str(e)) from e
    for provider_name in generated:
        click.echo(f'generated {provider_name} for {name}')


def _resolve_create_target(snowdb: SnowDb, name: str, template: str | None):
    """The (Dataset, DatasetConfig) to stage for ``dataset create``.

    With ``--template`` the dataset is brand-new: its config is the named built-in
    template and a fresh :class:`Dataset` is bound at ``data/<name>/``. Otherwise
    the dataset must already be registered under ``name`` -- its config is derived
    from the bound spec.
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
    ds = get_dataset(snowdb, name)
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
def add_dataset(manager: SnowDbManager, name: str, config_path: Path) -> None:
    """Register dataset NAME from its config at CONFIG_PATH (writes the link).

    The explicit-registration step: a dataset built/staged out of band (anywhere)
    goes live by linking its config into the root config. The config is validated
    (it must parse and its ingester must resolve) before the link is written.
    Idempotent -- re-adding a name overwrites its link. Going live also needs an
    ``pourpoint reindex`` + a service restart.
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

    manager.register_dataset(name, config_path)
    click.echo(
        f'registered {name} -> {config_path} '
        '(run `pourpoint reindex` + restart to go live)',
    )


@dataset.command('ingest')
@click.argument('name')
@click.argument(
    'archives',
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@config_option
@pass_manager
def ingest_dataset(
    manager: SnowDbManager,
    name: str,
    archives: tuple[Path, ...],
) -> None:
    """Ingest one or more source ARCHIVES into dataset NAME.

    Idempotent: re-ingesting an archive overwrites that date's COGs with the
    same (deterministic) result.
    """
    ds = get_dataset(manager.db, name)
    for archive in archives:
        try:
            dates = ds.ingest(archive, force=True)
        except (FileExistsError, SnowtoolError) as e:
            raise click.ClickException(str(e)) from e
        for ingested in dates:
            click.echo(f'ingested {name} {ingested.isoformat()} from {archive}')


@dataset.command('generate-zones')
@click.argument('name')
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
    name: str,
    provider_names: tuple[str, ...],
    sources: tuple[tuple[str, Path], ...],
    workers: int | None,
    block_size: int | None,
) -> None:
    """(Re)generate dataset NAME's zone layers, overwriting any existing ones.

    Generates every configured zone-layer provider (terrain, land cover, ...)
    unless ``--provider`` limits the selection; each uses its provider's default
    source unless ``--source PROVIDER PATH`` supplies a local file. Generation is
    heavy -- terrain reprojects the whole DEM source to a 10 m work grid (the
    default 3DEP source streams from S3) and land cover downloads the MRLC Annual
    NLCD national raster (~1.5 GB) on first use.
    """
    ds = get_dataset(manager.db, name)
    overrides = _resolve_source_overrides(manager.db, sources)

    try:
        generated = manager.generate_dataset_zone_layers(
            ds,
            provider_names=provider_names or None,
            source_overrides=overrides,
            force=True,
            options=GenerationOptions(workers=workers, block_size=block_size),
            progress_factory=lambda provider_name: ClickProgress(
                prefix=f'{name} {provider_name}: ',
            ),
        )
    except (ValueError, FileExistsError, SnowtoolError) as e:
        raise click.ClickException(str(e)) from e
    for provider_name in generated:
        click.echo(f'generated {provider_name} for {name}')


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
    """Remove a single ingested DATE from dataset NAME."""
    ds = get_dataset(manager.db, name)
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
    """Remove every ingested date in dataset NAME older than --before."""
    ds = get_dataset(manager.db, name)
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

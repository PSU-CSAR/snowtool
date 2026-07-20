"""The ``dataset`` command group: per-dataset management.

Thin wrappers over the ``Dataset`` API: each command resolves the dataset,
calls a domain method, and renders. Write commands (create/ingest/generate/
remove-date) first require an initialized snowdb root.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._confirm import confirm_destructive
from snowtool.cli._context import config_option, pass_manager, pass_snowdb
from snowtool.cli._datasets import format_option, get_dataset
from snowtool.cli._dates import DATE
from snowtool.cli._progress import RichProgress
from snowtool.cli._render import _emit, _emit_record
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

    ``cell_area_m2`` is ``null`` on a geographic grid (json/csv) -- the table
    form shows ``varies (geographic)`` instead. Likewise the elevation bracket
    is the two numeric fields ``min_elevation_m``/``max_elevation_m`` in
    json/csv, rendered as ``MIN .. MAX`` prose in the table.
    """
    from dataclasses import asdict

    from snowtool.snowdb.diagnostics import dataset_info_report

    ds = get_dataset(snowdb, name)
    report = dataset_info_report(snowdb, ds)

    record = asdict(report)
    record['extent'] = list(report.extent)
    record['variables'] = list(report.variables)
    record['dates'] = record.pop('date_count')
    record['first_date'] = report.first_date.isoformat() if report.first_date else None
    record['last_date'] = report.last_date.isoformat() if report.last_date else None
    if fmt == 'table':
        # Prose forms belong to the table only -- json/csv keep the typed fields
        # (`cell_area_m2: float | None`, numeric `min/max_elevation_m`).
        if report.cell_area_m2 is None:
            record['cell_area_m2'] = 'varies (geographic)'
        record['elevation_bracket_m'] = (
            f'{report.min_elevation_m} .. {report.max_elevation_m}'
        )
        del record['min_elevation_m'], record['max_elevation_m']
    _emit_record(record, fmt)


@dataset.command('dates')
@click.argument('name')
@click.option('--start', type=DATE, default=None, help='Only dates on/after this.')
@click.option('--end', type=DATE, default=None, help='Only dates on/before this.')
@click.option(
    '--missing',
    is_flag=True,
    help='List missing dates in the range instead of ingested ones.',
)
@format_option
@config_option
@pass_snowdb
def dataset_dates(
    snowdb: SnowDb,
    name: str,
    start: date | None,
    end: date | None,
    missing: bool,
    fmt: str,
) -> None:
    """Ingested (or, with --missing, missing) dates for dataset NAME.

    Without ``--missing``, lists every ingested date, optionally filtered by
    ``--start``/``--end``. With ``--missing``, lists every date absent between
    ``--start`` (default: the dataset's first ingested date) and ``--end``
    (default: today), inclusive.

    Resolves any *registered* dataset -- active or not -- like ``dataset info``.
    """
    from snowtool.snowdb import diagnostics

    ds = get_dataset(snowdb, name)
    if missing:
        dates = diagnostics.missing_dates(ds, start=start, end=end)
    else:
        dates = ds.available_dates(start=start, end=end)
    _emit([{'date': d.isoformat()} for d in dates], fmt)


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

    ds = get_dataset(snowdb, name)
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
    required=True,
    help='Stamp a built-in dataset template (e.g. snodas, swann-800m, instarr) as '
    'the new dataset NAME.',
)
@config_option
@pass_manager
def create_dataset(
    manager: SnowDbManager,
    name: str,
    template: str,
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
    live. The dataset's definition comes from ``--template`` (a built-in).
    Converge-by-default, like ingest: a re-create leaves current artifacts
    untouched (an AOI raster rebuilds only when its provenance tag reads
    stale) and never touches an existing registration (its active state and
    link are preserved). To force-rebuild AOI rasters regardless, use
    ``pourpoint rasterize --all --rebuild -d NAME``.
    """
    from snowtool.snowdb.datasets import DATASET_TEMPLATES, template_nodata_mask

    if template not in DATASET_TEMPLATES:
        known = ', '.join(sorted(DATASET_TEMPLATES))
        raise click.ClickException(
            f'No such template: {template!r}. Known templates: {known}.',
        )

    created = manager.create_dataset(
        name,
        DATASET_TEMPLATES[template],
        nodata_mask_source=template_nodata_mask(template),
        progress=RichProgress(),
    )

    if created.staged.created:
        click.echo(f'created dataset {name} at {created.staged.dataset.path}')
    else:
        click.echo(f'dataset {name} already created')

    if created.registered:
        click.echo(
            f'registered {name} (inactive; generate zones with '
            f"'dataset generate-zones {name}', activate with "
            f"'dataset activate {name}')",
        )


@dataset.command('register')
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
    band): ``dataset create`` already stages *and* registers, so ``register``
    exists only to link an externally built config into the root config.
    Registration is inactive: the dataset becomes manageable by name (ingest,
    generate-zones, diagnostics) but invisible to readers until
    ``dataset activate NAME``. The config is validated (it must parse and its
    ingester must resolve) before the link is written. Idempotent -- re-registering
    a name overwrites its link. Since ``register`` skips staging, pourpoint
    coverage for the new dataset is unknown until the next ``pourpoint reindex``.
    """
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
    manager.set_dataset_active(name, True)
    # Activation never needs a reindex: `dataset create` folds coverage into the
    # index at registration. The one exception (`dataset register`, which skips
    # staging) prints its own reindex guidance at registration time.
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
    manager.set_dataset_active(name, False)
    click.echo(f'deactivated {name} (restart the API to pick it up)')


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
@format_option
@config_option
@pass_manager
def ingest_dataset(
    manager: SnowDbManager,
    name: str,
    source: Path,
    force: bool,
    fmt: str,
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
    hash is left untouched (reported ``up-to-date``); a re-release under the
    same filename with different bytes rebuilds. ``--force`` rebuilds every
    date regardless.
    """
    ds = manager.resolve_dataset(name)
    result = ds.ingest(
        source,
        force=force,
        progress=RichProgress(prefix=f'{name} ingest: '),
    )

    rows = [
        {
            'dataset': name,
            'date': d.isoformat(),
            'action': action,
            'source': str(source),
        }
        for action, dates in (
            ('ingested', result.ingested),
            ('up-to-date', result.skipped),
        )
        for d in dates
    ]
    _emit(rows, fmt)


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
    datasets = [manager.resolve_dataset(name) for name in names]
    generated = manager.generate_zone_layers_for(
        datasets,
        provider_names=provider_names or None,
        source_overrides=dict(sources),
        force=True,
        options=GenerationOptions(workers=workers, block_size=block_size),
        progress_factory=lambda provider_name: RichProgress(
            prefix=f'{provider_name}: ',
        ),
    )
    for provider_name, hashes in generated.items():
        for ds_name in hashes:
            click.echo(f'generated {provider_name} for {ds_name}')


@dataset.command('remove-date')
@click.argument('name')
@click.argument('removal_date', metavar='DATE', type=DATE)
@click.option('--dry-run', is_flag=True, help='Show what would be removed only.')
@click.option('--yes', is_flag=True, help='Skip the confirmation prompt.')
@config_option
@pass_manager
def remove_date(
    manager: SnowDbManager,
    name: str,
    removal_date: date,
    dry_run: bool,
    yes: bool,
) -> None:
    """Remove a single ingested DATE from dataset NAME.

    NAME is a registered dataset name (active or not) or a dataset config path.
    """
    ds = manager.resolve_dataset(name)
    iso = removal_date.isoformat()

    if dry_run:
        present = ds.remove_date(removal_date, dry_run=True)
        click.echo(f'would remove {name} {iso}' if present else f'{name} {iso}: absent')
        return

    confirm_destructive(f'Remove {name} {iso}? This deletes its COGs.', yes=yes)

    if ds.remove_date(removal_date):
        click.echo(f'removed {name} {iso}')
    else:
        click.echo(f'{name} {iso}: absent (nothing removed)')

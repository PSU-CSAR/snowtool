"""The ``report`` command group: read-only diagnostics over a snowdb.

Every command resolves datasets, calls a builder in
:mod:`snowtool.snowdb.diagnostics` (where the scan logic lives and is unit
tested), and renders. Findings-style reports (completeness, missing-files,
aoi-coverage, aoi-health) print only problems -- empty output means clean, which
is what :command:`snowdb validate` rolls up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from snowtool.cli._context import pass_snowdb
from snowtool.cli._datasets import (
    dataset_option,
    format_option,
    get_dataset,
    resolve_datasets,
)
from snowtool.cli._render import DATE, _emit, _emit_record
from snowtool.snowdb import diagnostics

if TYPE_CHECKING:
    from datetime import date

    from snowtool.snowdb.db import SnowDb


@click.group()
def report() -> None:
    """Read-only diagnostics and health reports."""


@report.command('coverage')
@dataset_option
@format_option
@pass_snowdb
def coverage(snowdb: SnowDb, dataset_names: tuple[str, ...], fmt: str) -> None:
    """Date span, ingested-date count, and interior gaps per dataset."""
    rows = []
    for ds in resolve_datasets(snowdb, dataset_names):
        result = diagnostics.coverage_report(ds)
        rows.append(
            {
                'dataset': result.name,
                'dates': result.date_count,
                'first': result.first_date.isoformat() if result.first_date else '',
                'last': result.last_date.isoformat() if result.last_date else '',
                'gaps': len(result.gaps),
                'gap_ranges': '; '.join(
                    f'{start.isoformat()}..{end.isoformat()}'
                    for start, end in result.gaps
                ),
            },
        )
    _emit(rows, fmt)


@report.command('completeness')
@dataset_option
@click.option('--start', type=DATE, default=None, help='Only dates on/after this.')
@click.option('--end', type=DATE, default=None, help='Only dates on/before this.')
@format_option
@pass_snowdb
def completeness(
    snowdb: SnowDb,
    dataset_names: tuple[str, ...],
    start: date | None,
    end: date | None,
    fmt: str,
) -> None:
    """Ingested dates that are missing one or more of a dataset's variables."""
    rows = []
    for ds in resolve_datasets(snowdb, dataset_names):
        for finding in diagnostics.completeness_report(ds, start=start, end=end):
            rows.append(
                {
                    'dataset': finding.name,
                    'date': finding.date.isoformat(),
                    'missing': ', '.join(finding.missing),
                },
            )
    _emit(rows, fmt)


@report.command('missing-files')
@dataset_option
@format_option
@pass_snowdb
def missing_files(snowdb: SnowDb, dataset_names: tuple[str, ...], fmt: str) -> None:
    """Datasets missing an expected artifact (dem/area/cogs/aoi-rasters)."""
    rows = []
    for ds in resolve_datasets(snowdb, dataset_names):
        missing = diagnostics.missing_artifacts(ds)
        if missing:
            rows.append({'dataset': ds.spec.name, 'missing': ', '.join(missing)})
    _emit(rows, fmt)


@report.command('aoi-coverage')
@dataset_option
@format_option
@pass_snowdb
def aoi_coverage(snowdb: SnowDb, dataset_names: tuple[str, ...], fmt: str) -> None:
    """Global AOIs with no raster in a dataset, and orphaned AOI rasters."""
    rows: list[dict[str, str]] = []
    for ds in resolve_datasets(snowdb, dataset_names):
        result = diagnostics.aoi_coverage_report(snowdb, ds)
        rows.extend(
            {'dataset': result.name, 'triplet': triplet, 'issue': 'no raster'}
            for triplet in result.unrasterized
        )
        rows.extend(
            {'dataset': result.name, 'triplet': triplet, 'issue': 'orphan raster'}
            for triplet in result.orphan_rasters
        )
    _emit(rows, fmt)


@report.command('aoi-health')
@dataset_option
@format_option
@pass_snowdb
def aoi_health(snowdb: SnowDb, dataset_names: tuple[str, ...], fmt: str) -> None:
    """AOI rasters that won't read cleanly (bad tag / no DEM overlap / etc.)."""
    rows = []
    for ds in resolve_datasets(snowdb, dataset_names):
        for finding in diagnostics.aoi_health_report(ds):
            if not finding.ok:
                rows.append(
                    {
                        'dataset': finding.name,
                        'triplet': finding.triplet,
                        'issue': finding.issue,
                    },
                )
    _emit(rows, fmt)


@report.command('value-ranges')
@click.argument('name')
@click.option(
    '--date',
    'on_date',
    type=DATE,
    default=None,
    help='Date to report (default: latest ingested).',
)
@format_option
@pass_snowdb
def value_ranges(
    snowdb: SnowDb,
    name: str,
    on_date: date | None,
    fmt: str,
) -> None:
    """Per-variable min/max/mean (unit-scaled) and nodata % for one date."""
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


@report.command('grid')
@click.argument('name')
@format_option
@pass_snowdb
def grid_info(snowdb: SnowDb, name: str, fmt: str) -> None:
    """A dataset grid's CRS, extent, tiling, and cell area."""
    ds = get_dataset(snowdb, name)
    result = diagnostics.grid_report(ds)
    record = {
        'name': result.name,
        'crs': result.crs,
        'is_geographic': result.is_geographic,
        'rows': result.rows,
        'cols': result.cols,
        'px_size': result.px_size,
        'tile_size': result.tile_size,
        'n_tiles': result.n_tiles,
        'extent': list(result.extent),
        'cell_area_m2': (
            'varies (geographic)'
            if result.cell_area_m2 is None
            else result.cell_area_m2
        ),
    }
    _emit_record(record, fmt)

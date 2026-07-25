"""Diagnostic helpers over snowdb domain data, kept out of the click callbacks.

Two kinds live here, both returning plain dataclasses the CLI renders: pure
functions over already-gathered data (e.g. :func:`missing_dates`), and
dataset-scan *builders* (e.g. :func:`dataset_status`) that read a
:class:`Dataset` via its query helpers. Keeping the scan/finding logic here --
not in click callbacks -- makes it unit-testable without a CliRunner; the
commands just gather inputs and format the results.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.grid import grid_extent

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset, DatasetArtifacts
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.grid import Extent


@dataclass(frozen=True)
class DatasetStatus:
    """A one-line overview of a dataset's on-disk state (for ``snowtool status``)."""

    name: str
    present: bool  # the data/<name>/ directory exists
    artifacts: DatasetArtifacts
    date_count: int
    first_date: date | None
    last_date: date | None

    def to_row(self, *, active: bool) -> dict[str, object]:
        """Flatten to one ``snowtool status`` row.

        ``active`` (reader visibility) is not an on-disk fact this scan carries,
        so the caller passes it. The row interleaves a dynamic column per
        configured zone-layer provider (terrain, landcover, ...) between the
        presence columns and the artifact counts; ``first``/``last`` are ISO
        strings (empty when the dataset has no ingested dates).
        """
        row: dict[str, object] = {
            'dataset': self.name,
            'active': active,
            'present': self.present,
        }
        # One column per configured zone-layer provider (terrain, landcover, ...).
        for provider_name, present in sorted(self.artifacts.zone_layers.items()):
            row[provider_name] = present
        row.update(
            {
                'cogs': self.artifacts.cogs,
                'aoi_rasters': self.artifacts.aoi_rasters,
                'dates': self.date_count,
                'first': self.first_date.isoformat() if self.first_date else '',
                'last': self.last_date.isoformat() if self.last_date else '',
            },
        )
        return row


def dataset_status(dataset: Dataset) -> DatasetStatus:
    """Scan a dataset's directory into a :class:`DatasetStatus` snapshot."""
    dates = dataset.available_dates()
    return DatasetStatus(
        name=dataset.spec.name,
        present=dataset.path.is_dir(),
        artifacts=dataset.artifact_status(),
        date_count=len(dates),
        first_date=dates[0] if dates else None,
        last_date=dates[-1] if dates else None,
    )


def missing_dates(
    dataset: Dataset,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """Every date absent from ``dataset`` within ``[start, end]`` (inclusive).

    ``start`` defaults to the dataset's first ingested date; ``end`` defaults to
    today. Raises :class:`~snowtool.exceptions.QueryParameterError` if ``start``
    is omitted and the dataset has no ingested dates (there is no range start to
    infer). A ``start`` after ``end`` yields an empty list rather than erroring.
    """
    ingested = set(dataset.available_dates())
    if start is None:
        if not ingested:
            raise QueryParameterError(
                f'{dataset.spec.name} has no ingested dates; pass start explicitly',
            )
        start = min(ingested)
    if end is None:
        end = date.today()  # noqa: DTZ011 - a calendar date, not a timestamp

    n_days = (end - start).days + 1
    return [
        d for i in range(n_days) if (d := start + timedelta(days=i)) not in ingested
    ]


@dataclass(frozen=True)
class VariableRange:
    """The (unit-scaled) value range of one variable on one date."""

    variable: str
    unit: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    nodata_pct: float


def value_ranges_report(
    dataset: Dataset,
    on_date: date | None = None,
) -> list[VariableRange]:
    """Per-variable min/max/mean (unit-scaled) and nodata % for ``on_date``.

    ``on_date`` defaults to the dataset's latest ingested date; raises
    :class:`~snowtool.exceptions.QueryParameterError` if the dataset has no
    ingested dates (there is no "latest" to default to), or if the resolved
    date has no variable files at all (nothing to report).
    """
    import rasterio

    name = dataset.spec.name
    if on_date is None:
        dates = dataset.available_dates()
        if not dates:
            raise QueryParameterError(f'{name} has no ingested dates')
        on_date = dates[-1]

    findings: list[VariableRange] = []
    for _key, variable in sorted(dataset.spec.variables.items()):
        path = dataset.variable_path(on_date, variable)
        if path is None:
            continue
        with rasterio.open(path) as src:
            array = src.read(1)
        valid = array[array != variable.nodata]
        nodata_pct = (
            100.0 * (array.size - valid.size) / array.size if array.size else 0.0
        )
        scale = variable.unit.scale
        minimum = scale(float(valid.min())) if valid.size else None
        maximum = scale(float(valid.max())) if valid.size else None
        mean = scale(float(valid.mean())) if valid.size else None
        findings.append(
            VariableRange(
                variable=variable.key,
                unit=variable.unit.name,
                minimum=minimum,
                maximum=maximum,
                mean=mean,
                nodata_pct=nodata_pct,
            ),
        )
    if not findings:
        raise QueryParameterError(
            f'{name} has no variable files for {on_date.isoformat()}',
        )
    return findings


@dataclass(frozen=True)
class GridReport:
    """A dataset grid's geometry summary (spec-derived; no filesystem)."""

    name: str
    crs: str
    is_geographic: bool
    rows: int
    cols: int
    px_size: float
    tile_size: int
    n_tiles: int
    extent: Extent  # left, bottom, right, top
    cell_area_m2: float | None  # None on a geographic grid (per-pixel area raster)


def grid_report(dataset: Dataset) -> GridReport:
    spec = dataset.spec
    grid = spec.grid_params
    n_tiles = math.ceil(grid.rows / grid.tile_size) * math.ceil(
        grid.cols / grid.tile_size,
    )
    return GridReport(
        name=spec.name,
        crs=str(grid.crs),
        is_geographic=spec.is_geographic,
        rows=grid.rows,
        cols=grid.cols,
        px_size=grid.px_size,
        tile_size=grid.tile_size,
        n_tiles=n_tiles,
        extent=grid_extent(dataset.grid),
        cell_area_m2=None if spec.is_geographic else spec.cell_area,
    )


@dataclass(frozen=True)
class DatasetInfoReport:
    """Everything ``dataset info`` reports about one dataset.

    Nests :func:`dataset_status` (on-disk presence/artifacts/date span) and
    :func:`grid_report` (grid geometry) rather than flat-copying their fields,
    alongside the spec-level fields (variables, zone config, active flag) and
    per-provider zone-layer presence/provenance -- one scan each, no duplicated
    filesystem walks.
    ``grid.cell_area_m2`` is ``None`` on a geographic grid (per-pixel area
    raster, see :class:`GridReport`); ``min_elevation_m``/``max_elevation_m``
    are the shared elevation bracket
    (:data:`~snowtool.snowdb.constants.MIN_ELEVATION_M` /
    :data:`~snowtool.snowdb.constants.MAX_ELEVATION_M`) that elevation banding
    zones across, not a per-dataset measurement.
    """

    name: str
    active: bool
    status: DatasetStatus
    grid: GridReport
    zones: dict[str, dict[str, dict | None]]
    min_elevation_m: float
    max_elevation_m: float
    variables: tuple[str, ...]
    zone_layers: dict[str, dict[str, object]]  # provider -> {present, hash}

    def to_row(self) -> dict[str, object]:
        """Flatten to the format-independent ``dataset info`` output record.

        Spreads the nested ``status``/``grid`` reports back to the top level
        (dropping their redundant ``name``/``artifacts``) so output is a single
        flat record with a stable key order. ``date_count`` -> ``dates`` (the
        CLI's public name); ``extent``/``variables`` to plain lists (json/csv
        friendly); ``first_date``/``last_date`` to ISO strings (or ``None``).
        The record is presentation-neutral -- one typed shape for every
        ``--format``. The table form's prose substitutions
        (``'varies (geographic)'`` for a geographic grid's null cell area, the
        ``'MIN .. MAX'`` elevation bracket) live in the CLI renderer, not here.
        """
        status = self.status
        grid = self.grid
        return {
            'name': self.name,
            'active': self.active,
            'present': status.present,
            'crs': grid.crs,
            'is_geographic': grid.is_geographic,
            'rows': grid.rows,
            'cols': grid.cols,
            'tile_size': grid.tile_size,
            'cell_area_m2': grid.cell_area_m2,
            'px_size': grid.px_size,
            'n_tiles': grid.n_tiles,
            'extent': list(grid.extent),
            'zones': self.zones,
            'min_elevation_m': self.min_elevation_m,
            'max_elevation_m': self.max_elevation_m,
            'variables': list(self.variables),
            'zone_layers': self.zone_layers,
            'cogs': status.artifacts.cogs,
            'aoi_rasters': status.artifacts.aoi_rasters,
            'dates': status.date_count,
            'first_date': status.first_date.isoformat() if status.first_date else None,
            'last_date': status.last_date.isoformat() if status.last_date else None,
        }


def dataset_info_report(snowdb: SnowDb, dataset: Dataset) -> DatasetInfoReport:
    """Assemble a :class:`DatasetInfoReport` for ``dataset info``; see that class
    for the nesting rationale."""
    from snowtool.snowdb.constants import MAX_ELEVATION_M, MIN_ELEVATION_M

    spec = dataset.spec
    status = dataset_status(dataset)
    grid = grid_report(dataset)
    artifacts = status.artifacts

    return DatasetInfoReport(
        name=spec.name,
        active=spec.name in snowdb.datasets,
        status=status,
        grid=grid,
        zones={
            provider: {
                layer: params.model_dump() if params is not None else None
                for layer, params in layers.items()
            }
            for provider, layers in spec.zones.items()
        },
        min_elevation_m=MIN_ELEVATION_M,
        max_elevation_m=MAX_ELEVATION_M,
        variables=tuple(sorted(spec.variables)),
        zone_layers={
            name: {
                'present': artifacts.zone_layers[name],
                'hash': dataset.zones[name].provenance_hash(),
            }
            for name in dataset.zones
        },
    )

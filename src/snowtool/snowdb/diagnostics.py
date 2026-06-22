"""Diagnostic helpers over snowdb domain data, kept out of the click callbacks.

Two kinds live here, both returning plain dataclasses the CLI renders: pure
functions over already-gathered data (e.g. :func:`date_gaps`), and dataset-scan
*builders* (e.g. :func:`dataset_status`) that read a :class:`Dataset` via its
query helpers. Keeping the scan/finding logic here -- not in click callbacks --
makes it unit-testable without a CliRunner; the commands just gather inputs and
format the results.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from typing import TYPE_CHECKING

from snowtool import types

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    from snowtool.snowdb.dataset import Dataset, DatasetArtifacts
    from snowtool.snowdb.db import SnowDb


def date_gaps(dates: Iterable[date]) -> list[tuple[date, date]]:
    """Maximal runs of missing days *within* the span of ``dates``.

    Each gap is returned as an inclusive ``(first_missing, last_missing)`` pair.
    Only interior gaps are reported -- nothing before the earliest or after the
    latest date -- so a contiguous (or fewer-than-two-date) input yields ``[]``.
    Duplicate dates are ignored.
    """
    ordered = sorted(set(dates))
    one_day = timedelta(days=1)
    return [
        (earlier + one_day, later - one_day)
        for earlier, later in pairwise(ordered)
        if later - earlier > one_day
    ]


@dataclass(frozen=True)
class DatasetStatus:
    """A one-line overview of a dataset's on-disk state (for ``snowdb status``)."""

    name: str
    present: bool  # the data/<name>/ directory exists
    artifacts: DatasetArtifacts
    date_count: int
    first_date: date | None
    last_date: date | None


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


# --- report builders (read-only; the `report`/`validate` commands render these) -


@dataclass(frozen=True)
class CoverageReport:
    """A dataset's date span and the interior gaps in it."""

    name: str
    date_count: int
    first_date: date | None
    last_date: date | None
    gaps: tuple[tuple[date, date], ...]


def coverage_report(dataset: Dataset) -> CoverageReport:
    dates = dataset.available_dates()
    return CoverageReport(
        name=dataset.spec.name,
        date_count=len(dates),
        first_date=dates[0] if dates else None,
        last_date=dates[-1] if dates else None,
        gaps=tuple(date_gaps(dates)),
    )


@dataclass(frozen=True)
class IncompleteDate:
    """An ingested date that is missing one or more of its dataset's variables."""

    name: str
    date: date
    missing: tuple[str, ...]  # variable keys


def completeness_report(
    dataset: Dataset,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[IncompleteDate]:
    """Ingested dates (optionally within ``start``/``end``) missing variables."""
    findings: list[IncompleteDate] = []
    for d in dataset.available_dates():
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        missing = dataset.missing_variables(d)
        if missing:
            findings.append(
                IncompleteDate(
                    dataset.spec.name,
                    d,
                    tuple(sorted(variable.key for variable in missing)),
                ),
            )
    return findings


def missing_artifacts(dataset: Dataset) -> list[str]:
    """The dataset's expected on-disk artifacts that are absent.

    Skips the area raster on a projected grid, where it is not applicable
    (``DatasetArtifacts.area is None``). Every configured zone layer (terrain,
    land cover, ...) is expected -- ``snowdb init`` builds each from its default
    source -- so a missing one is a finding, reported by provider name.
    """
    artifacts = dataset.artifact_status()
    missing: list[str] = [
        name for name, present in artifacts.zone_layers.items() if not present
    ]
    if artifacts.area is False:
        missing.append('area')
    if not artifacts.cogs:
        missing.append('cogs')
    if not artifacts.aoi_rasters:
        missing.append('aoi-rasters')
    return missing


@dataclass(frozen=True)
class AoiCoverage:
    """How a dataset's burned AOI rasters line up with the global AOIs."""

    name: str
    unrasterized: tuple[str, ...]  # global AOIs with no raster in this dataset
    orphan_rasters: tuple[str, ...]  # rasters with no matching global AOI


def aoi_coverage_report(snowdb: SnowDb, dataset: Dataset) -> AoiCoverage:
    global_aois = snowdb.aoi_triplets()
    rasterized = dataset.aoi_raster_triplets()
    return AoiCoverage(
        name=dataset.spec.name,
        unrasterized=tuple(sorted(global_aois - rasterized)),
        orphan_rasters=tuple(sorted(rasterized - global_aois)),
    )


@dataclass(frozen=True)
class AoiRasterHealth:
    """The health of one burned AOI raster (opened to check its metadata)."""

    name: str
    triplet: str
    ok: bool
    issue: str | None  # None when healthy


def aoi_health_report(dataset: Dataset) -> list[AoiRasterHealth]:
    """Open each AOI raster and classify any that won't read cleanly."""
    from snowtool.snowdb.raster import AOIRaster

    findings: list[AoiRasterHealth] = []
    for path in dataset.aoi_raster_paths():
        triplet = types.stem_to_triplet(path.stem)
        issue: str | None = None
        try:
            aoi_raster = AOIRaster.open(path, dataset.grid)
        except ValueError:
            issue = 'missing SNOWTOOL_TILE_BBOX tag (run `migration aoi-tags`)'
        except Exception as e:  # noqa: BLE001 - a health scan reports any read failure
            issue = f'unreadable: {e}'
        else:
            # The mask burned to all-zero: the AOI polygon falls outside the grid,
            # so it would contribute no pixels to any query.
            if not aoi_raster.array.any():
                issue = 'empty mask (AOI does not overlap the grid)'
        findings.append(
            AoiRasterHealth(dataset.spec.name, triplet, issue is None, issue),
        )
    return findings


@dataclass(frozen=True)
class VariableRange:
    """The (unit-scaled) value range of one variable on one date."""

    variable: str
    unit: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    nodata_pct: float


def value_ranges_report(dataset: Dataset, d: date) -> list[VariableRange]:
    """Per-variable min/max/mean (unit-scaled) and nodata % for date ``d``."""
    import rasterio

    findings: list[VariableRange] = []
    for _key, variable in sorted(dataset.spec.variables.items()):
        path = dataset.variable_path(d, variable)
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
    extent: tuple[float, float, float, float]  # left, bottom, right, top
    cell_area_m2: float | None  # None on a geographic grid (per-pixel area raster)


def grid_report(dataset: Dataset) -> GridReport:
    spec = dataset.spec
    grid = spec.grid_params
    left = grid.origin_x
    top = grid.origin_y
    right = grid.origin_x + grid.cols * grid.px_size
    bottom = grid.origin_y - grid.rows * grid.px_size
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
        extent=(left, bottom, right, top),
        cell_area_m2=None if spec.is_geographic else spec.cell_area,
    )

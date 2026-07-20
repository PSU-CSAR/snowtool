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
from datetime import date, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING

from snowtool.exceptions import IncompleteDatasetDataError, QueryParameterError
from snowtool.snowdb import triplet_naming

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from affine import Affine

    from snowtool.snowdb.dataset import Dataset, DatasetArtifacts
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.grid import Extent


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
    """A one-line overview of a dataset's on-disk state (for ``snowtool status``)."""

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

    one_day = timedelta(days=1)
    missing: list[date] = []
    current = start
    while current <= end:
        if current not in ingested:
            missing.append(current)
        current += one_day
    return missing


# --- report builders (read-only; the `dataset`/`doctor` commands render these) --


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

    Every configured zone layer (terrain, land cover, ...) is expected --
    ``snowtool init`` builds each from its default source -- so a missing one is a
    finding. An incomplete zone-layer set names the specific layer files that are
    absent (``terrain (elevation.tif, aspect_majority.tif)``) so the finding is
    actionable, not just the provider name.
    """
    artifacts = dataset.artifact_status()
    missing: list[str] = []
    for name, present in artifacts.zone_layers.items():
        if present:
            continue
        absent = ', '.join(
            layer.filename for layer in dataset.zones[name].missing_layers()
        )
        missing.append(f'{name} ({absent})' if absent else name)
    if not artifacts.cogs:
        missing.append('cogs')
    if not artifacts.aoi_rasters:
        missing.append('aoi-rasters')
    return missing


@dataclass(frozen=True)
class ZoneLayerFormat:
    """A built zone-layer set whose stamped format version is out of date.

    ``stored`` is the version read off the set's provenance tag (``None`` for a
    missing/legacy tag); ``expected`` is the provider's current format version.
    Only stale sets are emitted, so ``stored != expected`` always holds.
    """

    name: str  # dataset name
    provider: str  # zone-layer provider (terrain, landcover, ...)
    stored: int | None
    expected: int


def stale_format_zone_layers(dataset: Dataset) -> list[ZoneLayerFormat]:
    """Built zone-layer sets stamped with an out-of-date on-disk format version.

    Skips sets that are not built (``missing_artifacts`` already reports those);
    a built set whose stamped version differs from the provider's current one --
    including a missing/legacy tag (stored ``None``) -- is flagged for a rebuild.
    """
    findings: list[ZoneLayerFormat] = []
    for provider_name, zone_set in dataset.zones.items():
        if zone_set.format_is_current() is False:
            findings.append(
                ZoneLayerFormat(
                    name=dataset.spec.name,
                    provider=provider_name,
                    stored=zone_set.stored_format_version(),
                    expected=zone_set.format_version,
                ),
            )
    return findings


@dataclass(frozen=True)
class PourpointCoverage:
    """How a dataset's grid + burned rasters line up with the stored pourpoints.

    ``unrasterized``/``orphan_rasters`` are about which AOI *rasters* exist;
    ``partial``/``uncovered`` are the geometric coverage of each pourpoint's basin
    by the dataset's grid (``partial`` = basin spills outside it, ``uncovered`` =
    basin entirely off-grid). A fully-covered pourpoint appears in none of these.
    """

    name: str
    unrasterized: tuple[str, ...]  # pourpoints with no AOI raster in this dataset
    orphan_rasters: tuple[str, ...]  # AOI rasters with no matching pourpoint
    partial: tuple[str, ...]  # basin only partially inside the grid
    uncovered: tuple[str, ...]  # basin entirely outside the grid


def pourpoint_coverage_report(snowdb: SnowDb, dataset: Dataset) -> PourpointCoverage:
    from snowtool.snowdb.coverage import Coverage, dataset_coverage

    triplets = snowdb.pourpoint_triplets()
    rasterized = dataset.aoi_raster_triplets()
    # Coverage is computed live from each stored basin (this is validation -- it
    # must not trust the derived index, which it exists to catch drift in).
    partial: list[str] = []
    uncovered: list[str] = []
    domain = dataset.coverage_domain
    for pourpoint in snowdb.pourpoints():
        match dataset_coverage(pourpoint, domain):
            case Coverage.PARTIAL:
                partial.append(pourpoint.station_triplet)
            case Coverage.NONE:
                uncovered.append(pourpoint.station_triplet)
            case _:
                pass
    return PourpointCoverage(
        name=dataset.spec.name,
        unrasterized=tuple(sorted(triplets - rasterized)),
        orphan_rasters=tuple(sorted(rasterized - triplets)),
        partial=tuple(sorted(partial)),
        uncovered=tuple(sorted(uncovered)),
    )


@dataclass(frozen=True)
class AoiRasterIssue:
    """A burned AOI raster that failed to open cleanly, or reads as empty."""

    name: str
    triplet: str
    issue: str


def aoi_health_report(dataset: Dataset) -> list[AoiRasterIssue]:
    """Open each AOI raster and report any that won't read cleanly."""
    from snowtool.snowdb.aoi_raster import AOIRaster

    findings: list[AoiRasterIssue] = []
    for path in dataset.aoi_raster_paths():
        triplet = triplet_naming.stem_to_triplet(path.stem)
        issue: str | None = None
        try:
            aoi_raster = AOIRaster.open(path, dataset.grid)
        except IncompleteDatasetDataError:
            issue = (
                'missing SNOWTOOL_TILE_BBOX tag '
                '(rebuild with `pourpoint rasterize --rebuild`)'
            )
        except Exception as e:  # noqa: BLE001 - a health scan reports any read failure
            issue = f'unreadable: {e}'
        else:
            # Burned to all-zero (no in-basin cell area): the AOI polygon falls
            # outside the grid, so it would contribute no pixels to any query.
            if not aoi_raster.array.any():
                issue = 'empty AOI (does not overlap the grid, or is entirely masked)'
        if issue is not None:
            findings.append(AoiRasterIssue(dataset.spec.name, triplet, issue))
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
    extent: Extent  # left, bottom, right, top
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


@dataclass(frozen=True)
class DatasetInfoReport:
    """Everything ``dataset info`` reports about one dataset, typed and flat.

    Composes :func:`dataset_status` (on-disk presence/artifacts/date span) and
    :func:`grid_report` (grid geometry) with the spec-level fields (variables,
    zone config, active flag) and per-provider zone-layer presence/provenance --
    one scan each, no duplicated filesystem walks. ``cell_area_m2`` is ``None``
    on a geographic grid (per-pixel area raster, see :class:`GridReport`);
    ``min_elevation_m``/``max_elevation_m`` are the shared elevation bracket
    (:data:`~snowtool.snowdb.constants.MIN_ELEVATION_M` /
    :data:`~snowtool.snowdb.constants.MAX_ELEVATION_M`) that elevation banding
    zones across, not a per-dataset measurement.
    """

    name: str
    active: bool
    present: bool
    crs: str
    is_geographic: bool
    rows: int
    cols: int
    tile_size: int
    cell_area_m2: float | None
    px_size: float
    n_tiles: int
    extent: Extent
    zones: dict[str, dict[str, dict | None]]
    min_elevation_m: float
    max_elevation_m: float
    variables: tuple[str, ...]
    zone_layers: dict[str, dict[str, object]]  # provider -> {present, hash}
    cogs: bool
    aoi_rasters: bool
    date_count: int
    first_date: date | None
    last_date: date | None


def dataset_info_report(snowdb: SnowDb, dataset: Dataset) -> DatasetInfoReport:
    """Assemble a :class:`DatasetInfoReport` for ``dataset info``.

    Composes :func:`dataset_status` and :func:`grid_report` rather than
    re-scanning the dataset -- this is the one place their fields are merged
    with the spec-derived ones (variables, zone config, per-provider zone-layer
    hashes) that neither builder carries.
    """
    from snowtool.snowdb.constants import MAX_ELEVATION_M, MIN_ELEVATION_M

    spec = dataset.spec
    status = dataset_status(dataset)
    grid = grid_report(dataset)
    artifacts = status.artifacts

    return DatasetInfoReport(
        name=spec.name,
        active=spec.name in snowdb.datasets,
        present=status.present,
        crs=grid.crs,
        is_geographic=grid.is_geographic,
        rows=grid.rows,
        cols=grid.cols,
        tile_size=grid.tile_size,
        cell_area_m2=grid.cell_area_m2,
        px_size=grid.px_size,
        n_tiles=grid.n_tiles,
        extent=grid.extent,
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
        cogs=artifacts.cogs,
        aoi_rasters=artifacts.aoi_rasters,
        date_count=status.date_count,
        first_date=status.first_date,
        last_date=status.last_date,
    )


def _first_present_cog(dataset: Dataset) -> Path | None:
    """The first variable COG present on disk (scanning dates ascending)."""
    for d in dataset.available_dates():
        for variable in dataset.spec.variables.values():
            path = dataset.variable_path(d, variable)
            if path is not None:
                return path
    return None


def _transforms_close(a: Affine, b: Affine) -> bool:
    """Whether two affine transforms agree to within float noise."""
    return all(
        math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
        for x, y in zip(tuple(a)[:6], tuple(b)[:6], strict=True)
    )


def grid_validation_report(dataset: Dataset) -> list[str]:
    """Cheap declaration-vs-reality checks for ``snowtool doctor``.

    Returns a list of human-readable problems (empty == consistent):

    1. **Ingester vs variables.** An ingester with no variables has nothing to
       write -- almost certainly a misconfiguration. (The reverse -- variables but
       no ingester -- is *not* flagged: that is a valid read-only/derived dataset,
       populated out of band.)
    2. **Declared grid vs the first present COG.** Opens the first variable COG on
       disk and checks its shape + transform against the declared grid, catching a
       config that has drifted from the real rasters. Skipped when no COG exists
       yet (that is a completeness concern, not an inconsistency).

    A deeper variables-vs-ingester check (the ingester's *required* variable keys
    being a subset of those declared) would need the ``Ingester`` protocol to
    expose its expected keys; that is left as a follow-up.
    """
    import rasterio

    issues: list[str] = []
    spec = dataset.spec

    if spec.ingester is not None and not spec.variables:
        issues.append('has an ingester but declares no variables')

    cog = _first_present_cog(dataset)
    if cog is not None:
        grid = spec.grid_params
        declared = dataset.grid.base_grid.transform
        with rasterio.open(cog) as src:
            actual = src.transform
            width, height = src.width, src.height
        if (width, height) != (grid.cols, grid.rows):
            issues.append(
                f'declared grid is {grid.cols}x{grid.rows} (cols x rows) but COG '
                f'{cog.name} is {width}x{height}',
            )
        if not _transforms_close(declared, actual):
            issues.append(
                f'declared grid transform {tuple(declared)[:6]} does not match '
                f'COG {cog.name} transform {tuple(actual)[:6]}',
            )
    return issues

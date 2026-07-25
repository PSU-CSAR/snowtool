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
from functools import partial
from typing import TYPE_CHECKING

from snowtool.exceptions import (
    IncompleteDatasetDataError,
    QueryParameterError,
    UnknownHealthCheckError,
)
from snowtool.snowdb import issues as issues_mod
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.grid import grid_extent
from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.query import DateRangeQuery

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from snowtool.snowdb.dataset import Dataset, DatasetArtifacts
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.grid import Extent
    from snowtool.snowdb.progress import ProgressReporter


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


# --- report builders (read-only; the `dataset`/`doctor` commands render these) --
#
# A ``Finding`` is one row of ``snowtool doctor`` output: the flat, uniform
# ``check``/``dataset``/``target``/``issue`` dict the CLI renders. The four
# doctor-only checks below compose their ``target``/``issue`` prose *where the
# condition is detected* and return these rows directly -- there is no separate
# typed-intermediate tier for the CLI to re-flatten. The facts each row encodes
# are pinned by ``tests/snowdb/test_report_diagnostics.py``.

type Finding = dict[str, str]


def _finding(check: str, dataset: str, target: str, issue: str) -> Finding:
    return {'check': check, 'dataset': dataset, 'target': target, 'issue': issue}


def completeness_report(
    dataset: Dataset,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[Finding]:
    """``dates`` findings: ingested dates (optionally within ``start``/``end``)
    missing one or more of their dataset's variables."""
    window = DateRangeQuery(start_date=start, end_date=end)
    findings: list[Finding] = []
    for d in window.select(dataset.available_dates()):
        findings.extend(_date_completeness(dataset, d))
    return findings


def _date_completeness(dataset: Dataset, d: date) -> list[Finding]:
    """The ``dates`` finding (if any) for a single ingested date ``d``."""
    unresolved = dataset.unresolved_variables(d)
    if not unresolved:
        return []
    return [
        _finding(
            'dates',
            dataset.spec.name,
            d.isoformat(),
            f'missing {", ".join(sorted(unresolved))}',
        ),
    ]


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


def stale_format_zone_layers(dataset: Dataset) -> list[Finding]:
    """``files`` findings: built zone-layer sets stamped with an out-of-date
    on-disk format version.

    Skips sets that are not built (``missing_artifacts`` already reports those);
    a built set whose stamped version differs from the provider's current one --
    including a missing/legacy tag (stored ``None``) -- is flagged for a rebuild.
    The ``target`` is the provider name; the ``issue`` names the stored vs.
    current versions.
    """
    name = dataset.spec.name
    findings: list[Finding] = []
    for provider_name, zone_set in dataset.zones.items():
        if zone_set.format_is_current() is False:
            findings.append(
                _finding(
                    'files',
                    name,
                    provider_name,
                    f'stale zone-layer format (stored '
                    f'{zone_set.stored_format_version()} != '
                    f'current {zone_set.format_version})',
                ),
            )
    return findings


def pourpoint_coverage_report(snowdb: SnowDb, dataset: Dataset) -> list[Finding]:
    """``pourpoints`` findings for how a dataset's grid + burned rasters line up
    with the stored pourpoints.

    ``no raster``/``orphan raster`` are about which AOI *rasters* exist;
    ``partial coverage``/``no coverage`` are the geometric coverage of each
    pourpoint's basin by the dataset's grid (partial = basin spills outside it,
    no coverage = basin entirely off-grid). A fully-covered, rasterized
    pourpoint yields no finding. The ``target`` is the station triplet.
    """
    from snowtool.snowdb.coverage import Coverage, dataset_coverage

    name = dataset.spec.name
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
    return [
        _finding('pourpoints', name, triplet, issue)
        for issue, triplet_set in (
            ('no raster', triplets - rasterized),
            ('orphan raster', rasterized - triplets),
            ('partial coverage', set(partial)),
            ('no coverage', set(uncovered)),
        )
        for triplet in sorted(triplet_set)
    ]


def aoi_health_report(dataset: Dataset) -> list[Finding]:
    """``pourpoints`` findings for burned AOI rasters that won't read cleanly.

    Opens each AOI raster; a read failure, a missing tile-bbox tag, or an
    all-zero (empty) raster becomes a finding whose ``target`` is the station
    triplet and whose ``issue`` describes the fault.
    """
    findings: list[Finding] = []
    for path in dataset.aoi_raster_paths():
        findings.extend(_aoi_raster_health(dataset, path))
    return findings


def _aoi_raster_health(dataset: Dataset, path: Path) -> list[Finding]:
    """The ``pourpoints`` finding (if any) for one burned AOI raster."""
    from snowtool.snowdb.aoi_raster import AOIRaster

    triplet = triplet_naming.stem_to_triplet(path.stem)
    issue: str | None = None
    try:
        aoi_raster = AOIRaster.open(path, dataset.grid)
    except IncompleteDatasetDataError:
        issue = (
            'missing SNOWTOOL_TILE_BBOX tag (rebuild with `pourpoint rasterize '
            '--rebuild`)'
        )
    except Exception as e:  # noqa: BLE001 - a health scan reports any read failure
        issue = f'unreadable: {e}'
    else:
        # Burned to all-zero (no in-basin cell area): the basin covers no in-grid
        # cells, so the raster would contribute no pixels to any query -- either
        # the basin is off-grid (a stray raster that should not exist) or it is
        # on-grid but entirely over masked/nodata pixels.
        if not aoi_raster.array.any():
            issue = 'empty AOI raster (covers no in-grid cells: off-grid or masked)'
    if issue is None:
        return []
    return [_finding('pourpoints', dataset.spec.name, triplet, issue)]


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


def _iter_cog_paths(dataset: Dataset) -> Iterator[Path]:
    """Every ingested COG on disk, ascending by date then filename.

    Lists each ``cogs/<date>/`` directory once and yields all its ``.tif`` files,
    so the grid check can validate *every* data file's header (not just a
    representative), and tolerate a stray/duplicate COG a per-variable resolve
    would raise on.
    """
    for d in dataset.available_dates():
        yield from sorted(dataset.date_dir(d).glob('*.tif'))


def _grid_declaration_issues(dataset: Dataset) -> list[str]:
    """The declaration-only grid problems (no raster I/O).

    An ingester with no variables has nothing to write -- almost certainly a
    misconfiguration. (The reverse -- variables but no ingester -- is *not*
    flagged: that is a valid read-only/derived dataset, populated out of band.)
    A deeper variables-vs-ingester check (the ingester's *required* keys being a
    subset of those declared) would need the ``Ingester`` protocol to expose its
    expected keys; that is left as a follow-up.
    """
    spec = dataset.spec
    if spec.ingester is not None and not spec.variables:
        return ['has an ingester but declares no variables']
    return []


def _cog_grid_issues(dataset: Dataset, cog: Path) -> list[str]:
    """Shape + transform problems for one COG against the declared grid.

    Opens ``cog``'s header and checks its dimensions and transform against the
    dataset's declared grid, catching a config that has drifted from the real
    rasters (or a file ingested onto a different lattice). Delegates the
    comparison to :func:`snowtool.snowdb.issues.grid_issues` -- the same check
    used for every other grid-bound artifact -- and renders each returned
    ``Issue`` back to a string so callers keep their ``list[str]`` shape.
    """
    import rasterio

    grid = dataset.spec.grid_params
    declared = dataset.grid.base_grid.transform
    with rasterio.open(cog) as src:
        actual = src.transform
        width, height = src.width, src.height
    found = issues_mod.grid_issues(
        declared_transform=declared,
        actual_transform=actual,
        declared_shape=(grid.rows, grid.cols),
        actual_shape=(height, width),
    )
    return [issue.message for issue in found]


def grid_validation_report(dataset: Dataset) -> list[str]:
    """Declaration-vs-reality grid problems (empty == consistent).

    The declaration check (:func:`_grid_declaration_issues`) plus the shape +
    transform check (:func:`_cog_grid_issues`) run against **every** ingested
    COG, so a lattice drift on any date is caught, not just the first. The
    ``doctor`` sweep enumerates the same work one COG at a time for progress
    (see :func:`_grid_steps`); this aggregate is for direct callers/tests.
    """
    issues = _grid_declaration_issues(dataset)
    for cog in _iter_cog_paths(dataset):
        issues.extend(_cog_grid_issues(dataset, cog))
    return issues


# --- doctor: enumerable health-check steps -----------------------------------
#
# `run_health_checks` shows one progress increment per unit of work, so a check
# is not a single ``(snowdb, dataset) -> findings`` call but an *enumeration* of
# `CheckStep`s: a live-progress label plus a deferred `run`. Enumeration only
# lists directories (cheap); each `run` does the expensive open/parse. Listing
# every step across all datasets up front lets the reporter set an exact total
# and announce each unit (a COG, an AOI raster, a date) as it runs.


@dataclass(frozen=True)
class CheckStep:
    """One unit of ``doctor`` work: a progress ``label`` and a deferred ``run``
    that produces its findings when executed."""

    label: str
    run: Callable[[], list[Finding]]


def _grid_declaration_findings(dataset: Dataset, name: str) -> list[Finding]:
    return [
        _finding('grid', name, '', issue) for issue in _grid_declaration_issues(dataset)
    ]


def _cog_grid_findings(
    dataset: Dataset,
    name: str,
    cog: Path,
    target: str,
) -> list[Finding]:
    return [
        _finding('grid', name, target, issue)
        for issue in _cog_grid_issues(dataset, cog)
    ]


def _grid_steps(_snowdb: SnowDb, dataset: Dataset) -> list[CheckStep]:
    name = dataset.spec.name
    steps = [
        CheckStep(
            f'{name} grid: declaration',
            partial(_grid_declaration_findings, dataset, name),
        ),
    ]
    # One step per COG so every data file's header is validated (not just a
    # representative), each advancing the bar under its own label.
    for cog in _iter_cog_paths(dataset):
        target = f'{cog.parent.name}/{cog.name}'
        steps.append(
            CheckStep(
                f'{name} grid: {target}',
                partial(_cog_grid_findings, dataset, name, cog, target),
            ),
        )
    return steps


def _dates_steps(_snowdb: SnowDb, dataset: Dataset) -> list[CheckStep]:
    name = dataset.spec.name
    return [
        CheckStep(
            f'{name} dates: {d.isoformat()}',
            partial(_date_completeness, dataset, d),
        )
        for d in dataset.available_dates()
    ]


def _missing_artifact_findings(dataset: Dataset, name: str) -> list[Finding]:
    return [
        _finding('files', name, artifact, 'missing')
        for artifact in missing_artifacts(dataset)
    ]


def _files_steps(_snowdb: SnowDb, dataset: Dataset) -> list[CheckStep]:
    name = dataset.spec.name
    return [
        CheckStep(
            f'{name} files: artifacts',
            partial(_missing_artifact_findings, dataset, name),
        ),
        CheckStep(
            f'{name} files: zone-layer formats',
            partial(stale_format_zone_layers, dataset),
        ),
    ]


def _pourpoint_coverage_findings(
    dataset: Dataset,
    record_path: Path,
    triplet: str,
    rasterized: frozenset[str],
) -> list[Finding]:
    """Coverage findings for one pourpoint basin (parses + reprojects it).

    This is the per-basin unit of the coverage scan -- the expensive part
    (parsing the record and reprojecting the basin into the grid CRS), one
    pourpoint at a time so ``doctor``'s bar advances through a large registry
    instead of stalling on a single monolithic step.

    An off-grid basin (``NONE``) cannot be rasterized -- ``rasterize_aoi`` refuses
    it and the batch path skips it -- so a *missing* raster is expected and only
    ``no coverage`` is reported (a *stray* all-zero raster is still caught by the
    per-raster AOI-validation steps). A ``PARTIAL`` basin can and should be
    rasterized, so ``no raster`` stands there. Issue order (``no raster`` before
    ``partial coverage``) matches the collapsed-row order the tests pin.
    """
    from snowtool.snowdb.coverage import Coverage, dataset_coverage
    from snowtool.snowdb.pourpoint import Pourpoint

    name = dataset.spec.name
    coverage = dataset_coverage(
        Pourpoint.from_basin_record(record_path),
        dataset.coverage_domain,
    )
    if coverage is Coverage.NONE:
        return [_finding('pourpoints', name, triplet, 'no coverage')]
    findings: list[Finding] = []
    if triplet not in rasterized:
        findings.append(_finding('pourpoints', name, triplet, 'no raster'))
    if coverage is Coverage.PARTIAL:
        findings.append(_finding('pourpoints', name, triplet, 'partial coverage'))
    return findings


def _orphan_raster_findings(
    name: str,
    rasterized: frozenset[str],
    triplets: set[str],
) -> list[Finding]:
    """``orphan raster`` findings: burned rasters with no backing record."""
    return [
        _finding('pourpoints', name, triplet, 'orphan raster')
        for triplet in sorted(rasterized - triplets)
    ]


def _pourpoints_steps(snowdb: SnowDb, dataset: Dataset) -> list[CheckStep]:
    name = dataset.spec.name
    # Cheap up front (filenames + a glob); the per-basin reprojection is deferred
    # into one step each so the progress total is exact and the bar advances.
    rasterized = frozenset(dataset.aoi_raster_triplets())
    steps = [
        CheckStep(
            f'{name} pourpoints: coverage {triplet_naming.stem_to_triplet(p.stem)}',
            partial(
                _pourpoint_coverage_findings,
                dataset,
                p,
                triplet_naming.stem_to_triplet(p.stem),
                rasterized,
            ),
        )
        for p in snowdb.pourpoint_paths()
    ]
    steps.append(
        CheckStep(
            f'{name} pourpoints: orphan rasters',
            partial(
                _orphan_raster_findings,
                name,
                rasterized,
                snowdb.pourpoint_triplets(),
            ),
        ),
    )
    for path in dataset.aoi_raster_paths():
        triplet = triplet_naming.stem_to_triplet(path.stem)
        steps.append(
            CheckStep(
                f'{name} pourpoints: AOI validation {triplet}',
                partial(_aoi_raster_health, dataset, path),
            ),
        )
    return steps


# Order is the ``doctor`` output/CLI-help order.
HEALTH_CHECKS: dict[str, Callable[[SnowDb, Dataset], list[CheckStep]]] = {
    'grid': _grid_steps,
    'dates': _dates_steps,
    'files': _files_steps,
    'pourpoints': _pourpoints_steps,
}

# The valid check names, in ``doctor``'s output/default-sweep order.
HEALTH_CHECK_NAMES: tuple[str, ...] = tuple(HEALTH_CHECKS)


def run_health_checks(
    snowdb: SnowDb,
    datasets: Sequence[Dataset],
    checks: Sequence[str],
    *,
    progress: ProgressReporter = NULL_PROGRESS,
) -> list[Finding]:
    """Sweep ``checks`` across ``datasets``, returning the flat finding rows
    ``snowtool doctor`` renders (empty means healthy).

    Resolves ``checks`` first: an unknown name raises
    :class:`~snowtool.exceptions.UnknownHealthCheckError`, duplicates are
    dropped (order-preserving), and an empty selection defaults to every check
    in :data:`HEALTH_CHECK_NAMES`. Then enumerates every check's steps (one per
    COG, AOI raster, date, ...) across all datasets up front so ``progress`` has
    an exact total, and runs each step, naming it on the bar and advancing one
    tick per step. Findings sharing a ``(check, dataset, target)`` are collapsed
    onto one row (:func:`_collapse_by_target`).
    """
    unknown = sorted(set(checks) - set(HEALTH_CHECK_NAMES))
    if unknown:
        raise UnknownHealthCheckError(
            f'Unknown check(s): {", ".join(unknown)}. '
            f'Known checks: {", ".join(HEALTH_CHECK_NAMES)}.',
        )
    selected = list(dict.fromkeys(checks)) if checks else list(HEALTH_CHECK_NAMES)

    steps: list[CheckStep] = []
    for dataset in datasets:
        for check in selected:
            steps.extend(HEALTH_CHECKS[check](snowdb, dataset))

    findings: list[Finding] = []
    with progress.track('doctor', total=len(steps)) as task:
        for step in steps:
            task.describe(step.label)
            findings.extend(step.run())
            task.advance()
    return _collapse_by_target(findings)


def _collapse_by_target(findings: Sequence[Finding]) -> list[Finding]:
    """Roll findings that share a ``(check, dataset, target)`` onto one row.

    A target that trips several issues (e.g. a basin that is both ``no raster``
    and ``partial coverage``) becomes a single row whose ``issue`` joins them with
    ``'; '`` -- so ``doctor`` never spreads one target across multiple lines --
    preserving first-seen order both across rows and within the joined issue.
    """
    collapsed: dict[tuple[str, str, str], Finding] = {}
    for finding in findings:
        key = (finding['check'], finding['dataset'], finding['target'])
        if (existing := collapsed.get(key)) is not None:
            existing['issue'] += f'; {finding["issue"]}'
        else:
            collapsed[key] = dict(finding)
    return list(collapsed.values())

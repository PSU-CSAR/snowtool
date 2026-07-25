"""The ``snowtool doctor`` health-check machinery.

A ``Finding`` is one row of ``doctor`` output: the flat, uniform
``check``/``dataset``/``target``/``issue`` dict the CLI renders. Checks express
their faults as typed :class:`~snowtool.snowdb.issues.Issue` objects (no prose
matching in code) and render them to findings where the condition is detected.

``run_health_checks`` shows one progress increment per unit of work, so a check
is not a single ``(snowdb, dataset) -> findings`` call but an *enumeration* of
:class:`CheckStep`s: a live-progress label plus a deferred ``run``. Enumeration
only lists directories (cheap); each ``run`` does the expensive open/parse.
Listing every step across all datasets up front lets the reporter set an exact
total and announce each unit (a COG, an AOI raster, a date) as it runs.

The facts each row encodes are pinned by
``tests/snowdb/test_report_diagnostics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from snowtool.exceptions import UnknownHealthCheckError
from snowtool.snowdb import issues as issues_mod
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.query import DateRangeQuery

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import date
    from pathlib import Path

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.progress import ProgressReporter


type Finding = dict[str, str]


def _finding(check: str, dataset: str, target: str, issue: str) -> Finding:
    return {'check': check, 'dataset': dataset, 'target': target, 'issue': issue}


def _findings_from_issues(
    check: str,
    dataset: str,
    target: str,
    issue_list: Sequence[issues_mod.Issue],
) -> list[Finding]:
    """Render a batch of typed :class:`~snowtool.snowdb.issues.Issue` to findings
    sharing one ``check``/``dataset``/``target``."""
    return [_finding(check, dataset, target, i.message) for i in issue_list]


# --- dates: completeness -----------------------------------------------------


def completeness_report(
    dataset: Dataset,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[Finding]:
    """``dates`` findings: ingested dates (optionally within ``start``/``end``)
    missing one or more of their dataset's variables.

    A thin windowed wrapper over :func:`_date_completeness` -- the same per-date
    helper the ``dates`` doctor step runs -- so the windowed and per-step paths
    cannot drift.
    """
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


# --- files: missing artifacts + stale zone-layer formats ---------------------


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


# --- grid: declaration + per-raster header validation ------------------------


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


def _raster_grid_findings(
    dataset: Dataset,
    name: str,
    path: Path,
    target: str,
) -> list[Finding]:
    """The ``grid`` findings for one full-grid raster -- a COG, a zone layer, or
    the nodata mask -- against the dataset's declared grid.

    Opens ``path``'s header and compares its shape + transform to the declared
    grid via :func:`snowtool.snowdb.issues.grid_issues` -- the same check every
    grid-bound artifact uses -- catching a config that has drifted from the real
    rasters, or a file written onto a different lattice. (An AOI raster is a
    windowed crop, checked under ``pourpoints`` via ``aoi_raster_issues``, not
    here.)
    """
    import rasterio

    grid = dataset.spec.grid_params
    with rasterio.open(path) as src:
        actual_transform = src.transform
        actual_shape = (src.height, src.width)
    found = issues_mod.grid_issues(
        declared_transform=dataset.grid.base_grid.transform,
        actual_transform=actual_transform,
        declared_shape=(grid.rows, grid.cols),
        actual_shape=actual_shape,
    )
    return _findings_from_issues('grid', name, target, found)


# --- doctor: enumerable health-check steps -----------------------------------


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


def _grid_target_rasters(dataset: Dataset) -> Iterator[tuple[Path, str]]:
    """Every present full-grid raster the grid check validates, as
    ``(path, target)`` pairs: each ingested COG, each built zone-layer file
    (terrain elevation/aspect, land-cover forest, ...), and the nodata mask when
    configured. A *missing* zone layer or mask is the ``files`` check's concern,
    so only files that actually exist on disk are yielded.
    """
    for cog in _iter_cog_paths(dataset):
        yield cog, f'{cog.parent.name}/{cog.name}'
    for provider_name, zone_set in dataset.zones.items():
        for layer in zone_set.layers:
            path = zone_set.layer_path(layer)
            if path.is_file():
                yield path, f'{provider_name}/{layer.filename}'
    if dataset.nodata_mask is not None and dataset.nodata_mask.is_file():
        yield dataset.nodata_mask, 'nodata-mask.tif'


def _grid_steps(_snowdb: SnowDb, dataset: Dataset) -> list[CheckStep]:
    name = dataset.spec.name
    steps = [
        CheckStep(
            f'{name} grid: declaration',
            partial(_grid_declaration_findings, dataset, name),
        ),
    ]
    # One step per present full-grid raster so every file's header is validated
    # (not just a representative), each advancing the bar under its own label.
    steps.extend(
        CheckStep(
            f'{name} grid: {target}',
            partial(_raster_grid_findings, dataset, name, path, target),
        )
        for path, target in _grid_target_rasters(dataset)
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
        return _findings_from_issues(
            'pourpoints',
            name,
            triplet,
            [issues_mod.NoCoverage()],
        )
    found: list[issues_mod.Issue] = []
    if triplet not in rasterized:
        found.append(issues_mod.NoRaster())
    if coverage is Coverage.PARTIAL:
        found.append(issues_mod.PartialCoverage())
    return _findings_from_issues('pourpoints', name, triplet, found)


def _orphan_raster_findings(
    name: str,
    rasterized: frozenset[str],
    triplets: set[str],
) -> list[Finding]:
    """``orphan raster`` findings: burned rasters with no backing record."""
    findings: list[Finding] = []
    for triplet in sorted(rasterized - triplets):
        findings.extend(
            _findings_from_issues(
                'pourpoints',
                name,
                triplet,
                [issues_mod.OrphanArtifact()],
            ),
        )
    return findings


def _pourpoints_steps(snowdb: SnowDb, dataset: Dataset) -> list[CheckStep]:
    name = dataset.spec.name
    # Cheap up front (filenames + a glob); the per-basin reprojection is deferred
    # into one step each so the progress total is exact and the bar advances.
    rasterized = frozenset(dataset.aoi_raster_triplets())
    steps: list[CheckStep] = []
    for p in snowdb.pourpoint_paths():
        triplet = triplet_naming.stem_to_triplet(p.stem)
        steps.append(
            CheckStep(
                f'{name} pourpoints: coverage {triplet}',
                partial(_pourpoint_coverage_findings, dataset, p, triplet, rasterized),
            ),
        )
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
                # Defer both the record parse (expected hash) and the raster read
                # into the step body -- computing them here, during enumeration,
                # parses every basin record before the progress bar opens.
                partial(_aoi_raster_step_findings, snowdb, dataset, path),
            ),
        )
    return steps


def _aoi_raster_findings(
    dataset: Dataset,
    stem: str,
    path: Path,
    expected_hash: str | None,
) -> list[Finding]:
    """The ``pourpoints`` findings for one burned AOI raster.

    Delegates to :func:`~snowtool.snowdb.aoi_raster.aoi_raster_issues` -- the
    shared structure/grid/freshness check -- and renders its typed
    :class:`~snowtool.snowdb.issues.Issue` list to findings.
    """
    from snowtool.snowdb.aoi_raster import aoi_raster_issues

    triplet = triplet_naming.stem_to_triplet(stem)
    found = aoi_raster_issues(path, grid=dataset.grid, expected_hash=expected_hash)
    return _findings_from_issues('pourpoints', dataset.spec.name, triplet, found)


def _aoi_raster_step_findings(
    snowdb: SnowDb,
    dataset: Dataset,
    path: Path,
) -> list[Finding]:
    """A doctor AOI-validation step: resolve the expected hash from the registry
    (deferred), then run the shared AOI check."""
    triplet = triplet_naming.stem_to_triplet(path.stem)
    expected_hash = _expected_aoi_hash(snowdb, dataset, triplet)
    return _aoi_raster_findings(dataset, path.stem, path, expected_hash)


def _expected_aoi_hash(
    snowdb: SnowDb,
    dataset: Dataset,
    triplet: str,
) -> str | None:
    """The AOI provenance hash a raster for ``triplet`` should carry, or ``None``
    when there is no backing basin record to compare against (an orphan raster,
    reported separately by :func:`_orphan_raster_findings`)."""
    from snowtool.snowdb.aoi_raster import aoi_provenance
    from snowtool.snowdb.pourpoint import Pourpoint

    record_path = snowdb.pourpoint_record_path(triplet)
    if not record_path.is_file():
        return None
    pourpoint = Pourpoint.from_basin_record(record_path)
    return aoi_provenance(pourpoint.geometry_hash, dataset.nodata_mask_hash)


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

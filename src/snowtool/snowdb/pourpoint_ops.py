"""Pourpoint lifecycle operations over a :class:`~snowtool.snowdb.db.SnowDb`.

The pourpoint import/sync/removal/rasterize half of the write layer, as
module-level functions that each take the read database (``db``) as their first
argument and reach the storage through it exactly as if they still lived on the
manager. :class:`~snowtool.snowdb.manager.SnowDbManager` keeps same-named
one-line delegators, so the write surface is still a single type; the functions
here carry the logic. Result dataclasses live here beside the operations that
produce them and are re-exported from
:mod:`snowtool.snowdb.manager` for callers that import them from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from snowtool import types
from snowtool.exceptions import (
    GeoJSONValidationError,
    PourpointPruneDestinationRequiredError,
)
from snowtool.snowdb.atomic import atomic_copy
from snowtool.snowdb.coverage import Coverage, dataset_coverage
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.pourpoint_index import PourpointIndex
from snowtool.snowdb.progress import NULL_PROGRESS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.progress import ProgressReporter


@dataclass(frozen=True)
class PourpointImportResult:
    """The outcome of an additive ``pourpoint import``.

    ``imported`` are the triplets written to ``records/`` (basin-bearing pourpoints);
    ``skipped`` are point-only pourpoints (valid pourpoints (no basin));
    ``invalid`` pairs each unparseable source path with its error message.
    """

    imported: list[types.StationTriplet]
    skipped: list[types.StationTriplet]
    invalid: list[tuple[Path, str]]


@dataclass(frozen=True)
class PourpointSyncResult(PourpointImportResult):
    """An :class:`PourpointImportResult` plus the triplets pruned (or, in a dry run,
    that would be pruned) because they are absent from the synced directory."""

    pruned: list[types.StationTriplet]


@dataclass(frozen=True)
class AOIRasterizeResult:
    """Per (AOI, dataset) rasterize outcomes: ``(triplet, dataset_name)`` pairs
    that were (re)built vs. skipped as already current, plus ``coverage`` -- the
    geometric coverage of each pair's dataset grid computed once during the pass
    (the same value that drives the ``Coverage.NONE`` skip), keyed by
    ``(triplet, dataset_name)`` so a caller need not recompute it."""

    built: list[tuple[types.StationTriplet, str]]
    skipped: list[tuple[types.StationTriplet, str]]
    coverage: dict[tuple[types.StationTriplet, str], Coverage]


def reindex_pourpoints(
    db: SnowDb,
    *,
    progress: ProgressReporter = NULL_PROGRESS,
) -> PourpointIndex:
    """Rebuild ``index.geojson`` from the ``records/`` dir and persist it.

    The explicit FULL rebuild: every record is re-parsed and the persisted
    index is ignored -- the recovery path for out-of-band ``records/`` edits
    and for a grid change to an already-registered dataset (the one change
    the incremental :func:`_update_index` cannot see). Coverage is re-derived
    against every *registered* dataset's current grid (active or not -- an
    inactive dataset carries real coverage the moment it is activated), so
    the manifest always reflects the live grids.
    """
    domains = _coverage_domains(db)
    index = PourpointIndex.build(
        db.pourpoint_paths(),
        domains,
        progress=progress,
    )
    index.save(db.pourpoint_index_path)
    return index


def _coverage_domains(db: SnowDb) -> dict[str, CoverageDomain]:
    """Every registered dataset's coverage domain, keyed by name."""
    return {name: ds.coverage_domain for name, ds in db.registered.items()}


def _update_index(
    db: SnowDb,
    imported: Mapping[types.StationTriplet, Pourpoint],
    *,
    progress: ProgressReporter = NULL_PROGRESS,
) -> PourpointIndex:
    """Incrementally rebuild ``index.geojson`` after an import/sync/remove.

    Iterates the *surviving* record files (so removed/pruned triplets fall
    out naturally) and, per record: a pourpoint parsed in-memory by this
    operation (``imported``) is indexed without a disk re-parse; an existing
    index entry whose coverage keys still equal the registered-dataset names
    is reused as-is; anything else -- a missing/stale index, or a dataset
    registered/removed since the entry was written -- is re-parsed from disk
    (the self-healing fallback). The one change this cannot see is a grid
    change for an already-registered dataset name; that requires an explicit
    :func:`reindex_pourpoints`. ``progress`` reports the pass, advancing once
    per record whether reused, rebuilt from memory, or parsed.
    """
    domains = _coverage_domains(db)
    previous = PourpointIndex.load(db.pourpoint_index_path)
    index = PourpointIndex.build(
        db.pourpoint_paths(),
        domains,
        reuse=previous.entries,
        preparsed=imported,
        progress=progress,
    )
    index.save(db.pourpoint_index_path)
    return index


def _resolve_sources(src: Path) -> list[Path]:
    """A file SRC -> ``[src]``; a directory SRC -> its sorted ``*.geojson``."""
    src = Path(src)
    if src.is_dir():
        return sorted(src.glob('*.geojson'))
    if src.is_file():
        return [src]
    raise FileNotFoundError(f'No such file or directory: {src}')


def _classify_sources(
    sources: Iterable[Path],
    *,
    progress: ProgressReporter = NULL_PROGRESS,
) -> tuple[
    list[tuple[Path, Pourpoint]],
    list[types.StationTriplet],
    list[tuple[Path, str]],
]:
    """Split source paths into importable pourpoints, skipped point-only, invalid.

    Pure (no writes, aside from advancing ``progress`` once per source): the
    caller decides whether to persist the result, so a dry run and a real run
    classify identically.
    """
    sources = list(sources)
    to_import: list[tuple[Path, Pourpoint]] = []
    skipped: list[types.StationTriplet] = []
    invalid: list[tuple[Path, str]] = []
    with progress.track(
        f'parsing {len(sources)} pourpoint source(s)',
        total=len(sources),
    ) as task:
        for path in sources:
            try:
                pourpoint = Pourpoint.from_geojson(path)
            except GeoJSONValidationError as e:
                invalid.append((path, str(e)))
            else:
                if pourpoint.polygon is None:
                    # A valid pourpoint, but with no basin it is skipped.
                    skipped.append(pourpoint.station_triplet)
                else:
                    to_import.append((path, pourpoint))
            task.advance()
    return to_import, skipped, invalid


def _write_records(db: SnowDb, to_import: Iterable[tuple[Path, Pourpoint]]) -> None:
    """Copy each source geojson verbatim to its canonical record path."""
    db.pourpoint_records_path.mkdir(parents=True, exist_ok=True)
    for path, pourpoint in to_import:
        atomic_copy(path, db.pourpoint_record_path(pourpoint.station_triplet))


def _commit(
    db: SnowDb,
    to_import: list[tuple[Path, Pourpoint]],
    *,
    progress: ProgressReporter,
) -> None:
    """Write records for ``to_import`` and incrementally update the index.

    The shared classify-then-write tail for both :func:`import_pourpoints` and
    :func:`sync_pourpoints` once a dry run has been ruled out: copy each source
    geojson to its canonical record path, then feed the just-parsed
    pourpoints to :func:`_update_index` as ``preparsed`` so they are indexed
    without a disk re-parse.
    """
    _write_records(db, to_import)
    _update_index(
        db,
        {pourpoint.station_triplet: pourpoint for _, pourpoint in to_import},
        progress=progress,
    )


def import_pourpoints(
    db: SnowDb,
    src: Path,
    *,
    dry_run: bool = False,
    progress: ProgressReporter = NULL_PROGRESS,
) -> PourpointImportResult:
    """Additively import Pourpoint(s) from a file or directory into ``records/``.

    Imports only polygon-bearing pourpoints (skips point-only ones, reports
    unparseable ones); never removes anything. Idempotent: re-importing a
    triplet overwrites its record. Updates the index incrementally
    (:func:`_update_index` -- untouched entries are reused, not re-parsed)
    unless ``dry_run``. ``progress`` reports the parse and index phases.
    """
    to_import, skipped, invalid = _classify_sources(
        _resolve_sources(src),
        progress=progress,
    )
    imported = [pourpoint.station_triplet for _, pourpoint in to_import]
    if not dry_run:
        _commit(db, to_import, progress=progress)
    return PourpointImportResult(imported, skipped, invalid)


def sync_pourpoints(
    db: SnowDb,
    src: Path,
    *,
    prune_to: Path | None = None,
    dry_run: bool = False,
    progress: ProgressReporter = NULL_PROGRESS,
) -> PourpointSyncResult:
    """Mirror a directory into storage: import it, then prune absent records.

    Imports ``src`` (directory only), then removes every stored pourpoint
    whose triplet is not present in ``src`` -- dumping each to ``prune_to``
    first. Removal is gated: if any pourpoint would be pruned and ``prune_to``
    is ``None`` (and not a dry run), raises
    :class:`PourpointPruneDestinationRequiredError` before writing anything, so
    the destructive step is never silent. The index is updated incrementally
    (:func:`_update_index`; pruned triplets simply fall out) unless
    ``dry_run``. ``progress`` reports the parse and index phases.
    """
    src = Path(src)
    if not src.is_dir():
        raise NotADirectoryError(f'pourpoint sync requires a directory: {src}')

    to_import, skipped, invalid = _classify_sources(
        _resolve_sources(src),
        progress=progress,
    )
    imported = [pourpoint.station_triplet for _, pourpoint in to_import]
    # Both basin-bearing and point-only pourpoints in the source represent
    # triplets the source "has"; only stored triplets absent from that set
    # are pruned.
    source_triplets = set(imported) | set(skipped)
    to_prune = sorted(t for t in db.pourpoint_triplets() if t not in source_triplets)

    if to_prune and not dry_run and prune_to is None:
        raise PourpointPruneDestinationRequiredError(to_prune)

    if not dry_run:
        for triplet in to_prune:
            _remove_pourpoint_files(db, triplet, dump_to=prune_to)
        _commit(db, to_import, progress=progress)

    return PourpointSyncResult(imported, skipped, invalid, to_prune)


def _remove_pourpoint_files(
    db: SnowDb,
    triplet: types.StationTriplet,
    *,
    dump_to: Path | None = None,
) -> None:
    """Delete a record and cascade to every dataset's burned AOI raster.

    Optionally dumps the record to ``dump_to`` first (the reversible prune
    path). Does not touch the index -- callers update it once after a batch
    (:func:`_update_index`).
    """
    if dump_to is not None:
        db.dump_pourpoint(triplet, dump_to)
    db.pourpoint_record_path(triplet).unlink(missing_ok=True)
    for dataset in db.registered.values():
        dataset.remove_aoi_raster(triplet)


def remove_pourpoint(
    db: SnowDb,
    triplet: types.StationTriplet,
    *,
    dry_run: bool = False,
    progress: ProgressReporter = NULL_PROGRESS,
) -> bool:
    """Remove a stored pourpoint and its per-dataset rasters; True if it existed.

    Cascade-deletes the record plus every ``aoi-rasters/<triplet>.tif`` and
    updates the index incrementally (:func:`_update_index` -- surviving
    entries are reused, the removed one falls out). Idempotent: removing an
    absent pourpoint is a no-op success.
    """
    existed = db.pourpoint_record_path(triplet).is_file()
    if not dry_run:
        _remove_pourpoint_files(db, triplet)
        _update_index(db, {}, progress=progress)
    return existed


def rasterize_aois(
    db: SnowDb,
    pourpoints: Iterable[Pourpoint],
    datasets: Iterable[Dataset],
    *,
    rebuild: bool = False,
    progress: ProgressReporter = NULL_PROGRESS,
) -> AOIRasterizeResult:
    """Burn each pourpoint's basin onto each dataset's grid when missing or stale.

    Builds the cartesian product of ``pourpoints`` x ``datasets``, (re)building a
    raster only when absent or its :attr:`Pourpoint.geometry_hash` tag no longer
    matches (``rebuild=True`` forces all). A pair whose basin does not intersect
    the dataset's coverage domain at all is *skipped* (an off-grid basin has no
    tile window to burn), so a batch over mixed-extent grids never trips
    :class:`~snowtool.exceptions.GeometryOutsideGridError`. ``progress`` reports
    the pass, advancing once per pourpoint-dataset pair (built or skipped) --
    the same seam zone-layer generation uses. Returns the built vs. skipped
    ``(triplet, dataset_name)`` pairs plus each pair's geometric ``coverage``
    (computed here for the skip and surfaced so callers need not recompute it).
    """
    pourpoints = list(pourpoints)
    datasets = list(datasets)
    domains = [dataset.coverage_domain for dataset in datasets]
    built: list[tuple[types.StationTriplet, str]] = []
    skipped: list[tuple[types.StationTriplet, str]] = []
    coverage: dict[tuple[types.StationTriplet, str], Coverage] = {}
    total = len(pourpoints) * len(datasets)
    with progress.track('rasterizing', total=total) as task:
        for pourpoint in pourpoints:
            for dataset, domain in zip(datasets, domains, strict=True):
                pair = (pourpoint.station_triplet, dataset.spec.name)
                pair_coverage = dataset_coverage(pourpoint, domain)
                coverage[pair] = pair_coverage
                if pair_coverage is Coverage.NONE:
                    # Entirely off this grid: nothing to burn.
                    skipped.append(pair)
                elif dataset.rasterize_aoi(pourpoint, rebuild=rebuild):
                    built.append(pair)
                else:
                    skipped.append(pair)
                task.advance()
    return AOIRasterizeResult(built, skipped, coverage)

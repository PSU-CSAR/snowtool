"""The pourpoint import/sync/lifecycle half of the write layer, split out by size.

This is *not* a public seam: the write surface is still a single
:class:`~snowtool.snowdb.manager.SnowDbManager` type. :class:`PourpointOpsMixin`
holds the pourpoint import/sync/removal/rasterize operations (everything below the
manager's ``pourpoint import / sync / lifecycle`` divider) purely to keep each file
under a readable size; ``SnowDbManager`` inherits it, so every method here is a
manager method and reaches the read database through ``self.db`` exactly as if it
still lived in ``manager.py``. The mixin carries no state of its own -- it declares
only the :attr:`db` attribute its owner supplies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

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
    that were (re)built vs. skipped as already current."""

    built: list[tuple[types.StationTriplet, str]]
    skipped: list[tuple[types.StationTriplet, str]]


class PourpointOpsMixin:
    """The pourpoint import/sync/lifecycle operations of :class:`SnowDbManager`.

    A file-size decomposition, not a public seam: these are ``SnowDbManager``
    methods that happen to live in their own module. The mixin has no state -- it
    only requires the :attr:`db` its owner supplies -- so it is never instantiated
    on its own.
    """

    db: SnowDb

    def reindex_pourpoints(
        self: Self,
        *,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> PourpointIndex:
        """Rebuild ``index.geojson`` from the ``records/`` dir and persist it.

        The explicit FULL rebuild: every record is re-parsed and the persisted
        index is ignored -- the recovery path for out-of-band ``records/`` edits
        and for a grid change to an already-registered dataset (the one change
        the incremental :meth:`_update_index` cannot see). Coverage is re-derived
        against every *registered* dataset's current grid (active or not -- an
        inactive dataset carries real coverage the moment it is activated), so
        the manifest always reflects the live grids.
        """
        domains = self._coverage_domains()
        index = PourpointIndex.build(
            self.db.pourpoint_paths(),
            domains,
            progress=progress,
        )
        index.save(self.db.pourpoint_index_path)
        return index

    def _coverage_domains(self: Self) -> dict[str, CoverageDomain]:
        """Every registered dataset's coverage domain, keyed by name."""
        return {name: ds.coverage_domain for name, ds in self.db.registered.items()}

    def _update_index(
        self: Self,
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
        :meth:`reindex_pourpoints`. ``progress`` reports the pass, advancing once
        per record whether reused, rebuilt from memory, or parsed.
        """
        domains = self._coverage_domains()
        previous = PourpointIndex.load(self.db.pourpoint_index_path)
        index = PourpointIndex.build(
            self.db.pourpoint_paths(),
            domains,
            reuse=previous.entries,
            preparsed=imported,
            progress=progress,
        )
        index.save(self.db.pourpoint_index_path)
        return index

    def _resolve_sources(self: Self, src: Path) -> list[Path]:
        """A file SRC -> ``[src]``; a directory SRC -> its sorted ``*.geojson``."""
        src = Path(src)
        if src.is_dir():
            return sorted(src.glob('*.geojson'))
        if src.is_file():
            return [src]
        raise FileNotFoundError(f'No such file or directory: {src}')

    @staticmethod
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

    def _write_records(self: Self, to_import: Iterable[tuple[Path, Pourpoint]]) -> None:
        """Copy each source geojson verbatim to its canonical record path."""
        self.db.pourpoint_records_path.mkdir(parents=True, exist_ok=True)
        for path, pourpoint in to_import:
            atomic_copy(path, self.db.pourpoint_record_path(pourpoint.station_triplet))

    def _commit(
        self: Self,
        to_import: list[tuple[Path, Pourpoint]],
        *,
        progress: ProgressReporter,
    ) -> None:
        """Write records for ``to_import`` and incrementally update the index.

        The shared classify-then-write tail for both ``import_pourpoints`` and
        ``sync_pourpoints`` once a dry run has been ruled out: copy each source
        geojson to its canonical record path, then feed the just-parsed
        pourpoints to :meth:`_update_index` as ``preparsed`` so they are indexed
        without a disk re-parse.
        """
        self._write_records(to_import)
        self._update_index(
            {pourpoint.station_triplet: pourpoint for _, pourpoint in to_import},
            progress=progress,
        )

    def import_pourpoints(
        self: Self,
        src: Path,
        *,
        dry_run: bool = False,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> PourpointImportResult:
        """Additively import Pourpoint(s) from a file or directory into ``records/``.

        Imports only polygon-bearing pourpoints (skips point-only ones, reports
        unparseable ones); never removes anything. Idempotent: re-importing a
        triplet overwrites its record. Updates the index incrementally
        (:meth:`_update_index` -- untouched entries are reused, not re-parsed)
        unless ``dry_run``. ``progress`` reports the parse and index phases.
        """
        to_import, skipped, invalid = self._classify_sources(
            self._resolve_sources(src),
            progress=progress,
        )
        imported = [pourpoint.station_triplet for _, pourpoint in to_import]
        if not dry_run:
            self._commit(to_import, progress=progress)
        return PourpointImportResult(imported, skipped, invalid)

    def sync_pourpoints(
        self: Self,
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
        (:meth:`_update_index`; pruned triplets simply fall out) unless
        ``dry_run``. ``progress`` reports the parse and index phases.
        """
        src = Path(src)
        if not src.is_dir():
            raise NotADirectoryError(f'pourpoint sync requires a directory: {src}')

        to_import, skipped, invalid = self._classify_sources(
            self._resolve_sources(src),
            progress=progress,
        )
        imported = [pourpoint.station_triplet for _, pourpoint in to_import]
        # Both basin-bearing and point-only pourpoints in the source represent
        # triplets the source "has"; only stored triplets absent from that set
        # are pruned.
        source_triplets = set(imported) | set(skipped)
        to_prune = sorted(
            t for t in self.db.pourpoint_triplets() if t not in source_triplets
        )

        if to_prune and not dry_run and prune_to is None:
            raise PourpointPruneDestinationRequiredError(to_prune)

        if not dry_run:
            for triplet in to_prune:
                self._remove_pourpoint_files(triplet, dump_to=prune_to)
            self._commit(to_import, progress=progress)

        return PourpointSyncResult(imported, skipped, invalid, to_prune)

    def _remove_pourpoint_files(
        self: Self,
        triplet: types.StationTriplet,
        *,
        dump_to: Path | None = None,
    ) -> None:
        """Delete a record and cascade to every dataset's burned AOI raster.

        Optionally dumps the record to ``dump_to`` first (the reversible prune
        path). Does not touch the index -- callers update it once after a batch
        (:meth:`_update_index`).
        """
        if dump_to is not None:
            self.db.dump_pourpoint(triplet, dump_to)
        self.db.pourpoint_record_path(triplet).unlink(missing_ok=True)
        for dataset in self.db.registered.values():
            dataset.remove_aoi_raster(triplet)

    def remove_pourpoint(
        self: Self,
        triplet: types.StationTriplet,
        *,
        dry_run: bool = False,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> bool:
        """Remove a stored pourpoint and its per-dataset rasters; True if it existed.

        Cascade-deletes the record plus every ``aoi-rasters/<triplet>.tif`` and
        updates the index incrementally (:meth:`_update_index` -- surviving
        entries are reused, the removed one falls out). Idempotent: removing an
        absent pourpoint is a no-op success.
        """
        existed = self.db.pourpoint_record_path(triplet).is_file()
        if not dry_run:
            self._remove_pourpoint_files(triplet)
            self._update_index({}, progress=progress)
        return existed

    def rasterize_aois(
        self: Self,
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
        ``(triplet, dataset_name)`` pairs.
        """
        pourpoints = list(pourpoints)
        datasets = list(datasets)
        domains = [dataset.coverage_domain for dataset in datasets]
        built: list[tuple[types.StationTriplet, str]] = []
        skipped: list[tuple[types.StationTriplet, str]] = []
        total = len(pourpoints) * len(datasets)
        with progress.track('rasterizing', total=total) as task:
            for pourpoint in pourpoints:
                for dataset, domain in zip(datasets, domains, strict=True):
                    pair = (pourpoint.station_triplet, dataset.spec.name)
                    if dataset_coverage(pourpoint, domain) is Coverage.NONE:
                        # Entirely off this grid: nothing to burn.
                        skipped.append(pair)
                    elif dataset.rasterize_aoi(pourpoint, rebuild=rebuild):
                        built.append(pair)
                    else:
                        skipped.append(pair)
                    task.advance()
        return AOIRasterizeResult(built, skipped)

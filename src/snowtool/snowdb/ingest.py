"""The ingest seam: how a dataset-kind turns source data into per-date COGs.

Ingest is dataset-*kind*-specific knowledge (source formats differ -- a SNODAS
tar of raw rasters, a directory of GeoTIFFs, a NetCDF, ...), so -- like a
dataset's variables -- it lives on the :class:`~snowtool.snowdb.spec.DatasetSpec`
as an :class:`Ingester`. A :class:`~snowtool.snowdb.dataset.Dataset` supplies the
generic side (a target ``cogs/<date>/`` directory via
:meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`); the ingester supplies
only the kind-specific *parsing*.

The seam is deliberately narrow: an ingester's sole job is to :meth:`~Ingester.plan`
its source into one :class:`DateIngest` per date -- the date, the source files
whose bytes hash that date's provenance, and a ``build_rasters`` callback that
turns the (driver-computed) source hash into the date's on-grid rasters. The
generic driver :func:`run_ingest` owns everything else: it computes the versioned
:data:`~snowtool.snowdb.dataset.INGEST_FORMAT_VERSION` source hash, drives the
write, and accumulates the :class:`IngestResult`. So a new dataset kind is a new
``plan`` (pure parsing) -- no hashing, no write orchestration, no result
bookkeeping. The driver covers both source shapes uniformly: one date per source
(SNODAS tar, SWANN NetCDF -> a single ``DateIngest``) and many dates per source
(an INSTARR tile tree -> one ``DateIngest`` per date).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.provenance import hash_files, versioned_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from datetime import date
    from pathlib import Path

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.progress import ProgressReporter


@dataclass(frozen=True)
class IngestResult:
    """The dates an ingest run wrote vs. skipped as already current.

    ``ingested`` are the dates whose ``cogs/<date>/`` dir this run (re)built;
    ``skipped`` are the dates left untouched because their stored source hash
    already matched (converge-by-default). One is returned per
    :func:`run_ingest`; the CLI reports the two sets on separate lines.
    """

    ingested: list[date]
    skipped: list[date]


class WritableRaster(Protocol):
    """Something that can write itself as a COG into a per-date directory.

    The minimal contract :meth:`Dataset.write_date_cogs` needs from each raster
    an ingester produces, so the generic write path is decoupled from any one
    dataset's input-raster type. ``out_name`` is the COG filename it writes into
    the date dir (``<source-stem>__<key>.tif``); the write path reads it up front
    to check the produced set covers every spec variable before any staging.
    """

    @property
    def out_name(self) -> str:
        # Read-only so both a plain instance attribute (SwannRaster,
        # InstarrMosaicRaster) and a property (SNODASInputRaster) satisfy it.
        ...

    def write_cog(self, output_dir: Path) -> None: ...


@dataclass(frozen=True)
class DateIngest:
    """One date's worth of work an :class:`Ingester` hands the driver.

    The whole ingester -> driver contract: ``date`` is the date this covers;
    ``source_files`` are the source artifacts whose bytes the driver hashes into
    that date's provenance (one tar/NetCDF, or a date's contributing tiles);
    ``build_rasters`` is called with the driver-computed versioned source hash and
    returns the date's on-grid rasters, tags already stamped -- so a COG can never
    be written without its ``SOURCE_HASH``.
    """

    date: date
    source_files: list[Path]
    build_rasters: Callable[[str], Iterable[WritableRaster]]


class Ingester(Protocol):
    """Parses a source artifact into per-date work items for the ingest driver.

    An implementation's sole job is :meth:`plan`: turn its own source format into
    one :class:`DateIngest` per date. All the shared machinery -- hashing the
    source into versioned provenance, driving the atomic per-date write, splitting
    ingested from skipped -- lives once in :func:`run_ingest`, not in each
    ingester. One lives on each dataset spec that supports ingest.
    """

    def plan(self, source: Path, dataset: Dataset) -> Iterator[DateIngest]: ...


def run_ingest(
    ingester: Ingester,
    source: Path,
    dataset: Dataset,
    *,
    force: bool = False,
    progress: ProgressReporter = NULL_PROGRESS,
) -> IngestResult:
    """Drive ``ingester``'s per-date plan into COGs on ``dataset``.

    The one place ingest orchestration lives: for each :class:`DateIngest` the
    ingester yields, compute the versioned source hash over its source files, ask
    it to build the date's rasters from that hash, and commit them via the generic
    :meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`. Accumulates and
    returns the :class:`IngestResult` splitting dates written from those skipped as
    already current. ``progress`` reports each date's per-variable COG writes; it
    defaults to the no-op :data:`~snowtool.snowdb.progress.NULL_PROGRESS`.
    """
    # Imported here (not at module top) to avoid an import cycle: dataset.py
    # imports this module's types under TYPE_CHECKING.
    from snowtool.snowdb.dataset import INGEST_FORMAT_VERSION

    ingested: list[date] = []
    skipped: list[date] = []
    for item in ingester.plan(source, dataset):
        # One versioned hash per date over its (sorted, by hash_files) source
        # files, stamped on every COG and compared by the skip check.
        source_hash = versioned_hash(
            INGEST_FORMAT_VERSION,
            hash_files(item.source_files),
        )
        wrote = dataset.write_date_cogs(
            item.date,
            item.build_rasters(source_hash),
            source_hash=source_hash,
            force=force,
            progress=progress,
        )
        (ingested if wrote else skipped).append(item.date)
    return IngestResult(ingested=ingested, skipped=skipped)

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
whose bytes hash that date's provenance, the COG filenames the date will land, and
a ``build_rasters`` callback that turns the (driver-computed) source hash into the
date's on-grid rasters. The generic driver :func:`run_ingest` owns everything else:
it computes the versioned :data:`INGEST_FORMAT_VERSION` source hash, drives the
write, and accumulates the :class:`IngestResult`. So a new
dataset kind is a new ``plan`` (pure parsing) -- no hashing, no write orchestration,
no result bookkeeping. The driver covers both source shapes uniformly: one date per
source (SNODAS tar, SWANN NetCDF -> a single ``DateIngest``) and many dates per
source (an INSTARR tile tree -> one ``DateIngest`` per date).

Planning stays cheap by design: an ingester derives ``out_names`` (and the date)
from source *metadata* alone -- the tar member names, a NetCDF filename -- never by
reading or extracting bytes. The per-date skip check
(:meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`) needs only those names and
the source hash, so an already-current date is skipped without ``build_rasters`` ever
running. Only a date that must (re)build pays to open its source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol

from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.provenance import hash_files, versioned_hash
from snowtool.snowdb.raster.cog import write_cog

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from datetime import date
    from pathlib import Path

    import numpy
    import rasterio

    from affine import Affine

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.progress import ProgressReporter


# On-disk format version of an ingested date's COGs, owned here by the ingest
# driver (:func:`run_ingest`). It rides along in the versioned SOURCE_HASH the skip
# compares, so bumping it makes every existing date read as stale (hash mismatch)
# and rebuild on the next ingest. Bump on a material change to the ingested-COG
# encoding (compression, band layout, nodata handling), not for source changes.
INGEST_FORMAT_VERSION = 1


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


class GridAlignedRaster(ABC):
    """Write plumbing shared by rasters that emit one grid-aligned COG.

    Both the SWANN single-band raster and the INSTARR mosaic write one COG on the
    dataset grid's *authoritative* geometry (its transform/CRS from the spec, not a
    source-file geotransform) with provenance tags. This base owns that common
    plumbing -- geometry, tags, and the :meth:`write_cog` call -- leaving each
    subclass only :meth:`read_array` (how it produces the grid-shaped array). It
    satisfies the :class:`WritableRaster` contract (``out_name`` + ``write_cog``).
    """

    def __init__(
        self,
        out_name: str,
        *,
        transform: Affine,
        crs: rasterio.crs.CRS,
        tile_size: int,
        nodata: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.out_name = out_name
        self.transform = transform
        self.crs = crs
        self.tile_size = tile_size
        self.nodata = nodata
        self.tags = tags

    @abstractmethod
    def read_array(self) -> numpy.ndarray:
        """The grid-shaped array this raster writes (subclass-specific)."""
        ...

    def write_cog(self, output_dir: Path) -> None:
        write_cog(
            output_dir / self.out_name,
            self.read_array(),
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
            tile_size=self.tile_size,
            tags=self.tags,
        )


@dataclass(frozen=True)
class DateIngest:
    """One date's worth of work an :class:`Ingester` hands the driver.

    The whole ingester -> driver contract: ``date`` is the date this covers;
    ``source_files`` are the source artifacts whose bytes the driver hashes into
    that date's provenance (one tar/NetCDF, or a date's contributing tiles);
    ``out_names`` are the COG filenames the date will land (the same
    ``WritableRaster.out_name``\\ s ``build_rasters`` produces), derived cheaply from
    source metadata so the per-date skip check can run *before* any bytes are read;
    ``build_rasters`` is called with the driver-computed versioned source hash and
    returns the date's on-grid rasters, tags already stamped -- so a COG can never
    be written without its ``SOURCE_HASH``. ``build_rasters`` runs only when the date
    is not skipped, so deferring the expensive source read into it (SNODAS tar
    extraction) keeps an already-current date free of that cost.
    """

    date: date
    source_files: list[Path]
    out_names: frozenset[str]
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
    ingester yields, compute the versioned source hash over its source files and hand
    the date's ``out_names`` plus a hash-bound ``build_rasters`` to the generic
    :meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`, which runs the per-date
    skip check on the names + hash alone and calls ``build_rasters`` only if it must
    (re)build -- so an already-current date never pays to read its source. Accumulates
    and returns the :class:`IngestResult` splitting dates written from those skipped as
    already current. ``progress`` reports each date's per-variable COG writes; it
    defaults to the no-op :data:`~snowtool.snowdb.progress.NULL_PROGRESS`.
    """
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
            item.out_names,
            # Defer the build: write_date_cogs invokes this only when the date is not
            # skipped, so the source is read at most once and never on the up-to-date
            # path. partial binds the driver's hash into the zero-arg callable.
            partial(item.build_rasters, source_hash),
            source_hash=source_hash,
            force=force,
            progress=progress,
        )
        (ingested if wrote else skipped).append(item.date)
    return IngestResult(ingested=ingested, skipped=skipped)

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
from typing import TYPE_CHECKING, ClassVar, Protocol

from snowtool.exceptions import IngestSourceError
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
    from snowtool.snowdb.variables import DatasetVariable


# On-disk format version of an ingested date's COGs, owned here by the ingest
# driver (:func:`run_ingest`). It rides along in the versioned SOURCE_HASH the skip
# compares, so bumping it makes every existing date read as stale (hash mismatch)
# and rebuild on the next ingest. Bump on a material change to the ingested-COG
# *decode contract* (band layout, nodata handling, value scaling) -- not on a
# codec swap that decodes byte-identically (the compression tag is per-file;
# DEFLATE- and ZSTD-era COGs coexist in one dataset).
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


@dataclass(frozen=True)
class GridGeometry:
    """A dataset grid's *authoritative* write geometry, threaded as one value.

    The four facts every :class:`GridAlignedRaster` needs to write a grid-aligned
    COG on the dataset grid's own lattice (its spec transform/CRS, not any source
    file's geotransform): the ``transform`` and ``crs`` a COG is written with, the
    ``tile_size`` (COG blocksize), and the grid ``shape`` (``(rows, cols)``) a
    produced array must match. Built once per dataset (see
    :attr:`~snowtool.snowdb.dataset.Dataset.grid_geometry`) and passed straight
    through each ingester's rasters, so the transform/crs/tile_size/shape are no
    longer re-derived and re-threaded as loose kwargs in every ingester.
    """

    transform: Affine
    crs: rasterio.crs.CRS
    tile_size: int
    shape: tuple[int, int]


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
        # Read-only so either a plain instance attribute (all three
        # GridAlignedRaster subclasses use one) or a property satisfies it.
        ...

    def write_cog(self, output_dir: Path) -> None: ...


class GridAlignedRaster(ABC):
    """Write plumbing shared by rasters that emit one grid-aligned COG.

    The SWANN single-band raster, the INSTARR mosaic, and the SNODAS header raster
    all write one COG on the dataset grid's *authoritative* geometry (its
    transform/CRS from the spec, not a source-file geotransform) with provenance
    tags. This base owns that common plumbing -- the :class:`GridGeometry`, tags,
    the shape guard, and the :meth:`write_cog` call -- leaving each subclass only
    :meth:`read_array` (how it produces the grid-shaped array). It satisfies the
    :class:`WritableRaster` contract (``out_name`` + ``write_cog``).
    """

    def __init__(
        self,
        out_name: str,
        geometry: GridGeometry,
        *,
        nodata: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.out_name = out_name
        self.geometry = geometry
        self.nodata = nodata
        self.tags = tags

    @abstractmethod
    def read_array(self) -> numpy.ndarray:
        """The grid-shaped array this raster writes (subclass-specific)."""
        ...

    def write_cog(self, output_dir: Path) -> None:
        array = self.read_array()
        # The grid's transform/CRS are authoritative, so an array of any other
        # shape would land silently mis-aligned under them; raise instead. INSTARR
        # is on-grid by construction and passes trivially -- a free guard.
        if array.shape != self.geometry.shape:
            raise IngestSourceError(
                f'{self.out_name!r} produced an array of shape {array.shape}, '
                f'expected the dataset grid shape {self.geometry.shape}.',
            )
        write_cog(
            output_dir / self.out_name,
            array,
            transform=self.geometry.transform,
            crs=self.geometry.crs,
            nodata=self.nodata,
            tile_size=self.geometry.tile_size,
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


def variable_out_name(stem: str, key: str) -> str:
    """The COG filename for a source ``stem`` + variable ``key``: ``{stem}__{key}.tif``.

    The single spelling of the ``__<key>.tif`` convention that couples an ingested
    COG's filename to the variable whose ``glob`` (``*__<key>.tif``) must resolve
    it. Every dataset variable's glob comment points here; changing the convention
    is a change in exactly one place.
    """
    return f'{stem}__{key}.tif'


def per_variable_ingest(
    stem: str,
    date: date,
    source_files: list[Path],
    dataset: Dataset,
    make_raster: Callable[[DatasetVariable, str, str], WritableRaster],
) -> DateIngest:
    """A :class:`DateIngest` with one grid-aligned COG per spec variable.

    The SWANN-and-INSTARR shape: name each COG ``{stem}__{key}.tif`` (via
    :func:`variable_out_name`), derive the date's ``out_names`` from the stem +
    spec alone (so the skip check has them without opening a source), and build one
    :class:`WritableRaster` per spec variable. ``make_raster(variable, out_name,
    source_hash)`` supplies only the genuinely dataset-specific part -- the source
    URI(s) and the per-variable (hash-stamped) tags; the driver-computed
    ``source_hash`` is threaded in when ``build_rasters`` runs. SNODAS (one raster
    per parsed archive member, not per spec variable) does not use this.
    """
    variables = list(dataset.spec.variables.values())
    out_by_key = {v.key: variable_out_name(stem, v.key) for v in variables}
    return DateIngest(
        date=date,
        source_files=source_files,
        out_names=frozenset(out_by_key.values()),
        build_rasters=lambda source_hash: [
            make_raster(v, out_by_key[v.key], source_hash) for v in variables
        ],
    )


class Ingester(Protocol):
    """Parses a source artifact into per-date work items for the ingest driver.

    An implementation's sole job is :meth:`plan`: turn its own source format into
    one :class:`DateIngest` per date. All the shared machinery -- hashing the
    source into versioned provenance, driving the atomic per-date write, splitting
    ingested from skipped -- lives once in :func:`run_ingest`, not in each
    ingester. One lives on each dataset spec that supports ingest.

    ``kind`` is the ingester's registry key (``'snodas'``, ``'swann'``,
    ``'instarr'``) -- the *kind* a dataset config names, distinct from a dataset
    *name* -- so :func:`~snowtool.snowdb.datasets.config_from_spec` reads the key
    straight off the ingester instead of reverse-mapping by type.
    """

    kind: ClassVar[str]

    def plan(self, source: Path, dataset: Dataset) -> Iterator[DateIngest]:
        """Yield one :class:`DateIngest` per date parsed from ``source``.

        The driver (:func:`run_ingest`) consumes each yielded ``DateIngest`` fully
        -- computing its source hash and running its (possibly skipped)
        ``build_rasters`` -- before advancing the generator. A ``build_rasters``
        callable may therefore hold generator-scoped resources bound in ``plan``
        (there are none today: SNODAS owns its extraction tempdir inside
        ``build_rasters`` itself). This is the plan -> driver ordering contract.
        """
        ...


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

"""The ingest seam: how a dataset-kind turns source data into per-date COGs.

Ingest is dataset-*kind*-specific knowledge (source formats differ -- a SNODAS
tar of raw rasters, a directory of GeoTIFFs, a NetCDF, ...), so -- like a
dataset's variables -- it lives on the :class:`~snowtool.snowdb.spec.DatasetSpec`
as an :class:`Ingester`. A :class:`~snowtool.snowdb.dataset.Dataset` supplies the
generic side (a target ``cogs/<date>/`` directory via
:meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`); the ingester supplies
the kind-specific parsing/resampling. The CLI's ``dataset ingest`` is therefore
dataset-agnostic: it just calls ``dataset.ingest(source)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from snowtool.snowdb.dataset import Dataset


@dataclass(frozen=True)
class IngestResult:
    """The dates an ingest run wrote vs. skipped as already current.

    ``ingested`` are the dates whose ``cogs/<date>/`` dir this run (re)built;
    ``skipped`` are the dates left untouched because their stored source hash
    already matched (converge-by-default). One is returned per
    :meth:`Ingester.ingest`; the CLI reports the two sets on separate lines.
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

    def write_cog(self, output_dir: Path, force: bool = False) -> None: ...


class Ingester(Protocol):
    """Turns a source artifact into per-date COGs on a dataset.

    Implementations parse their own source format and write the resulting
    rasters onto ``dataset`` (typically via
    :meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`), returning an
    :class:`IngestResult` splitting the dates written from those skipped as
    already current. One lives on each dataset spec that supports ingest.
    """

    def ingest(
        self,
        source: Path,
        dataset: Dataset,
        *,
        force: bool = False,
    ) -> IngestResult: ...

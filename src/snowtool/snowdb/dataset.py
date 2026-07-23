from __future__ import annotations

import shutil

from dataclasses import dataclass
from datetime import UTC, date, datetime
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Self

import rasterio

from snowtool import types
from snowtool.exceptions import (
    AOIRasterNotFoundError,
    IncompleteDatasetDataError,
    NodataMaskError,
    SnowtoolError,
)
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.aoi_raster import AOIRaster, aoi_provenance, write_aoi_raster
from snowtool.snowdb.atomic import staged_dir
from snowtool.snowdb.constants import AOI_HASH_TAG
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.provenance import hash_files
from snowtool.snowdb.raster.cog import SOURCE_HASH_TAG
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from griffine.grid import TiledAffineGrid

    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.ingest import IngestResult, WritableRaster
    from snowtool.snowdb.progress import ProgressReporter
    from snowtool.snowdb.query import DateQuery
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.variables import DatasetVariable
    from snowtool.snowdb.zones.zone_layer import (
        ZoneLayerProvider,
        ZoneLayerSet,
        ZoneLayerTarget,
    )


@dataclass(frozen=True)
class DatasetArtifacts:
    """Which of a dataset's on-disk artifacts are present.

    A read-only snapshot used by the ``dataset``/``doctor`` commands.
    ``zone_layers`` maps each configured zone-layer provider's name
    (``terrain``, ``landcover``, ...) to whether its complete set is present on
    disk.
    """

    zone_layers: dict[str, bool]
    aoi_rasters: bool
    cogs: bool


class Dataset:
    """A :class:`DatasetSpec` bound to its ``data/<name>/`` directory.

    Owns the per-dataset filesystem layout (``aoi-rasters/``, the per-provider
    zone-layer subdirs, ``cogs/``) and the operations on it; grid/variables are
    reached through ``self.spec``.
    """

    def __init__(
        self: Self,
        spec: DatasetSpec,
        path: Path,
        providers: Iterable[ZoneLayerProvider] = DEFAULT_ZONE_LAYER_PROVIDERS,
        *,
        nodata_mask: Path | None = None,
    ) -> None:
        self.spec = spec
        self.path = path
        # The dataset's valid-domain mask (config `nodata_mask`, resolved by the
        # SnowDb); None means every grid pixel is in-domain.
        self.nodata_mask = nodata_mask
        self._aoi_rasters = self.path / 'aoi-rasters'
        self._cogs = self.path / 'cogs'
        # One zone-layer set per provider this dataset *enables* (its config's
        # zones block), keyed by provider name (e.g. 'terrain', 'landcover'). A
        # registered provider the dataset does not enable is bound to neither
        # `providers` nor `zones`, so it is never generated, served, or reported
        # for this dataset. A new zone-layer kind is just a new provider in the
        # registry (enabled by a dataset's zones), with no edits here.
        self.providers = {
            provider.name: provider
            for provider in providers
            if spec.enables(provider.name)
        }
        self.zones: dict[str, ZoneLayerSet] = {
            provider.name: provider.layer_set(self.path / provider.subdir)
            for provider in self.providers.values()
        }

    @property
    def grid(self: Self) -> TiledAffineGrid:
        return self.spec.grid

    @property
    def coverage_domain(self: Self) -> CoverageDomain:
        """The static region this dataset can serve (for AOI coverage)."""
        return self.spec.coverage_domain

    @property
    def grid_crs(self: Self) -> rasterio.crs.CRS:
        # The rasterio view of the grid's CRS, used when writing COGs. Derived
        # from the spec's single parsed CRS (not re-parsed from grid_params) so
        # the pyproj and rasterio sides can never disagree.
        return rasterio.crs.CRS.from_user_input(self.spec.crs)

    @cached_property
    def nodata_mask_pair(self: Self) -> tuple[Path, str] | None:
        """The nodata mask paired with its sha256 digest, or ``None`` with no mask.

        The single source for both the mask path and its provenance hash: both
        halves come from the same config field, so the pair is never
        half-specified. It couples them into the one value ``write_aoi_raster``
        wants (path + digest) instead of two positionally-tied arguments; the
        digest alone is reachable via :attr:`nodata_mask_hash`. A configured mask
        whose file is missing raises :class:`NodataMaskError`.

        Cached per instance so a convergence loop over hundreds of pourpoints
        hashes the file once, not once per AOI. Management ops build short-lived
        Datasets, so a swapped mask file is picked up by the next run.
        """
        mask = self.nodata_mask
        if mask is None:
            return None
        if not mask.is_file():
            raise NodataMaskError(
                f'dataset {self.spec.name!r}: configured nodata_mask '
                f'{mask} is missing; restore the file or remove '
                'nodata_mask from the dataset config',
            )
        return (mask, hash_files([mask]))

    @property
    def nodata_mask_hash(self: Self) -> str | None:
        """sha256 of the configured nodata-mask file; ``None`` with no mask.

        The digest half of :attr:`nodata_mask_pair` (the cache lives there), as
        ``aoi_provenance`` wants just the hash.
        """
        pair = self.nodata_mask_pair
        return pair[1] if pair is not None else None

    @classmethod
    def create(
        cls: type[Self],
        spec: DatasetSpec,
        path: Path,
        *,
        nodata_mask: Path | None = None,
    ) -> tuple[Self, bool]:
        """Create the dataset's directory skeleton; converge-by-default.

        Idempotent: builds any missing part of the skeleton (the dataset dir plus
        its ``aoi-rasters/`` and ``cogs/`` subdirs) with ``exist_ok=True`` and
        never clobbers, so a fresh call and a re-run of a fully- or partially-built
        skeleton both succeed. Returns ``(dataset, created)`` where ``created`` is
        whether the skeleton was incomplete before this call (i.e. this call made
        it) -- the caller uses it to report new-vs-existing.

        Zone layers (terrain, land cover, ...) are *not* built here: each needs a
        source and is generated separately by
        :meth:`SnowDbManager.generate_zone_layers_for` (so generation can share one
        source read across every dataset).
        """
        self = cls(spec, path, nodata_mask=nodata_mask)
        created = not (self._aoi_rasters.is_dir() and self._cogs.is_dir())
        self.path.mkdir(parents=True, exist_ok=True)
        self._aoi_rasters.mkdir(exist_ok=True)
        self._cogs.mkdir(exist_ok=True)
        return self, created

    def zone_target(self: Self, provider: ZoneLayerProvider) -> ZoneLayerTarget:
        """This dataset's grid as a target for ``provider``'s generation engine."""
        from snowtool.snowdb.zones.zone_layer import ZoneLayerTarget

        return ZoneLayerTarget(
            name=self.spec.name,
            grid=self.grid,
            tile_size=self.spec.grid_params.tile_size,
            directory=self.zones[provider.name].directory,
        )

    @staticmethod
    def _format_date(date: date) -> str:
        return date.strftime('%Y%m%d')

    def raster_paths_from_query(
        self: Self,
        query: DateQuery,
        variable: DatasetVariable,
    ) -> Iterator[tuple[date, Path]]:
        # Filter the dates that actually exist (one directory listing) rather than
        # probing every calendar day in the range; this also lets a query carry an
        # open-ended interval, since selection is over a finite set. Each date's
        # single COG (and the multiple-match guard) comes from variable_path.
        for date_ in query.select(self.available_dates()):
            path = self.variable_path(date_, variable)
            if path is not None:
                yield date_, path

    def aoi_raster_path_from_triplet(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> Path:
        return (
            self._aoi_rasters / f'{triplet_naming.triplet_to_stem(station_triplet)}.tif'
        )

    def rasterize_aoi(
        self: Self,
        pourpoint: Pourpoint,
        *,
        rebuild: bool = False,
    ) -> AOIRaster | None:
        """Burn ``pourpoint``'s basin onto this dataset's grid as an AOI raster.

        Converge-by-default: build when the raster is missing or stale (see
        :meth:`aoi_raster_is_current`), skip when it is already current --
        ``rebuild=True`` forces a rebuild regardless of current state. Returns
        the built :class:`~snowtool.snowdb.aoi_raster.AOIRaster`, or ``None``
        when the existing raster was already current and nothing was written.
        This return shape serves both callers: production (the batch
        converge loop) only needs the did-it-write boolean, which the
        ``None``/raster split gives for free via truthiness; tests that want
        the raster itself get it directly on the build path, which is the
        common case in a synthetic-grid test that always starts from empty.

        The tile window is clamped to the grid (see
        :func:`~snowtool.snowdb.grid.bounding_tiles`), so a basin straddling a
        grid edge burns only its in-grid portion; a basin entirely outside the
        grid raises :class:`~snowtool.exceptions.GeometryOutsideGridError` (the
        batch paths pre-filter those by coverage instead of calling this).
        """
        if not rebuild and self.aoi_raster_is_current(pourpoint):
            return None

        # A management (write) op may run against a dataset that has no data yet,
        # so create the aoi-rasters dir if it is missing (but never the base
        # snowdb dirs -- those are SnowDbManager.initialize's job).
        self._aoi_rasters.mkdir(parents=True, exist_ok=True)

        path = self.aoi_raster_path_from_triplet(pourpoint.station_triplet)

        # The AOI is stored in WGS84; reproject it into this grid's CRS so its
        # tile extent and pixel mask are computed in the grid's own coordinates.
        # `spec.crs` is the single narrowed-CRS source (Optional resolved once).
        geometry = pourpoint.geometry_in_crs(self.spec.crs)

        # A projected grid burns its constant cell area; a geographic grid burns
        # per-row geodesic area, computed from the base grid (cell_area=None).
        cell_area = None if self.spec.is_geographic else self.spec.cell_area

        write_aoi_raster(
            path,
            geometry,
            self.grid,
            pourpoint.geometry_hash,
            cell_area=cell_area,
            nodata_mask=self.nodata_mask_pair,
        )

        return AOIRaster.open(path, self.grid)

    def aoi_raster_hash(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> str | None:
        """The AOI-geometry hash an existing AOI raster was burned from.

        Reads only the COG's tags (no array decode); returns ``None`` if the
        raster does not exist or predates the ``AOI_HASH_TAG`` tagging.
        """
        path = self.aoi_raster_path_from_triplet(station_triplet)
        if not path.is_file():
            return None
        with rasterio.open(path) as ds:
            return ds.tags().get(AOI_HASH_TAG)

    def aoi_raster_is_current(self: Self, pourpoint: Pourpoint) -> bool:
        """Whether a burned AOI raster exists AND matches ``pourpoint``'s geometry
        AND the current burned-raster format version.

        ``False`` means missing or stale (changed geometry *or* an old format
        version) -- either way :meth:`rasterize_aoi` should (re)build it.
        """
        return self.aoi_raster_hash(pourpoint.station_triplet) == aoi_provenance(
            pourpoint.geometry_hash,
            self.nodata_mask_hash,
        )

    def remove_aoi_raster(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> bool:
        """Delete this dataset's burned AOI raster for ``triplet``; True if present."""
        path = self.aoi_raster_path_from_triplet(station_triplet)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def ingest(
        self: Self,
        source: Path,
        *,
        force: bool = False,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> IngestResult:
        """Ingest a source artifact into per-date COGs, via this dataset's ingester.

        Drives ``spec.ingester``'s per-date plan through the generic
        :func:`~snowtool.snowdb.ingest.run_ingest` (which computes the versioned
        source hash and commits each date); raises if the dataset has no
        configured ingester. Returns an
        :class:`~snowtool.snowdb.ingest.IngestResult` splitting the dates written
        from those skipped as already current. ``progress`` reports each date's
        per-variable COG writes (see :meth:`write_date_cogs`).
        """
        from snowtool.snowdb.ingest import run_ingest

        ingester = self.spec.ingester
        if ingester is None:
            raise SnowtoolError(
                f"dataset '{self.spec.name}' has no configured ingester; "
                'nothing can be ingested into it.',
            )
        return run_ingest(ingester, source, self, force=force, progress=progress)

    def _unresolved_variables(self: Self, names: Iterable[str]) -> set[str]:
        """Spec variable keys whose glob does not match exactly one of ``names``.

        A key is *unresolved* when no filename matches its glob (missing) or more
        than one does (duplicated) -- both leave a date's data incomplete or
        ambiguous. Drives the pre-/post-write completeness checks in
        :meth:`write_date_cogs`.
        """
        names = list(names)
        return {
            variable.key
            for variable in self.spec.variables.values()
            if sum(fnmatch(name, variable.glob) for name in names) != 1
        }

    def _date_source_hash(self: Self, date_dir: Path) -> str | None:
        """The ``SOURCE_HASH`` provenance tag on a date's COGs, or ``None``.

        A header-only tags read of any one COG in the dir (they all carry the same
        per-date hash), the same cheap pattern as :meth:`aoi_raster_hash`. Returns
        ``None`` when the dir is empty or the COGs predate the tag (a legacy date
        dir), which the skip check reads as stale.
        """
        cogs = sorted(p for p in date_dir.iterdir() if p.is_file())
        if not cogs:
            return None
        with rasterio.open(cogs[0]) as ds:
            return ds.tags().get(SOURCE_HASH_TAG)

    def write_date_cogs(
        self: Self,
        date: date,
        out_names: Iterable[str],
        build_rasters: Callable[[], Iterable[WritableRaster]],
        *,
        source_hash: str,
        force: bool = False,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> bool:
        """Write a date's already-on-grid rasters into ``cogs/<YYYYMMDD>/`` atomically.

        The dataset-agnostic write side of ingest: it owns the date directory.
        ``out_names`` are the COG filenames this date will land (the ingester derives
        them cheaply from source metadata); ``build_rasters`` is a deferred callable
        producing the rasters (which know how to write themselves as COGs) -- it is
        invoked **only if the date is not skipped**, so an already-current date never
        pays to read the source. ``source_hash`` is the versioned hash of the source
        artifact this date came from (see
        :data:`~snowtool.snowdb.ingest.INGEST_FORMAT_VERSION`); it is both
        stamped on every COG (via the ingester's ``SOURCE_HASH`` tag) and used by the
        skip check below. Returns ``True`` if the date dir was (re)built, ``False`` if
        it was skipped as already current.

        The whole per-date directory is the unit of commit. Writes stage into a
        temp dir beside the target (:func:`~snowtool.snowdb.atomic.staged_dir`) and
        are swapped in wholesale, so (a) a crash mid-ingest never leaves a *partial*
        date on disk -- a reader sees the wholly-old dir or the wholly-new one --
        and (b) stale COGs from a prior, differently-named source vanish by
        construction rather than lingering beside the new ones and making a variable
        unresolvable (the finding-5 duplicate-``__swe.tif`` bug).

        Completeness is enforced at date granularity. Before any filesystem work the
        declared ``out_names`` must cover every spec variable, so a source short a
        required input variable raises :class:`IncompleteDatasetDataError` up front;
        after writing, every spec variable must resolve to exactly one COG in the
        staged dir or the swap is abandoned and the existing date dir left untouched.

        Idempotent-skip granularity is likewise **per-date, not per-file**: without
        ``force`` a date is skipped only when its dir already holds *exactly* the
        COGs this call would write (complete and free of stale members) **and** their
        stored ``SOURCE_HASH`` equals ``source_hash``. The filename set alone is not
        enough: source filenames embed provenance, so a *renamed* re-release is
        caught by a name mismatch, but a re-release under the *same* filename with
        different bytes would keep the names identical -- the hash equality catches
        that, forcing a rebuild. A missing tag (a date dir written before hashing)
        also reads as stale. Any divergence rebuilds the whole date dir; ``force``
        always rebuilds.
        """
        expected_names = frozenset(out_names)

        # Pre-validate the declared outputs before touching the filesystem (or
        # reading the source): the names must cover every spec variable, so a source
        # short an input variable fails fast rather than committing a partial date.
        missing = self._unresolved_variables(expected_names)
        if missing:
            raise IncompleteDatasetDataError.for_variables(
                self.spec.name,
                date,
                missing,
            )

        output_dir = self.date_dir(date)

        # Skip-if-current (per-date): an existing dir already holding exactly the
        # COGs this call would write *and* stamped with the same source hash is
        # complete and up to date -- nothing to do. Decided on the declared names +
        # hash alone, *before* build_rasters runs, so an up-to-date date is skipped
        # without reading (extracting) its source. Any divergence (a missing member,
        # a stale COG from an old source, or a same-name different-bytes re-release)
        # falls through to a full, atomic rebuild below.
        if (
            not force
            and output_dir.is_dir()
            and {p.name for p in output_dir.iterdir() if p.is_file()} == expected_names
            and self._date_source_hash(output_dir) == source_hash
        ):
            return False

        # Not skipped: now build the rasters (this is where SNODAS extracts its tar).
        rasters = list(build_rasters())

        # staged_dir stages beside the target, so the cogs/ parent must exist (a
        # management write may run before any date has been ingested).
        self._cogs.mkdir(parents=True, exist_ok=True)
        with (
            staged_dir(output_dir) as staging,
            progress.track(
                f'{self.spec.name} {date.isoformat()}',
                total=len(rasters),
            ) as task,
        ):
            for raster in rasters:
                raster.write_cog(output_dir=staging)
                task.advance()

            # Post-validate in the staged dir before the swap: every spec variable
            # must resolve to exactly one written COG. On failure this raises inside
            # the context, which discards the staged dir and leaves the existing
            # date dir untouched.
            missing = self._unresolved_variables(
                p.name for p in staging.iterdir() if p.is_file()
            )
            if missing:
                raise IncompleteDatasetDataError.for_variables(
                    self.spec.name,
                    date,
                    missing,
                )
        return True

    def load_aoi_raster(
        self: Self,
        station_triplet: types.StationTriplet,
    ) -> AOIRaster:
        """Open the burned AOI raster for ``triplet`` (the stats read input).

        Raises :class:`FileNotFoundError` (pointing at ``pourpoint rasterize``) when the
        raster has not been built for this dataset, so a stats query surfaces a
        clean missing-prerequisite error rather than a bare open failure.
        """
        path = self.aoi_raster_path_from_triplet(station_triplet)
        if not path.is_file():
            raise AOIRasterNotFoundError(
                f'No AOI raster for {station_triplet!r} in dataset '
                f'{self.spec.name!r}; run `pourpoint rasterize` first.',
            )
        return AOIRaster.open(path, self.grid)

    # --- read-only query helpers (drive the report/diagnostics commands) ------

    @staticmethod
    def _parse_date_dir(name: str) -> date | None:
        """The ``date`` a ``cogs/<name>/`` dir encodes, or ``None`` if it isn't one.

        ``name`` is a calendar label (``YYYYMMDD``); it is pinned to UTC purely
        to build a tz-aware value, which does not shift the date.
        """
        try:
            parsed = datetime.strptime(name, '%Y%m%d').replace(tzinfo=UTC)
        except ValueError:
            return None
        return parsed.date()

    def date_dir(self: Self, d: date) -> Path:
        """The ``cogs/<YYYYMMDD>/`` directory for date ``d`` (may not exist)."""
        return self._cogs / self._format_date(d)

    def available_dates(
        self: Self,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        """Every date with an ingested ``cogs/<YYYYMMDD>/`` directory, ascending.

        ``start``/``end`` bound the result inclusively; either may be omitted
        for an open end.
        """
        if not self._cogs.is_dir():
            return []
        dates = (
            self._parse_date_dir(child.name)
            for child in self._cogs.iterdir()
            if child.is_dir()
        )
        return sorted(
            d
            for d in dates
            if d is not None
            and (start is None or d >= start)
            and (end is None or d <= end)
        )

    def variable_path(
        self: Self,
        d: date,
        variable: DatasetVariable,
    ) -> Path | None:
        """The single COG for ``variable`` on date ``d``, or ``None`` if absent."""
        matching = list(self.date_dir(d).glob(variable.glob))
        if len(matching) > 1:
            # Two COGs match one variable's glob -- a stale duplicate from a
            # differently-named source that write_date_cogs' wholesale swap now
            # prevents on write, but an old date on disk may still carry. Surface
            # it as the typed integrity error rather than a bare RuntimeError.
            raise IncompleteDatasetDataError.for_variables(
                self.spec.name,
                d,
                [variable.key],
            )
        return matching[0] if matching else None

    def missing_variables(self: Self, d: date) -> set[DatasetVariable]:
        """Spec variables whose glob matches no file in date ``d``'s cogs dir.

        An absent date directory yields every variable (nothing is present).
        """
        return {
            variable
            for variable in self.spec.variables.values()
            if self.variable_path(d, variable) is None
        }

    def aoi_raster_paths(self: Self) -> list[Path]:
        """The burned ``aoi-rasters/*.tif`` files, sorted by path."""
        if not self._aoi_rasters.is_dir():
            return []
        return sorted(self._aoi_rasters.glob('*.tif'))

    def aoi_raster_triplets(self: Self) -> set[types.StationTriplet]:
        """Station triplets that have a burned ``aoi-rasters/<triplet>.tif``."""
        return {
            triplet_naming.stem_to_triplet(path.stem)
            for path in self.aoi_raster_paths()
        }

    def artifact_status(self: Self) -> DatasetArtifacts:
        """Which of this dataset's on-disk artifacts currently exist."""
        return DatasetArtifacts(
            zone_layers={
                name: zone_set.present() for name, zone_set in self.zones.items()
            },
            aoi_rasters=self._aoi_rasters.is_dir(),
            cogs=self._cogs.is_dir(),
        )

    def remove_date(self: Self, d: date, *, dry_run: bool = False) -> bool:
        """Delete a date's ``cogs/<YYYYMMDD>/`` directory; True if it existed.

        ``dry_run`` reports whether the date dir exists without deleting it.
        """
        date_dir = self.date_dir(d)
        if not date_dir.is_dir():
            return False
        if not dry_run:
            shutil.rmtree(date_dir)
        return True

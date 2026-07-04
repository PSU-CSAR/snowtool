from __future__ import annotations

import shutil

from dataclasses import dataclass
from datetime import UTC, date, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Self

import rasterio

from snowtool import types
from snowtool.exceptions import (
    AOIRasterNotFoundError,
    IncompleteDatasetDataError,
    SnowtoolError,
)
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.aoi_raster import AOIRaster, aoi_provenance, write_aoi_raster
from snowtool.snowdb.atomic import staged_dir
from snowtool.snowdb.constants import AOI_HASH_TAG
from snowtool.snowdb.grid import bounding_tiles, grid_extent_4326
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from griffine.grid import TiledAffineGrid

    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.ingest import WritableRaster
    from snowtool.snowdb.progress import ProgressReporter
    from snowtool.snowdb.query import DateQuery
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.variables import DatasetVariable
    from snowtool.snowdb.zones.zone_layer import (
        GenerationOptions,
        ZoneLayerProvider,
        ZoneLayerSet,
        ZoneLayerSource,
        ZoneLayerTarget,
    )


@dataclass(frozen=True)
class DatasetArtifacts:
    """Which of a dataset's on-disk artifacts are present.

    A read-only snapshot used by the diagnostics/report commands.
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
    ) -> None:
        self.spec = spec
        self.path = path
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

    def validate(self: Self) -> Self:
        if not self.path.exists():
            raise FileNotFoundError(f'Unable to read directory: {self.path}')

        if not self.path.is_dir():
            raise ValueError(f'Not a directory: {self.path}')

        return self

    @classmethod
    def create(
        cls: type[Self],
        spec: DatasetSpec,
        path: Path,
        force: bool = False,
    ) -> Self:
        """Create the dataset's directory skeleton.

        Zone layers (terrain, land cover, ...) are *not* built here: each needs a
        source and is generated separately by :meth:`generate_zone_layers` (so
        generation can share one source read across every dataset -- see
        :meth:`SnowDb.generate_zone_layers`).
        """
        self = cls(spec, path)

        try:
            # The dataset dir itself may already exist as an empty skeleton from
            # `snowdb init`, so tolerate it; whether the dataset is already
            # *populated* is enforced by the artifact guards below (the
            # aoi-rasters/cogs dirs), which still refuse to clobber existing data
            # without force.
            self.path.mkdir(parents=True, exist_ok=True)
            self._aoi_rasters.mkdir(exist_ok=force)
            self._cogs.mkdir(exist_ok=force)
        except FileExistsError as e:
            raise FileExistsError(
                f'Could not create {self.spec.name} dataset: {self.path} already '
                'exists. Remove and try again or use `force=True`.',
            ) from e

        return self

    def zone_target(self: Self, provider: ZoneLayerProvider) -> ZoneLayerTarget:
        """This dataset's grid as a target for ``provider``'s generation engine."""
        from snowtool.snowdb.zones.zone_layer import ZoneLayerTarget

        return ZoneLayerTarget(
            name=self.spec.name,
            grid=self.grid,
            tile_size=self.spec.grid_params.tile_size,
            directory=self.zones[provider.name].directory,
        )

    def generate_zone_layers(
        self: Self,
        provider: ZoneLayerProvider,
        source: ZoneLayerSource,
        *,
        force: bool = False,
        options: GenerationOptions | None = None,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> str:
        """Generate this dataset's zone-layer set for ``provider`` from ``source``.

        A single-grid pass over the source (binning only into this grid); for the
        multi-grid shared-source pass, see :meth:`SnowDb.generate_zone_layers`.
        ``options`` carries engine knobs (e.g. terrain's ``workers``/
        ``block_size``); ``progress`` reports the long step. Returns the set's
        provenance hash.
        """
        bounds = grid_extent_4326(self.grid)
        hashes = provider.generate(
            source,
            [self.zone_target(provider)],
            bounds,
            force=force,
            options=options,
            progress=progress,
        )
        return hashes[self.spec.name]

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

    def rasterize_aoi(self, aoi: Pourpoint, force: bool = False) -> AOIRaster:
        # A management (write) op may run against a dataset that has no data yet,
        # so create the aoi-rasters dir if it is missing (but never the base
        # snowdb dirs -- those are SnowDbManager.initialize's job).
        self._aoi_rasters.mkdir(parents=True, exist_ok=True)

        path = self.aoi_raster_path_from_triplet(aoi.station_triplet)
        if not force and path.exists():
            raise FileExistsError(
                f'Could not create AOI raster: {path} already exists. '
                'Remove and try again or use `force=True`.',
            )

        # The AOI is stored in WGS84; reproject it into this grid's CRS so its
        # tile extent and pixel mask are computed in the grid's own coordinates.
        crs = self.grid.crs
        if crs is None:  # pragma: no cover - make_grid always sets a CRS
            raise ValueError('grid has no CRS')
        geometry = aoi.geometry_in_crs(crs)
        ul_tile, br_tile = bounding_tiles(self.grid, geometry.bounds)

        # A projected grid burns its constant cell area; a geographic grid burns
        # per-row geodesic area, computed from the base grid (cell_area=None).
        cell_area = None if self.spec.is_geographic else self.spec.cell_area

        write_aoi_raster(
            path,
            geometry,
            self.grid_crs,
            ul_tile,
            br_tile,
            tile_size=self.spec.grid_params.tile_size,
            provenance=aoi_provenance(aoi.geometry_hash),
            base_grid=self.grid.base_grid,
            cell_area=cell_area,
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

    def aoi_raster_is_current(self: Self, aoi: Pourpoint) -> bool:
        """Whether a burned AOI raster exists AND matches ``aoi``'s geometry AND
        the current burned-raster format version.

        ``False`` means missing or stale (changed geometry *or* an old format
        version) -- either way :meth:`rasterize_aoi` should (re)build it.
        """
        return self.aoi_raster_hash(aoi.station_triplet) == aoi_provenance(
            aoi.geometry_hash,
        )

    def rasterize_aoi_if_needed(
        self: Self,
        aoi: Pourpoint,
        *,
        rebuild: bool = False,
    ) -> bool:
        """Build the AOI raster when missing or stale; True if it was (re)built.

        ``rebuild=True`` forces a rebuild regardless of current state. The
        converge-by-default path (``rebuild=False``) skips a raster only when it
        is already current (a matching :attr:`Pourpoint.geometry_hash` tag).
        """
        if not rebuild and self.aoi_raster_is_current(aoi):
            return False
        self.rasterize_aoi(aoi, force=True)
        return True

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

    def ingest(self: Self, source: Path, *, force: bool = False) -> list[date]:
        """Ingest a source artifact into per-date COGs, via this dataset's ingester.

        Delegates to ``spec.ingester`` (the dataset-kind-specific parser); raises
        if the dataset has no configured ingester. Returns the dates ingested.
        """
        ingester = self.spec.ingester
        if ingester is None:
            raise SnowtoolError(
                f"dataset '{self.spec.name}' has no configured ingester; "
                'nothing can be ingested into it.',
            )
        return ingester.ingest(source, self, force=force)

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

    def write_date_cogs(
        self: Self,
        date: date,
        rasters: Iterable[WritableRaster],
        *,
        force: bool = False,
    ) -> None:
        """Write a date's already-on-grid rasters into ``cogs/<YYYYMMDD>/`` atomically.

        The dataset-agnostic write side of ingest: it owns the date directory; the
        rasters (produced by an :class:`~snowtool.snowdb.ingest.Ingester`) know how
        to write themselves as COGs into it.

        The whole per-date directory is the unit of commit. Writes stage into a
        temp dir beside the target (:func:`~snowtool.snowdb.atomic.staged_dir`) and
        are swapped in wholesale, so (a) a crash mid-ingest never leaves a *partial*
        date on disk -- a reader sees the wholly-old dir or the wholly-new one --
        and (b) stale COGs from a prior, differently-named source vanish by
        construction rather than lingering beside the new ones and making a variable
        unresolvable (the finding-5 duplicate-``__swe.tif`` bug).

        Completeness is enforced at date granularity. Before any filesystem work the
        supplied rasters must cover every spec variable, so a source that is short a
        required input variable raises :class:`IncompleteDatasetDataError` up front;
        after writing, every spec variable must resolve to exactly one COG in the
        staged dir or the swap is abandoned and the existing date dir left untouched.

        Idempotent-skip granularity is likewise **per-date, not per-file**: without
        ``force`` a date is skipped only when its dir already holds *exactly* the
        COGs this call would write (complete and free of stale members); any missing
        or stale member rebuilds the whole date dir. ``force`` always rebuilds.
        """
        rasters = list(rasters)
        expected_names = {raster.out_name for raster in rasters}

        # Pre-validate the inputs before touching the filesystem: the produced
        # rasters must cover every spec variable, so a source short an input
        # variable fails fast rather than committing a partial date.
        missing = self._unresolved_variables(expected_names)
        if missing:
            raise IncompleteDatasetDataError.for_variables(
                self.spec.name,
                date,
                missing,
            )

        output_dir = self.date_dir(date)

        # Skip-if-current (per-date): an existing dir already holding exactly the
        # COGs this call would write is complete and up to date -- nothing to do.
        # Any divergence (a missing member, or a stale COG from an old source)
        # falls through to a full, atomic rebuild below.
        if not force and output_dir.is_dir():
            on_disk = {p.name for p in output_dir.iterdir() if p.is_file()}
            if on_disk == expected_names:
                return

        # staged_dir stages beside the target, so the cogs/ parent must exist (a
        # management write may run before any date has been ingested).
        self._cogs.mkdir(parents=True, exist_ok=True)
        with staged_dir(output_dir) as staging:
            for raster in rasters:
                # The staging dir is fresh and empty, so nothing can be clobbered;
                # ``force`` just keeps a raster's own existence guard from tripping.
                raster.write_cog(output_dir=staging, force=True)

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

    def aoi_rasters(self: Self) -> Iterator[AOIRaster]:
        yield from (
            AOIRaster.open(path, self.grid) for path in self._aoi_rasters.glob('*.tif')
        )

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

    def available_dates(self: Self) -> list[date]:
        """Every date with an ingested ``cogs/<YYYYMMDD>/`` directory, ascending."""
        if not self._cogs.is_dir():
            return []
        dates = (
            self._parse_date_dir(child.name)
            for child in self._cogs.iterdir()
            if child.is_dir()
        )
        return sorted(d for d in dates if d is not None)

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

    def dates_before(self: Self, before: date) -> list[date]:
        """Ingested dates strictly older than ``before`` (the prune selection)."""
        return [d for d in self.available_dates() if d < before]

    def remove_date(self: Self, d: date) -> bool:
        """Delete a date's ``cogs/<YYYYMMDD>/`` directory; True if it existed."""
        date_dir = self.date_dir(d)
        if not date_dir.is_dir():
            return False
        shutil.rmtree(date_dir)
        return True

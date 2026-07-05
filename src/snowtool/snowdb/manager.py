"""The snowdb admin/management layer: every write lives here, not on ``SnowDb``.

:class:`SnowDbManager` *has* a :class:`~snowtool.snowdb.db.SnowDb` (its lean
read/query surface, reachable as :attr:`SnowDbManager.db`) and owns every
operation that mutates the database -- creating the layout, registering datasets,
importing/syncing/removing pourpoints, rasterizing them, and generating zone layers.
The read path (the FastAPI app) builds only a :class:`SnowDb`; the CLI's write
commands and library admin code build a manager. "The management layer has a
snowdb, not the other way around."
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from snowtool import types
from snowtool.exceptions import (
    GeoJSONValidationError,
    PourpointPruneDestinationRequiredError,
    SnowDbConfigError,
)
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.atomic import atomic_copy
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    DatasetConfig,
    PathDatasetLink,
    RootConfig,
)
from snowtool.snowdb.coverage import Coverage, dataset_coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.pourpoint_index import PourpointIndex, PourpointIndexEntry
from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from snowtool.snowdb.aoi_raster import AOIRaster
    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.grid import Bounds
    from snowtool.snowdb.progress import ProgressReporter
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.zones.zone_layer import (
        GenerationOptions,
        ZoneLayerProvider,
        ZoneLayerSource,
    )


def _combined_extent(
    extents: Iterable[Bounds],
) -> Bounds:
    """Union of ``(west, south, east, north)`` extents."""
    boxes = list(extents)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


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


@dataclass(frozen=True)
class StagedDataset:
    """The product of :meth:`SnowDbManager.stage_dataset`: everything a new
    dataset needs built *under its own data directory* but not yet visible to
    readers, ready for :meth:`SnowDbManager.register_dataset` to commit.

    ``dataset`` is the built (still-unregistered) :class:`Dataset`;
    ``created`` is whether this stage created the skeleton (vs. found an existing
    one); ``rasterized`` is the AOI-raster pass over
    the new grid; ``coverage`` is the per-pourpoint geometric coverage of the new
    grid, which the commit writes into the index so a reader sees real coverage
    without waiting for a reindex.
    """

    dataset: Dataset
    created: bool
    rasterized: AOIRasterizeResult
    coverage: dict[types.StationTriplet, Coverage]


class SnowDbManager:
    """Owns every write against a held :class:`SnowDb` (its read/query surface).

    Built around an already-constructed :class:`SnowDb` (reachable as
    :attr:`db`); :meth:`open` and :meth:`initialize` are the convenience
    constructors that build the read database (or its layout) and wrap it.

    Concurrency: config and index writes are read-modify-write with no
    cross-process locking, so they assume a single writer at a time -- two admin
    commands mutating the root config or the pourpoint index concurrently can lose
    an update (last save wins). Ingest is different: it only writes per-date
    ``cogs/<date>/`` directories, each committed by an atomic whole-directory swap,
    so bulk ingest parallelizes freely across distinct dates. Just avoid
    deliberately ingesting the same date from two processes at once.
    """

    def __init__(self: Self, db: SnowDb) -> None:
        self.db = db

    @classmethod
    def open(
        cls: type[Self],
        path: Path,
        *,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
    ) -> Self:
        """Open the read :class:`SnowDb` at ``path`` and wrap it in a manager."""
        return cls(
            SnowDb.open(
                path,
                zone_layer_providers=zone_layer_providers,
            ),
        )

    @classmethod
    def initialize(
        cls: type[Self],
        path: Path,
        specs: Iterable[DatasetSpec] = (),
        *,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
    ) -> Self:
        """Create the base snowdb layout + an empty root config at ``path``.

        The one entry point that creates the root structure -- the
        ``snowdb_conf.json`` root config (with *no* datasets registered; a dataset
        exists only once :meth:`register_dataset` links it, and is served only
        while its link is active), ``pourpoints/``, ``data/``, and a ``data/<name>/``
        directory per ``specs`` entry (a convenience for staging; the CLI ``init``
        passes none). Idempotent: an existing config is loaded and left as is (its
        creation stamp and datasets preserved). Returns a manager over the root --
        its read database is empty unless datasets were already registered.
        """
        specs = list(specs)
        path = Path(path)
        # pourpoints/ holds the index.geojson manifest; pourpoints/records/ the
        # per-pourpoint record files.
        (path / 'pourpoints' / 'records').mkdir(parents=True, exist_ok=True)
        data_path = path / 'data'
        data_path.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            (data_path / spec.name).mkdir(parents=True, exist_ok=True)
        config_path = path / CONFIG_FILENAME
        if config_path.is_file():
            config = RootConfig.load(config_path)
        else:
            config = RootConfig.create()
            config.save(config_path)
        return cls(
            SnowDb(
                config,
                zone_layer_providers=zone_layer_providers,
            ),
        )

    def _read_root_config(self: Self) -> RootConfig:
        """Load this root's on-disk config (raises if it is absent)."""
        config_path = self.db.config_path
        if config_path is None or not config_path.is_file():
            raise SnowDbConfigError(self.db.root)
        return RootConfig.load(config_path)

    def register_dataset(
        self: Self,
        name: str,
        dataset_config_path: Path,
        *,
        link_type: str = 'path',
        active: bool = True,
        coverage: Mapping[types.StationTriplet, Coverage] | None = None,
    ) -> RootConfig:
        """Commit a dataset registration: the root-config write is the commit point.

        Writes ``datasets[name]`` -> a link at ``dataset_config_path``, stored
        relative to the root when the config lives under the tree (a relocatable
        tree) and absolute otherwise (a staged-elsewhere dataset). Re-registering a
        name overwrites its link. ``active`` sets the link's visibility flag:
        registration makes a dataset *exist* (manageable by name); only an active
        one is served by readers (toggle later with :meth:`set_dataset_active`).
        Returns the updated config.

        ``coverage`` (a triplet -> :class:`Coverage` map, produced by
        :meth:`stage_dataset`) is folded into every existing index entry under the
        new dataset's key *before* the config is written. The two writes are
        ordered index-first, config-second, and both are atomic (WS0), so every
        crash window is safe: a crash after the index write leaves only a harmless
        extra coverage key (readers still see the old dataset set from the config),
        and a crash before the config write leaves readers seeing exactly the old
        database. Without ``coverage`` (an out-of-band ``dataset add`` that skipped
        staging) only the config is written; the missing coverage key reads as
        ``Coverage.NONE`` until the next ``pourpoint reindex``. Going live still
        needs a service restart -- the ``SnowDb`` is built once at startup.

        ``name`` must be usable as a bare :meth:`resolve_dataset` token and a
        directory name, so a name containing a path separator or ending in
        ``.json`` (which that method's syntactic partition would read as a
        path) is rejected up front -- registration is the single choke point.
        """
        if '/' in name or '\\' in name or name.endswith('.json'):
            raise ValueError(
                f'Invalid dataset name {name!r}: a name must not contain a '
                "path separator or end with '.json' (it must be usable as a "
                'bare dataset token and a directory name).',
            )
        if link_type != 'path':
            raise ValueError(f'unknown dataset link type: {link_type!r}')
        config = self._read_root_config()
        config_path = self.db.config_path
        if config_path is None:  # pragma: no cover - _read_root_config guarantees it
            raise SnowDbConfigError(self.db.root)
        dataset_config_path = Path(dataset_config_path).resolve()
        root = config_path.parent.resolve()
        # Relative when under the tree (keeps the tree relocatable); absolute when
        # the dataset is staged elsewhere. Stored posix-normalized (via the
        # relative path's as_posix / the absolute path itself), which Path
        # round-trips on POSIX.
        if dataset_config_path.is_relative_to(root):
            link = Path(dataset_config_path.relative_to(root).as_posix())
        else:
            link = dataset_config_path

        # Commit order matters: fold the staged coverage into the index first, so a
        # crash before the config write leaves only an unreferenced coverage key.
        if coverage is not None:
            self._write_dataset_coverage(name, coverage)

        config.datasets[name] = PathDatasetLink(path=link, active=active)
        config.save(config_path)
        return config

    def set_dataset_active(self: Self, name: str, active: bool) -> RootConfig:
        """Toggle dataset ``name``'s ``active`` flag in the root config.

        The activation half of the register/activate split: registration says a
        dataset exists; this flips whether readers serve it. The config write is
        the commit point (atomic, like registration), and a running API server
        still needs a restart to see the change. Raises :class:`ValueError` for a
        name the root config does not register. Idempotent -- setting the current
        state re-saves harmlessly.
        """
        config = self._read_root_config()
        config_path = self.db.config_path
        if config_path is None:  # pragma: no cover - _read_root_config guarantees it
            raise SnowDbConfigError(self.db.root)
        if name not in config.datasets:
            registered = ', '.join(sorted(config.datasets)) or '(none)'
            raise ValueError(
                f'No registered dataset {name!r}. Registered datasets: {registered}.',
            )
        config.datasets[name].active = active
        config.save(config_path)
        return config

    def _write_dataset_coverage(
        self: Self,
        name: str,
        coverage: Mapping[types.StationTriplet, Coverage],
    ) -> None:
        """Add ``name``'s per-pourpoint coverage to the persisted index in place.

        Loads the on-disk index, sets ``entry.coverage[name]`` for every entry (an
        absent triplet reads as :attr:`Coverage.NONE`), and re-saves it atomically.
        A no-op when the index is empty -- there is nothing to annotate, and the
        coverage is re-derived for every dataset by the next reindex regardless.
        """
        index = PourpointIndex.load(self.db.pourpoint_index_path)
        if not index:
            return
        for triplet, entry in index.entries.items():
            entry.coverage[name] = coverage.get(triplet, Coverage.NONE)
        index.save(self.db.pourpoint_index_path)

    def _build_staged_dataset(
        self: Self,
        name: str,
        dataset_config_path: Path,
    ) -> Dataset:
        """Build a :class:`Dataset` from its config *directly*, bypassing the catalog.

        Mirrors the path-link binding :class:`SnowDb` does at construction (config
        location as the resolution base, ``data/<name>``-beside-config default), so
        a not-yet-registered dataset gets the same directory a later ``SnowDb.open``
        will resolve for it -- without appearing in ``self.db.datasets`` yet.
        """
        from snowtool.snowdb.spec import DatasetSpec

        resolved = Path(dataset_config_path).resolve()
        config = DatasetConfig.load(resolved)
        spec = DatasetSpec.from_config(config, name)
        directory = self.db.dataset_dir(
            name,
            config,
            base=resolved.parent,
            default=resolved.parent,
        )
        return Dataset(spec, directory, self.db.zone_layer_providers.values())

    def resolve_dataset(self: Self, token: str) -> Dataset:
        """Resolve a dataset NAME or a config path to a :class:`Dataset`.

        The token is partitioned *syntactically*, so a name and a file can
        never shadow each other: a token containing a path separator or ending
        in ``.json`` is a PATH; anything else is a NAME. A path token never
        consults the catalog -- it must be an existing dataset config file
        (its NAME taken from the parent directory), else :class:`ValueError`.
        A name token never touches the filesystem -- it resolves only against
        the root config's registered datasets (active or not: management ops
        -- ingest, zone generation, diagnostics -- never care about reader
        visibility); an unregistered name raises :class:`ValueError`. To
        target an unregistered (staged) config, pass its path.
        """
        is_path = (
            '/' in token or '\\' in token or os.sep in token or token.endswith('.json')
        )
        if is_path:
            path = Path(token)
            if not path.is_file():
                raise ValueError(f'No dataset config file at {path}.')
            return self._build_staged_dataset(path.parent.name, path)
        if token in self.db.registered:
            return self.db.registered[token]
        registered = ', '.join(sorted(self.db.registered)) or '(none)'
        raise ValueError(
            f'No registered dataset {token!r}. Registered datasets: '
            f'{registered}. To target an unregistered dataset config, pass '
            "its path (e.g. './dataset.json' or 'data/x/dataset.json').",
        )

    def stage_dataset(
        self: Self,
        name: str,
        dataset_config_path: Path,
        *,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> StagedDataset:
        """Build everything a new dataset needs, all *invisible* to readers.

        The staging half of the register split: it builds the dataset from its
        config (:meth:`_build_staged_dataset`, so it works before the dataset is in
        ``self.db.datasets``) and, entirely under ``data/<name>/`` -- a directory a
        reader ignores because datasets come only from the root config -- creates
        the skeleton, rasterizes every indexed (basin-bearing) pourpoint's basin
        onto the new grid, and computes each pourpoint's geometric coverage of
        that grid. Zone layers are *never* generated here -- that is a separate
        explicit operation (:meth:`generate_zone_layers_for`, which shares one
        source read across datasets). Nothing here touches the root config or
        the index, so a fresh ``SnowDb.open`` still does not see the dataset
        until :meth:`register_dataset` commits it (passing back
        :attr:`StagedDataset.coverage`).

        ``progress`` reports each slow phase as a sequential tracked task: parsing
        the pourpoint records, the per-pourpoint coverage computation, and the
        AOI rasterize pass. Coverage is computed *first* and only basins the new
        grid can serve (``PARTIAL``/``FULL``) are rasterized -- an off-grid basin
        has no window to burn, though its ``NONE`` coverage is still recorded so
        the index reports it as off-grid. Converge-by-default, like
        ingest: an existing skeleton is tolerated, and rasterization rebuilds an
        AOI raster only when it is absent or its provenance tag reads stale (a
        changed basin polygon or a format-version bump). A byte-level forced
        rebuild is :meth:`rasterize_aois` with ``rebuild=True`` (the
        ``pourpoint rasterize --rebuild`` command).
        """
        dataset = self._build_staged_dataset(name, dataset_config_path)

        try:
            Dataset.create(dataset.spec, dataset.path)
            created = True
        except FileExistsError:
            # Already staged (skeleton exists); the rasterize below is
            # idempotent, so continue rather than clobber existing artifacts.
            created = False

        # Only basin-bearing pourpoints are rasterized/covered (point-only ones
        # have no basin), matching what the index holds.
        record_paths = self.db.pourpoint_paths()
        basin_pourpoints: list[Pourpoint] = []
        with progress.track(
            f'parsing {len(record_paths)} pourpoint record(s)',
            total=len(record_paths),
        ) as task:
            for path in record_paths:
                pourpoint = Pourpoint.from_geojson(path)
                if pourpoint.polygon is not None:
                    basin_pourpoints.append(pourpoint)
                task.advance()
        domain = dataset.coverage_domain
        coverage: dict[types.StationTriplet, Coverage] = {}
        with progress.track(
            f'computing coverage for {len(basin_pourpoints)} pourpoint(s)',
            total=len(basin_pourpoints),
        ) as task:
            for pourpoint in basin_pourpoints:
                coverage[pourpoint.station_triplet] = dataset_coverage(
                    pourpoint,
                    domain,
                )
                task.advance()
        # Coverage gates the burn: an off-grid basin (NONE) has no tile window
        # on this grid, so only basins the grid at least partially serves are
        # rasterized.
        covered = [
            pourpoint
            for pourpoint in basin_pourpoints
            if coverage[pourpoint.station_triplet] is not Coverage.NONE
        ]
        rasterized = self.rasterize_aois(
            covered,
            [dataset],
            progress=progress,
        )
        return StagedDataset(
            dataset=dataset,
            created=created,
            rasterized=rasterized,
            coverage=coverage,
        )

    def rasterize_aoi(
        self: Self,
        aoi: Pourpoint,
        force: bool = False,
    ) -> dict[str, AOIRaster]:
        """Rasterize a pourpoint's basin onto every registered dataset's grid.

        Pourpoints are shared across datasets, but each dataset has its own grid, so an
        AOI must be burned once per dataset (different grids -> different tile
        windows and masks). Covers inactive datasets too, so activating one later
        is instant -- its AOI rasters already exist. Returns the resulting AOI
        raster keyed by dataset name.
        """
        return {
            name: dataset.rasterize_aoi(aoi, force=force)
            for name, dataset in self.db.registered.items()
        }

    def generate_zone_layers(
        self: Self,
        provider_name: str,
        datasets: Iterable[Dataset],
        *,
        source: ZoneLayerSource | None = None,
        force: bool = False,
        options: GenerationOptions | None = None,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> dict[str, str]:
        """Generate a provider's zone layers for several datasets in one pass.

        Reads ``source`` (default: this database's resolved source for
        ``provider_name``) once over the combined extent of ``datasets``' grids and
        bins it into all of them -- e.g. terrain's aspect must be computed at the
        source resolution, so sharing the read is the whole point. ``datasets`` are
        passed as objects (registered or merely staged), so *activation is
        irrelevant here*: zone layers live under ``data/<name>/`` regardless of
        whether the root config links the dataset. Only the datasets that *enable*
        ``provider_name`` are targeted (the rest have no such zone layer).
        ``options`` carries engine knobs (e.g. terrain's ``workers``/
        ``block_size``). Returns each generated dataset's provenance hash, keyed by
        name.
        """
        from snowtool.snowdb.grid import grid_extent_4326

        provider = self.db.zone_layer_providers[provider_name]
        # Only datasets whose config enables this provider have the layer to build.
        selected = [ds for ds in datasets if provider_name in ds.providers]
        if not selected:
            return {}

        if source is None:
            source = self.db.zone_layer_sources[provider_name]
        targets = [ds.zone_target(provider) for ds in selected]
        bounds = _combined_extent(grid_extent_4326(ds.grid) for ds in selected)

        return provider.generate(
            source,
            targets,
            bounds,
            force=force,
            options=options,
            progress=progress,
        )

    def generate_zone_layers_for(
        self: Self,
        datasets: Iterable[Dataset],
        provider_names: Iterable[str] | None = None,
        *,
        source_overrides: Mapping[str, Path] | None = None,
        force: bool = False,
        options: GenerationOptions | None = None,
        progress_factory: Callable[[str], ProgressReporter] | None = None,
    ) -> dict[str, dict[str, str]]:
        """Generate zone layers across ``datasets`` with one shared read per provider.

        The many-datasets orchestrator over :meth:`generate_zone_layers`: for each
        selected provider it resolves the source once (an override from
        ``source_overrides``, else the configured default) and reads it a single time
        over the combined extent of every dataset that enables that provider -- so
        standing up N datasets that share a provider pays that provider's expensive
        source read *once*, not N times. ``provider_names`` limits the providers
        (default: the union of every dataset's enabled providers); an unknown name
        raises :class:`ValueError`. ``progress_factory`` builds a per-provider
        reporter (default: silent). Returns ``{provider_name: {dataset_name: hash}}``,
        with provider keys that targeted no dataset omitted.
        """
        datasets = list(datasets)
        source_overrides = source_overrides or {}
        selected = (
            tuple(provider_names)
            if provider_names is not None
            else tuple(dict.fromkeys(p for ds in datasets for p in ds.providers))
        )
        for provider_name in selected:
            if provider_name not in self.db.zone_layer_providers:
                raise ValueError(f'No such zone-layer provider: {provider_name}')

        results: dict[str, dict[str, str]] = {}
        for provider_name in selected:
            provider = self.db.zone_layer_providers[provider_name]
            source = (
                provider.local_source(source_overrides[provider_name])
                if provider_name in source_overrides
                else self.db.zone_layer_sources[provider_name]
            )
            progress = (
                progress_factory(provider_name)
                if progress_factory is not None
                else NULL_PROGRESS
            )
            hashes = self.generate_zone_layers(
                provider_name,
                datasets,
                source=source,
                force=force,
                options=options,
                progress=progress,
            )
            if hashes:
                results[provider_name] = hashes
        return results

    # --- pourpoint import / sync / lifecycle ----------------------------------------

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
        index = PourpointIndex.from_records(
            self.db.pourpoint_records_path,
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
        paths = self.db.pourpoint_paths()
        entries: list[PourpointIndexEntry] = []
        with progress.track(
            f'indexing {len(paths)} pourpoint(s)',
            total=len(paths),
        ) as task:
            for path in paths:
                triplet = triplet_naming.stem_to_triplet(path.stem)
                if triplet in imported:
                    entries.append(
                        PourpointIndexEntry.from_pourpoint(imported[triplet], domains),
                    )
                elif triplet in previous and set(previous[triplet].coverage) == set(
                    domains,
                ):
                    entries.append(previous[triplet])
                else:
                    pourpoint = Pourpoint.from_geojson(path)
                    if pourpoint.polygon is not None:
                        entries.append(
                            PourpointIndexEntry.from_pourpoint(pourpoint, domains),
                        )
                task.advance()
        index = PourpointIndex.from_entries(entries)
        index.save(self.db.pourpoint_index_path)
        return index

    def _stored_triplets(self: Self) -> set[types.StationTriplet]:
        """Stored triplets read straight from record filenames (no geojson parse).

        Record files are written named for the pourpoint's own triplet, so the filename
        is authoritative -- cheaper than parsing every record for set diffs.
        """
        return {
            triplet_naming.stem_to_triplet(path.stem)
            for path in self.db.pourpoint_paths()
        }

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
        """Split source paths into importable AOIs, skipped point-only, invalid.

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
                    aoi = Pourpoint.from_geojson(path)
                except GeoJSONValidationError as e:
                    invalid.append((path, str(e)))
                else:
                    if aoi.polygon is None:
                        # A valid pourpoint, but with no basin it is skipped.
                        skipped.append(aoi.station_triplet)
                    else:
                        to_import.append((path, aoi))
                task.advance()
        return to_import, skipped, invalid

    def _write_records(self: Self, to_import: Iterable[tuple[Path, Pourpoint]]) -> None:
        """Copy each source geojson verbatim to its canonical record path."""
        self.db.pourpoint_records_path.mkdir(parents=True, exist_ok=True)
        for path, aoi in to_import:
            atomic_copy(path, self.db.pourpoint_record_path(aoi.station_triplet))

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
        imported = [aoi.station_triplet for _, aoi in to_import]
        if not dry_run:
            self._write_records(to_import)
            self._update_index(
                {aoi.station_triplet: aoi for _, aoi in to_import},
                progress=progress,
            )
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
            sorted(src.glob('*.geojson')),
            progress=progress,
        )
        imported = [aoi.station_triplet for _, aoi in to_import]
        # Both AOIs and point-only pourpoints in the source represent triplets the
        # source "has"; only stored triplets absent from that set are pruned.
        source_triplets = set(imported) | set(skipped)
        to_prune = sorted(
            t for t in self._stored_triplets() if t not in source_triplets
        )

        if to_prune and not dry_run and prune_to is None:
            raise PourpointPruneDestinationRequiredError(to_prune)

        if not dry_run:
            self._write_records(to_import)
            for triplet in to_prune:
                self._remove_pourpoint_files(triplet, dump_to=prune_to)
            self._update_index(
                {aoi.station_triplet: aoi for _, aoi in to_import},
                progress=progress,
            )

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
            for aoi in pourpoints:
                for dataset, domain in zip(datasets, domains, strict=True):
                    pair = (aoi.station_triplet, dataset.spec.name)
                    if dataset_coverage(aoi, domain) is Coverage.NONE:
                        # Entirely off this grid: nothing to burn.
                        skipped.append(pair)
                    elif dataset.rasterize_aoi_if_needed(aoi, rebuild=rebuild):
                        built.append(pair)
                    else:
                        skipped.append(pair)
                    task.advance()
        return AOIRasterizeResult(built, skipped)

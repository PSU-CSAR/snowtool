"""The snowdb admin/management layer: every write lives here, not on ``SnowDb``.

:class:`SnowDbManager` *has* a :class:`~snowtool.snowdb.db.SnowDb` (its lean
read/query surface, reachable as :attr:`SnowDbManager.db`) and owns every
operation that mutates the database -- creating the layout, registering datasets,
importing/syncing/removing pourpoints, rasterizing them, and generating zone layers.
The read path (the FastAPI app) builds only a :class:`SnowDb`; the CLI's write
commands and library admin code build a manager. "The management layer has a
snowdb, not the other way around."

The pourpoint import/sync/lifecycle operations live in
:mod:`snowtool.snowdb.manager_pourpoints` as :class:`PourpointOpsMixin` -- a pure
file-size decomposition, not a public seam. ``SnowDbManager`` inherits that mixin,
so the write surface is still a single type; its result dataclasses
(:class:`PourpointImportResult`, :class:`PourpointSyncResult`,
:class:`AOIRasterizeResult`) are re-exported here so existing imports keep working.
"""

from __future__ import annotations

import os
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import ValidationError

from snowtool import types
from snowtool.exceptions import (
    InvalidDatasetNameError,
    SnowDbConfigError,
    UnknownDatasetError,
    UnknownZoneLayerProviderError,
)
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    DATASET_CONFIG_FILENAME,
    DatasetConfig,
    PathDatasetLink,
    RootConfig,
)
from snowtool.snowdb.coverage import Coverage, dataset_coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.manager_pourpoints import (
    AOIRasterizeResult,
    PourpointImportResult,
    PourpointOpsMixin,
    PourpointSyncResult,
)
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.pourpoint_index import PourpointIndex
from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.spec import DatasetSpec
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from snowtool.snowdb.aoi_raster import AOIRaster
    from snowtool.snowdb.grid import Bounds
    from snowtool.snowdb.progress import ProgressReporter
    from snowtool.snowdb.zones.zone_layer import (
        GenerationOptions,
        ZoneLayerProvider,
        ZoneLayerSource,
    )

# Re-exported for backward compatibility: these result dataclasses moved to
# manager_pourpoints alongside the operations that produce them, but importers
# still reach them via ``snowtool.snowdb.manager``.
__all__ = [
    'AOIRasterizeResult',
    'PourpointImportResult',
    'PourpointSyncResult',
    'SnowDbManager',
]


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


def _is_path_token(token: str) -> bool:
    """Whether ``token`` reads as a config *path* rather than a dataset *name*.

    The single partition rule shared by :meth:`SnowDbManager.resolve_dataset` (which
    routes a path token to the filesystem and a name token to the catalog) and
    :meth:`SnowDbManager.register_dataset` (which rejects a name that would read as a
    path). A token containing a path separator (``'/'``, ``'\\'``, or the platform's
    ``os.sep``, a superset covering both spellings on either OS) or ending in
    ``.json`` is a path; anything else is a name.
    """
    return '/' in token or '\\' in token or os.sep in token or token.endswith('.json')


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


@dataclass(frozen=True)
class CreatedDataset:
    """The product of :meth:`SnowDbManager.create_dataset`: the full stamp-a-new-
    dataset lifecycle in one result.

    ``staged`` is the :class:`StagedDataset` from the staging pass (its
    ``dataset.path`` is the on-disk directory, ``created`` whether staging built
    the skeleton). ``registered`` is whether *this* call added the root-config
    registration -- ``False`` when the name was already registered and its link
    was deliberately left untouched (see :meth:`create_dataset`), so a caller can
    render the "registered ... (inactive)" follow-up guidance only when it
    actually happened.
    """

    staged: StagedDataset
    registered: bool


class SnowDbManager(PourpointOpsMixin):
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
        database. Without ``coverage`` (an out-of-band ``dataset register`` that skipped
        staging) only the config is written; the missing coverage key reads as
        ``Coverage.NONE`` until the next ``pourpoint reindex``. Going live still
        needs a service restart -- the ``SnowDb`` is built once at startup.

        ``name`` must be usable as a bare :meth:`resolve_dataset` token and a
        directory name, so a name containing a path separator or ending in
        ``.json`` (which that method's syntactic partition would read as a
        path) is rejected up front -- registration is the single choke point.

        For a ``path`` link that already exists on disk, the config it points at
        is parsed and resolved (:meth:`~snowtool.snowdb.spec.DatasetSpec.from_config`)
        before anything is written, so a caller cannot commit a link to a config
        that exists but fails to parse or resolve; a malformed or unresolvable
        config raises :class:`~snowtool.exceptions.SnowDbConfigError` (mirroring
        how :meth:`SnowDb.open` wraps an unreadable linked config). A link to a
        *missing* path is still committed as-is and only surfaces as the existing
        "dangling link" error when a reader opens the database.
        """
        if _is_path_token(name):
            raise InvalidDatasetNameError(
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
        # A missing path is deliberately not validated here: it defers to the
        # existing dangling-link error at SnowDb.open() time, preserving the
        # documented contract that register commits the link, not the target.
        if dataset_config_path.is_file():
            try:
                dataset_config = DatasetConfig.load(dataset_config_path)
                # Validate it resolves (ingester, ...), not just parses.
                DatasetSpec.from_config(dataset_config, name)
            except (ValidationError, ValueError, UnicodeDecodeError) as e:
                raise SnowDbConfigError(
                    self.db.root,
                    f'Not a usable dataset config ({dataset_config_path}): {e}',
                ) from e
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
        still needs a restart to see the change. Raises
        :class:`~snowtool.exceptions.UnknownDatasetError` for a name the root
        config does not register. Idempotent -- setting the current state
        re-saves harmlessly.
        """
        config = self._read_root_config()
        config_path = self.db.config_path
        if config_path is None:  # pragma: no cover - _read_root_config guarantees it
            raise SnowDbConfigError(self.db.root)
        if name not in config.datasets:
            registered = ', '.join(sorted(config.datasets)) or '(none)'
            raise UnknownDatasetError(
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

        Binding goes through :meth:`SnowDb.bind_dataset` with the config's own
        location as the resolution base (the same call a path link gets at
        ``SnowDb.open``), so a not-yet-registered dataset resolves exactly as it
        will once registered -- without appearing in ``self.db.datasets`` yet.
        """
        from snowtool.snowdb.spec import DatasetSpec

        resolved = Path(dataset_config_path).resolve()
        config = DatasetConfig.load(resolved)
        spec = DatasetSpec.from_config(config, name)
        return self.db.bind_dataset(name, spec, config, base=resolved.parent)

    def resolve_dataset(self: Self, token: str) -> Dataset:
        """Resolve a dataset NAME or a config path to a :class:`Dataset`.

        The token is partitioned *syntactically*, so a name and a file can
        never shadow each other: a token containing a path separator or ending
        in ``.json`` is a PATH; anything else is a NAME. A path token never
        consults the catalog -- it must be an existing dataset config file
        (its NAME taken from the parent directory), else
        :class:`~snowtool.exceptions.UnknownDatasetError`. A name token never
        touches the filesystem -- it resolves only against the root config's
        registered datasets (active or not: management ops -- ingest, zone
        generation, diagnostics -- never care about reader visibility); an
        unregistered name raises the same error. To target an unregistered
        (staged) config, pass its path.
        """
        if _is_path_token(token):
            path = Path(token)
            if not path.is_file():
                raise UnknownDatasetError(f'No dataset config file at {path}.')
            return self._build_staged_dataset(path.parent.name, path)
        if token in self.db.registered:
            return self.db.registered[token]
        registered = ', '.join(sorted(self.db.registered)) or '(none)'
        raise UnknownDatasetError(
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

    def create_dataset(
        self: Self,
        name: str,
        config: DatasetConfig,
        *,
        nodata_mask_source: Path | None = None,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> CreatedDataset:
        """Stamp a brand-new dataset ``name`` from ``config``: stage it, then
        register it inactive -- the whole lifecycle the ``dataset create`` command
        used to orchestrate step-by-step in the CLI.

        Resolves the dataset's data directory the way a later ``SnowDb.open`` will
        (:meth:`~snowtool.snowdb.db.SnowDb.dataset_dir`), writes ``config`` beside
        its data as ``data/<name>/dataset.json`` so :meth:`stage_dataset` can build
        from it and :meth:`register_dataset` can link it, stages every artifact
        (skeleton, AOI rasters, coverage -- but never zone layers; those are the
        separate :meth:`generate_zone_layers_for` pass), and registers the staged
        dataset. Converge-by-default like ingest and staging: the directory mkdir
        and the config write are idempotent overwrites, and staging rebuilds an AOI
        raster only when its provenance tag reads stale. When ``nodata_mask_source``
        is given (e.g. a template's packaged mask), it is copied into the dataset
        directory and ``config`` is updated to reference it before either is used,
        so the first staging pass burns AOI rasters with the mask already applied.

        The one real invariant it enforces: an existing registration is *never*
        clobbered. Registration happens only when ``name`` is not already in the
        root config -- so a re-create of a live dataset never deactivates it or
        relinks its config out from under readers (its ``active`` state and link
        survive verbatim). A fresh registration is committed *inactive*
        (``active=False``) with the staged coverage folded into the index, so the
        dataset exists (manageable by name) but stays invisible to readers until an
        explicit :meth:`set_dataset_active`. Returns a :class:`CreatedDataset`
        carrying the staging result and whether this call registered the dataset.
        """
        directory = self.db.dataset_dir(name, config)
        directory.mkdir(parents=True, exist_ok=True)

        if nodata_mask_source is not None:
            # Materialize the mask beside the config and point the config at it
            # (relative, so the dataset dir stays relocatable). Copied before
            # staging so the very first AOI rasterize pass burns with the mask.
            shutil.copyfile(nodata_mask_source, directory / 'nodata-mask.tif')
            config = config.model_copy(update={'nodata_mask': Path('nodata-mask.tif')})

        config_path = directory / DATASET_CONFIG_FILENAME
        config.save(config_path)

        staged = self.stage_dataset(name, config_path, progress=progress)

        registered = name not in self.db.registered
        if registered:
            self.register_dataset(
                name,
                config_path,
                coverage=staged.coverage,
                active=False,
            )
        return CreatedDataset(staged=staged, registered=registered)

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
        (default: the union of every dataset's enabled providers); an unknown
        name -- selected or overridden -- raises
        :class:`~snowtool.exceptions.UnknownZoneLayerProviderError`.
        ``progress_factory`` builds a per-provider reporter (default: silent).
        Returns ``{provider_name: {dataset_name: hash}}``, with provider keys
        that targeted no dataset omitted.
        """
        datasets = list(datasets)
        source_overrides = source_overrides or {}
        selected = (
            tuple(provider_names)
            if provider_names is not None
            else tuple(dict.fromkeys(p for ds in datasets for p in ds.providers))
        )
        # Override keys are validated alongside the selection so a typo'd
        # ``--source PROVIDER PATH`` fails loudly instead of silently not applying.
        for provider_name in (*selected, *source_overrides):
            if provider_name not in self.db.zone_layer_providers:
                raise UnknownZoneLayerProviderError(
                    f'No such zone-layer provider: {provider_name}',
                )

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

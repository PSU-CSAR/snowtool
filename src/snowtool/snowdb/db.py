"""The snow database read/query surface: the global ``pourpoints/`` plus per-dataset
``data/``.

``SnowDb`` is the lean, read-only *catalog* of a snowdb: built from a root
:class:`~snowtool.snowdb.config.RootConfig`, it binds every registered dataset to its
directory whether or not that directory exists (a dataset is defined by its config; a
missing directory just means no data yet), so the read path tolerates an
un-initialized root (serving nothing and logging a warning). Every operation that
*mutates* the database -- creating the layout, registering datasets, importing and
rasterizing pourpoints, generating zone layers -- lives on
:class:`~snowtool.snowdb.manager.SnowDbManager`, which *has* a ``SnowDb``. It is
constructed per entrypoint (the API at app-lifespan scope, the CLI per invocation).

``SnowDb`` itself holds only constants (config, paths, specs, datasets, coverage) and
cache-free disk reads. The one piece of non-constant read-path state -- the
:class:`~snowtool.snowdb.raster.tiff_cache.TiffCache` shared across a database's COG
reads -- lives on its sibling :class:`~snowtool.snowdb.reader.SnowDbReader`, which
*has* a ``SnowDb`` and owns ``zonal_stats`` (the sole cache consumer).
"""

from __future__ import annotations

import shutil

from pathlib import Path
from typing import TYPE_CHECKING, Self

from snowtool import types
from snowtool.exceptions import (
    IndexedPourpointMissingBasinError,
    PourpointNotFoundError,
    SnowDbConfigError,
    UnknownDatasetError,
)
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    DatasetConfig,
    InlineDatasetLink,
    RootConfig,
    resolve_path,
)
from snowtool.snowdb.coverage import (
    Coverage,
    require_full_coverage,
)
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.pourpoint_index import PourpointIndex
from snowtool.snowdb.spec import DatasetSpec, load_dataset_spec, resolve_spec
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from geojson_pydantic import MultiPolygon, Polygon

    from snowtool.snowdb.pourpoint_index import PourpointIndexEntry
    from snowtool.snowdb.zones.zone_layer import (
        ZoneLayerProvider,
        ZoneLayerSource,
    )


class SnowDb:
    def __init__(
        self: Self,
        config: RootConfig,
        *,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
    ) -> None:
        """Build a snowdb from a root ``config``.

        The single constructor: it takes a :class:`~snowtool.snowdb.config.RootConfig`
        -- loaded from a file (:meth:`open`) or built in code -- and resolves
        everything the config defines (the root for relative links, the pourpoint
        index/records locations, and each registered dataset). A dataset is either
        embedded *inline* (its config carried in the link) or *referenced* by a path
        link to a ``dataset.json``; either way it is deserialized into a
        :class:`DatasetSpec` and bound to its data directory (see
        :meth:`~snowtool.snowdb.config.DatasetConfig.resolve_data_dir`).
        The code follows the config rather than assuming paths.
        """
        self.config = config
        self.config_path = config.path
        # The root is the config file's directory: the base relative links resolve
        # against. A config built in code (no path) has no root -- fine as long as
        # every link it uses is absolute; a relative one raises when resolved.
        self.root = config.path.parent if config.path is not None else None
        self.data_path = self.root / 'data' if self.root is not None else None

        self.pourpoint_records_path = resolve_path(
            config.pourpoint_records,
            root=self.root,
        )
        self.pourpoint_index_path = resolve_path(config.pourpoint_index, root=self.root)

        # The zone-layer providers (terrain, land cover, ...) every dataset is
        # built/read with. Injected (not a global) so tests/entrypoints can supply
        # their own set; adding a kind is one entry in the default registry.
        # `bind_dataset` reads these, so they must be set before the bind loop.
        self.zone_layer_providers = {p.name: p for p in zone_layer_providers}

        # `registered` is everything the root config knows (the management
        # surface: ingest, zone generation, diagnostics); `datasets` is the
        # active subset -- the read surface (query/API/available_zones) sees
        # only those, so a link's `active` flag is the visibility toggle.
        self.registered: dict[str, Dataset] = {}
        for name, link in config.datasets.items():
            if isinstance(link, InlineDatasetLink):
                dataset_config = link.dataset
                base = None
                spec = resolve_spec(
                    dataset_config,
                    name,
                    location=self.root,
                    detail=f'inline dataset {name!r} is not usable',
                )
            else:  # PathDatasetLink
                resolved = resolve_path(link.path, root=self.root)
                if not resolved.is_file():
                    raise SnowDbConfigError(
                        self.root,
                        f'dataset {name!r} link points at a missing config: {resolved}',
                    )
                dataset_config, spec = load_dataset_spec(resolved, name)
                base = resolved.parent
            self.registered[name] = self.bind_dataset(
                name,
                spec,
                dataset_config,
                base=base,
            )
        self.datasets = {
            name: ds
            for name, ds in self.registered.items()
            if config.datasets[name].active
        }
        # The source each provider reads from during generation -- declared in the
        # config (provider name -> path, resolved like any other config path). A
        # source belongs to the whole database (one source bins into every grid in
        # a single pass), not to any one dataset. A provider with no configured
        # source falls back to its default (3DEP / the MRLC bundle), which needs a
        # root. Reads never touch sources -- only generation does.
        self.zone_layer_sources: dict[str, ZoneLayerSource] = {}
        for name, provider in self.zone_layer_providers.items():
            configured = config.sources.get(name)
            if configured is not None:
                self.zone_layer_sources[name] = provider.local_source(
                    resolve_path(configured, root=self.root),
                )
            elif self.root is not None:
                self.zone_layer_sources[name] = provider.default_source(self.root)

        # The pourpoint index is read on every listing/coverage call (every API
        # request), so it is cached on the instance rather than re-parsed each
        # time. The cache is *mtime-revalidated* (see `pourpoint_index`):
        # re-read only when `index.geojson`'s mtime changes, so a long-running
        # API server picks up an out-of-band `pourpoint import`/`reindex`
        # without a restart at the cost of one stat.
        self._index: PourpointIndex | None = None
        self._index_mtime: int | None = None

    def bind_dataset(
        self: Self,
        name: str,
        spec: DatasetSpec,
        dataset_config: DatasetConfig,
        *,
        base: Path | None = None,
    ) -> Dataset:
        """Resolve ``dataset_config``'s paths and construct its :class:`Dataset`.

        The single place a dataset config becomes a bound Dataset: the data
        directory and the optional nodata mask resolve against ``base`` --
        ``None`` for an inline dataset (the root, with the ``data/<name>``
        convention), the config file's own directory for a referenced one. The
        manager's staged-dataset path shares this, so an unregistered config
        binds exactly as a later ``SnowDb.open`` will bind it.
        """
        return Dataset(
            spec,
            dataset_config.resolve_data_dir(name, root=self.root, base=base),
            self.zone_layer_providers.values(),
            nodata_mask=dataset_config.resolve_nodata_mask(root=self.root, base=base),
        )

    @classmethod
    def open(
        cls: type[Self],
        path: Path,
        *,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
    ) -> Self:
        """Open a snowdb from its root config file -- the "from file" constructor.

        ``path`` is the snowdb root directory (holding ``snowdb_conf.json``) or the
        config file itself. The config is *required*: a root without one is not a
        snowdb this version understands, so this raises
        :class:`~snowtool.exceptions.SnowDbConfigError` pointing at ``snowtool
        init``. The I/O half of construction: it reads + parses the root config,
        then hands it to the constructor.
        """
        path = Path(path)
        config_path = path / CONFIG_FILENAME if path.is_dir() else path
        if not config_path.is_file():
            raise SnowDbConfigError(path)
        config = RootConfig.load(config_path)
        return cls(
            config,
            zone_layer_providers=zone_layer_providers,
        )

    # --- global pourpoint query helpers (drive the pourpoint commands) ---

    def pourpoint_paths(self: Self) -> list[Path]:
        """The per-pourpoint record geojson under ``pourpoints/records/`` (sorted)."""
        if not self.pourpoint_records_path.is_dir():
            return []
        return sorted(self.pourpoint_records_path.glob('*.geojson'))

    def pourpoints(self: Self) -> Iterator[Pourpoint]:
        """Parse and yield every stored pourpoint record."""
        for path in self.pourpoint_paths():
            yield Pourpoint.from_geojson(path)

    def pourpoint_triplets(self: Self) -> set[types.StationTriplet]:
        """The station triplets of every stored pourpoint, read from filenames.

        Record files are written named for the pourpoint's own triplet, so the
        filename is authoritative -- cheaper than parsing every record just for
        the triplet set (e.g. for set diffs or a coverage report).
        """
        return {triplet_naming.stem_to_triplet(p.stem) for p in self.pourpoint_paths()}

    def pourpoint_record_path(self: Self, triplet: types.StationTriplet) -> Path:
        """The canonical ``records/<triplet>.geojson`` path (``:`` -> ``_``)."""
        return (
            self.pourpoint_records_path
            / f'{triplet_naming.triplet_to_stem(triplet)}.geojson'
        )

    def load_pourpoint(
        self: Self,
        triplet: types.StationTriplet,
        *,
        index: PourpointIndex | None = None,
    ) -> Pourpoint:
        """Parse the stored record for an *indexed* pourpoint ``triplet``.

        The single owner of the ``indexed => basin-bearing`` invariant. The index
        is the availability gate: only basin-bearing pourpoints are indexed
        (``PourpointIndex.build`` refuses point-only records), so a triplet absent
        from the index -- anything dropped into ``records/`` out of band without a
        ``pourpoint reindex`` -- is not served and raises
        :class:`PourpointNotFoundError`. A triplet that *is* indexed but whose
        stored record has been swapped out-of-band to a point-only Feature (no
        polygon) is a data-integrity bug, surfaced as the typed
        :class:`IndexedPourpointMissingBasinError` (an unmapped -- server 500 --
        condition) rather than returning a basin-less :class:`Pourpoint` that
        every caller would have to re-check. Callers already holding the index
        (e.g. a listing loop) pass it in to avoid re-reading it.
        """
        if index is None:
            index = self.pourpoint_index()
        if triplet not in index:
            raise PourpointNotFoundError.for_triplet(triplet)
        path = self.pourpoint_record_path(triplet)
        if not path.is_file():
            raise PourpointNotFoundError.for_triplet(triplet)
        pourpoint = Pourpoint.from_geojson(path)
        if pourpoint.polygon is None:
            raise IndexedPourpointMissingBasinError(
                f'Indexed pourpoint {triplet!r} has no basin polygon.',
            )
        return pourpoint

    def pourpoint_index(self: Self) -> PourpointIndex:
        """The persisted ``index.geojson`` manifest (empty if absent), mtime-cached.

        Serves ``pourpoint list`` without parsing the (large) basin records; see
        :mod:`~snowtool.snowdb.pourpoint_index` for the incremental-vs-reindex
        maintenance contract. Cached and revalidated against the file's mtime: it
        ``stat``s the index file and reloads only when the mtime differs from
        the cached one (a missing file is cached as an empty index, mtime
        ``None``), so a single ``SnowDb`` -- e.g. an app-lifespan API instance --
        stays correct after an out-of-band reindex at the cost of one stat per
        access.
        """
        try:
            mtime: int | None = self.pourpoint_index_path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if self._index is None or mtime != self._index_mtime:
            self._index = PourpointIndex.load(self.pourpoint_index_path)
            self._index_mtime = mtime
        return self._index

    def pourpoint_page(
        self: Self,
        *,
        offset: int,
        limit: int,
        with_basins: bool = False,
        contains: Callable[[float, float], bool] | None = None,
    ) -> tuple[list[tuple[PourpointIndexEntry, Polygon | MultiPolygon | None]], int]:
        """One page of the (triplet-sorted) index: entries paired with geometry.

        The catalog read behind ``GET /pourpoints``, kept in the domain so the
        response model shrinks to pure feature/link shaping. ``contains``, when
        given, filters entries on their point ``(lon, lat)`` -- the caller
        supplies the predicate (e.g. an OGC ``bbox`` containment test), keeping
        this method free of any HTTP query type. The *filtered* count is the
        ``total`` (second return value), computed before the ``offset``/``limit``
        slice so pagination reports the full match count. Each page entry is
        paired with its basin polygon when ``with_basins`` -- a per-record
        :meth:`load_pourpoint` (the expensive view; the index stores points only),
        which enforces the ``indexed => basin-bearing`` invariant -- or ``None``
        otherwise (the caller uses the entry's point).
        """
        index = self.pourpoint_index()
        entries = [
            entry
            for entry in index  # PourpointIndex iterates entries sorted by triplet
            if contains is None or contains(*entry.point.coordinates[:2])
        ]
        total = len(entries)
        page: list[tuple[PourpointIndexEntry, Polygon | MultiPolygon | None]] = []
        for entry in entries[offset : offset + limit]:
            geometry = (
                self.load_pourpoint(entry.triplet, index=index).polygon
                if with_basins
                else None
            )
            page.append((entry, geometry))
        return page, total

    def pourpoint_dataset_coverage(
        self: Self,
        triplet: types.StationTriplet,
        dataset_name: str,
    ) -> Coverage:
        """How fully ``dataset_name``'s grid covers pourpoint ``triplet``'s basin.

        Read straight from the index's cached per-dataset coverage (computed at
        reindex/registration against each dataset's grid). A grid change is by
        definition a new dataset, so the cached value never goes stale -- and
        reading it avoids re-parsing the (large) basin record on every query.

        A dataset registered *after* the index entry was written (a legacy
        out-of-order registration, or before a ``pourpoint reindex``) has no key
        in the entry's ``coverage`` dict; that reads as
        :attr:`~snowtool.snowdb.coverage.Coverage.NONE` (no coverage) rather than
        an error, so a not-yet-recomputed dataset degrades to "off grid" instead
        of a 500. Raises :class:`~snowtool.exceptions.UnknownDatasetError` if the
        dataset is unknown, or :class:`~snowtool.exceptions.PourpointNotFoundError`
        if the pourpoint is unindexed.
        """
        # Route the dataset check through __getitem__ so an inactive-but-registered
        # name gets its pointed "activate it" hint rather than a generic miss.
        self[dataset_name]
        index = self.pourpoint_index()
        if triplet not in index:
            raise PourpointNotFoundError.for_triplet(triplet)
        return index[triplet].coverage.get(dataset_name, Coverage.NONE)

    def require_pourpoint_coverage(
        self: Self,
        triplet: types.StationTriplet,
        dataset_name: str,
        *,
        allow_partial: bool = False,
    ) -> Coverage:
        """Query guard: raise unless ``dataset_name`` fully covers ``triplet``.

        The seam a stats/query call uses before reading rasters, closing the
        silent-partial-stats gap. ``allow_partial`` permits a knowingly-clipped
        query over a partially-covered pourpoint; a wholly off-grid one always
        raises. Returns the computed :class:`Coverage` for callers that want to
        log it.
        """
        coverage = self.pourpoint_dataset_coverage(triplet, dataset_name)
        require_full_coverage(
            coverage,
            triplet=triplet,
            dataset=dataset_name,
            allow_partial=allow_partial,
        )
        return coverage

    def dump_pourpoint(
        self: Self,
        triplet: types.StationTriplet,
        dest_dir: Path,
    ) -> Path:
        """Copy a stored pourpoint record out to ``dest_dir`` (round-trip / archive).

        A pure read/export -- it copies a record out without touching the database,
        so it lives on the read side even though the prune cascade
        (:class:`~snowtool.snowdb.manager.SnowDbManager`) also uses it.
        """
        source = self.pourpoint_record_path(triplet)
        if not source.is_file():
            raise PourpointNotFoundError.for_triplet(triplet)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copyfile(source, dest)
        return dest

    def __getitem__(self: Self, name: str) -> Dataset:
        """Look up active dataset ``name``.

        Raises :class:`~snowtool.exceptions.UnknownDatasetError` (not
        ``KeyError``) for a name that is unregistered *or* registered but
        inactive -- this surface serves only active datasets. A
        registered-but-inactive name gets a pointed "activate it" hint instead
        of a generic miss, since the fix differs.
        """
        try:
            return self.datasets[name]
        except KeyError:
            if name in self.registered:
                raise UnknownDatasetError(
                    f'Dataset {name!r} is registered but inactive. '
                    f"Activate it with 'snowtool dataset activate {name}'.",
                ) from None
            active = ', '.join(sorted(self.datasets)) or '(none)'
            raise UnknownDatasetError(
                f'No such dataset {name!r}. Active datasets: {active}.',
            ) from None

    def registered_dataset(self: Self, name: str, *, hint: str = '') -> Dataset:
        """Look up a *registered* dataset ``name`` (active or not).

        The management/diagnostics counterpart to :meth:`__getitem__`: it
        resolves anything registered (activation is irrelevant to ingest, zone
        generation, and the report surfaces), where ``__getitem__`` serves only
        the active subset. A miss raises
        :class:`~snowtool.exceptions.UnknownDatasetError` listing the registered
        names, mirroring ``__getitem__``'s wording. ``hint`` is an optional
        trailing clause a caller with more context appends (e.g.
        :meth:`SnowDbManager.resolve_dataset` pointing an unregistered name at
        the path form) -- so both the CLI helper and the manager resolve through
        this one lookup-with-error home instead of each rebuilding the message.
        """
        try:
            return self.registered[name]
        except KeyError:
            registered = ', '.join(sorted(self.registered)) or '(none)'
            raise UnknownDatasetError(
                f'No such dataset {name!r}. Registered datasets: {registered}.{hint}',
            ) from None

    def __iter__(self: Self) -> Iterator[str]:
        return iter(self.datasets)

    def __contains__(self: Self, name: str) -> bool:
        return name in self.datasets

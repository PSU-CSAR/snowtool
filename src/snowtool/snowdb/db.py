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
    PourpointNotFoundError,
    SnowDbConfigError,
    UnknownDatasetError,
)
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    DatasetConfig,
    RootConfig,
)
from snowtool.snowdb.coverage import (
    Coverage,
    require_full_coverage,
)
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.pourpoint_index import PourpointIndex
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.zones.zone_layer import (
        AvailableZone,
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
        :class:`DatasetSpec` and bound to its data directory (see :meth:`dataset_dir`).
        The code follows the config rather than assuming paths.
        """
        from snowtool.snowdb.config import InlineDatasetLink
        from snowtool.snowdb.spec import DatasetSpec

        self.config = config
        self.config_path = config.path
        # The root is the config file's directory: the base relative links resolve
        # against. A config built in code (no path) has no root -- fine as long as
        # every link it uses is absolute; a relative one raises when resolved.
        self.root = config.path.parent if config.path is not None else None
        self.data_path = self.root / 'data' if self.root is not None else None

        self.pourpoint_records_path = self._resolve_path(config.pourpoint_records)
        self.pourpoint_index_path = self._resolve_path(config.pourpoint_index)

        # Resolve every registered dataset (inline or referenced) into a spec,
        # keeping each config with its resolution base so `bind_dataset` -- the
        # one place a config's paths become a bound Dataset -- can resolve them.
        # Inline uses the root (base None: the `data/<name>` convention); a
        # referenced one resolves and defaults beside its own config file.
        specs: list[DatasetSpec] = []
        self._dataset_links: dict[str, tuple[DatasetConfig, Path | None]] = {}
        for name, link in config.datasets.items():
            if isinstance(link, InlineDatasetLink):
                dataset_config = link.dataset
                base = None
            else:  # PathDatasetLink
                resolved = self._resolve_path(link.path)
                if not resolved.is_file():
                    raise SnowDbConfigError(
                        self.root,
                        f'dataset {name!r} link points at a missing config: {resolved}',
                    )
                # DatasetConfig.load raises a clean SnowDbConfigError naming
                # `resolved` if the linked config exists but doesn't parse/
                # validate; nothing further to wrap here.
                dataset_config = DatasetConfig.load(resolved)
                base = resolved.parent
            self._dataset_links[name] = (dataset_config, base)
            specs.append(DatasetSpec.from_config(dataset_config, name))

        self._specs = self._index_specs(specs)
        # The zone-layer providers (terrain, land cover, ...) every dataset is
        # built/read with. Injected (not a global) so tests/entrypoints can supply
        # their own set; adding a kind is one entry in the default registry.
        self.zone_layer_providers = {p.name: p for p in zone_layer_providers}
        # Each configured dataset is always bound to its directory, present or not.
        # A dataset with no directory simply has no data yet, which keeps the read
        # path resilient to an un-initialized root. `registered` is everything the
        # root config knows (the management surface: ingest, zone generation,
        # diagnostics); `datasets` is the active subset -- the read surface
        # (query/API/available_zones) sees only those, so a link's `active` flag
        # is the visibility toggle.
        self.registered = self._bind_datasets()
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
                    self._resolve_path(configured),
                )
            elif self.root is not None:
                self.zone_layer_sources[name] = provider.default_source(self.root)

        # The pourpoint index is read on every listing/coverage call (every API
        # request), so it is cached on the instance rather than re-parsed each
        # time. The cache is *mtime-revalidated* (see `pourpoint_index`): primed
        # here alongside the config reads, then re-read only when `index.geojson`'s
        # mtime changes, so a long-running API server picks up an out-of-band
        # `pourpoint import`/`reindex` without a restart at the cost of one stat.
        self._index: PourpointIndex | None = None
        self._index_mtime: int | None = None
        self._load_index()

    def _resolve_path(self: Self, link: str | Path, base: Path | None = None) -> Path:
        """Resolve a config path: absolute -> as-is; relative -> against ``base``
        (the root by default). A relative path with no root has nothing to resolve
        against, so it raises."""
        p = Path(link)
        if p.is_absolute():
            return p
        anchor = base if base is not None else self.root
        if anchor is None:
            raise SnowDbConfigError(
                self.root,
                f'cannot resolve relative path {str(p)!r}: this config has no '
                'location (built in code, not saved). Make the path absolute, or '
                'save the config first.',
            )
        return anchor / p

    def dataset_dir(
        self: Self,
        name: str,
        dataset_config: DatasetConfig,
        *,
        base: Path | None = None,
        default: Path | None = None,
    ) -> Path:
        """Where ``name``'s data lives -- the single rule reads and writes share.

        The config's ``data_dir`` (absolute -> anywhere; relative -> against
        ``base``), else the convention ``default`` (``data/<name>``). ``base``
        defaults to the root.
        """
        location = dataset_config.data_dir
        if location is None:
            location = default if default is not None else Path('data') / name
        return self._resolve_path(location, base)

    def dataset_nodata_mask(
        self: Self,
        dataset_config: DatasetConfig,
        *,
        base: Path | None = None,
    ) -> Path | None:
        """Where ``dataset_config``'s nodata mask lives, or ``None``.

        Resolved exactly like ``data_dir``: absolute -> anywhere; relative ->
        against ``base`` (the root by default). There is no convention default:
        no config entry means no mask.
        """
        if dataset_config.nodata_mask is None:
            return None
        return self._resolve_path(dataset_config.nodata_mask, base)

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
        directory (:meth:`dataset_dir`) and the optional nodata mask
        (:meth:`dataset_nodata_mask`) resolve against ``base`` -- ``None`` for
        an inline dataset (the root, with the ``data/<name>`` convention), the
        config file's own directory for a referenced one. The manager's
        staged-dataset path shares this, so an unregistered config binds
        exactly as a later ``SnowDb.open`` will bind it.
        """
        return Dataset(
            spec,
            self.dataset_dir(name, dataset_config, base=base, default=base),
            self.zone_layer_providers.values(),
            nodata_mask=self.dataset_nodata_mask(dataset_config, base=base),
        )

    @staticmethod
    def _index_specs(specs: Iterable[DatasetSpec]) -> dict[str, DatasetSpec]:
        indexed: dict[str, DatasetSpec] = {}
        for spec in specs:
            if spec.name in indexed:
                raise ValueError(f'Duplicate dataset spec name: {spec.name!r}')
            indexed[spec.name] = spec
        return indexed

    def _bind_datasets(self: Self) -> dict[str, Dataset]:
        datasets: dict[str, Dataset] = {}
        for name, spec in self._specs.items():
            dataset_config, base = self._dataset_links[name]
            datasets[name] = self.bind_dataset(name, spec, dataset_config, base=base)
        return datasets

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
        init`` (the deliberate no-backwards-compat call -- there is no
        lenient un-initialized read path). The I/O half of construction: it reads +
        parses the root config, then hands it to the constructor.
        """
        path = Path(path)
        config_path = path / CONFIG_FILENAME if path.is_dir() else path
        if not config_path.is_file():
            raise SnowDbConfigError(path)
        # RootConfig.load raises a clean SnowDbConfigError naming `config_path`
        # if it exists but doesn't parse/validate; nothing further to wrap here.
        config = RootConfig.load(config_path)
        return cls(
            config,
            zone_layer_providers=zone_layer_providers,
        )

    def available_zones(self: Self) -> dict[str, AvailableZone]:
        """The query-able zone layers across this database's *enabled* providers.

        Keyed ``'<provider>.<layer.key>'`` (e.g. ``'terrain.elevation'``); the union
        over every dataset's enabled providers, so a zone appears only if some
        dataset serves it. Only layers that declare a zoning scheme appear. The
        terrain aspect-orientation components are each their own banded axis
        (``terrain.northness`` / ``terrain.eastness``), so they *are* listed. The
        representation of a zone's valid values is its scheme's ``zones()``.
        """
        from snowtool.snowdb.zones.zone_layer import available_zones

        zones: dict[str, AvailableZone] = {}
        for dataset in self.datasets.values():
            zones.update(available_zones(dataset.providers.values()))
        return zones

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
        """The station triplets of every stored pourpoint (parsed from the id)."""
        return {pp.station_triplet for pp in self.pourpoints()}

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

        The index is the availability gate: only basin-bearing pourpoints are
        indexed (``PourpointIndex.from_records`` skips point-only ones), so a
        triplet absent from the index -- a point-only record or anything dropped
        into ``records/`` out of band without a ``pourpoint reindex`` -- is not
        served and raises :class:`PourpointNotFoundError`. Callers already holding
        the index (e.g. a listing loop) pass it in to avoid re-reading it.
        """
        if index is None:
            index = self.pourpoint_index()
        if triplet not in index:
            raise PourpointNotFoundError(
                f'No stored pourpoint for triplet {triplet!r}.',
            )
        path = self.pourpoint_record_path(triplet)
        if not path.is_file():
            raise PourpointNotFoundError(
                f'No stored pourpoint for triplet {triplet!r}.',
            )
        return Pourpoint.from_geojson(path)

    def _load_index(self: Self) -> PourpointIndex:
        """Return the cached index, re-reading it only if ``index.geojson`` changed.

        The revalidation primitive behind :meth:`pourpoint_index`: it ``stat``s the
        index file and reloads only when the mtime differs from the cached one (a
        missing file is cached as an empty index, mtime ``None``). One stat per
        access keeps a single ``SnowDb`` -- e.g. an app-lifespan API instance --
        correct after an out-of-band reindex without re-parsing on every request.
        """
        try:
            mtime: int | None = self.pourpoint_index_path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if self._index is None or mtime != self._index_mtime:
            self._index = PourpointIndex.load(self.pourpoint_index_path)
            self._index_mtime = mtime
        return self._index

    def pourpoint_index(self: Self) -> PourpointIndex:
        """The persisted ``index.geojson`` manifest (empty if absent), mtime-cached.

        Serves ``pourpoint list`` without parsing the (large) basin records. The
        index is maintained *incrementally* by import/sync/remove (an entry is
        reused as-is while its record and the registered-dataset set are
        unchanged); ``pourpoint reindex`` is the explicit full rebuild -- required
        after out-of-band ``records/`` edits and after a grid change to an
        already-registered dataset name (the one change incremental maintenance
        cannot see; registering/removing a dataset self-heals). The result is
        cached and revalidated against the file's mtime (see :meth:`_load_index`),
        so repeated reads within one process are cheap yet still reflect an
        out-of-band rewrite.
        """
        return self._load_index()

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
        if dataset_name not in self.datasets:
            active = ', '.join(sorted(self.datasets)) or '(none)'
            raise UnknownDatasetError(
                f'No such dataset {dataset_name!r}. Active datasets: {active}.',
            )
        index = self.pourpoint_index()
        if triplet not in index:
            raise PourpointNotFoundError(
                f'No stored pourpoint for triplet {triplet!r}.',
            )
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
            raise PourpointNotFoundError(
                f'No stored pourpoint for triplet {triplet!r}.',
            )
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copyfile(source, dest)
        return dest

    def __getitem__(self: Self, name: str) -> Dataset:
        """Look up active dataset ``name``.

        Raises :class:`~snowtool.exceptions.UnknownDatasetError` (not
        ``KeyError``) for a name that is unregistered *or* registered but
        inactive -- this surface serves only active datasets.
        """
        try:
            return self.datasets[name]
        except KeyError:
            active = ', '.join(sorted(self.datasets)) or '(none)'
            raise UnknownDatasetError(
                f'No such dataset {name!r}. Active datasets: {active}.',
            ) from None

    def __iter__(self: Self) -> Iterator[str]:
        return iter(self.datasets)

    def __contains__(self: Self, name: str) -> bool:
        return name in self.datasets

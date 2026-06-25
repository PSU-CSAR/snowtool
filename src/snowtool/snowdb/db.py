"""The snow database read/query surface: the global ``aois/`` plus per-dataset
``data/``.

``SnowDb`` is the lean, read-only view of a snowdb: it is built from a root
:class:`~snowtool.snowdb.config.RootConfig` and binds every registered dataset to
its directory, present on disk or not (a dataset is defined by its config, and a
missing directory just means it has no data yet). The read path therefore
tolerates an un-initialized root (it serves no data and logs a warning). Every
operation that *mutates* the database -- creating the layout, registering
datasets, importing/rasterizing AOIs, generating zone layers -- lives on
:class:`~snowtool.snowdb.manager.SnowDbManager`, which *has* a ``SnowDb``; the
FastAPI app builds only this read side. It is constructed per entrypoint (the API
builds one at app-lifespan scope, the CLI one per invocation). It also owns the
:class:`~snowtool.snowdb.tiff_cache.TiffCache` shared by all of its datasets'
reads.
"""

from __future__ import annotations

import shutil

from pathlib import Path
from typing import TYPE_CHECKING, Self

from snowtool import types
from snowtool.exceptions import SnowDbConfigError
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.aoi_index import AOIIndex
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    DatasetConfig,
    RootConfig,
)
from snowtool.snowdb.coverage import (
    Coverage,
    dataset_coverage,
    require_full_coverage,
)
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.tiff_cache import TiffCache
from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.variables import DatasetVariable
    from snowtool.snowdb.zonal_stats import ZonalStats, ZoneSelection
    from snowtool.snowdb.zone_layer import (
        AvailableZone,
        ZoneLayerProvider,
        ZoneLayerSource,
    )


class SnowDb:
    def __init__(
        self: Self,
        config: RootConfig,
        *,
        tiff_cache: TiffCache | None = None,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
    ) -> None:
        """Build a snowdb from a root ``config``.

        The single constructor: it takes a :class:`~snowtool.snowdb.config.RootConfig`
        -- loaded from a file (:meth:`open`) or built in code -- and resolves
        everything the config defines. The config's own
        :attr:`~snowtool.snowdb.config.RootConfig.path` gives the root (its parent),
        the base relative links resolve against. A config built in code (no path)
        has no root, so it works only if every link it uses is absolute -- a
        relative link raises when resolved. Each registered dataset is either
        embedded *inline* (its config carried in the link) or *referenced* by a path
        link to a ``dataset.json``; the linked config is deserialized into a
        :class:`DatasetSpec` and the dataset bound to its directory, which comes from
        the dataset config's ``data_dir`` (absolute, or relative to that config's own
        location) and defaults to the convention (beside a referenced config,
        ``data/<name>/`` for an inline one). The AOI index/records locations come
        from the config too, so the code follows the config rather than assuming
        paths.
        """
        from snowtool.snowdb.config import InlineDatasetLink
        from snowtool.snowdb.spec import DatasetSpec

        self.config = config
        self.config_path = config.path
        # The root is the config file's directory: the base relative links resolve
        # against. A config built in code (no path) has no root -- fine as long as
        # every link it uses is absolute; a relative one raises when resolved.
        self.root = config.path.parent if config.path is not None else None
        # `path` is kept as an alias for `root` (the many read helpers below, and
        # callers, refer to `self.path`).
        self.path = self.root
        self.data_path = self.root / 'data' if self.root is not None else None

        self.aoi_records_path = self._resolve_path(config.aoi_records)
        self.aoi_index_path = self._resolve_path(config.aoi_index)

        # Resolve every registered dataset (inline or referenced) into a spec and
        # record where its data lives -- the `dataset_dir` rule reads and writes
        # share. Inline uses the root + `data/<name>` convention; a referenced one
        # defaults beside its own config file.
        specs: list[DatasetSpec] = []
        self._dataset_paths: dict[str, Path] = {}
        for name, link in config.datasets.items():
            if isinstance(link, InlineDatasetLink):
                dataset_config = link.dataset
                self._dataset_paths[name] = self.dataset_dir(name, dataset_config)
            else:  # PathDatasetLink
                resolved = self._resolve_path(link.path)
                if not resolved.is_file():
                    raise SnowDbConfigError(
                        self.root,
                        f'dataset {name!r} link points at a missing config: {resolved}',
                    )
                dataset_config = DatasetConfig.load(resolved)
                self._dataset_paths[name] = self.dataset_dir(
                    name,
                    dataset_config,
                    base=resolved.parent,
                    default=resolved.parent,
                )
            specs.append(DatasetSpec.from_config(dataset_config, name))

        self._specs = self._index_specs(specs)
        # The zone-layer providers (terrain, land cover, ...) every dataset is
        # built/read with. Injected (not a global) so tests/entrypoints can supply
        # their own set; adding a kind is one entry in the default registry.
        self.zone_layer_providers = {p.name: p for p in zone_layer_providers}
        # Each configured dataset is always bound to its directory, present or not.
        # A dataset with no directory simply has no data yet, which keeps the read
        # path resilient to an un-initialized root.
        self.datasets = self._bind_datasets()
        # One COG-handle cache shared across all datasets' reads (keyed by path).
        # Injected so the entrypoint can size it from settings; defaulted so
        # tests/CLI can build a SnowDb without wiring one up.
        self.tiff_cache = tiff_cache if tiff_cache is not None else TiffCache()
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

    @staticmethod
    def _index_specs(specs: Iterable[DatasetSpec]) -> dict[str, DatasetSpec]:
        indexed: dict[str, DatasetSpec] = {}
        # Generated response-model names come from spec.model_prefix, and names
        # that differ only by case or -/_ collapse to the same prefix. Reject
        # such collisions here so two datasets can't share an OpenAPI schema name.
        prefixes: dict[str, str] = {}
        for spec in specs:
            if spec.name in indexed:
                raise ValueError(f'Duplicate dataset spec name: {spec.name!r}')
            if spec.model_prefix in prefixes:
                raise ValueError(
                    f'Dataset specs {prefixes[spec.model_prefix]!r} and '
                    f'{spec.name!r} generate the same response-model name '
                    f'{spec.model_prefix!r} (their names differ only by case or '
                    '-/_ separators). Rename one.',
                )
            prefixes[spec.model_prefix] = spec.name
            indexed[spec.name] = spec
        return indexed

    def _bind_datasets(self: Self) -> dict[str, Dataset]:
        return {
            name: Dataset(
                spec,
                # The dataset's directory, resolved from its config's data_dir
                # (or the convention) in __init__.
                self._dataset_paths[name],
                self.zone_layer_providers.values(),
            )
            for name, spec in self._specs.items()
        }

    @classmethod
    def open(
        cls: type[Self],
        path: Path,
        *,
        tiff_cache: TiffCache | None = None,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
    ) -> Self:
        """Open a snowdb from its root config file -- the "from file" constructor.

        ``path`` is the snowdb root directory (holding ``snowdb_conf.json``) or the
        config file itself. The config is *required*: a root without one is not a
        snowdb this version understands, so this raises
        :class:`~snowtool.exceptions.SnowDbConfigError` pointing at ``snowtool
        migration stamp`` (the deliberate no-backwards-compat call -- there is no
        lenient un-initialized read path). The I/O half of construction: it reads +
        parses the root config, then hands it to the constructor.
        """
        path = Path(path)
        config_path = path / CONFIG_FILENAME if path.is_dir() else path
        if not config_path.is_file():
            raise SnowDbConfigError(path)
        config = RootConfig.load(config_path)
        return cls(
            config,
            tiff_cache=tiff_cache,
            zone_layer_providers=zone_layer_providers,
        )

    def available_zones(self: Self) -> dict[str, AvailableZone]:
        """The query-able zone layers across this database's *enabled* providers.

        Keyed ``'<provider>.<layer.key>'`` (e.g. ``'terrain.elevation'``); the union
        over every dataset's enabled providers, so a zone appears only if some
        dataset serves it. Only layers that declare a zoning scheme appear (the
        aspect components, which have no scheme, are excluded). The representation
        of a zone's valid values is its scheme's ``zones()``.
        """
        from snowtool.snowdb.zone_layer import available_zones

        zones: dict[str, AvailableZone] = {}
        for dataset in self.datasets.values():
            zones.update(available_zones(dataset.providers.values()))
        return zones

    # --- global AOI query helpers (drive the aoi/report commands) -------------

    def aoi_paths(self: Self) -> list[Path]:
        """The per-AOI record geojson under ``aois/records/``, sorted by path."""
        if not self.aoi_records_path.is_dir():
            return []
        return sorted(self.aoi_records_path.glob('*.geojson'))

    def aois(self: Self) -> Iterator[AOI]:
        """Parse and yield every stored AOI record."""
        for path in self.aoi_paths():
            yield AOI.from_geojson(path)

    def aoi_triplets(self: Self) -> set[types.StationTriplet]:
        """The station triplets of every stored AOI record (parsed from the id)."""
        return {aoi.station_triplet for aoi in self.aois()}

    def aoi_record_path(self: Self, triplet: types.StationTriplet) -> Path:
        """The canonical ``records/<triplet>.geojson`` path (``:`` -> ``_``)."""
        return self.aoi_records_path / f'{types.triplet_to_stem(triplet)}.geojson'

    def load_aoi(self: Self, triplet: types.StationTriplet) -> AOI:
        """Parse the stored AOI record for ``triplet`` (raises if it is absent)."""
        path = self.aoi_record_path(triplet)
        if not path.is_file():
            raise FileNotFoundError(f'No stored AOI for triplet {triplet!r}.')
        return AOI.from_geojson(path)

    def aoi_index(self: Self) -> AOIIndex:
        """Load the persisted ``index.geojson`` manifest (empty if absent).

        Serves ``aoi list`` without parsing the (large) basin records. The index
        is maintained by import/sync/remove/reindex; run ``aoi reindex`` if the
        ``records/`` dir was edited out of band.
        """
        return AOIIndex.load(self.aoi_index_path)

    def aoi_dataset_coverage(
        self: Self,
        triplet: types.StationTriplet,
        dataset_name: str,
    ) -> Coverage:
        """How fully ``dataset_name``'s grid covers AOI ``triplet``'s basin.

        Computed live from the stored basin geometry (not the cached index), so it
        is authoritative even if the index is stale. Raises if either the AOI or
        the dataset is unknown.
        """
        aoi = self.load_aoi(triplet)
        return dataset_coverage(aoi, self.datasets[dataset_name].coverage_domain)

    def require_aoi_coverage(
        self: Self,
        triplet: types.StationTriplet,
        dataset_name: str,
        *,
        allow_partial: bool = False,
    ) -> Coverage:
        """Query guard: raise unless ``dataset_name`` fully covers AOI ``triplet``.

        The seam a stats/query call uses before reading rasters, closing the
        silent-partial-stats gap. ``allow_partial`` permits a knowingly-clipped
        query over a partially-covered AOI; a wholly off-grid AOI always raises.
        Returns the computed :class:`Coverage` for callers that want to log it.
        """
        coverage = self.aoi_dataset_coverage(triplet, dataset_name)
        require_full_coverage(
            coverage,
            triplet=triplet,
            dataset=dataset_name,
            allow_partial=allow_partial,
        )
        return coverage

    @staticmethod
    def _resolve_variables(
        dataset: Dataset,
        variable_keys: Iterable[str] | None,
    ) -> set[DatasetVariable]:
        """The :class:`DatasetVariable`\\ s named by ``variable_keys``.

        ``None`` (or an empty selection) means every variable the dataset defines;
        an unknown key raises a clean ``ValueError`` listing the choices.
        """
        available = dataset.spec.variables
        keys = None if variable_keys is None else list(variable_keys)
        if not keys:
            return set(available.values())
        resolved: set[DatasetVariable] = set()
        for key in keys:
            try:
                resolved.add(available[key])
            except KeyError as e:
                raise ValueError(
                    f'Unknown variable {key!r} for dataset {dataset.spec.name!r}; '
                    f'available: {", ".join(sorted(available))}.',
                ) from e
        return resolved

    async def zonal_stats(
        self: Self,
        triplet: types.StationTriplet,
        dataset_name: str,
        query: types.DateQuery,
        *,
        variable_keys: Iterable[str] | None = None,
        zone_selections: Sequence[ZoneSelection] = (),
        allow_partial: bool = False,
        max_zone_cells: int | None = None,
    ) -> ZonalStats:
        """Compute zonal statistics for one AOI over one dataset.

        The shared read seam behind the ``query stats`` CLI command (and the future
        HTTP route): it guards coverage, loads the burned AOI raster, resolves the
        requested variables, builds the raster collection for ``query``, and runs
        the crossed-zone reduction. ``variable_keys`` defaults to every variable the
        dataset defines; ``zone_selections`` defaults to none (a whole-basin
        reduction). Raises a clean error when the dataset/variable is unknown, the
        AOI is not covered (:class:`~snowtool.exceptions.AOICoverageError`), or the
        AOI raster has not been rasterized (:class:`FileNotFoundError`).
        """
        from snowtool.snowdb.raster_collection import RasterCollection
        from snowtool.snowdb.zonal_stats import DEFAULT_MAX_ZONE_CELLS, ZonalStats

        dataset = self.datasets[dataset_name]
        # Refuse a silently-clipped result: the AOI must be inside the dataset's
        # served footprint (fully, unless allow_partial), checked before any read.
        self.require_aoi_coverage(triplet, dataset_name, allow_partial=allow_partial)

        variables = self._resolve_variables(dataset, variable_keys)
        aoi_raster = dataset.load_aoi_raster(triplet)
        collection = RasterCollection.from_variables_query(query, variables, dataset)
        return await ZonalStats.calculate(
            aoi_raster,
            collection,
            self.tiff_cache,
            dataset,
            zone_selections,
            max_zone_cells=(
                DEFAULT_MAX_ZONE_CELLS if max_zone_cells is None else max_zone_cells
            ),
        )

    def dump_aoi(self: Self, triplet: types.StationTriplet, dest_dir: Path) -> Path:
        """Copy a stored AOI record out to ``dest_dir`` (round-trip / archive).

        A pure read/export -- it copies a record out without touching the database,
        so it lives on the read side even though the prune cascade
        (:class:`~snowtool.snowdb.manager.SnowDbManager`) also uses it.
        """
        source = self.aoi_record_path(triplet)
        if not source.is_file():
            raise FileNotFoundError(f'No stored AOI for triplet {triplet!r}.')
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copyfile(source, dest)
        return dest

    def __getitem__(self: Self, name: str) -> Dataset:
        return self.datasets[name]

    def __iter__(self: Self) -> Iterator[str]:
        return iter(self.datasets)

    def __contains__(self: Self, name: str) -> bool:
        return name in self.datasets

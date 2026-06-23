"""The snow database root: the global ``aois/`` plus per-dataset ``data/``.

``SnowDb`` is configured with the dataset specs it supports (passed in) and binds
every one of them to its ``data/<name>/`` directory, present on disk or not: a
dataset is defined by its spec, and a missing directory just means it has no data
yet. The read path therefore tolerates an un-initialized root (it serves no data
and logs a warning); :meth:`SnowDb.initialize` -- driven by ``snowtool snowdb
init`` -- is the one place that creates the base layout. It is constructed per
entrypoint (the API builds one at app-lifespan scope, the CLI one per
invocation); the built-in spec set lives in
:data:`snowtool.snowdb.datasets.DEFAULT_DATASET_SPECS`. It also owns the
:class:`~snowtool.snowdb.tiff_cache.TiffCache` shared by all of its datasets'
reads.
"""

from __future__ import annotations

import logging
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from snowtool import types
from snowtool.exceptions import (
    AOIPruneDestinationRequiredError,
    GeoJSONValidationError,
    SnowDbConfigError,
)
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.aoi_index import AOIIndex
from snowtool.snowdb.config import CONFIG_FILENAME, RootConfig
from snowtool.snowdb.coverage import (
    Coverage,
    dataset_coverage,
    require_full_coverage,
)
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.tiff_cache import TiffCache
from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from snowtool.snowdb.raster import AOIRaster
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.zone_layer import (
        AvailableZone,
        ZoneLayerProvider,
        ZoneLayerSource,
    )


logger = logging.getLogger(__name__)


def _combined_extent(
    extents: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Union of ``(west, south, east, north)`` extents."""
    boxes = list(extents)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


@dataclass(frozen=True)
class AOIImportResult:
    """The outcome of an additive ``aoi import``.

    ``imported`` are the triplets written to ``records/`` (polygon-bearing AOIs);
    ``skipped`` are point-only pourpoints (valid pourpoints, not AOIs);
    ``invalid`` pairs each unparseable source path with its error message.
    """

    imported: list[types.StationTriplet]
    skipped: list[types.StationTriplet]
    invalid: list[tuple[Path, str]]


@dataclass(frozen=True)
class AOISyncResult(AOIImportResult):
    """An :class:`AOIImportResult` plus the triplets pruned (or, in a dry run,
    that would be pruned) because they are absent from the synced directory."""

    pruned: list[types.StationTriplet]


@dataclass(frozen=True)
class AOIRasterizeResult:
    """Per (AOI, dataset) rasterize outcomes: ``(triplet, dataset_name)`` pairs
    that were (re)built vs. skipped as already current."""

    built: list[tuple[types.StationTriplet, str]]
    skipped: list[tuple[types.StationTriplet, str]]


class SnowDb:
    def __init__(
        self: Self,
        path: Path,
        specs: Iterable[DatasetSpec],
        *,
        tiff_cache: TiffCache | None = None,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
        zone_layer_sources: dict[str, ZoneLayerSource] | None = None,
    ) -> None:
        self.path = Path(path)
        self.aois_path = self.path / 'aois'
        # Per-AOI source-of-truth geojson live in a records/ subdir; the derived
        # FeatureCollection manifest sits beside it at aois/index.geojson.
        self.aoi_records_path = self.aois_path / 'records'
        self.aoi_index_path = self.aois_path / 'index.geojson'
        self.data_path = self.path / 'data'
        self._specs = self._index_specs(specs)
        # The zone-layer providers (terrain, land cover, ...) every dataset is
        # built/read with. Injected (not a global) so tests/entrypoints can supply
        # their own set; adding a kind is one entry in the default registry.
        self.zone_layer_providers = {p.name: p for p in zone_layer_providers}
        # Datasets are defined by their specs, not by what's on disk: every
        # configured spec is always bound to its data/<name>/ dir, present or
        # not. A dataset with no directory simply has no data yet, which keeps
        # the read path resilient to an un-initialized root.
        self.datasets = self._bind_datasets()
        # One COG-handle cache shared across all datasets' reads (keyed by path).
        # Injected so the entrypoint can size it from settings; defaulted so
        # tests/CLI can build a SnowDb without wiring one up.
        self.tiff_cache = tiff_cache if tiff_cache is not None else TiffCache()
        # The source each provider reads from during generation. A source belongs
        # to the whole database (one source bins into every grid in a single pass),
        # not to any one dataset. Per-provider overrides win (the CLI/tests inject
        # local files to avoid 3DEP/the MRLC download); otherwise the provider's
        # default source is used (3DEP for terrain, the MRLC bundle for land cover),
        # so `init` works out of the box.
        overrides = zone_layer_sources or {}
        self.zone_layer_sources: dict[str, ZoneLayerSource] = {
            name: overrides.get(name) or provider.default_source(self.path)
            for name, provider in self.zone_layer_providers.items()
        }
        self._warn_if_uninitialized()

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
                    "-/_ separators). Rename one.",
                )
            prefixes[spec.model_prefix] = spec.name
            indexed[spec.name] = spec
        return indexed

    def _bind_datasets(self: Self) -> dict[str, Dataset]:
        return {
            name: Dataset(
                spec,
                self.data_path / name,
                self.zone_layer_providers.values(),
            )
            for name, spec in self._specs.items()
        }

    def _missing_base_dirs(self: Self) -> list[Path]:
        """Base directories (``aois/`` + ``data/``) the root lacks."""
        return [p for p in (self.aois_path, self.data_path) if not p.is_dir()]

    def _missing_dirs(self: Self) -> list[Path]:
        """Base/dataset directories the root is expected to have but doesn't."""
        missing = self._missing_base_dirs()
        # Only enumerate per-dataset dirs when data/ exists; a missing data/
        # already implies every dataset dir is absent.
        if self.data_path.is_dir():
            missing.extend(
                dataset.path
                for dataset in self.datasets.values()
                if not dataset.path.is_dir()
            )
        return missing

    def _warn_if_uninitialized(self: Self) -> None:
        missing = self._missing_dirs()
        if missing:
            logger.warning(
                'snowdb at %s is missing expected directories (%s); affected '
                'datasets will serve no data. Run `snowtool snowdb init` to '
                'create the layout.',
                self.path,
                ', '.join(str(p) for p in missing),
            )

    def require_initialized(self: Self) -> Self:
        """Raise unless the root has its base structure (``aois/`` + ``data/``).

        Read paths tolerate a missing layout (they just serve no data), but
        management commands that write call this first so they refuse to operate
        on a root that was never ``snowdb init``-ed rather than silently creating
        the base directories themselves.
        """
        missing = self._missing_base_dirs()
        if missing:
            raise FileNotFoundError(
                f'{self.path} is not an initialized snowdb (missing '
                f'{", ".join(str(p) for p in missing)}). '
                'Run `snowtool snowdb init` first.',
            )
        return self

    @classmethod
    def initialize(
        cls: type[Self],
        path: Path,
        specs: Iterable[DatasetSpec],
        *,
        tiff_cache: TiffCache | None = None,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
        zone_layer_sources: dict[str, ZoneLayerSource] | None = None,
    ) -> Self:
        """Create the base snowdb layout + root config at ``path`` and open it.

        The one entry point that creates the root structure -- the
        ``snowdb_conf.json`` root config, ``aois/``, ``data/``, and a
        ``data/<name>/`` directory per configured spec. No datasets are registered
        in the config: a dataset goes live by adding its link (the config carries
        an empty ``datasets`` list). Other (management) commands may create missing
        dataset dirs but never the base ``aois/``/``data/`` dirs (see
        :meth:`require_initialized`). Idempotent -- an existing config is left as
        is (its creation stamp and any links are preserved), and the result is
        constructed through :meth:`open` so init exercises the same config
        requirement every read does.
        """
        specs = list(specs)
        path = Path(path)
        # aois/ holds the index.geojson manifest; aois/records/ the per-AOI files.
        (path / 'aois' / 'records').mkdir(parents=True, exist_ok=True)
        data_path = path / 'data'
        data_path.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            (data_path / spec.name).mkdir(parents=True, exist_ok=True)
        config_path = path / CONFIG_FILENAME
        if not config_path.is_file():
            RootConfig.create().save(config_path)
        return cls.open(
            path,
            specs,
            tiff_cache=tiff_cache,
            zone_layer_providers=zone_layer_providers,
            zone_layer_sources=zone_layer_sources,
        )

    @classmethod
    def open(
        cls: type[Self],
        path: Path,
        specs: Iterable[DatasetSpec],
        *,
        tiff_cache: TiffCache | None = None,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
        zone_layer_sources: dict[str, ZoneLayerSource] | None = None,
    ) -> Self:
        """Open an existing snowdb from its root config -- the construction seam.

        ``path`` is the snowdb root directory (holding ``snowdb_conf.json``) or the
        config file itself. The config is *required*: a root without it is not a
        snowdb this version understands, so this raises
        :class:`~snowtool.exceptions.SnowDbConfigError` pointing at ``snowtool
        migration stamp`` rather than serving an un-initialized root (the
        deliberate no-backwards-compat call). The config is loaded and validated
        here; the given ``specs`` are then bound to the root exactly as today.
        Following the config's dataset *links* to resolve the specs (instead of
        taking them as an argument) lands in a later phase -- every entrypoint
        already constructs through this one factory, so that change is contained to
        this method.
        """
        path = Path(path)
        config_path = path / CONFIG_FILENAME if path.is_dir() else path
        root = config_path.parent
        if not config_path.is_file():
            raise SnowDbConfigError(path)
        # Parse + validate now (surfaces a malformed/foreign config as a clean
        # error); the links it carries are consumed once link-following lands.
        RootConfig.load(config_path)
        return cls(
            root,
            specs,
            tiff_cache=tiff_cache,
            zone_layer_providers=zone_layer_providers,
            zone_layer_sources=zone_layer_sources,
        )

    def rasterize_aoi(
        self: Self,
        aoi: AOI,
        force: bool = False,
    ) -> dict[str, AOIRaster]:
        """Rasterize a global AOI onto every active dataset's grid.

        AOIs are shared across datasets, but each dataset has its own grid, so an
        AOI must be burned once per dataset (different grids -> different tile
        windows and masks). Returns the resulting AOI raster keyed by dataset
        name.
        """
        return {
            name: dataset.rasterize_aoi(aoi, force=force)
            for name, dataset in self.datasets.items()
        }

    def generate_zone_layers(
        self: Self,
        provider_name: str,
        names: Iterable[str] | None = None,
        *,
        source: ZoneLayerSource | None = None,
        force: bool = False,
        **options: object,
    ) -> dict[str, str]:
        """Generate a provider's zone layers for several datasets in one pass.

        Reads ``source`` (default: this database's resolved source for
        ``provider_name``) once over the combined extent of the selected datasets'
        grids and bins it into all of them -- e.g. terrain's aspect must be computed
        at the source resolution, so sharing the read is the whole point. ``names``
        selects datasets (default: all). ``**options`` carries engine-specific knobs
        (e.g. terrain's ``workers``/``block_size``). Returns each dataset's
        provenance hash, keyed by name.
        """
        from snowtool.snowdb.grid import grid_extent_4326

        provider = self.zone_layer_providers[provider_name]
        selected = (
            list(self.datasets.values())
            if names is None
            else [self.datasets[name] for name in names]
        )
        if not selected:
            return {}

        if source is None:
            source = self.zone_layer_sources[provider_name]
        targets = [ds.zone_target(provider) for ds in selected]
        bounds = _combined_extent(grid_extent_4326(ds.grid) for ds in selected)

        return provider.generate(source, targets, bounds, force=force, **options)

    def available_zones(self: Self) -> dict[str, AvailableZone]:
        """The query-able zone layers across this database's providers.

        Keyed ``'<provider>.<layer.key>'`` (e.g. ``'terrain.elevation'``); only
        layers that declare a zoning scheme appear (the aspect components, which
        have no scheme, are excluded). The representation of a zone's valid values
        is its scheme's ``zones()``.
        """
        from snowtool.snowdb.zone_layer import available_zones

        return available_zones(self.zone_layer_providers.values())

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

    def _stored_triplets(self: Self) -> set[types.StationTriplet]:
        """Stored triplets read straight from record filenames (no geojson parse).

        Record files are written named for the AOI's own triplet, so the filename
        is authoritative -- cheaper than :meth:`aoi_triplets` for set diffs.
        """
        return {types.stem_to_triplet(path.stem) for path in self.aoi_paths()}

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

    def reindex_aois(self: Self) -> AOIIndex:
        """Rebuild ``index.geojson`` from the ``records/`` dir and persist it.

        Coverage is re-derived against every dataset's current grid, so the
        manifest always reflects the live grids (a grid change is picked up by
        re-running this).
        """
        domains = {
            name: ds.coverage_domain for name, ds in self.datasets.items()
        }
        index = AOIIndex.from_records(self.aoi_records_path, domains)
        index.save(self.aoi_index_path)
        return index

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

    # --- AOI import / sync / lifecycle ----------------------------------------

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
    ) -> tuple[
        list[tuple[Path, AOI]],
        list[types.StationTriplet],
        list[tuple[Path, str]],
    ]:
        """Split source paths into importable AOIs, skipped point-only, invalid.

        Pure (no writes): the caller decides whether to persist the result, so a
        dry run and a real run classify identically.
        """
        to_import: list[tuple[Path, AOI]] = []
        skipped: list[types.StationTriplet] = []
        invalid: list[tuple[Path, str]] = []
        for path in sources:
            try:
                aoi = AOI.from_geojson(path)
            except GeoJSONValidationError as e:
                invalid.append((path, str(e)))
                continue
            if aoi.polygon is None:
                # A valid pourpoint, but with no basin it is not an AOI.
                skipped.append(aoi.station_triplet)
                continue
            to_import.append((path, aoi))
        return to_import, skipped, invalid

    def _write_records(self: Self, to_import: Iterable[tuple[Path, AOI]]) -> None:
        """Copy each source geojson verbatim to its canonical record path."""
        self.aoi_records_path.mkdir(parents=True, exist_ok=True)
        for path, aoi in to_import:
            shutil.copyfile(path, self.aoi_record_path(aoi.station_triplet))

    def import_aois(
        self: Self,
        src: Path,
        *,
        dry_run: bool = False,
    ) -> AOIImportResult:
        """Additively import AOI(s) from a file or directory into ``records/``.

        Imports only polygon-bearing pourpoints (skips point-only ones, reports
        unparseable ones); never removes anything. Idempotent: re-importing a
        triplet overwrites its record. Rebuilds the index unless ``dry_run``.
        """
        to_import, skipped, invalid = self._classify_sources(
            self._resolve_sources(src),
        )
        imported = [aoi.station_triplet for _, aoi in to_import]
        if not dry_run:
            self._write_records(to_import)
            self.reindex_aois()
        return AOIImportResult(imported, skipped, invalid)

    def sync_aois(
        self: Self,
        src: Path,
        *,
        prune_to: Path | None = None,
        dry_run: bool = False,
    ) -> AOISyncResult:
        """Mirror a directory into storage: import it, then prune absent records.

        Imports ``src`` (directory only), then removes every stored AOI whose
        triplet is not present in ``src`` -- dumping each to ``prune_to`` first.
        Removal is gated: if any AOI would be pruned and ``prune_to`` is ``None``
        (and not a dry run), raises :class:`AOIPruneDestinationRequiredError` before
        writing anything, so the destructive step is never silent.
        """
        src = Path(src)
        if not src.is_dir():
            raise NotADirectoryError(f'aoi sync requires a directory: {src}')

        to_import, skipped, invalid = self._classify_sources(
            sorted(src.glob('*.geojson')),
        )
        imported = [aoi.station_triplet for _, aoi in to_import]
        # Both AOIs and point-only pourpoints in the source represent triplets the
        # source "has"; only stored triplets absent from that set are pruned.
        source_triplets = set(imported) | set(skipped)
        to_prune = sorted(
            t for t in self._stored_triplets() if t not in source_triplets
        )

        if to_prune and not dry_run and prune_to is None:
            raise AOIPruneDestinationRequiredError(to_prune)

        if not dry_run:
            self._write_records(to_import)
            for triplet in to_prune:
                self._remove_aoi_files(triplet, dump_to=prune_to)
            self.reindex_aois()

        return AOISyncResult(imported, skipped, invalid, to_prune)

    def dump_aoi(self: Self, triplet: types.StationTriplet, dest_dir: Path) -> Path:
        """Copy a stored AOI record out to ``dest_dir`` (round-trip / archive)."""
        source = self.aoi_record_path(triplet)
        if not source.is_file():
            raise FileNotFoundError(f'No stored AOI for triplet {triplet!r}.')
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copyfile(source, dest)
        return dest

    def _remove_aoi_files(
        self: Self,
        triplet: types.StationTriplet,
        *,
        dump_to: Path | None = None,
    ) -> None:
        """Delete a record and cascade to every dataset's burned AOI raster.

        Optionally dumps the record to ``dump_to`` first (the reversible prune
        path). Does not touch the index -- callers reindex once after a batch.
        """
        if dump_to is not None:
            self.dump_aoi(triplet, dump_to)
        self.aoi_record_path(triplet).unlink(missing_ok=True)
        for dataset in self.datasets.values():
            dataset.remove_aoi_raster(triplet)

    def remove_aoi(
        self: Self,
        triplet: types.StationTriplet,
        *,
        dry_run: bool = False,
    ) -> bool:
        """Remove a stored AOI and its per-dataset rasters; True if it existed.

        Cascade-deletes the record plus every ``aoi-rasters/<triplet>.tif`` and
        rebuilds the index. Idempotent: removing an absent AOI is a no-op success.
        """
        existed = self.aoi_record_path(triplet).is_file()
        if not dry_run:
            self._remove_aoi_files(triplet)
            self.reindex_aois()
        return existed

    def rasterize_aois(
        self: Self,
        aois: Iterable[AOI],
        datasets: Iterable[Dataset],
        *,
        rebuild: bool = False,
    ) -> AOIRasterizeResult:
        """Burn each AOI onto each dataset's grid when missing or stale.

        Builds the cartesian product of ``aois`` x ``datasets``, (re)building a
        raster only when absent or its :attr:`AOI.geometry_hash` tag no longer
        matches (``rebuild=True`` forces all). Returns the built vs. skipped
        ``(triplet, dataset_name)`` pairs.
        """
        datasets = list(datasets)
        built: list[tuple[types.StationTriplet, str]] = []
        skipped: list[tuple[types.StationTriplet, str]] = []
        for aoi in aois:
            for dataset in datasets:
                pair = (aoi.station_triplet, dataset.spec.name)
                if dataset.rasterize_aoi_if_needed(aoi, rebuild=rebuild):
                    built.append(pair)
                else:
                    skipped.append(pair)
        return AOIRasterizeResult(built, skipped)

    def __getitem__(self: Self, name: str) -> Dataset:
        return self.datasets[name]

    def __iter__(self: Self) -> Iterator[str]:
        return iter(self.datasets)

    def __contains__(self: Self, name: str) -> bool:
        return name in self.datasets

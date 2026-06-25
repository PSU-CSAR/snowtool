"""The snowdb admin/management layer: every write lives here, not on ``SnowDb``.

:class:`SnowDbManager` *has* a :class:`~snowtool.snowdb.db.SnowDb` (its lean
read/query surface, reachable as :attr:`SnowDbManager.db`) and owns every
operation that mutates the database -- creating the layout, registering datasets,
importing/syncing/removing AOIs, rasterizing them, and generating zone layers.
The read path (the FastAPI app) builds only a :class:`SnowDb`; the CLI's write
commands and library admin code build a manager. "The management layer has a
snowdb, not the other way around."
"""

from __future__ import annotations

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
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    PathDatasetLink,
    RootConfig,
)
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.raster import AOIRaster
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.tiff_cache import TiffCache
    from snowtool.snowdb.zone_layer import ZoneLayerProvider, ZoneLayerSource


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


class SnowDbManager:
    """Owns every write against a held :class:`SnowDb` (its read/query surface).

    Built around an already-constructed :class:`SnowDb` (reachable as
    :attr:`db`); :meth:`open` and :meth:`initialize` are the convenience
    constructors that build the read database (or its layout) and wrap it.
    """

    def __init__(self: Self, db: SnowDb) -> None:
        self.db = db

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
        """Open the read :class:`SnowDb` at ``path`` and wrap it in a manager."""
        return cls(
            SnowDb.open(
                path,
                tiff_cache=tiff_cache,
                zone_layer_providers=zone_layer_providers,
            ),
        )

    @classmethod
    def initialize(
        cls: type[Self],
        path: Path,
        specs: Iterable[DatasetSpec] = (),
        *,
        tiff_cache: TiffCache | None = None,
        zone_layer_providers: Iterable[ZoneLayerProvider] = (
            DEFAULT_ZONE_LAYER_PROVIDERS
        ),
    ) -> Self:
        """Create the base snowdb layout + an empty root config at ``path``.

        The one entry point that creates the root structure -- the
        ``snowdb_conf.json`` root config (with *no* datasets registered; a dataset
        goes live only by adding it), ``aois/``, ``data/``, and a ``data/<name>/``
        directory per ``specs`` entry (a convenience for staging; the CLI ``init``
        passes none). Idempotent: an existing config is loaded and left as is (its
        creation stamp and datasets preserved). Returns a manager over the root --
        its read database is empty unless datasets were already registered.
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
        if config_path.is_file():
            config = RootConfig.load(config_path)
        else:
            config = RootConfig.create()
            config.save(config_path)
        return cls(
            SnowDb(
                config,
                tiff_cache=tiff_cache,
                zone_layer_providers=zone_layer_providers,
            ),
        )

    def _read_root_config(self: Self) -> RootConfig:
        """Load this root's on-disk config (raises if it is absent)."""
        config_path = self.db.config_path
        if config_path is None or not config_path.is_file():
            raise SnowDbConfigError(self.db.path)
        return RootConfig.load(config_path)

    def register_dataset(
        self: Self,
        name: str,
        dataset_config_path: Path,
        *,
        link_type: str = 'path',
    ) -> RootConfig:
        """Register a dataset link in the on-disk root config (idempotent).

        Writes ``datasets[name]`` -> a link at ``dataset_config_path``, stored
        relative to the root when the config lives under the tree (a relocatable
        tree) and absolute otherwise (a staged-elsewhere dataset). Re-registering a
        name overwrites its link. Returns the updated config. Going live still
        needs an ``aoi reindex`` + restart -- this only records the registration.
        """
        if link_type != 'path':
            raise ValueError(f'unknown dataset link type: {link_type!r}')
        config = self._read_root_config()
        config_path = self.db.config_path
        if config_path is None:  # pragma: no cover - _read_root_config guarantees it
            raise SnowDbConfigError(self.db.path)
        dataset_config_path = Path(dataset_config_path).resolve()
        root = config_path.parent.resolve()
        # Relative when under the tree (keeps the tree relocatable); absolute when
        # the dataset is staged elsewhere.
        if dataset_config_path.is_relative_to(root):
            link = dataset_config_path.relative_to(root).as_posix()
        else:
            link = str(dataset_config_path)
        config.datasets[name] = PathDatasetLink(path=link)
        config.save(config_path)
        return config

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
            for name, dataset in self.db.datasets.items()
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
        selects datasets (default: all); either way only the datasets that *enable*
        ``provider_name`` are targeted (the rest have no such zone layer).
        ``**options`` carries engine-specific knobs (e.g. terrain's
        ``workers``/``block_size``). Returns each generated dataset's provenance
        hash, keyed by name.
        """
        from snowtool.snowdb.grid import grid_extent_4326

        provider = self.db.zone_layer_providers[provider_name]
        candidates = (
            list(self.db.datasets.values())
            if names is None
            else [self.db.datasets[name] for name in names]
        )
        # Only datasets whose config enables this provider have the layer to build.
        selected = [ds for ds in candidates if provider_name in ds.providers]
        if not selected:
            return {}

        if source is None:
            source = self.db.zone_layer_sources[provider_name]
        targets = [ds.zone_target(provider) for ds in selected]
        bounds = _combined_extent(grid_extent_4326(ds.grid) for ds in selected)

        return provider.generate(source, targets, bounds, force=force, **options)

    # --- AOI import / sync / lifecycle ----------------------------------------

    def reindex_aois(self: Self) -> AOIIndex:
        """Rebuild ``index.geojson`` from the ``records/`` dir and persist it.

        Coverage is re-derived against every dataset's current grid, so the
        manifest always reflects the live grids (a grid change is picked up by
        re-running this).
        """
        domains = {name: ds.coverage_domain for name, ds in self.db.datasets.items()}
        index = AOIIndex.from_records(self.db.aoi_records_path, domains)
        index.save(self.db.aoi_index_path)
        return index

    def _stored_triplets(self: Self) -> set[types.StationTriplet]:
        """Stored triplets read straight from record filenames (no geojson parse).

        Record files are written named for the AOI's own triplet, so the filename
        is authoritative -- cheaper than parsing every record for set diffs.
        """
        return {types.stem_to_triplet(path.stem) for path in self.db.aoi_paths()}

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
        self.db.aoi_records_path.mkdir(parents=True, exist_ok=True)
        for path, aoi in to_import:
            shutil.copyfile(path, self.db.aoi_record_path(aoi.station_triplet))

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
            self.db.dump_aoi(triplet, dump_to)
        self.db.aoi_record_path(triplet).unlink(missing_ok=True)
        for dataset in self.db.datasets.values():
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
        existed = self.db.aoi_record_path(triplet).is_file()
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

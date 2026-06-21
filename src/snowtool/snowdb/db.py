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

from pathlib import Path
from typing import TYPE_CHECKING, Self

from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.tiff_cache import TiffCache

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from snowtool.snowdb.aoi import AOI
    from snowtool.snowdb.raster import AOIRaster
    from snowtool.snowdb.spec import DatasetSpec


logger = logging.getLogger(__name__)


class SnowDb:
    def __init__(
        self: Self,
        path: Path,
        specs: Iterable[DatasetSpec],
        *,
        tiff_cache: TiffCache | None = None,
    ) -> None:
        self.path = Path(path)
        self.aois_path = self.path / 'aois'
        self.data_path = self.path / 'data'
        self._specs = self._index_specs(specs)
        # Datasets are defined by their specs, not by what's on disk: every
        # configured spec is always bound to its data/<name>/ dir, present or
        # not. A dataset with no directory simply has no data yet, which keeps
        # the read path resilient to an un-initialized root.
        self.datasets = self._bind_datasets()
        # One COG-handle cache shared across all datasets' reads (keyed by path).
        # Injected so the entrypoint can size it from settings; defaulted so
        # tests/CLI can build a SnowDb without wiring one up.
        self.tiff_cache = tiff_cache if tiff_cache is not None else TiffCache()
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
            name: Dataset(spec, self.data_path / name)
            for name, spec in self._specs.items()
        }

    def _missing_dirs(self: Self) -> list[Path]:
        """Base/dataset directories the root is expected to have but doesn't."""
        missing = [p for p in (self.aois_path, self.data_path) if not p.is_dir()]
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
        missing = [p for p in (self.aois_path, self.data_path) if not p.is_dir()]
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
    ) -> Self:
        """Create the base snowdb layout at ``path`` and return a SnowDb over it.

        The one entry point that creates the root structure -- ``aois/``,
        ``data/``, and a ``data/<name>/`` directory per configured spec. Other
        (management) commands may create missing dataset dirs but never the base
        ``aois/``/``data/`` dirs (see :meth:`require_initialized`). Idempotent.
        """
        specs = list(specs)
        path = Path(path)
        (path / 'aois').mkdir(parents=True, exist_ok=True)
        data_path = path / 'data'
        data_path.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            (data_path / spec.name).mkdir(parents=True, exist_ok=True)
        return cls(path, specs, tiff_cache=tiff_cache)

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

    def __getitem__(self: Self, name: str) -> Dataset:
        return self.datasets[name]

    def __iter__(self: Self) -> Iterator[str]:
        return iter(self.datasets)

    def __contains__(self: Self, name: str) -> bool:
        return name in self.datasets

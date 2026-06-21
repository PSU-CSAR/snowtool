"""The snow database root: the global ``aois/`` plus per-dataset ``data/``.

``SnowDb`` is configured with the dataset specs it supports (passed in) and
discovers which of them are present under ``data/`` by binding each directory to
its :class:`DatasetSpec`. It is constructed per entrypoint (the API builds one at
app-lifespan scope, the CLI one per invocation); the built-in spec set lives in
:data:`snowtool.snowdb.datasets.DEFAULT_DATASET_SPECS`. It also owns the
:class:`~snowtool.snowdb.tiff_cache.TiffCache` shared by all of its datasets'
reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.tiff_cache import TiffCache

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from snowtool.snowdb.aoi import AOI
    from snowtool.snowdb.raster import AOIRaster
    from snowtool.snowdb.spec import DatasetSpec


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
        self.datasets = self._discover()
        # One COG-handle cache shared across all datasets' reads (keyed by path).
        # Injected so the entrypoint can size it from settings; defaulted so
        # tests/CLI can build a SnowDb without wiring one up.
        self.tiff_cache = tiff_cache if tiff_cache is not None else TiffCache()

    @staticmethod
    def _index_specs(specs: Iterable[DatasetSpec]) -> dict[str, DatasetSpec]:
        indexed: dict[str, DatasetSpec] = {}
        for spec in specs:
            if spec.name in indexed:
                raise ValueError(f'Duplicate dataset spec name: {spec.name!r}')
            indexed[spec.name] = spec
        return indexed

    def _discover(self: Self) -> dict[str, Dataset]:
        if not self.data_path.is_dir():
            raise FileNotFoundError(
                f'No data directory in snow database: {self.data_path}',
            )

        datasets: dict[str, Dataset] = {}
        for entry in sorted(self.data_path.iterdir()):
            # Skip stray files and hidden entries (e.g. macOS .DS_Store).
            if not entry.is_dir() or entry.name.startswith('.'):
                continue

            try:
                spec = self._specs[entry.name]
            except KeyError as e:
                raise ValueError(
                    f'Unknown dataset directory {entry.name!r} in {self.data_path}: '
                    f'not in the configured specs ({sorted(self._specs)}).',
                ) from e

            datasets[entry.name] = Dataset(spec, entry)

        return datasets

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

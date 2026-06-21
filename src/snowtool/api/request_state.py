from collections.abc import Iterator, Mapping
from typing import Any, Self

from fastapi import FastAPI

from snowtool.settings import Settings
from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.tiff_cache import TiffCache


class RequestState(Mapping):
    def __init__(self, app: FastAPI, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        # One SnowDb for the app's lifetime: all configured datasets, their
        # grids, generated response models, and the shared COG-handle cache
        # (sized from settings), discovered under the configured root. Exposed on
        # request.state for the routers; the read path uses snowdb.tiff_cache.
        self.snowdb = SnowDb(
            settings.snowdb_path,
            DEFAULT_DATASET_SPECS,
            tiff_cache=TiffCache(settings.tiff_cache_size),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

    def __getitem__(self, key) -> Any:
        return self.__dict__[key]

    def __iter__(self) -> Iterator[str]:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return len(self.__dict__)

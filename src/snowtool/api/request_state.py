from collections.abc import Iterator, Mapping
from typing import Any, Self

from fastapi import FastAPI


from .settings import Settings


class RequestState(Mapping):
    def __init__(self, app: FastAPI, settings: Settings) -> None:
        self.app = app
        self.settings = settings

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

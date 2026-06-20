"""A bounded, async LRU cache of open async-tiff ``TIFF`` handles.

The read path (zonal stats) fans tile reads across many COGs with
``asyncio.gather``. Re-opening a COG (parsing its IFD metadata) for every tile
would be wasteful, so handles are cached and reused. ``LocalStore`` holds no
file descriptors, so the bound is purely about retained IFD metadata.

The cache is an :func:`async_lru.alru_cache` built fresh in ``__init__`` rather
than a module-level decorator, so each instance owns an independent cache.
Create one per :class:`~snowtool.api.request_state.RequestState`. ``alru_cache``
dedupes concurrent gets for a cold key (the first opens; the rest await the same
in-flight task), bounds the cache to ``maxsize`` entries, and does not cache a
failed open. Its in-flight tasks are bound to the event loop that first awaits a
get, so a single instance must only be used from one event loop.
"""

from __future__ import annotations

from pathlib import Path

from async_lru import alru_cache
from async_tiff import TIFF
from async_tiff.store import LocalStore


class TiffCache:
    def __init__(self, maxsize: int) -> None:
        self._cached_open = alru_cache(maxsize=max(1, maxsize))(self._open)

    def __len__(self) -> int:
        return self._cached_open.cache_info().currsize

    async def get(self, path: Path | str) -> TIFF:
        """Return an open ``TIFF`` for ``path``, opening (once) on a miss."""
        return await self._cached_open(Path(path))

    @staticmethod
    async def _open(path: Path) -> TIFF:
        store = LocalStore(prefix=str(path.parent))
        return await TIFF.open(path.name, store=store)

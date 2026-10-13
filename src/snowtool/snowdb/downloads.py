from __future__ import annotations

import calendar

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar, Protocol, Self


@dataclass
class DownloadResult:
    status: bool
    detail: str = ''


class BaseUrl(ABC):
    @classmethod
    def _build_dest(cls, import_path: str | Path) -> Path:
        return Path(import_path)

    @classmethod
    @abstractmethod
    def _for_date(cls, target_date: date) -> Self:
        raise NotImplementedError

    @abstractmethod
    def _iter_downloads(self) -> Iterator[tuple[tuple[str, ...], Path]]:
        raise NotImplementedError


@dataclass
class INSTARRUrls(BaseUrl):
    urls: dict[str, Path]

    TILES: ClassVar[tuple[str, ...]] = (
        'h08v04',
        'h08v05',
        'h09v04',
        'h09v05',
        'h10v04',
    )
    BASE_URL: ClassVar[str] = (
        'ftp://dtn.rc.colorado.edu/shares/snow-today/gridded_data/SPIRES_NRT_V01/{tile}/{year}/SPIRES_NRT_{tile}_MOD09GA061_{year}{month}{day}_V1.0.nc'
    )

    @classmethod
    def _for_date(cls, target_date: date) -> INSTARRUrls:
        tile_downloads = {}
        for tile in cls.TILES:
            download_url = cls.BASE_URL.format(
                tile=tile,
                year=target_date.year,
                month=f'{target_date.month:02d}',
                day=f'{target_date.day:02d}',
            )
            dest = cls._build_dest(
                f'{tile}/{target_date.year}/{target_date.month:02d}/',
            )
            tile_downloads[download_url] = dest
        return cls(
            tile_downloads,
        )

    def _iter_downloads(self) -> Iterator[tuple[tuple[str, ...], Path]]:
        for url, dest in self.urls.items():
            yield (url,), dest


@dataclass
class SWANNUrls(BaseUrl):
    urls: tuple[str, ...]
    dest: Path

    BASE_URL: ClassVar[str] = (
        'https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{water_year}/UA_SWE_Depth_800m_v1_{year}{month}{day}_{qualifier}.nc'
    )

    # We don't know exactly what version of a daily upload UA will provide,
    # so we need to iterate over every possible qualifier to get what we need
    # Also useful for when snowtool download retry attempts to grab any missing
    # uploads that only exist as provisional/stable versions
    QUALIFIERS: ClassVar[tuple[str, ...]] = ('early', 'provisional', 'stable')

    @staticmethod
    def _water_year(d: date) -> int:
        """
        _water_year returns the water year for a given date.
        Water year N runs from Oct 1 of year (N-1) through Sep 30 of year N.

        Inputs:
            d: date to check
        Outputs:
            Water Year
        """
        return d.year + 1 if d.month >= 10 else d.year

    @classmethod
    def _for_date(cls, target_date: date) -> SWANNUrls:
        swann_urls: tuple[str, ...] = ()
        wy = cls._water_year(target_date)
        for qual in cls.QUALIFIERS:
            swann_urls += (
                cls.BASE_URL.format(
                    water_year=wy,
                    year=target_date.year,
                    month=f'{target_date.month:02d}',
                    day=f'{target_date.day:02d}',
                    qualifier=qual,
                ),
            )
        return cls(
            urls=swann_urls,
            dest=cls._build_dest(
                f'{wy!s}/{target_date.month:02d}/',
            ),
        )

    def _iter_downloads(self) -> Iterator[tuple[tuple[str, ...], Path]]:
        yield self.urls, self.dest


@dataclass
class SNODASUrl(BaseUrl):
    url: str
    dest: Path

    BASE_URL: ClassVar[str] = (
        'https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{year}/{month}_{month_abbr}/SNODAS_{year}{month}{day}.tar'
    )

    @classmethod
    def _for_date(cls, target_date: date) -> SNODASUrl:
        download_url = cls.BASE_URL.format(
            year=target_date.year,
            month=f'{target_date.month:02d}',
            month_abbr=calendar.month_abbr[target_date.month],
            day=f'{target_date.day:02d}',
        )
        return cls(
            url=download_url,
            dest=cls._build_dest(
                f'{target_date.year}/{target_date.month:02d}',
            ),
        )

    def _iter_downloads(self) -> Iterator[tuple[tuple[str, ...], Path]]:
        yield (self.url,), self.dest


class StreamingResponse(Protocol):
    """The slice of a streamed response ``_get_file`` consumes."""

    def raise_for_status(self) -> object: ...

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class Downloader(Protocol):
    """The slice of ``curl_cffi.requests.Session`` that ``_get_file`` needs.

    Declared as a protocol so tests can inject a fake transport through the
    ``session`` seam rather than patching the module or reaching the network.
    """

    def get(
        self,
        url: str,
        *,
        impersonate: str,
        stream: bool,
    ) -> StreamingResponse: ...

    def close(self) -> None: ...

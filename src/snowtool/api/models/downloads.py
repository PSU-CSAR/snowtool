from __future__ import annotations

import calendar

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar, Self


class BaseUrl(ABC):
    BASE_DEST: ClassVar[str] = '/d/projects/gisdata/{source}/unprocessed/{import_path}'

    @classmethod
    def _build_dest(cls, source: str, import_path: str | Path) -> Path:
        return Path(cls.BASE_DEST.format(source=source, import_path=import_path))

    @classmethod
    @abstractmethod
    def _for_date(cls, target_date: date) -> Self:
        raise NotImplementedError

    @abstractmethod
    def _iter_downloads(self) -> Iterator[tuple[str, Path]]:
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
                day=f'{target_date.day}',
            )
            dest = cls._build_dest(
                'instarr',
                f'{tile}/{target_date.year}/{target_date.month:02d}/',
            )
            tile_downloads[download_url] = dest
        return cls(
            tile_downloads,
        )

    def _iter_downloads(self) -> Iterator[tuple[str, Path]]:
        yield from self.urls.items()


@dataclass
class SWANNUrl(BaseUrl):
    url: str
    dest: Path
    qualifier: str

    BASE_URL: ClassVar[str] = (
        'https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{year}/UA_SWE_Depth_800m_v1_{year}{month}{day}_{qualifier}.nc'
    )

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
    def _for_date(cls, target_date: date) -> SWANNUrl:
        qualifier = cls._get_qualifier(target_date)
        wy = cls._water_year(target_date)
        download_url = cls.BASE_URL.format(
            year=wy,
            month=f'{target_date.month:02d}',
            day=f'{target_date.day:02d}',
            qualifier=qualifier,
        )
        return cls(
            url=download_url,
            dest=cls._build_dest('swann', f'{wy!s}/{target_date.month:02d}/'),
            qualifier=qualifier,
        )

    @staticmethod
    def _get_qualifier(d: date):
        """
        _swann_qualifier_for_date will determine which SWANN qualifier
        to request for a given date, per the UA readme's documented schedule.
        For backfill purposes, we treat 'early' as unavailable beyond the current month.
        Inputs:
            d: date to check against
            today: Todays date to compare with
        Outputs:
            Qualifier based on time difference between files
        """
        age_days = (date.today() - d).days  # noqa: DTZ011
        if age_days < 0:
            raise ValueError(f'Cannot backfill a future date: {d}')
        if age_days <= 31:
            return 'early'
        if age_days <= 186:
            return 'provisional'
        return 'stable'

    def _iter_downloads(self) -> Iterator[tuple[str, Path]]:
        yield self.url, self.dest


@dataclass
class SNODASUrl(BaseUrl):
    url: str
    dest: Path

    BASE_URL: ClassVar[str] = (
        'https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{year}/{month}_{month_abbr}/SNODAS_{year}{month}{day}.tar'
    )
    BASE_DEST: ClassVar[str] = '/d/projects/gisdata/snodas/in_db/masked/{import_path}/'

    @classmethod
    def _for_date(cls, target_date: date) -> SNODASUrl:
        download_url = cls.BASE_URL.format(
            year=target_date.year,
            month=f'{target_date.month:02d}',
            month_abbr=calendar.month_abbr[target_date.month],
            day=f'{target_date.day:02d}',
        )
        dest = Path(
            cls.BASE_DEST.format(
                import_path=f'{target_date.year}/{target_date.month:02d}',
            ),
        )
        return cls(
            url=download_url,
            dest=dest,
        )

    def _iter_downloads(self) -> Iterator[tuple[str, Path]]:
        yield self.url, self.dest

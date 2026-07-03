from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar


@dataclass
class DownloadResult:
    url: str
    dest: Path
    status: str  # "downloaded", "missing", "error", "verify_failed"
    detail: str = ''


@dataclass(frozen=True)
class BaseUrl:
    url: str
    dest: str

    BASE_DEST: str = '/d/projects/gisdata/{source}/unprocessed/{import_path}'

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
    def get_file(cls, url_type: str):
        raise NotImplementedError


@dataclass(frozen=True)
class INSTARRUrls(BaseUrl):
    TILES: ClassVar[tuple[str, ...]] = (
        'h08v04',
        'h08v05',
        'h09v04',
        'h09v05',
        'h10v04',
    )
    BASE_URL: str = 'ftp://dtn.rc.colorado.edu/shares/snow-today/gridded_data/SPIRES_NRT_V01/{tile}/{year}/SPIRES_NRT_{tile}_MOD09GA061_{year}{month}{day}_V1.0.nc'

    @classmethod
    def _for_date(cls, target_date: date) -> INSTARRUrls:
        wy = cls._water_year(target_date)
        download_url = cls.BASE_URL.format(
            year=wy,
            month=target_date.month,
            day=target_date.day,
        )
        return cls(
            url=download_url,
            dest=cls.BASE_DEST.format('instarr', wy),
        )

    @classmethod
    def _build_(cls):
        for tile in cls.TILES:
            return tile
        return None

    # Stuff to build INSTARR URL


@dataclass(frozen=True)
class SWANNUrl(BaseUrl):
    BASE_URL: str = 'https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{year}/UA_SWE_Depth_800m_v1_{year}{month}{day}_{qualifier}.nc'
    qualifier: str

    @classmethod
    def _for_date(cls, target_date: date) -> SWANNUrl:
        qualifier = cls._get_qualifier(target_date)
        wy = cls._water_year(target_date)
        download_url = cls.BASE_URL.format(
            wy,
            wy,
            target_date.month,
            target_date.day,
            qualifier,
        )
        import_path = f'{wy!s}/{target_date.month!s}/'
        return cls(
            url=download_url,
            dest=cls.BASE_DEST.format('swann', import_path),
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


@dataclass(frozen=True)
class SNODASUrl(BaseUrl):
    url: str
    dest: Path

    # Stuff to build SNODAS URL
